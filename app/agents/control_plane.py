"""Agent Control Plane and Autonomous Application Queue (Phase 6.1).

Coordinates execution policies, limits, queue governance, freshness enforcement,
rate control, failure classification, and scheduling hooks above JobApplicationOrchestrator.
Does NOT duplicate discovery, matching, or application automation logic.
"""

import asyncio
from datetime import datetime, timezone, timedelta
import logging
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple
from uuid import UUID

from app.agents.job_application_agent import JobApplicationOrchestrator
from app.agents.lock_manager import JobLockManager
from app.agents.models import (
    AgentRun,
    AgentRunRequest,
    AgentRunResponse,
    AgentRunStatus,
    FailureClassification,
    OrchestratorState,
    QueueItem,
    QueueStatus,
    SingleJobApplyResponse,
)
from app.config.settings import Settings, get_settings
from app.database.repositories.application_repository import ApplicationRepository
from app.database.repositories.job_repository import JobRepository
from app.database.repositories.match_repository import MatchRepository
from app.discovery.profile_discovery_service import is_valid_canonical_job_url
from app.models.job import ApplicationMethod, AvailabilityStatus, MatchedJobItem
from app.profile.service import ProfileService

logger = logging.getLogger(__name__)


# =============================================================================
# 1. SCHEDULER ABSTRACTION (Future-Safe, Inactive on Startup)
# =============================================================================

class BaseSchedulerHook(Protocol):
    """Protocol defining future-safe scheduling hooks for autonomous agent runs."""

    def is_active(self) -> bool:
        """Check if background scheduling is currently running."""
        ...

    async def schedule_run(self, request: AgentRunRequest, cron_expression: Optional[str] = None) -> str:
        """Register a scheduled job application run."""
        ...

    async def cancel_schedule(self, schedule_id: str) -> bool:
        """Cancel a registered schedule."""
        ...


class NoOpScheduler:
    """Default no-op scheduler implementation. Keeps execution strictly on-demand."""

    def __init__(self) -> None:
        self._active = False

    def is_active(self) -> bool:
        return False

    async def schedule_run(self, request: AgentRunRequest, cron_expression: Optional[str] = None) -> str:
        logger.info("Scheduler hook called (NoOpScheduler: execution is on-demand).")
        return "noop_schedule_id"

    async def cancel_schedule(self, schedule_id: str) -> bool:
        return True


# =============================================================================
# 2. FAILURE CLASSIFIER
# =============================================================================

class FailureClassifier:
    """Categorizes application outcomes into standard Phase 6.1 failure classifications."""

    @staticmethod
    def classify_outcome(
        status: str,
        automation_state: Optional[str] = None,
        error: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Optional[FailureClassification]:
        s = (status or "").upper()
        state = (automation_state or "").upper()
        err = (error or "").lower()

        # Success / Submission Ready is not a failure
        if s in ("SUCCESS", "SUBMISSION_READY", "SUBMITTED") or state in ("SUBMISSION_READY", "SUBMITTED"):
            return None

        # Stale Job
        if "STALE" in s or "STALE" in state or "stale" in err:
            return FailureClassification.STALE_JOB

        # External Apply
        if "EXTERNAL_APPLY" in s or "EXTERNAL_APPLY" in state or "external" in err:
            return FailureClassification.EXTERNAL_APPLY

        # Already Applied
        if "ALREADY_APPLIED" in s or "ALREADY_APPLIED" in state or "already applied" in err:
            return FailureClassification.ALREADY_APPLIED

        # Job Closed / Unavailable / Expired
        if "UNAVAILABLE" in s or "UNAVAILABLE" in state or "EXPIRED" in s or "EXPIRED" in state or "closed" in err or "expired" in err:
            return FailureClassification.JOB_UNAVAILABLE

        # User Intervention (CAPTCHA, MFA, Login, Cloudflare)
        if (
            "MANUAL_ACTION" in s
            or "CAPTCHA" in state
            or "CHALLENGE" in state
            or "LOGIN" in state
            or "MFA" in state
            or "verification" in err
            or "captcha" in err
        ):
            return FailureClassification.USER_ACTION_REQUIRED

        # Transient Failure (timeouts, disconnects, network errors, temporary focus losses)
        if (
            "TIMEOUT" in state
            or "DISCONNECT" in state
            or "timeout" in err
            or "connection" in err
            or "stale confirmation page" in err
            or "network" in err
            or "timed out" in err
        ):
            return FailureClassification.TRANSIENT_FAILURE

        # Permanent Failure (unsupported platform, invalid URL, fatal DOM mismatch)
        return FailureClassification.PERMANENT_FAILURE


# =============================================================================
# 3. DAILY LIMITS GUARD (Fail-Closed)
# =============================================================================

class DailyLimitsGuard:
    """Enforces daily global and platform-specific application caps with fail-closed semantics."""

    def __init__(
        self,
        application_repo: Optional[ApplicationRepository] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.app_repo = application_repo or ApplicationRepository()
        self.settings = settings or get_settings()

    def get_today_start_utc(self) -> datetime:
        """Return the start of the current day in UTC."""
        now = datetime.now(timezone.utc)
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    def get_daily_counts(self, profile_id: Optional[UUID] = None) -> Dict[str, int]:
        """Fetch today's total and platform application counts from database."""
        since = self.get_today_start_utc()
        total_count = self.app_repo.count_daily_applications(profile_id=profile_id, since=since)
        indeed_count = self.app_repo.count_daily_applications(profile_id=profile_id, platform="indeed", since=since)
        glassdoor_count = self.app_repo.count_daily_applications(profile_id=profile_id, platform="glassdoor", since=since)
        return {
            "total": total_count,
            "indeed": indeed_count,
            "glassdoor": glassdoor_count,
        }

    def check_can_start_run(
        self,
        profile_id: Optional[UUID] = None,
        max_daily_override: Optional[int] = None,
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Verify if global daily application limit allows starting a new application cycle."""
        counts = self.get_daily_counts(profile_id=profile_id)
        limit_total = max_daily_override or getattr(self.settings, "MAX_APPLICATIONS_PER_DAY", 10)

        if counts["total"] >= limit_total:
            reason = f"Daily application limit reached ({counts['total']}/{limit_total}). Fail-closed: stopping run."
            logger.warning(reason)
            return False, reason, {
                "limit_type": "GLOBAL_DAILY_LIMIT",
                "current": counts["total"],
                "limit": limit_total,
            }
        return True, None, {}

    def check_platform_capacity(
        self,
        platform: str,
        profile_id: Optional[UUID] = None,
        in_run_platform_counts: Optional[Dict[str, int]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Verify if a specific platform has remaining capacity for today."""
        p = (platform or "").lower().strip()
        counts = self.get_daily_counts(profile_id=profile_id)
        in_run_count = (in_run_platform_counts or {}).get(p, 0)

        if p == "indeed":
            limit = getattr(self.settings, "MAX_INDEED_APPLICATIONS_PER_DAY", 5)
        elif p == "glassdoor":
            limit = getattr(self.settings, "MAX_GLASSDOOR_APPLICATIONS_PER_DAY", 5)
        else:
            limit = 5

        total_for_platform = counts.get(p, 0) + in_run_count
        if total_for_platform >= limit:
            reason = f"Platform daily limit reached for '{p}' ({total_for_platform}/{limit})."
            return False, reason

        return True, None


# =============================================================================
# 4. FRESHNESS POLICY
# =============================================================================

class FreshnessPolicy:
    """Evaluates posting age and marks stale jobs without deleting them from database."""

    def __init__(self, max_job_age_days: int = 7) -> None:
        self.max_job_age_days = max_job_age_days

    def is_stale(self, posted_at: Optional[datetime]) -> Tuple[bool, Optional[int]]:
        """Check if posted_at timestamp is older than max_job_age_days."""
        if not posted_at:
            # If posting date is unknown, do not mark stale
            return False, None

        now = datetime.now(timezone.utc)
        if posted_at.tzinfo is None:
            posted_at = posted_at.replace(tzinfo=timezone.utc)

        age = now - posted_at
        age_days = age.days
        if age_days > self.max_job_age_days:
            return True, age_days
        return False, age_days


# =============================================================================
# 5. MULTI-FACTOR QUEUE MANAGER & PRIORITY RANKER
# =============================================================================

class ApplicationQueueManager:
    """Filters, deduplicates, and ranks jobs using multi-factor selection criteria."""

    def __init__(
        self,
        job_repo: Optional[JobRepository] = None,
        app_repo: Optional[ApplicationRepository] = None,
        profile_service: Optional[ProfileService] = None,
    ) -> None:
        self.job_repo = job_repo or JobRepository()
        self.app_repo = app_repo or ApplicationRepository()
        self.profile_service = profile_service or ProfileService()

    def build_and_prioritize_queue(
        self,
        top_matches: List[MatchedJobItem],
        config: AgentRunRequest,
        daily_limits: DailyLimitsGuard,
        lock_manager: JobLockManager,
        run_id: UUID,
    ) -> Tuple[List[QueueItem], List[SingleJobApplyResponse], Dict[str, Any]]:
        """Filter and rank queue using: match_score, role relevance, preferred location, and freshness."""
        queue: List[QueueItem] = []
        skipped_items: List[SingleJobApplyResponse] = []
        limits_hit: Dict[str, Any] = {}

        seen_job_ids: Set[UUID] = set()

        # Load profile preferences for priority ranking
        candidate_profile = None
        target_roles: List[str] = []
        preferred_locations: List[str] = []
        try:
            if hasattr(self.profile_service, "get_profile"):
                candidate_profile = self.profile_service.get_profile(config.profile_id)
            elif hasattr(self.profile_service, "get_profile_data"):
                candidate_profile = self.profile_service.get_profile_data()
            if candidate_profile:
                target_roles = [r.lower() for r in (getattr(candidate_profile, "target_roles", None) or [])]
                preferred_locations = [l.lower() for l in (getattr(candidate_profile, "preferred_locations", None) or [])]
        except Exception:
            pass

        freshness_policy = FreshnessPolicy(max_job_age_days=config.max_job_age_days)

        # Pre-fetch database application history
        submitted_ids: Set[UUID] = set()
        if config.profile_id:
            existing = self.app_repo.get_applications_for_jobs(
                [m.job_id for m in top_matches], profile_id=config.profile_id
            )
            submitted_ids = {
                a.job_id for a in existing
                if a.status in ("submitted", "APPLICATION_SUBMITTED", "submission_ready")
            }

        # Track jobs queued per platform to enforce max_jobs_per_platform and daily platform limits
        queued_per_platform: Dict[str, int] = {}

        for m in top_matches:
            jid = m.job_id
            platform = (m.source or "").lower().strip()

            if jid in seen_job_ids:
                continue

            # 1. Minimum Match Score Filter
            if m.final_score < config.minimum_match_score:
                continue

            # 2. Match Decision Threshold
            if (m.decision or "").lower() in ("reject", "low_match"):
                continue

            # 3. Canonical URL & Platform Support
            if not is_valid_canonical_job_url(platform, m.url):
                continue

            # 4. Already Applied in Database
            if jid in submitted_ids:
                skipped_items.append(
                    SingleJobApplyResponse(
                        job_id=jid,
                        title=m.title,
                        company=m.company,
                        platform=platform,
                        match_score=m.final_score,
                        status="already_applied",
                        automation_state="ALREADY_APPLIED",
                        application_eligible=False,
                        error="Already applied in database",
                    )
                )
                continue

            # 5. Lock Check
            if lock_manager.is_locked(jid):
                skipped_items.append(
                    SingleJobApplyResponse(
                        job_id=jid,
                        title=m.title,
                        company=m.company,
                        platform=platform,
                        match_score=m.final_score,
                        status="skipped",
                        automation_state="LOCKED",
                        application_eligible=False,
                        error="Job is locked by another agent run",
                    )
                )
                continue

            # 6. Job Freshness Check (date_posted)
            real_job = self.job_repo.get_job_by_id(jid)
            posted_at = getattr(real_job, "posted_at", None) if real_job else None
            is_stale, age_days = freshness_policy.is_stale(posted_at)
            if is_stale:
                logger.info("Job [%s] is stale (%d days old > max %d days). Marking SKIPPED_STALE_JOB.", jid, age_days or 0, config.max_job_age_days)
                skipped_items.append(
                    SingleJobApplyResponse(
                        job_id=jid,
                        title=m.title,
                        company=m.company,
                        platform=platform,
                        match_score=m.final_score,
                        status="SKIPPED_STALE_JOB",
                        automation_state="SKIPPED_STALE_JOB",
                        application_eligible=False,
                        error=f"Job posting is {age_days} days old (exceeds max_job_age_days={config.max_job_age_days})",
                    )
                )
                continue

            # 7. Platform Daily Limit Check (Fail Closed)
            can_apply_plat, plat_reason = daily_limits.check_platform_capacity(
                platform=platform,
                profile_id=config.profile_id,
                in_run_platform_counts=queued_per_platform,
            )
            if not can_apply_plat:
                limits_hit[platform] = plat_reason
                skipped_items.append(
                    SingleJobApplyResponse(
                        job_id=jid,
                        title=m.title,
                        company=m.company,
                        platform=platform,
                        match_score=m.final_score,
                        status="SKIPPED_DAILY_LIMIT",
                        automation_state="SKIPPED_DAILY_LIMIT",
                        application_eligible=False,
                        error=plat_reason or f"Daily limit reached for platform {platform}",
                    )
                )
                continue

            # 8. Per-Platform Cap in Current Run
            current_queued_for_plat = queued_per_platform.get(platform, 0)
            if current_queued_for_plat >= config.max_jobs_per_platform:
                continue

            # 9. Multi-factor Priority Calculation
            # Factors: match_score (primary), role relevance, preferred location, freshness
            role_match = 1 if any(t in m.title.lower() for t in target_roles) else 0
            loc_lower = (m.location or "").lower()
            loc_match = 1 if any(pl in loc_lower for pl in preferred_locations) or "remote" in loc_lower else 0
            freshness_timestamp = posted_at.timestamp() if posted_at else 0.0

            seen_job_ids.add(jid)
            queued_per_platform[platform] = current_queued_for_plat + 1

            queue_item = QueueItem(
                job_id=jid,
                source=platform,
                external_id=getattr(m, "external_id", None) or getattr(real_job, "external_id", None),
                title=m.title,
                company=m.company,
                location=m.location,
                url=m.url or (real_job.url if real_job else ""),
                match_score=m.final_score,
                status=QueueStatus.QUEUED,
                priority=0,
                decision=m.decision,
                posted_at=posted_at,
            )
            # Store tuple for sort: (-match_score, -freshness, -loc_match, -role_match)
            setattr(queue_item, "_sort_key", (-m.final_score, -freshness_timestamp, -loc_match, -role_match))
            queue.append(queue_item)

        # Sort queue according to multi-factor criteria without introducing arbitrary scores
        queue.sort(key=lambda item: getattr(item, "_sort_key", (-item.match_score, 0, 0, 0)))

        return queue, skipped_items, limits_hit


# =============================================================================
# 6. AGENT CONTROL PLANE
# =============================================================================

class AgentControlPlane:
    """Central Control Plane governing autonomous job application cycles."""

    def __init__(
        self,
        orchestrator: Optional[JobApplicationOrchestrator] = None,
        daily_limits: Optional[DailyLimitsGuard] = None,
        queue_manager: Optional[ApplicationQueueManager] = None,
        lock_manager: Optional[JobLockManager] = None,
        scheduler_hook: Optional[BaseSchedulerHook] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.orchestrator = orchestrator or JobApplicationOrchestrator()
        self.settings = settings or get_settings()
        self.daily_limits = daily_limits or DailyLimitsGuard(
            application_repo=getattr(self.orchestrator, "app_repo", None),
            settings=self.settings,
        )
        self.queue_manager = queue_manager or ApplicationQueueManager(
            job_repo=getattr(self.orchestrator, "job_repo", None),
            app_repo=getattr(self.orchestrator, "app_repo", None),
            profile_service=getattr(self.orchestrator, "profile_service", None),
        )
        self.job_repo = getattr(self.orchestrator, "job_repo", None) or JobRepository()
        self.app_repo = getattr(self.orchestrator, "app_repo", None) or ApplicationRepository()
        self.lock_manager = lock_manager or getattr(self.orchestrator, "lock_manager", None) or JobLockManager()
        self.scheduler_hook = scheduler_hook or NoOpScheduler()

        # Strict single browser execution lock
        self._concurrency_lock = asyncio.Lock()

    async def run_autonomous_cycle(self, request: AgentRunRequest) -> AgentRunResponse:
        """Execute a controlled autonomous job application run with all Phase 6.1 safeguards."""
        async with self._concurrency_lock:
            # 1. Candidate Profile Validation (Section 5)
            target_profile = None
            if request.profile_id:
                target_profile = self.orchestrator.profile_service.get_profile(request.profile_id)
            else:
                target_profile = self.orchestrator.profile_service.get_profile()

            if not target_profile:
                logger.error("Candidate profile not found: %s", request.profile_id)
                empty_run = self.orchestrator.run_manager.create_run(
                    platform="multi",
                    profile_id=request.profile_id,
                    configuration=request.model_dump(mode="json"),
                )
                self.orchestrator.run_manager.update_status(
                    empty_run,
                    AgentRunStatus.FAILED,
                    "Candidate profile not found",
                )
                resp = self.orchestrator._build_response(empty_run, OrchestratorState.FAILED)
                resp.queue_empty_reason = "PROFILE_NOT_FOUND"
                resp.stop_reason = "PROFILE_NOT_FOUND"
                resp.pipeline = {
                    "discovery_called": False,
                    "discovery_service": "ProfileDiscoveryService",
                    "error": "PROFILE_NOT_FOUND",
                }
                resp.report = self.generate_run_report(resp, stop_reason="PROFILE_NOT_FOUND")
                return resp

            effective_profile_id = target_profile.id

            # 2. Daily Limits Guard Check (Fail Closed)
            can_start, limit_reason, limit_meta = self.daily_limits.check_can_start_run(
                profile_id=effective_profile_id
            )
            if not can_start:
                logger.warning("AgentControlPlane: Cannot start run. %s", limit_reason)
                empty_run = self.orchestrator.run_manager.create_run(
                    platform="multi",
                    profile_id=effective_profile_id,
                    configuration=request.model_dump(mode="json"),
                )
                self.orchestrator.run_manager.update_status(
                    empty_run,
                    AgentRunStatus.COMPLETED,
                    limit_reason or "Daily limit reached",
                )
                resp = self.orchestrator._build_response(empty_run, OrchestratorState.COMPLETED)
                resp.limits_hit = limit_meta
                resp.queue_empty_reason = "DAILY_LIMIT_REACHED"
                resp.stop_reason = "DAILY_LIMIT_REACHED"
                resp.pipeline = {
                    "discovery_called": False,
                    "discovery_service": "ProfileDiscoveryService",
                    "error": "DAILY_LIMIT_REACHED",
                }
                resp.report = self.generate_run_report(resp, limits_hit=limit_meta, stop_reason="DAILY_LIMIT_REACHED")
                return resp

            # 3. Run Profile Discovery & Matching via Orchestrator's services
            run = self.orchestrator.run_manager.create_run(
                platform="multi",
                profile_id=effective_profile_id,
                configuration=request.model_dump(mode="json"),
            )
            run_id_str = str(run.run_id)

            self.orchestrator.run_manager.update_status(
                run, AgentRunStatus.DISCOVERING, "Generating queries and searching job boards"
            )
            self.orchestrator._record_log(run, None, "orchestrator", "state_transition", OrchestratorState.DISCOVERING.value)

            from app.models.job import ProfileJobSearchRequest
            disc_request = ProfileJobSearchRequest(
                sites=request.sites,
                results_per_query=25,
                hours_old=request.max_job_age_days * 24,
                max_queries=request.max_queries,
                minimum_score=request.minimum_match_score,
                run_matching=True,
                use_semantic=request.use_semantic,
                limit=100,
            )

            discovery_called = True
            try:
                disc_response = await self.orchestrator.discovery_service.discover_jobs_from_profile(
                    request=disc_request,
                    profile_id=effective_profile_id,
                )
            except Exception as exc:
                logger.error("Discovery failed: %s", exc, exc_info=True)
                self.orchestrator.run_manager.update_status(run, AgentRunStatus.FAILED, f"Discovery failed: {exc}")
                resp = self.orchestrator._build_response(run, OrchestratorState.FAILED)
                resp.stop_reason = f"Discovery failed: {exc}"
                resp.pipeline = {
                    "discovery_called": True,
                    "discovery_service": "ProfileDiscoveryService",
                    "error": str(exc),
                }
                resp.report = self.generate_run_report(resp, stop_reason=f"Discovery failed: {exc}")
                return resp

            top_matches = getattr(disc_response, "top_matches", []) or []
            disc_stats = getattr(disc_response, "discovery", None)
            matching_stats = getattr(disc_response, "matching", None)
            search_plan = getattr(disc_response, "search_plan", None)

            raw_jobs = getattr(disc_stats, "raw_jobs", 0) if disc_stats else 0
            unique_jobs = getattr(disc_stats, "unique_jobs", 0) if disc_stats else len(top_matches)
            queries_count = getattr(search_plan, "queries_generated", len(getattr(search_plan, "queries", []))) if search_plan else 0
            query_results = getattr(disc_stats, "query_results", []) if disc_stats else []
            discovery_status = getattr(disc_response, "status", "success")

            jobs_disc = max(raw_jobs, unique_jobs, len(top_matches))
            jobs_sc = getattr(matching_stats, "jobs_scored", len(top_matches)) if matching_stats else len(top_matches)

            # Fallback & Supplement: ensure requested sites are represented if live discovery yielded 0 matches for a platform
            req_sites = [s.lower() for s in (request.sites or ["indeed", "glassdoor"])]
            present_sites = {m.source.lower() for m in top_matches if m.source}
            missing_sites = [s for s in req_sites if s not in present_sites]
            if missing_sites or not top_matches:
                try:
                    match_repo = getattr(self.orchestrator, "match_repo", None) or MatchRepository()
                    stored_matches = match_repo.get_top_matches(
                        profile_id=effective_profile_id, limit=50
                    )
                    target_sites = missing_sites if top_matches else req_sites
                    existing_job_ids = {m.job_id for m in top_matches}
                    supplement_items: List[MatchedJobItem] = []
                    for sm in stored_matches:
                        j_data = sm.get("jobs") or {}
                        src = (j_data.get("source") or "").lower()
                        score = float(sm.get("match_score") or 0.0)
                        j_id = UUID(str(sm["job_id"]))
                        if src in target_sites and score >= request.minimum_match_score and j_id not in existing_job_ids:
                            supplement_items.append(
                                MatchedJobItem(
                                    job_id=j_id,
                                    source=src,
                                    external_id=j_data.get("external_id"),
                                    title=j_data.get("title", ""),
                                    company=j_data.get("company", ""),
                                    location=j_data.get("location"),
                                    url=j_data.get("url"),
                                    final_score=score,
                                    decision=sm.get("decision", "review"),
                                )
                            )
                    if supplement_items:
                        logger.info("Supplemented %d matches from database fallback for sites %s", len(supplement_items), target_sites)
                        if not top_matches:
                            top_matches = supplement_items
                        else:
                            top_matches.extend(supplement_items)
                        jobs_disc = max(jobs_disc, len(top_matches))
                        jobs_sc = max(jobs_sc, len(top_matches))
                except Exception as fb_err:
                    logger.debug("Database fallback matching skipped: %s", fb_err)

            # 4. Queue Construction & Governance via Control Plane QueueManager
            self.orchestrator.run_manager.update_status(
                run, AgentRunStatus.BUILDING_QUEUE, "Governing and prioritizing application queue"
            )
            queue, pre_skipped_items, limits_hit = self.queue_manager.build_and_prioritize_queue(
                top_matches=top_matches,
                config=request,
                daily_limits=self.daily_limits,
                lock_manager=self.lock_manager,
                run_id=run.run_id,
            )

            # Record pre-skipped items in metrics
            results: List[SingleJobApplyResponse] = list(pre_skipped_items)
            for item in pre_skipped_items:
                if item.status == "SKIPPED_STALE_JOB":
                    run.jobs_stale += 1
                    run.jobs_skipped += 1
                elif item.status == "SKIPPED_DAILY_LIMIT":
                    run.jobs_skipped += 1
                elif item.status == "already_applied":
                    run.jobs_already_applied += 1

            self.orchestrator.run_manager.record_discovery_and_matching(
                run=run,
                jobs_discovered=jobs_disc,
                jobs_scored=jobs_sc,
                jobs_eligible=len(queue),
            )

            # Compute structured eligibility diagnostics
            disc_jobs = getattr(disc_response, "jobs", []) or []
            diag_items = disc_jobs if disc_jobs else top_matches
            diag_breakdown, diag_samples = self.compute_eligibility_diagnostics(
                candidate_items=diag_items,
                profile_id=effective_profile_id,
                max_job_age_days=request.max_job_age_days,
                top_matches=top_matches,
            )

            # Pipeline diagnostics dictionary
            pipeline_diagnostics = {
                "discovery_called": True,
                "discovery_service": "ProfileDiscoveryService",
                "search_plan_queries": queries_count,
                "discovery_status": discovery_status,
                "discovery_raw_jobs": raw_jobs,
                "discovery_unique_jobs": unique_jobs,
                "discovery_query_results_count": len(query_results),
                "jobs_discovered": jobs_disc,
                "jobs_unique": unique_jobs,
                "jobs_matched": jobs_sc,
                "jobs_eligible": len(queue),
                "jobs_queued": len(queue),
            }

            if not queue:
                # Classify precise queue_empty_reason
                if jobs_disc == 0:
                    empty_reason = "DISCOVERY_ZERO_JOBS"
                elif jobs_sc == 0 or len(top_matches) == 0:
                    empty_reason = "MATCHING_ZERO_QUALIFIED"
                elif pre_skipped_items and all(item.status == "already_applied" for item in pre_skipped_items):
                    empty_reason = "ALL_JOBS_ALREADY_APPLIED"
                elif pre_skipped_items and all(item.status == "SKIPPED_STALE_JOB" for item in pre_skipped_items):
                    empty_reason = "ALL_JOBS_STALE"
                elif pre_skipped_items and all(item.status == "SKIPPED_DAILY_LIMIT" for item in pre_skipped_items):
                    empty_reason = "ALL_JOBS_PLATFORM_LIMIT"
                elif pre_skipped_items and all(item.automation_state == "LOCKED" for item in pre_skipped_items):
                    empty_reason = "ALL_JOBS_LOCKED"
                else:
                    empty_reason = "QUEUE_EMPTY"

                self.orchestrator.run_manager.update_status(run, AgentRunStatus.COMPLETED, f"No eligible unapplied jobs in queue: {empty_reason}")
                resp = self.orchestrator._build_response(run, OrchestratorState.COMPLETED, results=results)
                resp.jobs_discovered = jobs_disc
                resp.jobs_matched = jobs_sc
                resp.jobs_stale = run.jobs_stale
                resp.limits_hit = limits_hit
                resp.queue_empty_reason = empty_reason
                resp.stop_reason = empty_reason
                resp.pipeline = pipeline_diagnostics
                resp.eligibility_breakdown = diag_breakdown
                resp.eligibility_samples = diag_samples
                resp.jobs_external_apply = diag_breakdown["external_apply"]
                resp.jobs_expired = diag_breakdown["expired"]
                resp.jobs_unavailable = diag_breakdown["unavailable"]
                resp.jobs_already_applied = diag_breakdown["already_applied"]
                resp.jobs_ineligible = diag_breakdown["ineligible"]
                resp.jobs_application_eligible = diag_breakdown["eligible"]
                resp.report = self.generate_run_report(resp, limits_hit=limits_hit, stop_reason=empty_reason)
                return resp

            # 5. Sequential Application Loop under Single Browser Lock & Cooldown
            self.orchestrator.run_manager.update_status(
                run, AgentRunStatus.APPLYING, f"Processing queue (max {request.max_jobs} jobs, rate controlled)"
            )

            delay_seconds = max(
                request.application_delay_seconds,
                getattr(self.settings, "COOLDOWN_SECONDS_BETWEEN_APPLICATIONS", 15),
            )

            failure_class_counts: Dict[str, int] = {}
            jobs_attempted_in_run = 0

            while queue and jobs_attempted_in_run < request.max_jobs:
                if run_id_str in self.orchestrator._abort_requested:
                    self.orchestrator._abort_requested.discard(run_id_str)
                    self.orchestrator.run_manager.update_status(run, AgentRunStatus.CANCELLED, "Run aborted by user")
                    self.lock_manager.release_all_for_run(run.run_id)
                    resp = self.orchestrator._build_response(run, OrchestratorState.COMPLETED, results=results, remaining_queue=queue)
                    resp.eligibility_breakdown = diag_breakdown
                    resp.eligibility_samples = diag_samples
                    resp.jobs_external_apply = diag_breakdown["external_apply"]
                    resp.jobs_expired = diag_breakdown["expired"]
                    resp.jobs_unavailable = diag_breakdown["unavailable"]
                    resp.jobs_already_applied = diag_breakdown["already_applied"]
                    resp.jobs_ineligible = diag_breakdown["ineligible"]
                    resp.jobs_application_eligible = diag_breakdown["eligible"]
                    resp.report = self.generate_run_report(resp, stop_reason="USER_ABORT")
                    return resp

                next_item = queue.pop(0)
                job_id = next_item.job_id
                platform = next_item.source

                # Daily platform capacity check again before dispatch (fail-closed)
                can_apply_plat, plat_reason = self.daily_limits.check_platform_capacity(
                    platform=platform,
                    profile_id=request.profile_id,
                )
                if not can_apply_plat:
                    logger.info("Skipping job [%s] on platform [%s]: %s", job_id, platform, plat_reason)
                    skip_resp = SingleJobApplyResponse(
                        job_id=job_id,
                        title=next_item.title,
                        company=next_item.company,
                        platform=platform,
                        match_score=next_item.match_score,
                        status="SKIPPED_DAILY_LIMIT",
                        automation_state="SKIPPED_DAILY_LIMIT",
                        application_eligible=False,
                        error=plat_reason,
                    )
                    results.append(skip_resp)
                    run.jobs_skipped += 1
                    continue

                # Acquire execution lock
                if not self.lock_manager.acquire_lock(job_id=job_id, agent_run_id=run.run_id):
                    results.append(SingleJobApplyResponse(
                        job_id=job_id,
                        title=next_item.title,
                        company=next_item.company,
                        platform=platform,
                        match_score=next_item.match_score,
                        status="skipped",
                        error="Job is locked by another agent run",
                    ))
                    run.jobs_skipped += 1
                    continue

                self.orchestrator._record_log(run, job_id, platform, "state_transition", OrchestratorState.JOB_SELECTED.value)

                try:
                    # Live Eligibility Gate
                    eligibility = await self.orchestrator.eligibility_gate.evaluate_job_live(
                        job_id=job_id,
                        url=next_item.url or "",
                        platform=platform,
                        profile_id=request.profile_id,
                        job_title=next_item.title,
                        company=next_item.company,
                    )

                    if not eligibility.is_eligible and not (request.allow_external_apply and eligibility.skip_reason == "SKIPPED_EXTERNAL_APPLY"):
                        next_item.status = QueueStatus.SKIPPED
                        self.orchestrator._record_eligibility_metrics(run, eligibility)
                        skip_resp = SingleJobApplyResponse(
                            job_id=job_id,
                            title=next_item.title,
                            company=next_item.company,
                            platform=platform,
                            match_score=next_item.match_score,
                            status=eligibility.status_code,
                            automation_state=eligibility.status_code,
                            application_method=eligibility.application_method.value,
                            availability_status=eligibility.availability_status.value,
                            application_eligible=False,
                            diagnostics=eligibility.diagnostics,
                            error=eligibility.message,
                        )
                        f_class = FailureClassifier.classify_outcome(
                            status=skip_resp.status,
                            automation_state=skip_resp.automation_state,
                            error=skip_resp.error,
                        )
                        if f_class is not None:
                            failure_class_counts[f_class.value] = failure_class_counts.get(f_class.value, 0) + 1
                        results.append(skip_resp)
                        continue

                    run.jobs_application_eligible += 1

                    # Process application via Orchestrator
                    jobs_attempted_in_run += 1
                    run.jobs_attempted += 1
                    res = await self.orchestrator.process_job(
                        job=next_item,
                        run=run,
                        auto_submit=request.auto_submit,
                        profile_id=request.profile_id,
                    )
                    if res.status == "failed" and run.jobs_failed == 0:
                        run.jobs_failed += 1

                    f_class = FailureClassifier.classify_outcome(
                        status=res.status,
                        automation_state=res.automation_state,
                        error=res.error,
                    )
                    if f_class is not None:
                        failure_class_counts[f_class.value] = failure_class_counts.get(f_class.value, 0) + 1

                    # Transient retry policy check
                    if (
                        f_class == FailureClassification.TRANSIENT_FAILURE
                        and request.retry_failed_jobs
                        and next_item.attempt_count < 1
                    ):
                        logger.warning("Job [%s] suffered transient failure (%s). Retrying once...", job_id, res.error)
                        next_item.attempt_count += 1
                        await asyncio.sleep(2)
                        res = await self.orchestrator.process_job(
                            job=next_item,
                            run=run,
                            auto_submit=request.auto_submit,
                            profile_id=request.profile_id,
                        )
                        # Reclassify after retry
                        f_class_retry = FailureClassifier.classify_outcome(
                            status=res.status,
                            automation_state=res.automation_state,
                            error=res.error,
                        )
                        if f_class_retry is None:
                            failure_class_counts[FailureClassification.TRANSIENT_FAILURE.value] = max(
                                0, failure_class_counts.get(FailureClassification.TRANSIENT_FAILURE.value, 1) - 1
                            )
                        elif f_class_retry != FailureClassification.TRANSIENT_FAILURE:
                            failure_class_counts[f_class_retry.value] = failure_class_counts.get(f_class_retry.value, 0) + 1

                    results.append(res)

                    # Pause condition: human intervention required
                    if res.status == "manual_action_required":
                        logger.warning("Job [%s] requires human intervention. Pausing run.", job_id)
                        self.orchestrator.run_manager.pause_run_for_manual_action(
                            run=run,
                            current_job_id=job_id,
                            pause_reason=res.error or "Challenge / Human Input Required",
                            remaining_queue=[],
                        )
                        resp = self.orchestrator._build_response(
                            run,
                            OrchestratorState.WAITING_FOR_HUMAN_INPUT,
                            results=results,
                            remaining_queue=queue,
                        )
                        resp.jobs_stale = run.jobs_stale
                        resp.failure_classification = failure_class_counts
                        resp.pipeline = pipeline_diagnostics
                        resp.eligibility_breakdown = diag_breakdown
                        resp.eligibility_samples = diag_samples
                        resp.jobs_external_apply = diag_breakdown["external_apply"]
                        resp.jobs_expired = diag_breakdown["expired"]
                        resp.jobs_unavailable = diag_breakdown["unavailable"]
                        resp.jobs_already_applied = diag_breakdown["already_applied"]
                        resp.jobs_ineligible = diag_breakdown["ineligible"]
                        resp.jobs_application_eligible = diag_breakdown["eligible"]
                        resp.stop_reason = "HUMAN_ACTION_REQUIRED"
                        resp.report = self.generate_run_report(resp, failure_classes=failure_class_counts, stop_reason="HUMAN_ACTION_REQUIRED")
                        return resp

                finally:
                    self.lock_manager.release_lock(job_id=job_id, agent_run_id=run.run_id)

                # Rate control cooldown between applications
                if queue and jobs_attempted_in_run < request.max_jobs:
                    logger.info("Control Plane: rate cooldown for %d seconds before next job...", delay_seconds)
                    await asyncio.sleep(delay_seconds)

            final_state = OrchestratorState.COMPLETED
            if run.jobs_failed > 0 and run.applications_submitted == 0 and not results:
                final_state = OrchestratorState.FAILED

            self.orchestrator.run_manager.update_status(run, AgentRunStatus.COMPLETED, "Control plane run completed")
            self.lock_manager.release_all_for_run(run.run_id)

            resp = self.orchestrator._build_response(run, final_state, results=results, remaining_queue=queue)
            resp.jobs_stale = run.jobs_stale
            resp.limits_hit = limits_hit
            resp.failure_classification = failure_class_counts
            resp.pipeline = pipeline_diagnostics
            resp.eligibility_breakdown = diag_breakdown
            resp.eligibility_samples = diag_samples
            resp.jobs_external_apply = diag_breakdown["external_apply"]
            resp.jobs_expired = diag_breakdown["expired"]
            resp.jobs_unavailable = diag_breakdown["unavailable"]
            resp.jobs_already_applied = diag_breakdown["already_applied"]
            resp.jobs_ineligible = diag_breakdown["ineligible"]
            resp.jobs_application_eligible = diag_breakdown["eligible"]
            resp.stop_reason = "MAX_JOBS_REACHED" if jobs_attempted_in_run >= request.max_jobs else "QUEUE_EXHAUSTED"
            resp.report = self.generate_run_report(
                resp,
                limits_hit=limits_hit,
                failure_classes=failure_class_counts,
                stop_reason=resp.stop_reason,
            )
            return resp

    def compute_eligibility_diagnostics(
        self,
        candidate_items: List[Any],
        profile_id: Optional[UUID],
        max_job_age_days: int = 7,
        top_matches: Optional[List[MatchedJobItem]] = None,
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        """Compute structured eligibility breakdown and per-job diagnostic samples."""
        breakdown = {
            "easy_apply": 0,
            "external_apply": 0,
            "unknown_method": 0,
            "available": 0,
            "expired": 0,
            "unavailable": 0,
            "unknown_availability": 0,
            "already_applied": 0,
            "stale": 0,
            "eligible": 0,
            "ineligible": 0,
        }
        samples: List[Dict[str, Any]] = []

        freshness_policy = FreshnessPolicy(max_job_age_days=max_job_age_days)
        match_map = {m.job_id: m for m in (top_matches or [])}

        # Pre-fetch applied job IDs for candidate
        submitted_ids: Set[UUID] = set()
        all_jids = [
            getattr(j, "id", None) or getattr(j, "job_id", None)
            for j in candidate_items
            if (getattr(j, "id", None) or getattr(j, "job_id", None)) is not None
        ]

        if profile_id and all_jids:
            try:
                existing = self.app_repo.get_applications_for_jobs(all_jids, profile_id=profile_id)
                submitted_ids = {
                    a.job_id for a in existing
                    if a.status in ("submitted", "APPLICATION_SUBMITTED", "submission_ready")
                }
            except Exception as exc:
                logger.warning("Error pre-fetching applications for diagnostics: %s", exc)

        for item in candidate_items:
            jid = getattr(item, "id", None) or getattr(item, "job_id", None)
            title = getattr(item, "title", "Unknown Title")
            company = getattr(item, "company", "Unknown Company")
            platform = (getattr(item, "source", "") or "").lower().strip()
            url = getattr(item, "url", "")

            # Match score & decision
            matched_item = match_map.get(jid)
            if matched_item:
                match_score = matched_item.final_score
                decision = matched_item.decision
            else:
                match_score = getattr(item, "final_score", None) or getattr(item, "match_score", 0.0)
                decision = getattr(item, "decision", None) or getattr(item, "match_decision", "reject")

            # Check database for real job model if needed for dates/easy_apply
            real_job = None
            if jid:
                try:
                    real_job = self.job_repo.get_job_by_id(jid)
                except Exception:
                    pass

            posted_at = getattr(real_job, "posted_at", None) if real_job else getattr(item, "posted_at", None)
            is_stale, _ = freshness_policy.is_stale(posted_at)

            # Application method
            app_method = getattr(item, "application_method", None)
            ea_val = getattr(real_job, "easy_apply", None) if real_job else getattr(item, "easy_apply", None)
            if ea_val is True or app_method == ApplicationMethod.EASY_APPLY.value:
                app_method = ApplicationMethod.EASY_APPLY.value
            elif (ea_val is False and platform in ("indeed", "glassdoor")) or app_method == ApplicationMethod.EXTERNAL_APPLY.value:
                app_method = ApplicationMethod.EXTERNAL_APPLY.value
            else:
                app_method = ApplicationMethod.UNKNOWN.value

            # Availability status
            avail_status = getattr(item, "availability_status", None) or AvailabilityStatus.UNKNOWN.value
            if real_job and getattr(real_job, "availability_status", None):
                avail_status = real_job.availability_status

            already_applied = jid in submitted_ids if jid else False

            # Update breakdown counters
            if app_method == ApplicationMethod.EASY_APPLY.value:
                breakdown["easy_apply"] += 1
            elif app_method == ApplicationMethod.EXTERNAL_APPLY.value:
                breakdown["external_apply"] += 1
            else:
                breakdown["unknown_method"] += 1

            if avail_status == AvailabilityStatus.AVAILABLE.value:
                breakdown["available"] += 1
            elif avail_status == AvailabilityStatus.EXPIRED.value:
                breakdown["expired"] += 1
            elif avail_status == AvailabilityStatus.UNAVAILABLE.value:
                breakdown["unavailable"] += 1
            else:
                breakdown["unknown_availability"] += 1

            if already_applied:
                breakdown["already_applied"] += 1

            if is_stale:
                breakdown["stale"] += 1

            # Determine eligibility and rejection reasons
            rejection_reasons: List[str] = []
            if not is_valid_canonical_job_url(platform, url):
                rejection_reasons.append("INVALID_CANONICAL_URL")
            if already_applied:
                rejection_reasons.append("ALREADY_APPLIED")
            if is_stale:
                rejection_reasons.append("STALE_POSTING")
            if app_method == ApplicationMethod.EXTERNAL_APPLY.value:
                rejection_reasons.append("EXTERNAL_APPLY")
            elif app_method == ApplicationMethod.UNKNOWN.value:
                rejection_reasons.append("METHOD_UNVERIFIED")
            if avail_status in (AvailabilityStatus.EXPIRED.value, AvailabilityStatus.UNAVAILABLE.value):
                rejection_reasons.append(f"STATUS_{avail_status}")
            if decision in ("reject", "low_match"):
                rejection_reasons.append(f"MATCH_DISQUALIFIED_{decision.upper()}")

            is_eligible = (
                len(rejection_reasons) == 0
                and app_method == ApplicationMethod.EASY_APPLY.value
                and avail_status == AvailabilityStatus.AVAILABLE.value
            )
            if is_eligible:
                breakdown["eligible"] += 1
            else:
                breakdown["ineligible"] += 1

            # Collect up to 10 samples
            if len(samples) < 10:
                samples.append({
                    "job_id": str(jid) if jid else None,
                    "title": title,
                    "company": company,
                    "platform": platform,
                    "match_score": float(match_score),
                    "match_decision": str(decision),
                    "application_method": app_method,
                    "availability_status": avail_status,
                    "application_eligible": is_eligible,
                    "is_stale": is_stale,
                    "already_applied": already_applied,
                    "rejection_reason": "; ".join(rejection_reasons) if rejection_reasons else "ELIGIBLE",
                })

        return breakdown, samples

    def generate_run_report(
        self,
        resp: AgentRunResponse,
        limits_hit: Optional[Dict[str, Any]] = None,
        failure_classes: Optional[Dict[str, int]] = None,
        stop_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate structured run report as required by Phase 6.1 Section 9."""
        return {
            "discovered": resp.jobs_discovered,
            "matched": resp.jobs_matched,
            "eligible": resp.jobs_application_eligible,
            "queued": resp.jobs_queued,
            "attempted": resp.jobs_attempted,
            "submitted": resp.jobs_submitted,
            "skipped": resp.jobs_skipped,
            "already_applied": resp.jobs_already_applied,
            "external_apply": resp.jobs_external_apply,
            "stale": resp.jobs_stale,
            "failed": resp.jobs_failed,
            "manual_action_required": resp.manual_actions_required,
            "stop_reason": stop_reason or resp.stop_reason,
            "queue_empty_reason": resp.queue_empty_reason,
            "pipeline": resp.pipeline or {},
            "limits_hit": limits_hit or {},
            "failure_classification": failure_classes or {},
            "eligibility_breakdown": resp.eligibility_breakdown or {},
            "eligibility_samples": resp.eligibility_samples or [],
        }
