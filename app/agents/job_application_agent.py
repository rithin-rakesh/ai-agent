"""Autonomous Job Application Orchestrator (Phase 6.0).

Connects discovery, matching, candidate-answer bank, Indeed automation, and Glassdoor automation
into a single, unified, sequential, 17-state lifecycle machine.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from app.agents.lock_manager import JobLockManager
from app.agents.models import (
    AgentAbortRequest,
    AgentApplyOneRequest,
    AgentResumeRequest,
    AgentRun,
    AgentRunRequest,
    AgentRunResponse,
    AgentRunStatus,
    AgentSkipRequest,
    OrchestratorState,
    QueueItem,
    QueueStatus,
    SingleJobApplyResponse,
)
from app.agents.eligibility_gate import ApplicationEligibilityGate, EligibilityResult
from app.agents.platform_router import PlatformRouter
from app.agents.run_manager import AgentRunManager
from app.config.settings import Settings, get_settings
from app.database.repositories.application_repository import ApplicationRepository
from app.database.repositories.job_repository import JobRepository
from app.discovery.profile_discovery_service import (
    ProfileDiscoveryService,
    is_valid_canonical_job_url,
)
from app.models.job import (
    ApplicationMethod,
    AvailabilityStatus,
    MatchedJobItem,
    ProfileJobSearchRequest,
)
from app.profile.service import ProfileService

logger = logging.getLogger(__name__)


class JobApplicationOrchestrator:
    """Central autonomous agent orchestrating discovery, matching, queueing, and assisted applications."""

    def __init__(
        self,
        discovery_service: Optional[ProfileDiscoveryService] = None,
        job_repository: Optional[JobRepository] = None,
        application_repository: Optional[ApplicationRepository] = None,
        run_manager: Optional[AgentRunManager] = None,
        lock_manager: Optional[JobLockManager] = None,
        platform_router: Optional[PlatformRouter] = None,
        profile_service: Optional[ProfileService] = None,
        settings: Optional[Settings] = None,
        eligibility_gate: Optional[ApplicationEligibilityGate] = None,
    ) -> None:
        self.discovery_service = discovery_service or ProfileDiscoveryService()
        self.job_repo = job_repository or JobRepository()
        self.app_repo = application_repository or ApplicationRepository()
        self.run_manager = run_manager or AgentRunManager()
        self.lock_manager = lock_manager or JobLockManager()
        self.platform_router = platform_router or PlatformRouter()
        self.profile_service = profile_service or ProfileService()
        self.settings = settings or get_settings()
        self.eligibility_gate = eligibility_gate or ApplicationEligibilityGate(
            platform_router=self.platform_router,
            application_repository=self.app_repo,
            settings=self.settings,
        )

        # Strict session safety: apply ONE job at a time
        self._execution_lock = asyncio.Lock()
        self._abort_requested: Set[str] = set()

    # =========================================================================
    # 1. MULTI-JOB AUTONOMOUS CYCLE (POST /agent/run)
    # =========================================================================
    async def run_application_cycle(self, request: AgentRunRequest) -> AgentRunResponse:
        """Execute a full autonomous job application cycle: discovery -> matching -> queue -> apply."""
        run = self.run_manager.create_run(
            platform="multi",
            profile_id=request.profile_id,
            configuration=request.model_dump(mode="json"),
        )
        run_id_str = str(run.run_id)
        logger.info("Starting Autonomous Agent Run [%s]", run.run_id)

        # 1. IDLE -> DISCOVERING
        self.run_manager.update_status(run, AgentRunStatus.DISCOVERING, "Generating queries and searching job boards")
        self._record_log(run, None, "orchestrator", "state_transition", OrchestratorState.DISCOVERING.value)

        disc_response = None
        try:
            disc_request = ProfileJobSearchRequest(
                sites=request.sites,
                results_per_query=25,
                hours_old=168,
                max_queries=request.max_queries,
                minimum_score=request.minimum_match_score,
                run_matching=True,
                use_semantic=request.use_semantic,
                limit=100,
            )
            disc_response = await self.discovery_service.discover_jobs_from_profile(
                request=disc_request,
                profile_id=request.profile_id,
            )
        except Exception as exc:
            logger.error("Discovery failed in Agent Run [%s]: %s", run.run_id, exc, exc_info=True)
            self.run_manager.update_status(run, AgentRunStatus.FAILED, f"Discovery failed: {exc}")
            self._record_log(run, None, "orchestrator", "discovery_error", "failed", error=str(exc))
            return self._build_response(run, OrchestratorState.FAILED)

        # 2. DISCOVERY_COMPLETE
        self._record_log(run, None, "orchestrator", "state_transition", OrchestratorState.DISCOVERY_COMPLETE.value)

        # 3. MATCHING
        self.run_manager.update_status(run, AgentRunStatus.MATCHING, "Filtering matches against criteria")
        self._record_log(run, None, "orchestrator", "state_transition", OrchestratorState.MATCHING.value)

        top_matches = getattr(disc_response, "top_matches", []) or []
        disc_stats = getattr(disc_response, "discovery", None)
        matching_stats = getattr(disc_response, "matching", None)
        jobs_disc = getattr(disc_stats, "unique_jobs", len(top_matches)) if disc_stats else len(top_matches)
        jobs_sc = getattr(matching_stats, "jobs_scored", len(top_matches)) if matching_stats else len(top_matches)

        # 4. QUEUE_READY
        self.run_manager.update_status(run, AgentRunStatus.BUILDING_QUEUE, "Constructing and ranking application queue")
        queue = self._build_queue(
            top_matches=top_matches,
            minimum_score=request.minimum_match_score,
            profile_id=request.profile_id,
        )

        self.run_manager.record_discovery_and_matching(
            run=run,
            jobs_discovered=jobs_disc,
            jobs_scored=jobs_sc,
            jobs_eligible=len(queue),
        )
        self._record_log(run, None, "orchestrator", "state_transition", OrchestratorState.QUEUE_READY.value, details={"queue_size": len(queue)})

        if not queue:
            self.run_manager.update_status(run, AgentRunStatus.COMPLETED, "No eligible unapplied jobs in queue")
            return self._build_response(run, OrchestratorState.COMPLETED)

        # 5. SEQUENTIAL APPLICATION LOOP
        self.run_manager.update_status(run, AgentRunStatus.APPLYING, f"Processing queue (max {request.max_jobs} jobs)")

        results: List[SingleJobApplyResponse] = []
        delay_seconds = getattr(self.settings, "APPLICATION_DELAY_SECONDS", 5)

        while queue and len(results) < request.max_jobs:
            if run_id_str in self._abort_requested:
                logger.info("Agent Run [%s] abort requested. Terminating processing.", run.run_id)
                self._abort_requested.discard(run_id_str)
                self.run_manager.update_status(run, AgentRunStatus.CANCELLED, "Run aborted by user")
                self.lock_manager.release_all_for_run(run.run_id)
                return self._build_response(run, OrchestratorState.COMPLETED, results=results, remaining_queue=queue)

            # SELECTING_JOB
            next_item = self.select_next_job(queue)
            if not next_item:
                break
            queue.remove(next_item)

            job_id = next_item.job_id
            self._record_log(run, job_id, next_item.source, "state_transition", OrchestratorState.SELECTING_JOB.value)

            # Check Lock
            locked = self.lock_manager.acquire_lock(job_id=job_id, agent_run_id=run.run_id)
            if not locked:
                logger.info("Job [%s] is locked by another run. Skipping.", job_id)
                next_item.status = QueueStatus.SKIPPED
                run.jobs_skipped += 1
                results.append(SingleJobApplyResponse(
                    job_id=job_id,
                    title=next_item.title,
                    company=next_item.company,
                    platform=next_item.source,
                    match_score=next_item.match_score,
                    status="skipped",
                    error="Job is locked by another agent run",
                ))
                continue

            # JOB_SELECTED
            self._record_log(run, job_id, next_item.source, "state_transition", OrchestratorState.JOB_SELECTED.value)

            # Process Job under single-job execution lock
            try:
                # Mandatory Live Pre-Application Eligibility Gate
                eligibility = await self.eligibility_gate.evaluate_job_live(
                    job_id=job_id,
                    url=next_item.url or "",
                    platform=next_item.source,
                    profile_id=request.profile_id,
                    job_title=next_item.title,
                    company=next_item.company,
                )

                if not eligibility.is_eligible:
                    logger.info(
                        "Job [%s] is not application-eligible: %s (%s). Skipping.",
                        job_id,
                        eligibility.skip_reason,
                        eligibility.message,
                    )
                    next_item.status = QueueStatus.SKIPPED
                    self._record_eligibility_metrics(run, eligibility)
                    self._record_log(
                        run,
                        job_id,
                        next_item.source,
                        "eligibility_check",
                        eligibility.status_code,
                        details=eligibility.diagnostics,
                        error=eligibility.message,
                    )
                    results.append(SingleJobApplyResponse(
                        job_id=job_id,
                        title=next_item.title,
                        company=next_item.company,
                        platform=next_item.source,
                        match_score=next_item.match_score,
                        status=eligibility.status_code,
                        automation_state=eligibility.status_code,
                        application_method=eligibility.application_method.value,
                        availability_status=eligibility.availability_status.value,
                        application_eligible=False,
                        diagnostics=eligibility.diagnostics,
                        error=eligibility.message,
                    ))
                    continue

                run.jobs_application_eligible += 1

                res = await self.process_job(
                    job=next_item,
                    run=run,
                    auto_submit=request.auto_submit,
                    profile_id=request.profile_id,
                )
                results.append(res)

                # Check if paused for human input
                if res.status == "manual_action_required":
                    logger.warning("Job [%s] requires human action (%s). Pausing run [%s].", job_id, res.error, run.run_id)
                    self.run_manager.pause_run_for_manual_action(
                        run=run,
                        current_job_id=job_id,
                        pause_reason=res.error or "Challenge / Human Input Required",
                        remaining_queue=[],
                    )
                    return self._build_response(
                        run,
                        OrchestratorState.WAITING_FOR_HUMAN_INPUT,
                        results=results,
                        remaining_queue=queue,
                    )

            finally:
                self.lock_manager.release_lock(job_id=job_id, agent_run_id=run.run_id)

            # Rate / Safety delay between jobs
            if queue and len(results) < request.max_jobs:
                logger.info("Waiting %d seconds before next job application...", delay_seconds)
                await asyncio.sleep(delay_seconds)

        final_state = OrchestratorState.COMPLETED
        status_enum = AgentRunStatus.COMPLETED
        if run.jobs_failed > 0 and run.applications_submitted == 0 and not results:
            final_state = OrchestratorState.FAILED
            status_enum = AgentRunStatus.FAILED

        self.run_manager.update_status(run, status_enum, "Run finished")
        self.lock_manager.release_all_for_run(run.run_id)
        return self._build_response(run, final_state, results=results, remaining_queue=queue)

    # =========================================================================
    # 2. SINGLE-JOB MODE (POST /agent/apply-one)
    # =========================================================================
    async def run_single_job(self, request: AgentApplyOneRequest) -> SingleJobApplyResponse:
        """Run a single job through the entire state machine without full discovery cycle."""
        job_id = request.job_id
        real_job = self.job_repo.get_job_by_id(job_id)
        if not real_job:
            raise ValueError(f"Job with id '{job_id}' does not exist in database.")

        run = self.run_manager.create_run(
            platform=real_job.source,
            profile_id=request.profile_id,
            configuration=request.model_dump(mode="json"),
        )
        logger.info("Starting Single-Job Agent Run [%s] for job [%s]", run.run_id, job_id)

        # Eligibility check
        if not self.platform_router.is_platform_supported(real_job.source):
            return SingleJobApplyResponse(
                job_id=job_id,
                title=real_job.title,
                company=real_job.company,
                platform=real_job.source,
                match_score=0.0,
                status="failed",
                error=f"Platform '{real_job.source}' is not supported.",
            )

        if not is_valid_canonical_job_url(real_job.source, real_job.url):
            return SingleJobApplyResponse(
                job_id=job_id,
                title=real_job.title,
                company=real_job.company,
                platform=real_job.source,
                match_score=0.0,
                status="failed",
                error=f"Invalid or non-canonical job URL: '{real_job.url}'",
            )

        # Acquire lock
        if not self.lock_manager.acquire_lock(job_id=job_id, agent_run_id=run.run_id):
            return SingleJobApplyResponse(
                job_id=job_id,
                title=real_job.title,
                company=real_job.company,
                platform=real_job.source,
                match_score=0.0,
                status="skipped",
                error="Job is currently locked by another agent run.",
            )

        try:
            # Mandatory Live Pre-Application Eligibility Gate
            eligibility = await self.eligibility_gate.evaluate_job_live(
                job_id=real_job.id,
                url=real_job.url or "",
                platform=real_job.source,
                profile_id=request.profile_id,
                job_title=real_job.title,
                company=real_job.company,
            )

            if not eligibility.is_eligible:
                logger.info(
                    "Job [%s] is not application-eligible: %s (%s)",
                    job_id,
                    eligibility.skip_reason,
                    eligibility.message,
                )
                self._record_eligibility_metrics(run, eligibility)
                self._record_log(
                    run,
                    job_id,
                    real_job.source,
                    "eligibility_check",
                    eligibility.status_code,
                    details=eligibility.diagnostics,
                    error=eligibility.message,
                )
                return SingleJobApplyResponse(
                    job_id=job_id,
                    title=real_job.title,
                    company=real_job.company,
                    platform=real_job.source,
                    match_score=100.0,
                    status=eligibility.status_code,
                    automation_state=eligibility.status_code,
                    application_method=eligibility.application_method.value,
                    availability_status=eligibility.availability_status.value,
                    application_eligible=False,
                    diagnostics=eligibility.diagnostics,
                    error=eligibility.message,
                )

            run.jobs_application_eligible += 1
            queue_item = QueueItem(
                job_id=real_job.id,
                source=real_job.source,
                external_id=real_job.external_id,
                title=real_job.title,
                company=real_job.company,
                location=real_job.location,
                url=real_job.url or "",
                match_score=100.0,
                status=QueueStatus.QUEUED,
            )

            return await self.process_job(
                job=queue_item,
                run=run,
                auto_submit=request.auto_submit,
                profile_id=request.profile_id,
            )
        finally:
            self.lock_manager.release_lock(job_id=job_id, agent_run_id=run.run_id)
            self.run_manager.update_status(run, AgentRunStatus.COMPLETED, "Single job finished")

    # =========================================================================
    # 3. CORE JOB APPLICATION PROCESSOR
    # =========================================================================
    async def process_job(
        self,
        job: QueueItem,
        run: AgentRun,
        auto_submit: bool,
        profile_id: Optional[UUID] = None,
    ) -> SingleJobApplyResponse:
        """Process a single job application under mutual exclusion lock."""
        async with self._execution_lock:
            job_id = job.job_id
            platform = job.source.lower().strip()
            url = job.url

            # Duplicate Application Guard (Database level)
            if profile_id:
                existing_app = self.app_repo.get_application_by_job_and_profile(job_id, profile_id)
                if existing_app and existing_app.status in ("submitted", "APPLICATION_SUBMITTED"):
                    logger.info("Job [%s] has already been submitted in database. Marking ALREADY_APPLIED.", job_id)
                    job.status = QueueStatus.ALREADY_APPLIED
                    run.jobs_already_applied += 1
                    self._record_log(run, job_id, platform, "duplicate_guard", OrchestratorState.ALREADY_APPLIED.value)
                    return SingleJobApplyResponse(
                        job_id=job_id,
                        title=job.title,
                        company=job.company,
                        platform=platform,
                        match_score=job.match_score,
                        status="already_applied",
                        automation_state="ALREADY_APPLIED",
                    )

            # APPLICATION_STARTING
            self._record_log(run, job_id, platform, "state_transition", OrchestratorState.APPLICATION_STARTING.value)
            run.jobs_attempted += 1

            # Dispatch to platform apply service
            apply_service = self.platform_router.get_apply_service_for_platform(platform)

            # APPLICATION_IN_PROGRESS
            self._record_log(run, job_id, platform, "state_transition", OrchestratorState.APPLICATION_IN_PROGRESS.value)

            result = None
            try:
                if platform == "glassdoor":
                    from app.automation.glassdoor.models import (
                        GlassdoorAutomationApplyRequest,
                        GlassdoorAutomationNavigateRequest,
                    )
                    if auto_submit:
                        req = GlassdoorAutomationApplyRequest(
                            job_id=job_id,
                            url=url,
                            source="glassdoor",
                            profile_id=profile_id,
                        )
                        # Final Submit Guard
                        self._record_log(run, job_id, platform, "state_transition", OrchestratorState.SUBMITTING.value)
                        result = await self._maybe_await(apply_service.apply_to_job(req))
                    else:
                        req = GlassdoorAutomationNavigateRequest(
                            job_id=job_id,
                            url=url,
                            source="glassdoor",
                            profile_id=profile_id,
                        )
                        result = await self._maybe_await(apply_service.navigate_to_submit(req))

                elif platform == "indeed":
                    from app.automation.indeed.models import (
                        IndeedAutomationApplyRequest,
                        IndeedAutomationNavigateRequest,
                    )
                    if auto_submit:
                        req = IndeedAutomationApplyRequest(
                            job_id=job_id,
                            url=url,
                            source="indeed",
                            profile_id=profile_id,
                            submit=True,
                        )
                        self._record_log(run, job_id, platform, "state_transition", OrchestratorState.SUBMITTING.value)
                        result = await self._maybe_await(apply_service.apply_to_job(req))
                    else:
                        req = IndeedAutomationNavigateRequest(
                            job_id=job_id,
                            url=url,
                            source="indeed",
                            profile_id=profile_id,
                        )
                        result = await self._maybe_await(apply_service.navigate_to_submit(req))
                else:
                    raise ValueError(f"Unsupported platform '{platform}'")

            except Exception as exc:
                logger.error("Exception during application of job [%s]: %s", job_id, exc, exc_info=True)
                job.status = QueueStatus.FAILED
                job.error = str(exc)
                run.jobs_failed += 1
                self._record_log(run, job_id, platform, "automation_exception", "failed", error=str(exc))
                return SingleJobApplyResponse(
                    job_id=job_id,
                    title=job.title,
                    company=job.company,
                    platform=platform,
                    match_score=job.match_score,
                    status="failed",
                    error=str(exc),
                )

            # Interpret Automation Result
            curr_state = str(getattr(result, "current_state", ""))
            if hasattr(getattr(result, "current_state", None), "value"):
                curr_state = result.current_state.value

            status_str = getattr(result, "status", "failed")
            is_manual = getattr(result, "manual_action_required", False)
            diag = getattr(result, "diagnostics", {}) or {}
            msg = getattr(result, "message", None)

            # State-specific mappings
            if is_manual or "CHALLENGE" in curr_state or "LOGIN" in curr_state or "MFA" in curr_state or "CAPTCHA" in curr_state:
                job.status = QueueStatus.NEEDS_USER_INPUT
                job.error = msg or "Challenge / Human intervention required"
                run.manual_action_required = True
                self._record_log(run, job_id, platform, "state_transition", OrchestratorState.WAITING_FOR_HUMAN_INPUT.value, details=diag)
                return SingleJobApplyResponse(
                    job_id=job_id,
                    title=job.title,
                    company=job.company,
                    platform=platform,
                    match_score=job.match_score,
                    status="manual_action_required",
                    automation_state=curr_state,
                    diagnostics=diag,
                    error=msg,
                )

            if "ALREADY_APPLIED" in curr_state:
                job.status = QueueStatus.ALREADY_APPLIED
                run.jobs_already_applied += 1
                self._record_log(run, job_id, platform, "state_transition", OrchestratorState.ALREADY_APPLIED.value, details=diag)
                return SingleJobApplyResponse(
                    job_id=job_id,
                    title=job.title,
                    company=job.company,
                    platform=platform,
                    match_score=job.match_score,
                    status="already_applied",
                    automation_state=curr_state,
                    application_method="UNKNOWN",
                    availability_status="AVAILABLE",
                    application_eligible=False,
                    diagnostics=diag,
                )

            if "UNAVAILABLE" in curr_state or "CLOSED" in curr_state:
                job.status = QueueStatus.FAILED
                job.error = "Job listing is closed or unavailable"
                run.jobs_failed += 1
                self._record_log(run, job_id, platform, "state_transition", OrchestratorState.JOB_UNAVAILABLE.value, details=diag)
                return SingleJobApplyResponse(
                    job_id=job_id,
                    title=job.title,
                    company=job.company,
                    platform=platform,
                    match_score=job.match_score,
                    status="blocked",
                    automation_state=curr_state,
                    application_method="UNKNOWN",
                    availability_status="UNAVAILABLE",
                    application_eligible=False,
                    diagnostics=diag,
                    error=job.error,
                )

            if getattr(result, "submission_confirmed", False) or "SUBMITTED" in curr_state:
                job.status = QueueStatus.SUBMITTED
                run.applications_submitted += 1
                self._record_log(run, job_id, platform, "state_transition", OrchestratorState.SUBMITTED.value, details=diag)
                if profile_id:
                    self.app_repo.record_application_submission(
                        job_id=job_id,
                        profile_id=profile_id,
                        platform=platform,
                        application_url=url,
                        match_score=job.match_score,
                        agent_run_id=run.run_id,
                        status="submitted",
                    )
                return SingleJobApplyResponse(
                    job_id=job_id,
                    title=job.title,
                    company=job.company,
                    platform=platform,
                    match_score=job.match_score,
                    status="success",
                    automation_state=curr_state,
                    application_method="EASY_APPLY",
                    availability_status="AVAILABLE",
                    application_eligible=True,
                    diagnostics=diag,
                )

            if getattr(result, "submit_verified", False) or "SUBMISSION_READY" in curr_state:
                job.status = QueueStatus.SUBMISSION_READY
                self._record_log(run, job_id, platform, "state_transition", OrchestratorState.SUBMISSION_READY.value, details=diag)
                if profile_id:
                    self.app_repo.record_application_submission(
                        job_id=job_id,
                        profile_id=profile_id,
                        platform=platform,
                        application_url=url,
                        match_score=job.match_score,
                        agent_run_id=run.run_id,
                        status="submission_ready",
                    )
                return SingleJobApplyResponse(
                    job_id=job_id,
                    title=job.title,
                    company=job.company,
                    platform=platform,
                    match_score=job.match_score,
                    status="success" if not auto_submit else "submission_ready",
                    automation_state=curr_state,
                    application_method="EASY_APPLY",
                    availability_status="AVAILABLE",
                    application_eligible=True,
                    diagnostics=diag,
                )

            # General blocked or failed
            job.status = QueueStatus.FAILED
            job.error = msg or "Application stopped without reaching submission ready"
            run.jobs_failed += 1
            self._record_log(run, job_id, platform, "application_failed", curr_state, details=diag, error=msg)
            return SingleJobApplyResponse(
                job_id=job_id,
                title=job.title,
                company=job.company,
                platform=platform,
                match_score=job.match_score,
                status="failed" if status_str == "failed" else "blocked",
                automation_state=curr_state,
                diagnostics=diag,
                error=msg,
            )

    # =========================================================================
    # 4. RESUME, SKIP, ABORT ACTIONS
    # =========================================================================
    async def resume_run(self, request: AgentResumeRequest) -> AgentRunResponse:
        """Resume an active or paused agent run after user completed manual action."""
        run = self.run_manager.get_run(request.agent_run_id)
        if not run:
            raise ValueError(f"Agent run '{request.agent_run_id}' not found.")

        if not run.manual_action_required and run.status != AgentRunStatus.PAUSED_MANUAL_ACTION:
            raise ValueError(f"Agent run '{request.agent_run_id}' is not in paused/manual action state.")

        logger.info("Resuming Agent Run [%s] for job [%s]", run.run_id, run.current_job_id)
        run.manual_action_required = False
        run.pause_reason = None
        self.run_manager.update_status(run, AgentRunStatus.APPLYING, "Resuming application cycle")

        # Re-dispatch current job if set
        results = []
        if run.current_job_id:
            real_job = self.job_repo.get_job_by_id(run.current_job_id)
            if real_job:
                q_item = QueueItem(
                    job_id=real_job.id,
                    source=real_job.source,
                    external_id=real_job.external_id,
                    title=real_job.title,
                    company=real_job.company,
                    location=real_job.location,
                    url=real_job.url or "",
                    match_score=100.0,
                    status=QueueStatus.QUEUED,
                )
                auto_sub = run.configuration.get("auto_submit", False)
                res = await self.process_job(q_item, run, auto_sub, run.profile_id)
                results.append(res)

        self.run_manager.update_status(run, AgentRunStatus.COMPLETED, "Resume cycle complete")
        return self._build_response(run, OrchestratorState.COMPLETED, results=results)

    async def skip_job(self, request: AgentSkipRequest) -> AgentRunResponse:
        """Skip the current or specified job, release locks, and continue."""
        run = self.run_manager.get_run(request.agent_run_id)
        if not run:
            raise ValueError(f"Agent run '{request.agent_run_id}' not found.")

        target_job_id = request.job_id or run.current_job_id
        if target_job_id:
            self.lock_manager.release_lock(job_id=target_job_id, agent_run_id=run.run_id)
            run.jobs_skipped += 1
            if run.current_job_id == target_job_id:
                run.current_job_id = None
                run.manual_action_required = False
                run.pause_reason = None

        self.run_manager.update_status(run, AgentRunStatus.COMPLETED, f"Skipped job {target_job_id}")
        return self._build_response(run, OrchestratorState.COMPLETED)

    async def abort_run(self, request: AgentAbortRequest) -> AgentRunResponse:
        """Cancel run execution, release all acquired locks, and preserve logs."""
        run = self.run_manager.get_run(request.agent_run_id)
        if not run:
            raise ValueError(f"Agent run '{request.agent_run_id}' not found.")

        run_id_str = str(run.run_id)
        self._abort_requested.add(run_id_str)
        self.lock_manager.release_all_for_run(run.run_id)
        self.run_manager.update_status(run, AgentRunStatus.CANCELLED, "Aborted by user")
        self._record_log(run, None, "orchestrator", "run_aborted", "cancelled")

        return self._build_response(run, OrchestratorState.COMPLETED)

    # =========================================================================
    # 5. INTERNAL HELPERS
    # =========================================================================
    def select_next_job(self, queue: List[QueueItem]) -> Optional[QueueItem]:
        """Rank and select the highest priority eligible job from queue."""
        if not queue:
            return None
        # Sort by: -match_score, then priority
        queue.sort(key=lambda item: (-item.match_score, item.priority))
        return queue[0]

    def _build_queue(
        self,
        top_matches: List[MatchedJobItem],
        minimum_score: float,
        profile_id: Optional[UUID] = None,
    ) -> List[QueueItem]:
        """Filter, deduplicate, and construct application queue using real Supabase job IDs."""
        queue: List[QueueItem] = []
        seen_job_ids: Set[UUID] = set()

        # Pre-fetch existing submitted applications
        submitted_ids: Set[UUID] = set()
        if profile_id:
            existing = self.app_repo.get_applications_for_jobs(
                [m.job_id for m in top_matches], profile_id=profile_id
            )
            submitted_ids = {a.job_id for a in existing if a.status in ("submitted", "APPLICATION_SUBMITTED")}

        for m in top_matches:
            # Rejection filters
            if m.job_id in seen_job_ids or m.job_id in submitted_ids:
                continue
            if m.final_score < minimum_score:
                continue
            if (m.decision or "").lower() in ("reject", "low_match"):
                continue
            if not self.platform_router.is_platform_supported(m.source):
                continue
            if not is_valid_canonical_job_url(m.source, m.url):
                continue

            seen_job_ids.add(m.job_id)
            queue.append(
                QueueItem(
                    job_id=m.job_id,
                    source=m.source,
                    title=m.title,
                    company=m.company,
                    location=m.location,
                    url=m.url or "",
                    match_score=m.final_score,
                    status=QueueStatus.QUEUED,
                    decision=m.decision,
                )
            )

        return queue

    def _record_eligibility_metrics(self, run: AgentRun, eligibility: EligibilityResult) -> None:
        """Update run counters for skipped / ineligible jobs."""
        run.jobs_ineligible += 1
        if (
            eligibility.skip_reason in ("SKIPPED_EXTERNAL_APPLY", "EXTERNAL_APPLY")
            or eligibility.application_method == ApplicationMethod.EXTERNAL_APPLY
        ):
            run.jobs_external_apply += 1
            run.jobs_skipped += 1
        elif (
            eligibility.skip_reason in ("SKIPPED_EXPIRED", "EXPIRED")
            or eligibility.availability_status == AvailabilityStatus.EXPIRED
        ):
            run.jobs_expired += 1
            run.jobs_skipped += 1
        elif (
            eligibility.skip_reason in ("SKIPPED_UNAVAILABLE", "JOB_UNAVAILABLE", "UNAVAILABLE")
            or eligibility.availability_status == AvailabilityStatus.UNAVAILABLE
        ):
            run.jobs_unavailable += 1
            run.jobs_skipped += 1
        elif eligibility.skip_reason == "ALREADY_APPLIED":
            run.jobs_already_applied += 1
        else:
            run.jobs_skipped += 1

    def _record_log(
        self,
        run: AgentRun,
        job_id: Optional[UUID],
        platform: str,
        action: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Persist automation log entry via ApplicationRepository."""
        try:
            self.app_repo.record_automation_log(
                platform=platform,
                action=action,
                status=status,
                job_id=job_id,
                agent_run_id=run.run_id,
                details=details,
                error=error,
            )
        except Exception as exc:
            logger.debug("Failed to record automation log: %s", exc)

    def _build_response(
        self,
        run: AgentRun,
        status: OrchestratorState,
        results: Optional[List[SingleJobApplyResponse]] = None,
        remaining_queue: Optional[List[QueueItem]] = None,
    ) -> AgentRunResponse:
        """Construct AgentRunResponse from domain run state."""
        return AgentRunResponse(
            run_id=run.run_id,
            platform=run.platform,
            status=status,
            started_at=run.started_at,
            completed_at=run.completed_at,
            jobs_discovered=run.jobs_discovered,
            jobs_matched=run.jobs_scored,
            jobs_queued=run.jobs_eligible,
            jobs_attempted=run.jobs_attempted,
            jobs_submitted=run.applications_submitted,
            jobs_skipped=run.jobs_skipped,
            jobs_already_applied=run.jobs_already_applied,
            jobs_ineligible=run.jobs_ineligible,
            jobs_external_apply=run.jobs_external_apply,
            jobs_expired=run.jobs_expired,
            jobs_unavailable=run.jobs_unavailable,
            jobs_application_eligible=run.jobs_application_eligible,
            jobs_failed=run.jobs_failed,
            manual_actions_required=run.manual_action_required,
            pause_reason=run.pause_reason,
            current_job_id=run.current_job_id,
            results=results or [],
            remaining_queue=remaining_queue or [],
        )

    async def _maybe_await(self, val: Any) -> Any:
        """Safely await coroutines or return raw sync values."""
        if asyncio.iscoroutine(val):
            return await val
        if hasattr(val, "__await__"):
            return await val
        return val
