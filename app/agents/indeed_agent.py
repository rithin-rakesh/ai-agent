"""Indeed Job Application Agent (Phase 5.5).

Orchestrates candidate profile loading, broad Indeed job discovery, deterministic & semantic matching,
eligibility filtering, deduplication, queue sorting, and automated assisted application execution.
"""

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Set, Tuple
from uuid import UUID

from app.agents.models import (
    AgentRun,
    AgentRunStatus,
    AgentRunSummary,
    ApplicationTask,
    ApplicationTaskResult,
    IndeedAgentRunRequest,
)
from app.agents.run_manager import AgentRunManager
from app.automation.indeed.apply_service import IndeedApplyService
from app.automation.indeed.models import (
    AutomationState,
    IndeedAutomationApplyRequest,
    IndeedAutomationNavigateRequest,
    IndeedAutomationResumeRequest,
)
from app.automation.indeed.url_validator import extract_indeed_jk
from app.database.repositories.application_repository import ApplicationRepository
from app.discovery.profile_discovery_service import ProfileDiscoveryService
from app.models.application import ApplicationCreate, AutomationLogCreate
from app.models.job import MatchedJobItem, ProfileJobSearchRequest
from app.profile.service import ProfileService

logger = logging.getLogger(__name__)


def _extract_indeed_jk(url: str) -> Optional[str]:
    """Extract Indeed job key (jk) query param from URL if present."""
    return extract_indeed_jk(url)


class IndeedApplicationAgent:
    """Agent orchestrating end-to-end Indeed job search and assisted application."""

    def __init__(
        self,
        discovery_service: Optional[ProfileDiscoveryService] = None,
        apply_service: Optional[IndeedApplyService] = None,
        run_manager: Optional[AgentRunManager] = None,
        application_repository: Optional[ApplicationRepository] = None,
        profile_service: Optional[ProfileService] = None,
    ) -> None:
        self.discovery_service = discovery_service or ProfileDiscoveryService()
        self.apply_service = apply_service or IndeedApplyService()
        self.run_manager = run_manager or AgentRunManager()
        self.app_repo = application_repository or ApplicationRepository()
        self.profile_service = profile_service or ProfileService()

    async def run(self, request: IndeedAgentRunRequest) -> AgentRunSummary:
        """Execute a full Indeed job application workflow from discovery through submission."""
        # 1. Initialize Run in CREATED state
        run = self.run_manager.create_run(
            platform="indeed",
            profile_id=request.profile_id,
            configuration=request.model_dump(mode="json"),
        )
        logger.info("Starting Indeed Agent Run [%s]", run.run_id)

        try:
            # 2. Discovery Stage
            self.run_manager.update_status(run, AgentRunStatus.DISCOVERING, "Generating queries and querying Indeed")
            disc_request = ProfileJobSearchRequest(
                results_per_query=request.results_per_query,
                hours_old=request.hours_old,
                max_queries=request.max_queries,
                minimum_score=request.minimum_score,
                use_semantic=request.use_semantic,
                use_nvidia_expansion=request.use_nvidia_expansion,
                sites=["indeed"],  # Strictly Indeed only
                limit=100,
            )

            disc_response = await self.discovery_service.discover_jobs_from_profile(
                request=disc_request,
                profile_id=request.profile_id,
            )

            # 3. Matching & Metrics Stage
            self.run_manager.update_status(run, AgentRunStatus.MATCHING, "Scoring and evaluating candidates")
            disc_stats = getattr(disc_response, "discovery", None) or getattr(disc_response, "stats", None)
            matching_stats = getattr(disc_response, "matching", None)

            jobs_disc = getattr(disc_stats, "unique_jobs", getattr(disc_stats, "raw_jobs", getattr(disc_stats, "jobs_discovered", len(disc_response.top_matches)))) if disc_stats else len(disc_response.top_matches)
            jobs_sc = getattr(matching_stats, "jobs_scored", getattr(disc_stats, "jobs_scored", len(disc_response.top_matches))) if (matching_stats or disc_stats) else len(disc_response.top_matches)

            self.run_manager.record_discovery_and_matching(
                run=run,
                jobs_discovered=jobs_disc,
                jobs_scored=jobs_sc,
                jobs_eligible=0,  # Updated after filtering
            )

            # 4. Building Queue Stage
            self.run_manager.update_status(run, AgentRunStatus.BUILDING_QUEUE, "Filtering and deduplicating eligible queue")
            queue = self._build_application_queue(
                top_matches=disc_response.top_matches,
                allowed_decisions=set(d.lower() for d in request.allowed_decisions),
                minimum_score=request.minimum_score,
                profile_id=request.profile_id,
            )

            run.jobs_eligible = len(queue)
            self.run_manager.repository.save_run(run)
            logger.info("Application queue built with %d eligible tasks", len(queue))

            if not queue:
                self.run_manager.update_status(run, AgentRunStatus.COMPLETED, "No eligible unapplied jobs in queue")
                summary = self.run_manager.get_run_summary(run.run_id)
                assert summary is not None
                return summary

            # 5. Applying Stage
            self.run_manager.update_status(run, AgentRunStatus.APPLYING, f"Processing queue (max {request.max_applications} attempts)")
            return await self._process_application_queue(
                run=run,
                queue=queue,
                max_applications=request.max_applications,
                submit=request.submit,
                profile_id=request.profile_id,
            )

        except Exception as exc:
            logger.error("Indeed Agent Run [%s] failed with unexpected error: %s", run.run_id, exc, exc_info=True)
            self.run_manager.update_status(run, AgentRunStatus.FAILED, f"Unexpected error: {exc}")
            summary = self.run_manager.get_run_summary(run.run_id)
            assert summary is not None
            return summary

    async def resume_run(self, run_id: UUID) -> AgentRunSummary:
        """Resume an existing paused agent run after user completed manual action."""
        run = self.run_manager.get_run(run_id)
        if not run:
            raise ValueError(f"AgentRun [{run_id}] not found.")

        if run.status != AgentRunStatus.PAUSED_MANUAL_ACTION:
            raise ValueError(f"AgentRun [{run_id}] is in state {run.status.value}, not PAUSED_MANUAL_ACTION.")

        logger.info("Resuming AgentRun [%s] for paused job [%s]", run_id, run.current_job_id)

        # 1. Locate current job details
        paused_task = None
        for res in run.results:
            if res.job_id == run.current_job_id:
                paused_task = res
                break

        if not paused_task:
            # Fallback: check remaining queue
            if run.remaining_queue:
                t = run.remaining_queue[0]
                paused_task = ApplicationTaskResult(
                    job_id=t.job_id,
                    external_job_id=t.external_job_id,
                    platform=t.platform,
                    url=t.url,
                    title=t.title,
                    company=t.company,
                    match_score=t.match_score,
                    match_decision=t.match_decision,
                    status="manual_action_required",
                )

        if not paused_task:
            raise ValueError(f"Cannot locate task details for paused job [{run.current_job_id}].")

        # 2. Attempt resume-submit on the active application window
        resume_req = IndeedAutomationResumeRequest(
            job_id=paused_task.job_id,
            url=paused_task.url,
            source="indeed",
            profile_id=run.profile_id,
        )
        resume_result = self.apply_service.resume_submit(resume_req)

        # 3. Check outcome
        if resume_result.current_state in (
            AutomationState.CAPTCHA_OR_CHALLENGE,
            AutomationState.LOGIN_REQUIRED,
            AutomationState.MFA_REQUIRED,
        ) or resume_result.manual_action_required:
            logger.warning("Challenge is still active on resume attempt. Remaining paused.")
            run.manual_action_required = True
            run.pause_reason = resume_result.current_state.value
            self.run_manager.repository.save_run(run)
            summary = self.run_manager.get_run_summary(run.run_id)
            assert summary is not None
            return summary

        # Challenge resolved: update paused task outcome
        if resume_result.current_state == AutomationState.APPLICATION_SUBMITTED:
            logger.info("Paused job [%s] confirmed submitted after resume!", paused_task.job_id)
            now = datetime.now(timezone.utc)
            paused_task.status = "success"
            paused_task.automation_state = AutomationState.APPLICATION_SUBMITTED.value
            paused_task.submitted_at = now
            paused_task.diagnostics = resume_result.diagnostics

            run.applications_submitted += 1
            run.manual_action_required = False
            run.pause_reason = None
            run.current_job_id = None

            # Persist application record
            if run.profile_id:
                self._persist_application_record(paused_task, run.profile_id, now)
        else:
            logger.warning("Resume submit for job [%s] ended in %s", paused_task.job_id, resume_result.current_state)
            paused_task.status = "failed"
            paused_task.automation_state = resume_result.current_state.value
            paused_task.failure_reason = resume_result.message
            run.jobs_failed += 1
            run.manual_action_required = False
            run.pause_reason = None
            run.current_job_id = None

        self.run_manager.repository.save_run(run)

        # 4. Continue with remaining queue
        queue = list(run.remaining_queue)
        run.remaining_queue = []
        max_apps = run.configuration.get("max_applications", 3)
        submit_flag = run.configuration.get("submit", True)

        return await self._process_application_queue(
            run=run,
            queue=queue,
            max_applications=max_apps,
            submit=submit_flag,
            profile_id=run.profile_id,
        )

    def _build_application_queue(
        self,
        top_matches: List[MatchedJobItem],
        allowed_decisions: Set[str],
        minimum_score: float,
        profile_id: Optional[UUID] = None,
    ) -> List[ApplicationTask]:
        """Filter, deduplicate, and sort eligible tasks for the application queue."""
        # 1. Fetch existing submitted applications to prevent duplicate attempts
        job_ids = [m.job_id for m in top_matches]
        existing_apps = self.app_repo.get_applications_for_jobs(job_ids, profile_id=profile_id)
        submitted_job_ids = {
            app.job_id for app in existing_apps
            if app.status in ("submitted", "APPLICATION_SUBMITTED", "pending", "in_progress")
        }

        # Track seen IDs, jks, and URLs in current run to guarantee queue uniqueness
        seen_job_ids: Set[UUID] = set()
        seen_jks: Set[str] = set()
        seen_urls: Set[str] = set()

        queue: List[ApplicationTask] = []

        for m in top_matches:
            dec = (m.decision or "").lower()
            if dec not in allowed_decisions:
                continue
            if m.final_score < minimum_score:
                continue
            if m.job_id in submitted_job_ids:
                logger.info("Skipping job [%s] - already submitted in database", m.job_id)
                continue
            if m.job_id in seen_job_ids:
                continue

            jk = _extract_indeed_jk(m.url)
            if jk and jk in seen_jks:
                logger.info("Skipping job [%s] - duplicate jk [%s]", m.job_id, jk)
                continue

            if m.url in seen_urls:
                continue

            seen_job_ids.add(m.job_id)
            if jk:
                seen_jks.add(jk)
            seen_urls.add(m.url)

            task = ApplicationTask(
                job_id=m.job_id,
                external_job_id=jk,
                platform="indeed",
                url=m.url,
                title=m.title,
                company=m.company,
                location=m.location,
                match_score=m.final_score,
                match_decision=m.decision,
                profile_id=profile_id,
                status="PENDING",
            )
            queue.append(task)

        # Sort descending by final match score
        queue.sort(key=lambda t: t.match_score, reverse=True)
        return queue

    async def _process_application_queue(
        self,
        run: AgentRun,
        queue: List[ApplicationTask],
        max_applications: int,
        submit: bool,
        profile_id: Optional[UUID],
    ) -> AgentRunSummary:
        """Iterate through application queue and execute platform automation."""
        idx = 0
        while idx < len(queue) and run.jobs_attempted < max_applications:
            task = queue[idx]
            idx += 1

            logger.info("Processing application task [%s]: '%s' at '%s'", task.job_id, task.title, task.company)
            task.attempt_count += 1
            task.last_attempt_at = datetime.now(timezone.utc)

            if not submit:
                # Dry-Run Mode: Navigate up to Submit verification without clicking final Submit
                nav_req = IndeedAutomationNavigateRequest(
                    job_id=task.job_id,
                    url=task.url,
                    source="indeed",
                    profile_id=profile_id,
                )
                nav_res = self.apply_service.navigate_to_submit(nav_req)

                if nav_res.current_state == AutomationState.SUBMISSION_READY:
                    res = ApplicationTaskResult(
                        job_id=task.job_id,
                        external_job_id=task.external_job_id,
                        platform="indeed",
                        url=task.url,
                        title=task.title,
                        company=task.company,
                        match_score=task.match_score,
                        match_decision=task.match_decision,
                        status="success",
                        automation_state=AutomationState.SUBMISSION_READY.value,
                        diagnostics=nav_res.diagnostics or {},
                    )
                    self.run_manager.record_attempt_result(run, res)
                else:
                    outcome = "skipped" if nav_res.current_state in (
                        AutomationState.ALREADY_APPLIED,
                        AutomationState.EXTERNAL_APPLY,
                        AutomationState.APPLY_WITH_INDEED_NOT_VERIFIED,
                    ) else "failed"
                    res = ApplicationTaskResult(
                        job_id=task.job_id,
                        external_job_id=task.external_job_id,
                        platform="indeed",
                        url=task.url,
                        title=task.title,
                        company=task.company,
                        match_score=task.match_score,
                        match_decision=task.match_decision,
                        status=outcome,
                        automation_state=nav_res.current_state.value,
                        failure_reason=nav_res.message,
                        diagnostics=nav_res.diagnostics or {},
                    )
                    self.run_manager.record_attempt_result(run, res)
                continue

            # Live Assisted Apply Mode
            apply_req = IndeedAutomationApplyRequest(
                job_id=task.job_id,
                url=task.url,
                source="indeed",
                profile_id=profile_id,
            )
            apply_res = self.apply_service.apply_to_job(apply_req)

            # 1. Success Outcome
            if apply_res.current_state == AutomationState.APPLICATION_SUBMITTED:
                now = datetime.now(timezone.utc)
                task_res = ApplicationTaskResult(
                    job_id=task.job_id,
                    external_job_id=task.external_job_id,
                    platform="indeed",
                    url=task.url,
                    title=task.title,
                    company=task.company,
                    match_score=task.match_score,
                    match_decision=task.match_decision,
                    status="success",
                    automation_state=AutomationState.APPLICATION_SUBMITTED.value,
                    submitted_at=now,
                    diagnostics=apply_res.diagnostics or {},
                )
                self.run_manager.record_attempt_result(run, task_res)
                if profile_id:
                    self._persist_application_record(task_res, profile_id, now)
                continue

            # 2. Human Verification / Challenge Pause Outcome
            if apply_res.current_state in (
                AutomationState.CAPTCHA_OR_CHALLENGE,
                AutomationState.LOGIN_REQUIRED,
                AutomationState.MFA_REQUIRED,
            ) or apply_res.manual_action_required:
                task_res = ApplicationTaskResult(
                    job_id=task.job_id,
                    external_job_id=task.external_job_id,
                    platform="indeed",
                    url=task.url,
                    title=task.title,
                    company=task.company,
                    match_score=task.match_score,
                    match_decision=task.match_decision,
                    status="manual_action_required",
                    automation_state=apply_res.current_state.value,
                    failure_reason=apply_res.message,
                    diagnostics=apply_res.diagnostics or {},
                )
                run.results.append(task_res)
                run.jobs_attempted += 1
                remaining = queue[idx:]
                self.run_manager.pause_run_for_manual_action(
                    run=run,
                    current_job_id=task.job_id,
                    pause_reason=apply_res.current_state.value,
                    remaining_queue=remaining,
                )
                summary = self.run_manager.get_run_summary(run.run_id)
                assert summary is not None
                return summary

            # 3. Critical Environment / Browser Failure Outcome
            if apply_res.current_state in (
                AutomationState.BROWSER_NOT_VERIFIED,
                AutomationState.UNSUPPORTED_PLATFORM,
            ):
                task_res = ApplicationTaskResult(
                    job_id=task.job_id,
                    external_job_id=task.external_job_id,
                    platform="indeed",
                    url=task.url,
                    title=task.title,
                    company=task.company,
                    match_score=task.match_score,
                    match_decision=task.match_decision,
                    status="failed",
                    automation_state=apply_res.current_state.value,
                    failure_reason=apply_res.message,
                    diagnostics=apply_res.diagnostics or {},
                )
                self.run_manager.record_attempt_result(run, task_res)
                self.run_manager.update_status(run, AgentRunStatus.FAILED, apply_res.message)
                summary = self.run_manager.get_run_summary(run.run_id)
                assert summary is not None
                return summary

            # 4. Ordinary Job Failure Outcome (skip or failed, continue to next job)
            outcome = "skipped" if apply_res.current_state in (
                AutomationState.ALREADY_APPLIED,
                AutomationState.EXTERNAL_APPLY,
                AutomationState.APPLY_WITH_INDEED_NOT_VERIFIED,
            ) else "failed"
            task_res = ApplicationTaskResult(
                job_id=task.job_id,
                external_job_id=task.external_job_id,
                platform="indeed",
                url=task.url,
                title=task.title,
                company=task.company,
                match_score=task.match_score,
                match_decision=task.match_decision,
                status=outcome,
                automation_state=apply_res.current_state.value,
                failure_reason=apply_res.message,
                diagnostics=apply_res.diagnostics or {},
            )
            self.run_manager.record_attempt_result(run, task_res)

        # Queue processing finished
        if run.jobs_failed > 0 and run.applications_submitted == 0:
            final_status = AgentRunStatus.COMPLETED_WITH_ERRORS
        elif run.jobs_failed > 0:
            final_status = AgentRunStatus.COMPLETED_WITH_ERRORS
        else:
            final_status = AgentRunStatus.COMPLETED

        self.run_manager.update_status(run, final_status, "Run completed queue processing")
        summary = self.run_manager.get_run_summary(run.run_id)
        assert summary is not None
        return summary

    def _persist_application_record(
        self,
        task_res: ApplicationTaskResult,
        profile_id: UUID,
        submitted_at: datetime,
    ) -> None:
        """Record successful application entry and automation log in Supabase."""
        try:
            app_create = ApplicationCreate(
                job_id=task_res.job_id,
                profile_id=profile_id,
                platform="indeed",
                status="submitted",
                started_at=task_res.attempted_at,
                submitted_at=submitted_at,
                application_url=task_res.url,
                confirmation_text="Application submitted and confirmed via Indeed PyWinAuto agent.",
            )
            app_rec = self.app_repo.create_application(app_create)
            if app_rec:
                log_create = AutomationLogCreate(
                    application_id=app_rec.id,
                    platform="indeed",
                    action="apply_and_submit",
                    status="success",
                    details=task_res.diagnostics,
                )
                self.app_repo.create_automation_log(log_create)
        except Exception as exc:
            logger.warning("Failed to persist application record for job %s: %s", task_res.job_id, exc)
