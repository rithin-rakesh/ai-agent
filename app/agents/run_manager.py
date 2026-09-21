"""Agent Run Manager for managing lifecycle states and run persistence (Phase 5.5).

Handles run state transitions, progress tracking, result accumulation, and state recovery.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.agents.models import (
    AgentRun,
    AgentRunStatus,
    AgentRunSummary,
    ApplicationTask,
    ApplicationTaskResult,
)
from app.database.repositories.agent_run_repository import AgentRunRepository

logger = logging.getLogger(__name__)


class AgentRunManager:
    """Manages the creation, state transitions, and persistence of Agent Runs."""

    def __init__(self, repository: Optional[AgentRunRepository] = None) -> None:
        self.repository = repository or AgentRunRepository()

    def create_run(
        self,
        platform: str,
        profile_id: Optional[UUID] = None,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> AgentRun:
        """Initialize a new AgentRun in CREATED state."""
        run = AgentRun(
            run_id=uuid4(),
            platform=platform,
            profile_id=profile_id,
            status=AgentRunStatus.CREATED,
            started_at=datetime.now(timezone.utc),
            configuration=configuration or {},
        )
        self.repository.save_run(run)
        logger.info("Created AgentRun [run_id=%s, platform=%s]", run.run_id, platform)
        return run

    def update_status(self, run: AgentRun, status: AgentRunStatus, message: Optional[str] = None) -> AgentRun:
        """Update run status and persist state."""
        run.status = status
        if status in (AgentRunStatus.COMPLETED, AgentRunStatus.COMPLETED_WITH_ERRORS, AgentRunStatus.FAILED, AgentRunStatus.CANCELLED):
            if not run.completed_at:
                run.completed_at = datetime.now(timezone.utc)
        self.repository.save_run(run)
        logger.info("AgentRun [%s] transitioned to %s (%s)", run.run_id, status.value, message or "")
        return run

    def record_discovery_and_matching(
        self,
        run: AgentRun,
        jobs_discovered: int,
        jobs_scored: int,
        jobs_eligible: int,
    ) -> AgentRun:
        """Update metrics for discovery and matching stages."""
        run.jobs_discovered = jobs_discovered
        run.jobs_scored = jobs_scored
        run.jobs_eligible = jobs_eligible
        self.repository.save_run(run)
        return run

    def record_attempt_result(
        self,
        run: AgentRun,
        result: ApplicationTaskResult,
    ) -> AgentRun:
        """Record the outcome of a single job application attempt."""
        run.results.append(result)
        run.jobs_attempted += 1

        if result.status == "success" and result.submitted_at:
            run.applications_submitted += 1
        elif result.status == "skipped":
            run.jobs_skipped += 1
        elif result.status in ("failed", "blocked") and not result.failure_reason:
            run.jobs_failed += 1
        elif result.status == "failed":
            run.jobs_failed += 1

        self.repository.save_run(run)
        return run

    def pause_run_for_manual_action(
        self,
        run: AgentRun,
        current_job_id: UUID,
        pause_reason: str,
        remaining_queue: List[ApplicationTask],
    ) -> AgentRun:
        """Pause the run for human action (CAPTCHA, Login, MFA) and preserve state."""
        run.status = AgentRunStatus.PAUSED_MANUAL_ACTION
        run.manual_action_required = True
        run.pause_reason = pause_reason
        run.current_job_id = current_job_id
        run.remaining_queue = remaining_queue
        self.repository.save_run(run)
        logger.warning(
            "AgentRun [%s] PAUSED for manual action on job [%s]: %s. Preserved %d remaining tasks.",
            run.run_id,
            current_job_id,
            pause_reason,
            len(remaining_queue),
        )
        return run

    def get_run(self, run_id: UUID) -> Optional[AgentRun]:
        """Fetch an existing agent run by ID."""
        return self.repository.get_run_by_id(run_id)

    def get_run_summary(self, run_id: UUID) -> Optional[AgentRunSummary]:
        """Fetch run summary for API reporting."""
        run = self.get_run(run_id)
        if not run:
            return None
        return AgentRunSummary(
            run_id=run.run_id,
            platform=run.platform,
            status=run.status,
            started_at=run.started_at,
            completed_at=run.completed_at,
            jobs_discovered=run.jobs_discovered,
            jobs_scored=run.jobs_scored,
            jobs_eligible=run.jobs_eligible,
            jobs_attempted=run.jobs_attempted,
            applications_submitted=run.applications_submitted,
            jobs_skipped=run.jobs_skipped,
            jobs_failed=run.jobs_failed,
            manual_action_required=run.manual_action_required,
            pause_reason=run.pause_reason,
            current_job_id=run.current_job_id,
            results=run.results,
        )
