"""Agent Run Repository for Supabase database operations.

Handles persistence, retrieval, and updates for the 'agent_runs' table in Supabase.
Includes graceful fallback to in-memory storage when offline or unmigrated.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID
from supabase import Client

from app.agents.models import AgentRun, AgentRunStatus
from app.database.supabase import get_supabase_client, get_supabase_service_client

logger = logging.getLogger(__name__)


def _serialize_agent_run(run: AgentRun) -> Dict[str, Any]:
    """Serialize an AgentRun into a dictionary suitable for Supabase PostgreSQL."""
    payload = {
        "id": str(run.run_id),
        "platform": run.platform,
        "profile_id": str(run.profile_id) if run.profile_id else None,
        "status": run.status.value if isinstance(run.status, AgentRunStatus) else str(run.status),
        "current_job_id": str(run.current_job_id) if run.current_job_id else None,
        "configuration_json": run.configuration,
        "summary_json": {
            "jobs_discovered": run.jobs_discovered,
            "jobs_scored": run.jobs_scored,
            "jobs_eligible": run.jobs_eligible,
            "jobs_attempted": run.jobs_attempted,
            "applications_submitted": run.applications_submitted,
            "jobs_skipped": run.jobs_skipped,
            "jobs_failed": run.jobs_failed,
            "results": [r.model_dump(mode="json") for r in run.results],
            "remaining_queue": [t.model_dump(mode="json") for t in run.remaining_queue],
        },
        "pause_reason": run.pause_reason,
        "manual_action_required": run.manual_action_required,
        "started_at": run.started_at.isoformat() if run.started_at else datetime.now(timezone.utc).isoformat(),
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }
    return payload


def _deserialize_agent_run(data: Dict[str, Any]) -> AgentRun:
    """Reconstruct an AgentRun model from a Supabase row dictionary."""
    summary = data.get("summary_json") or {}
    results = summary.get("results", [])
    remaining_queue = summary.get("remaining_queue", [])

    return AgentRun(
        run_id=UUID(data["id"]),
        platform=data.get("platform", "indeed"),
        profile_id=UUID(data["profile_id"]) if data.get("profile_id") else None,
        status=AgentRunStatus(data.get("status", "CREATED")),
        started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else datetime.now(timezone.utc),
        completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        current_job_id=UUID(data["current_job_id"]) if data.get("current_job_id") else None,
        jobs_discovered=summary.get("jobs_discovered", 0),
        jobs_scored=summary.get("jobs_scored", 0),
        jobs_eligible=summary.get("jobs_eligible", 0),
        jobs_attempted=summary.get("jobs_attempted", 0),
        applications_submitted=summary.get("applications_submitted", 0),
        jobs_skipped=summary.get("jobs_skipped", 0),
        jobs_failed=summary.get("jobs_failed", 0),
        manual_action_required=data.get("manual_action_required", False),
        pause_reason=data.get("pause_reason"),
        configuration=data.get("configuration_json") or {},
        results=results,
        remaining_queue=remaining_queue,
    )


class AgentRunRepository:
    """Repository handling CRUD operations for AgentRun records."""

    _shared_memory_store: Dict[UUID, AgentRun] = {}

    def __init__(self, client: Optional[Client] = None) -> None:
        self._client = client
        self._memory_store: Dict[UUID, AgentRun] = self._shared_memory_store

    @property
    def client(self) -> Optional[Client]:
        """Retrieve active Supabase client or None if unconfigured."""
        if self._client is not None:
            return self._client
        try:
            return get_supabase_service_client()
        except Exception:
            try:
                return get_supabase_client()
            except Exception:
                return None

    def save_run(self, run: AgentRun) -> AgentRun:
        """Persist or update an AgentRun in Supabase and memory."""
        self._memory_store[run.run_id] = run
        c = self.client
        if c is not None:
            try:
                payload = _serialize_agent_run(run)
                c.table("agent_runs").upsert(payload, on_conflict="id").execute()
            except Exception as exc:
                logger.debug("Database persistence skipped/failed, keeping in-memory state: %s", exc)
        return run

    def get_run_by_id(self, run_id: UUID) -> Optional[AgentRun]:
        """Retrieve an AgentRun by its UUID."""
        if run_id in self._memory_store:
            return self._memory_store[run_id]

        c = self.client
        if c is not None:
            try:
                resp = c.table("agent_runs").select("*").eq("id", str(run_id)).limit(1).execute()
                if resp.data:
                    run = _deserialize_agent_run(resp.data[0])
                    self._memory_store[run_id] = run
                    return run
            except Exception as exc:
                logger.debug("Failed to query database for agent run %s: %s", run_id, exc)

        return None
