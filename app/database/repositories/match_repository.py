"""Supabase Database Repository for JobMatch entities.

Handles persistence, retrieval, filtering, and ranking of job match evaluations
in the Supabase 'job_matches' table.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from supabase import Client

from app.database.supabase import get_supabase_client, get_supabase_service_client
from app.models.match import JobMatch, JobMatchCreate

logger = logging.getLogger(__name__)


def _serialize_match_create(match: JobMatchCreate) -> Dict[str, Any]:
    """Convert JobMatchCreate model to dict suitable for Supabase PostgreSQL."""
    payload = match.model_dump(mode="python")
    payload["job_id"] = str(payload["job_id"])
    payload["profile_id"] = str(payload["profile_id"])

    # In Phase 4, persist llm_score if present, otherwise None
    if payload.get("llm_score") is not None:
        payload["llm_score"] = float(payload["llm_score"])
    else:
        payload["llm_score"] = None
    return payload


class MatchRepository:
    """Repository handling CRUD and search operations for JobMatch records in Supabase."""

    def __init__(self, client: Optional[Client] = None) -> None:
        self._client = client

    @property
    def client(self) -> Client:
        """Retrieve active Supabase client. Prefers service-role client for background DB operations."""
        if self._client is not None:
            return self._client
        try:
            return get_supabase_service_client()
        except ValueError:
            return get_supabase_client()

    def upsert_match(self, match_create: JobMatchCreate) -> Optional[JobMatch]:
        """Insert or update a job match record based on unique (job_id, profile_id)."""
        try:
            payload = _serialize_match_create(match_create)
            response = (
                self.client.table("job_matches")
                .upsert(payload, on_conflict="job_id,profile_id")
                .execute()
            )
            if response.data:
                return JobMatch.model_validate(response.data[0])
            return None
        except Exception as exc:
            logger.error(
                "Failed to upsert job match (job_id=%s, profile_id=%s): %s",
                match_create.job_id,
                match_create.profile_id,
                exc,
            )
            return None

    def get_match(self, job_id: UUID, profile_id: UUID) -> Optional[JobMatch]:
        """Fetch a match record for a specific job and candidate profile."""
        try:
            response = (
                self.client.table("job_matches")
                .select("*")
                .eq("job_id", str(job_id))
                .eq("profile_id", str(profile_id))
                .limit(1)
                .execute()
            )
            if not response.data:
                return None
            return JobMatch.model_validate(response.data[0])
        except Exception as exc:
            logger.error("Failed to get match (job_id=%s, profile_id=%s): %s", job_id, profile_id, exc)
            return None

    def list_matches(
        self,
        profile_id: UUID,
        limit: int = 50,
        offset: int = 0,
        min_score: Optional[float] = None,
        decision: Optional[str] = None,
    ) -> Tuple[List[JobMatch], int]:
        """Retrieve a paginated list of job matches with optional filtering."""
        try:
            query = (
                self.client.table("job_matches")
                .select("*", count="exact")
                .eq("profile_id", str(profile_id))
            )

            if min_score is not None:
                query = query.gte("match_score", min_score)
            if decision:
                query = query.eq("decision", decision.lower())

            query = query.order("match_score", desc=True).range(offset, offset + limit - 1)
            response = query.execute()

            total_count = response.count if response.count is not None else len(response.data or [])
            matches = [JobMatch.model_validate(item) for item in response.data] if response.data else []
            return matches, total_count
        except Exception as exc:
            logger.error("Failed to list matches for profile %s: %s", profile_id, exc)
            return [], 0

    def get_top_matches(self, profile_id: UUID, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve top ranked job matches joined with job details."""
        try:
            response = (
                self.client.table("job_matches")
                .select("*, jobs(*)")
                .eq("profile_id", str(profile_id))
                .order("match_score", desc=True)
                .limit(limit)
                .execute()
            )
            if not response.data:
                return []
            return response.data
        except Exception as exc:
            logger.error("Failed to fetch top matches for profile %s: %s", profile_id, exc)
            return []
