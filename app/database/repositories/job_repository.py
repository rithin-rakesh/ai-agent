"""Job Repository for Supabase database operations.

Encapsulates all database interactions for the 'jobs' table.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from supabase import Client

from app.database.supabase import get_supabase_client, get_supabase_service_client
from app.models.job import Job, JobCreate, JobStats

logger = logging.getLogger(__name__)


def _serialize_job_create(job: JobCreate) -> Dict[str, Any]:
    """Convert a JobCreate Pydantic model into a dictionary suitable for Supabase PostgreSQL."""
    payload = job.model_dump(mode="python")

    # Strip runtime/transient fields not present in Supabase 'jobs' table
    for field in ("application_method", "availability_status", "application_eligible"):
        payload.pop(field, None)

    # Serialize datetime to ISO 8601 string
    if payload.get("posted_at") and isinstance(payload["posted_at"], datetime):
        payload["posted_at"] = payload["posted_at"].isoformat()

    # Ensure raw_data is a valid JSON dict
    if "raw_data" in payload and payload["raw_data"] is None:
        payload["raw_data"] = {}

    return payload


class JobRepository:
    """Repository handling CRUD and search operations for Job records in Supabase."""

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

    def get_job_by_source_external_id(self, source: str, external_id: str) -> Optional[Job]:
        """Fetch a single job by source and external ID."""
        try:
            response = (
                self.client.table("jobs")
                .select("*")
                .eq("source", source)
                .eq("external_id", external_id)
                .limit(1)
                .execute()
            )
            if response.data and len(response.data) > 0:
                return Job.model_validate(response.data[0])
            return None
        except Exception as exc:
            logger.error("Error fetching job by (%s, %s): %s", source, external_id, exc)
            return None

    def get_job_by_id(self, job_id: UUID) -> Optional[Job]:
        """Fetch a single job by its primary UUID."""
        try:
            response = (
                self.client.table("jobs")
                .select("*")
                .eq("id", str(job_id))
                .limit(1)
                .execute()
            )
            if response.data and len(response.data) > 0:
                return Job.model_validate(response.data[0])
            return None
        except Exception as exc:
            logger.error("Error fetching job %s: %s", job_id, exc)
            return None

    def upsert_job(self, job_create: JobCreate) -> Tuple[Optional[Job], bool]:
        """Upsert a single job into Supabase.

        Returns:
            Tuple of (saved Job or None, is_new: bool)
        """
        existing = self.get_job_by_source_external_id(job_create.source, job_create.external_id)
        payload = _serialize_job_create(job_create)

        try:
            # Use on_conflict matching unique constraint (source, external_id)
            response = (
                self.client.table("jobs")
                .upsert(payload, on_conflict="source,external_id")
                .execute()
            )

            if response.data and len(response.data) > 0:
                saved = Job.model_validate(response.data[0])
                return saved, (existing is None)

            # Fallback re-fetch if upsert didn't return representation
            refetched = self.get_job_by_source_external_id(job_create.source, job_create.external_id)
            return refetched, (existing is None)

        except Exception as exc:
            logger.error(
                "Failed to upsert job (%s, %s): %s",
                job_create.source,
                job_create.external_id,
                exc,
            )
            return None, False

    def upsert_jobs(self, jobs: List[JobCreate]) -> Tuple[List[Job], int, int]:
        """Upsert a batch of jobs into Supabase, tracking new vs duplicate/updated counts.

        Args:
            jobs: List of JobCreate entities to persist

        Returns:
            Tuple of (list of persisted Job models, new_count, duplicate_count)
        """
        persisted: List[Job] = []
        new_count = 0
        duplicate_count = 0

        for job_in in jobs:
            saved_job, is_new = self.upsert_job(job_in)
            if saved_job is not None:
                persisted.append(saved_job)
                if is_new:
                    new_count += 1
                else:
                    duplicate_count += 1

        logger.info(
            "Batch upsert completed: %d total processed, %d new, %d duplicates/updated",
            len(jobs),
            new_count,
            duplicate_count,
        )
        return persisted, new_count, duplicate_count

    def list_jobs(
        self,
        limit: int = 50,
        offset: int = 0,
        source: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[Job], int]:
        """List jobs with pagination, filtering, and total count.

        Returns:
            Tuple of (list of Job models, total matching count)
        """
        try:
            query = self.client.table("jobs").select("*", count="exact")

            if source:
                query = query.eq("source", source.lower().strip())

            if search:
                # Search across title or company or location
                s = search.strip()
                query = query.or_(f"title.ilike.%{s}%,company.ilike.%{s}%,location.ilike.%{s}%")

            # Order by posted_at or created_at descending
            query = query.order("posted_at", desc=True, nullsfirst=False).order("created_at", desc=True)
            query = query.range(offset, offset + limit - 1)

            response = query.execute()
            total_count = response.count if response.count is not None else len(response.data)
            jobs = [Job.model_validate(item) for item in response.data]
            return jobs, total_count
        except Exception as exc:
            logger.error("Error listing jobs: %s", exc)
            return [], 0

    def count_jobs(self, source: Optional[str] = None) -> int:
        """Count total jobs stored, optionally filtered by source platform."""
        try:
            query = self.client.table("jobs").select("id", count="exact", head=True)
            if source:
                query = query.eq("source", source.lower().strip())
            response = query.execute()
            return response.count or 0
        except Exception as exc:
            logger.error("Error counting jobs: %s", exc)
            return 0

    def get_job_stats(self) -> JobStats:
        """Compute summary statistics for jobs in the database."""
        try:
            # Total jobs
            total = self.count_jobs()

            # Source breakdowns
            sources_counts: Dict[str, int] = {}
            for src in ["linkedin", "indeed", "naukri", "glassdoor"]:
                sources_counts[src] = self.count_jobs(source=src)

            # Remote jobs count
            remote_resp = self.client.table("jobs").select("id", count="exact", head=True).eq("remote", True).execute()
            remote_count = remote_resp.count or 0

            # Easy Apply jobs count
            easy_resp = self.client.table("jobs").select("id", count="exact", head=True).eq("easy_apply", True).execute()
            easy_count = easy_resp.count or 0

            # Latest posted_at
            latest_resp = (
                self.client.table("jobs")
                .select("posted_at")
                .not_.is_("posted_at", "null")
                .order("posted_at", desc=True)
                .limit(1)
                .execute()
            )
            latest_posted_at = None
            if latest_resp.data and len(latest_resp.data) > 0 and latest_resp.data[0].get("posted_at"):
                val = latest_resp.data[0]["posted_at"]
                if isinstance(val, str):
                    if val.endswith("Z"):
                        val = val[:-1] + "+00:00"
                    latest_posted_at = datetime.fromisoformat(val)

            return JobStats(
                total_jobs=total,
                sources=sources_counts,
                remote_jobs=remote_count,
                easy_apply_jobs=easy_count,
                latest_posted_at=latest_posted_at,
            )
        except Exception as exc:
            logger.error("Error computing job stats: %s", exc)
            return JobStats(total_jobs=0, sources={}, remote_jobs=0, easy_apply_jobs=0)
