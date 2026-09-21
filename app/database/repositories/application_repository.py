"""Application Repository for Supabase database operations.

Encapsulates CRUD interactions for 'applications', 'application_answers', and 'automation_logs'.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from supabase import Client

from app.database.supabase import get_supabase_client, get_supabase_service_client
from app.models.application import (
    Application,
    ApplicationCreate,
    ApplicationUpdate,
    AutomationLog,
    AutomationLogCreate,
)

logger = logging.getLogger(__name__)


class ApplicationRepository:
    """Repository handling CRUD operations for Application and AutomationLog records."""

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

    def get_application_by_job_and_profile(
        self, job_id: UUID, profile_id: UUID
    ) -> Optional[Application]:
        """Fetch an application record by job_id and profile_id."""
        try:
            response = (
                self.client.table("applications")
                .select("*")
                .eq("job_id", str(job_id))
                .eq("profile_id", str(profile_id))
                .limit(1)
                .execute()
            )
            if response.data:
                return Application.model_validate(response.data[0])
            return None
        except Exception as exc:
            logger.warning("Error fetching application for job %s and profile %s: %s", job_id, profile_id, exc)
            return None

    def get_applications_for_jobs(
        self, job_ids: List[UUID], profile_id: Optional[UUID] = None
    ) -> List[Application]:
        """Fetch existing applications for a list of job IDs."""
        if not job_ids:
            return []
        try:
            query = self.client.table("applications").select("*").in_("job_id", [str(j) for j in job_ids])
            if profile_id:
                query = query.eq("profile_id", str(profile_id))
            response = query.execute()
            if response.data:
                return [Application.model_validate(item) for item in response.data]
            return []
        except Exception as exc:
            logger.warning("Error fetching applications for batch jobs: %s", exc)
            return []

    def create_application(self, app_create: ApplicationCreate) -> Optional[Application]:
        """Create a new application record in Supabase."""
        try:
            payload = app_create.model_dump(mode="python")
            payload["job_id"] = str(payload["job_id"])
            payload["profile_id"] = str(payload["profile_id"])
            if payload.get("started_at") and isinstance(payload["started_at"], datetime):
                payload["started_at"] = payload["started_at"].isoformat()
            if payload.get("submitted_at") and isinstance(payload["submitted_at"], datetime):
                payload["submitted_at"] = payload["submitted_at"].isoformat()

            response = self.client.table("applications").insert(payload).execute()
            if response.data:
                return Application.model_validate(response.data[0])
            return None
        except Exception as exc:
            logger.error("Failed to create application record: %s", exc)
            return None

    def update_application(
        self, application_id: UUID, app_update: ApplicationUpdate
    ) -> Optional[Application]:
        """Update an existing application record."""
        try:
            payload = app_update.model_dump(mode="python", exclude_unset=True)
            if payload.get("started_at") and isinstance(payload["started_at"], datetime):
                payload["started_at"] = payload["started_at"].isoformat()
            if payload.get("submitted_at") and isinstance(payload["submitted_at"], datetime):
                payload["submitted_at"] = payload["submitted_at"].isoformat()

            response = (
                self.client.table("applications")
                .update(payload)
                .eq("id", str(application_id))
                .execute()
            )
            if response.data:
                return Application.model_validate(response.data[0])
            return None
        except Exception as exc:
            logger.error("Failed to update application %s: %s", application_id, exc)
            return None

    def create_automation_log(self, log_create: AutomationLogCreate) -> Optional[AutomationLog]:
        """Create an automation log entry in Supabase."""
        try:
            payload = log_create.model_dump(mode="python")
            if payload.get("application_id"):
                payload["application_id"] = str(payload["application_id"])
            response = self.client.table("automation_logs").insert(payload).execute()
            if response.data:
                return AutomationLog.model_validate(response.data[0])
            return None
        except Exception as exc:
            logger.warning("Failed to create automation log: %s", exc)
            return None

    def get_answers_by_profile(self, profile_id: Optional[UUID] = None) -> Dict[str, str]:
        """Retrieve a dictionary mapping question text/keys to user-approved answer strings."""
        try:
            query = self.client.table("application_answers").select("question, answer, confidence")
            if profile_id:
                # Optionally filter by applications belonging to this profile
                pass
            response = query.execute()
            answers = {}
            if response.data:
                for row in response.data:
                    q = (row.get("question") or "").strip().lower()
                    a = (row.get("answer") or "").strip()
                    if q and a:
                        answers[q] = a
            return answers
        except Exception as exc:
            logger.warning("Failed to fetch application answers: %s", exc)
            return {}

    def record_application_submission(
        self,
        job_id: UUID,
        profile_id: UUID,
        platform: str,
        application_url: Optional[str] = None,
        match_score: Optional[float] = None,
        agent_run_id: Optional[UUID] = None,
        status: str = "submitted",
        confirmation_text: Optional[str] = None,
    ) -> Optional[Application]:
        """Create or update an application submission record in Supabase avoiding duplicate rows."""
        now = datetime.now()
        existing = self.get_application_by_job_and_profile(job_id, profile_id)
        if existing:
            update_data = ApplicationUpdate(
                status=status,
                submitted_at=now if status == "submitted" else None,
                application_url=application_url or existing.application_url,
                confirmation_text=confirmation_text or existing.confirmation_text,
            )
            return self.update_application(existing.id, update_data)
        else:
            app_create = ApplicationCreate(
                job_id=job_id,
                profile_id=profile_id,
                platform=platform,
                status=status,
                started_at=now,
                submitted_at=now if status == "submitted" else None,
                application_url=application_url,
                confirmation_text=confirmation_text,
            )
            return self.create_application(app_create)

    def record_automation_log(
        self,
        platform: str,
        action: str,
        status: str,
        job_id: Optional[UUID] = None,
        agent_run_id: Optional[UUID] = None,
        application_id: Optional[UUID] = None,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[AutomationLog]:
        """Record an automation log entry with strict sanitization of credentials and sensitive tokens."""
        sanitized_details = dict(details or {})
        if job_id:
            sanitized_details["job_id"] = str(job_id)
        if agent_run_id:
            sanitized_details["agent_run_id"] = str(agent_run_id)
        if error:
            sanitized_details["error"] = str(error)

        # Redact sensitive keys
        SENSITIVE_KEYS = {"password", "token", "auth", "authorization", "secret", "cookie", "session"}
        for k in list(sanitized_details.keys()):
            if any(s in k.lower() for s in SENSITIVE_KEYS):
                sanitized_details[k] = "[REDACTED]"

        log_create = AutomationLogCreate(
            application_id=application_id,
            platform=platform,
            action=action,
            status=status,
            details=sanitized_details,
        )
        return self.create_automation_log(log_create)

    def save_application_answer(
        self,
        question: str,
        answer: str,
        application_id: Optional[UUID] = None,
        source: str = "user",
        confidence: float = 1.0,
    ) -> bool:
        """Persist a user-approved question-answer mapping for future memory."""
        try:
            payload = {
                "question": question.strip(),
                "answer": answer.strip(),
                "source": source,
                "confidence": confidence,
            }
            if application_id:
                payload["application_id"] = str(application_id)
            response = self.client.table("application_answers").insert(payload).execute()
            return bool(response.data)
        except Exception as exc:
            logger.warning("Failed to save application answer: %s", exc)
            return False

    def count_daily_applications(
        self,
        profile_id: Optional[UUID] = None,
        platform: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> int:
        """Count total applications recorded for today (since start of day UTC or specified timestamp)."""
        try:
            if since is None:
                from datetime import timezone
                now_utc = datetime.now(timezone.utc)
                since = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

            since_iso = since.isoformat()
            query = self.client.table("applications").select("id", count="exact")
            query = query.gte("created_at", since_iso)
            if profile_id:
                query = query.eq("profile_id", str(profile_id))
            if platform:
                query = query.eq("platform", platform.lower().strip())

            response = query.execute()
            return response.count if response.count is not None else len(response.data or [])
        except Exception as exc:
            logger.warning("Error counting daily applications: %s", exc)
            return 0


