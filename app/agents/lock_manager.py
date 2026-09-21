"""Application Locking Manager for Autonomous Job Application Runs (Phase 6.0).

Prevents concurrent agent runs from applying to the same job simultaneously.
Provides atomic acquisition, configurable expiration timeouts, re-entrancy for the same run,
and resilient dual-layer persistence (Supabase with persistent local JSON fallback).
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID
from supabase import Client

from app.database.supabase import get_supabase_client, get_supabase_service_client

logger = logging.getLogger(__name__)

DEFAULT_LOCK_DIR = Path("data")
DEFAULT_LOCK_FILE = DEFAULT_LOCK_DIR / "job_locks.json"
DEFAULT_LOCK_TIMEOUT_SECONDS = 1800  # 30 minutes


class JobLockManager:
    """Manages acquisition, renewal, and release of job application locks."""

    def __init__(
        self,
        lock_file_path: Optional[Any] = None,
        lock_file: Optional[Any] = None,
        supabase_client: Optional[Client] = None,
        default_timeout_seconds: int = DEFAULT_LOCK_TIMEOUT_SECONDS,
    ) -> None:
        target_path = lock_file or lock_file_path or DEFAULT_LOCK_FILE
        self.lock_file_path = Path(target_path)
        self._supabase_client = supabase_client
        self.default_timeout_seconds = default_timeout_seconds
        self._thread_lock = threading.Lock()
        self._memory_locks: Dict[str, Dict[str, Any]] = {}
        self._ensure_lock_file()
        self._load_local_locks()

    @property
    def client(self) -> Optional[Client]:
        """Retrieve active Supabase client or None if unconfigured."""
        if self._supabase_client is not None:
            return self._supabase_client
        try:
            return get_supabase_service_client()
        except Exception:
            try:
                return get_supabase_client()
            except Exception:
                return None

    def _ensure_lock_file(self) -> None:
        """Ensure the local lock file directory and file exist."""
        try:
            self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.lock_file_path.exists():
                self.lock_file_path.write_text("{}", encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not initialize local lock file: %s", exc)

    def _load_local_locks(self) -> None:
        """Load stored locks from local JSON file into memory cache."""
        try:
            if self.lock_file_path.exists():
                raw = self.lock_file_path.read_text(encoding="utf-8").strip()
                if raw:
                    self._memory_locks = json.loads(raw)
        except Exception as exc:
            logger.warning("Failed to load local locks from %s: %s", self.lock_file_path, exc)
            self._memory_locks = {}

    def _save_local_locks(self) -> None:
        """Persist memory locks to local JSON file atomically."""
        try:
            self._ensure_lock_file()
            temp_file = self.lock_file_path.with_suffix(".tmp")
            temp_file.write_text(json.dumps(self._memory_locks, indent=2), encoding="utf-8")
            temp_file.replace(self.lock_file_path)
        except Exception as exc:
            logger.warning("Failed to persist local locks to %s: %s", self.lock_file_path, exc)

    def acquire_lock(
        self,
        job_id: UUID,
        agent_run_id: UUID,
        timeout_seconds: Optional[int] = None,
    ) -> bool:
        """Attempt to acquire an exclusive lock on a job for an agent run.

        Returns:
            bool: True if lock was acquired or already held by the same run.
                  False if an active lock is held by another agent run.
        """
        now = datetime.now(timezone.utc)
        timeout = timeout_seconds or self.default_timeout_seconds
        expires_at = datetime.fromtimestamp(now.timestamp() + timeout, tz=timezone.utc)
        jid_str = str(job_id)
        run_id_str = str(agent_run_id)

        with self._thread_lock:
            # Check local memory/file state first
            existing = self._memory_locks.get(jid_str)
            if existing:
                try:
                    exp = datetime.fromisoformat(existing["expires_at"])
                    if exp > now:
                        if existing.get("agent_run_id") != run_id_str:
                            logger.info(
                                "Job [%s] is currently locked by another run [%s] until %s. Skipping.",
                                jid_str,
                                existing.get("agent_run_id"),
                                exp.isoformat(),
                            )
                            return False
                        else:
                            # Re-entrant: refresh lock expiry
                            existing["expires_at"] = expires_at.isoformat()
                            existing["locked_at"] = now.isoformat()
                            self._save_local_locks()
                            return True
                except Exception:
                    pass

            # Acquire locally
            lock_data = {
                "job_id": jid_str,
                "agent_run_id": run_id_str,
                "locked_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
            }
            self._memory_locks[jid_str] = lock_data
            self._save_local_locks()

        # Best-effort sync with Supabase if table exists
        c = self.client
        if c is not None:
            try:
                c.table("job_locks").upsert(
                    {
                        "job_id": jid_str,
                        "agent_run_id": run_id_str,
                        "locked_at": now.isoformat(),
                        "expires_at": expires_at.isoformat(),
                    },
                    on_conflict="job_id",
                ).execute()
            except Exception as exc:
                logger.debug("Database job_locks sync skipped/unavailable: %s", exc)

        logger.info("Acquired exclusive lock on job [%s] for run [%s]", jid_str, run_id_str)
        return True

    def release_lock(self, job_id: UUID, agent_run_id: UUID) -> bool:
        """Release an exclusive lock held by an agent run on a job.

        Returns:
            bool: True if lock was found and released; False otherwise.
        """
        jid_str = str(job_id)
        run_id_str = str(agent_run_id)

        with self._thread_lock:
            existing = self._memory_locks.get(jid_str)
            if existing and existing.get("agent_run_id") == run_id_str:
                del self._memory_locks[jid_str]
                self._save_local_locks()
            else:
                return False

        # Best-effort sync with Supabase
        c = self.client
        if c is not None:
            try:
                c.table("job_locks").delete().eq("job_id", jid_str).eq("agent_run_id", run_id_str).execute()
            except Exception as exc:
                logger.debug("Database job_locks delete skipped/unavailable: %s", exc)

        logger.info("Released lock on job [%s] for run [%s]", jid_str, run_id_str)
        return True

    def release_all_for_run(self, agent_run_id: UUID) -> int:
        """Release all locks held by a specific agent run."""
        run_id_str = str(agent_run_id)
        released_count = 0

        with self._thread_lock:
            to_remove = [
                jid for jid, data in self._memory_locks.items()
                if data.get("agent_run_id") == run_id_str
            ]
            for jid in to_remove:
                del self._memory_locks[jid]
                released_count += 1
            if released_count > 0:
                self._save_local_locks()

        # Best-effort sync with Supabase
        c = self.client
        if c is not None and released_count > 0:
            try:
                c.table("job_locks").delete().eq("agent_run_id", run_id_str).execute()
            except Exception as exc:
                logger.debug("Database job_locks batch release skipped/unavailable: %s", exc)

        logger.info("Released %d locks for agent run [%s]", released_count, run_id_str)
        return released_count

    def is_locked(self, job_id: UUID) -> bool:
        """Check if an unexpired lock exists on the specified job."""
        now = datetime.now(timezone.utc)
        jid_str = str(job_id)

        with self._thread_lock:
            existing = self._memory_locks.get(jid_str)
            if not existing:
                return False
            try:
                exp = datetime.fromisoformat(existing["expires_at"])
                return exp > now
            except Exception:
                return False
