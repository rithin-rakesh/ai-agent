"""LinkedIn Lead and Outreach Draft Repository (Phase 6.2).

Provides dual-layer persistence for LinkedIn leads and generated outreach email drafts:
- Primary layer: Supabase PostgreSQL (tables: linkedin_leads, linkedin_outreach_drafts)
- Resilient fallback: Local JSON storage (data/linkedin_leads.json, data/linkedin_drafts.json)
- Deduplication by canonical post_url
- Email contact deduplication check
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from supabase import Client

from app.database.supabase import get_supabase_client, get_supabase_service_client
from app.outreach.linkedin.models import (
    DraftStatus,
    LeadStatus,
    LinkedInLead,
    LinkedInLeadCreate,
    LinkedInLeadUpdate,
    LinkedInOutreachDraft,
)

logger = logging.getLogger(__name__)

DEFAULT_LEADS_FILE = Path("data/linkedin_leads.json")
DEFAULT_DRAFTS_FILE = Path("data/linkedin_drafts.json")


class LinkedInLeadRepository:
    """Repository handling CRUD operations for LinkedIn leads and outreach drafts."""

    def __init__(
        self,
        supabase_client: Optional[Client] = None,
        leads_file_path: Optional[Path] = None,
        drafts_file_path: Optional[Path] = None,
    ) -> None:
        self._supabase_client = supabase_client
        self.leads_file = Path(leads_file_path or DEFAULT_LEADS_FILE)
        self.drafts_file = Path(drafts_file_path or DEFAULT_DRAFTS_FILE)
        self._lock = threading.Lock()
        self._leads_cache: Dict[str, Dict[str, Any]] = {}
        self._drafts_cache: Dict[str, Dict[str, Any]] = {}
        self._init_local_storage()

    @property
    def client(self) -> Optional[Client]:
        """Retrieve active Supabase client or None."""
        if self._supabase_client is not None:
            return self._supabase_client
        try:
            return get_supabase_service_client()
        except Exception:
            try:
                return get_supabase_client()
            except Exception:
                return None

    def _init_local_storage(self) -> None:
        """Initialize local JSON files and load cached data."""
        with self._lock:
            try:
                self.leads_file.parent.mkdir(parents=True, exist_ok=True)
                if self.leads_file.exists():
                    with open(self.leads_file, "r", encoding="utf-8") as f:
                        self._leads_cache = json.load(f)
                else:
                    self._save_local_leads()

                if self.drafts_file.exists():
                    with open(self.drafts_file, "r", encoding="utf-8") as f:
                        self._drafts_cache = json.load(f)
                else:
                    self._save_local_drafts()
            except Exception as exc:
                logger.warning("Failed initializing local LinkedIn repository files: %s", exc)

    def _save_local_leads(self) -> None:
        try:
            with open(self.leads_file, "w", encoding="utf-8") as f:
                json.dump(self._leads_cache, f, indent=2, default=str)
        except Exception as exc:
            logger.error("Error saving local LinkedIn leads: %s", exc)

    def _save_local_drafts(self) -> None:
        try:
            with open(self.drafts_file, "w", encoding="utf-8") as f:
                json.dump(self._drafts_cache, f, indent=2, default=str)
        except Exception as exc:
            logger.error("Error saving local LinkedIn drafts: %s", exc)

    # -------------------------------------------------------------------------
    # LEAD OPERATIONS
    # -------------------------------------------------------------------------

    def create_lead(self, lead_create: LinkedInLeadCreate) -> LinkedInLead:
        """Create a new LinkedIn lead, checking for post_url uniqueness."""
        existing = self.get_lead_by_post_url(lead_create.post_url)
        if existing:
            return existing

        lead_id = uuid4()
        now = datetime.now(timezone.utc)
        payload = lead_create.model_dump(mode="json")
        payload["id"] = str(lead_id)
        payload["created_at"] = now.isoformat()
        payload["updated_at"] = now.isoformat()

        # Try Supabase first
        if self.client:
            try:
                resp = self.client.table("linkedin_leads").insert(payload).execute()
                if resp.data:
                    logger.info("Persisted LinkedIn lead %s to Supabase", lead_id)
            except Exception as exc:
                logger.debug("Supabase insert for linkedin_leads failed, using local JSON fallback: %s", exc)

        # Fallback / Dual write to local storage
        with self._lock:
            self._leads_cache[str(lead_id)] = payload
            self._save_local_leads()

        return LinkedInLead.model_validate(payload)

    def get_lead_by_id(self, lead_id: UUID) -> Optional[LinkedInLead]:
        """Fetch lead by its UUID."""
        str_id = str(lead_id)
        if self.client:
            try:
                resp = self.client.table("linkedin_leads").select("*").eq("id", str_id).limit(1).execute()
                if resp.data:
                    return LinkedInLead.model_validate(resp.data[0])
            except Exception:
                pass

        with self._lock:
            data = self._leads_cache.get(str_id)
            if data:
                return LinkedInLead.model_validate(data)
        return None

    def get_lead_by_post_url(self, post_url: str) -> Optional[LinkedInLead]:
        """Fetch lead by canonical post URL."""
        if not post_url:
            return None

        if self.client:
            try:
                resp = self.client.table("linkedin_leads").select("*").eq("post_url", post_url).limit(1).execute()
                if resp.data:
                    return LinkedInLead.model_validate(resp.data[0])
            except Exception:
                pass

        with self._lock:
            for item in self._leads_cache.values():
                if item.get("post_url") == post_url:
                    return LinkedInLead.model_validate(item)
        return None

    def list_leads(
        self,
        status: Optional[LeadStatus] = None,
        min_relevance_score: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[LinkedInLead]:
        """List leads matching criteria, ordered by relevance_score descending."""
        if self.client:
            try:
                q = self.client.table("linkedin_leads").select("*")
                if status:
                    q = q.eq("status", status.value)
                if min_relevance_score is not None:
                    q = q.gte("relevance_score", min_relevance_score)
                q = q.order("relevance_score", desc=True).range(offset, offset + limit - 1)
                resp = q.execute()
                if resp.data is not None:
                    return [LinkedInLead.model_validate(d) for d in resp.data]
            except Exception:
                pass

        with self._lock:
            results = []
            for item in self._leads_cache.values():
                if status and item.get("status") != status.value:
                    continue
                if min_relevance_score is not None and float(item.get("relevance_score", 0)) < min_relevance_score:
                    continue
                results.append(LinkedInLead.model_validate(item))

            results.sort(key=lambda l: l.relevance_score, reverse=True)
            return results[offset : offset + limit]

    def update_lead(self, lead_id: UUID, lead_update: LinkedInLeadUpdate) -> Optional[LinkedInLead]:
        """Update existing lead status or contact fields."""
        str_id = str(lead_id)
        current = self.get_lead_by_id(lead_id)
        if not current:
            return None

        update_dict = lead_update.model_dump(exclude_unset=True, mode="json")
        update_dict["updated_at"] = datetime.now(timezone.utc).isoformat()

        if self.client:
            try:
                resp = self.client.table("linkedin_leads").update(update_dict).eq("id", str_id).execute()
                if resp.data:
                    return LinkedInLead.model_validate(resp.data[0])
            except Exception:
                pass

        with self._lock:
            if str_id in self._leads_cache:
                self._leads_cache[str_id].update(update_dict)
                self._save_local_leads()
                return LinkedInLead.model_validate(self._leads_cache[str_id])
        return None

    def has_contacted_email(self, email: str) -> bool:
        """Check if an email has already been drafted or contacted."""
        if not email:
            return False
        clean = email.strip().lower()

        with self._lock:
            for draft in self._drafts_cache.values():
                if str(draft.get("recipient_email", "")).lower() == clean:
                    if draft.get("status") in (DraftStatus.SENT.value, DraftStatus.DRAFT.value, DraftStatus.APPROVED.value):
                        return True
        return False

    # -------------------------------------------------------------------------
    # DRAFT OPERATIONS
    # -------------------------------------------------------------------------

    def create_draft(self, draft: LinkedInOutreachDraft) -> LinkedInOutreachDraft:
        """Save a new outreach email draft."""
        draft_id_str = str(draft.id)
        payload = draft.model_dump(mode="json")

        if self.client:
            try:
                resp = self.client.table("linkedin_outreach_drafts").insert(payload).execute()
                if resp.data:
                    logger.info("Persisted LinkedIn draft %s to Supabase", draft.id)
            except Exception as exc:
                logger.debug("Supabase insert for linkedin_outreach_drafts failed, using fallback: %s", exc)

        with self._lock:
            self._drafts_cache[draft_id_str] = payload
            self._save_local_drafts()

        return draft

    def get_draft_by_lead_id(self, lead_id: UUID) -> Optional[LinkedInOutreachDraft]:
        """Fetch draft created for a specific lead."""
        str_lead_id = str(lead_id)
        if self.client:
            try:
                resp = self.client.table("linkedin_outreach_drafts").select("*").eq("lead_id", str_lead_id).limit(1).execute()
                if resp.data:
                    return LinkedInOutreachDraft.model_validate(resp.data[0])
            except Exception:
                pass

        with self._lock:
            for draft in self._drafts_cache.values():
                if draft.get("lead_id") == str_lead_id:
                    return LinkedInOutreachDraft.model_validate(draft)
        return None
