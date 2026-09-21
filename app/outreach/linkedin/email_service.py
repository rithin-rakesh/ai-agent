"""LinkedIn Outreach Email Service and Provider Abstraction (Phase 6.2).

Enforces strict email safety policies:
- Default draft mode: OUTREACH_AUTO_SEND=False strictly enforced
- Abstract EmailProvider interface (avoids vendor lock-in or hardcoding Gmail)
- DraftOnlyEmailProvider: rejects actual dispatch and ensures zero emails sent in this phase
- Duplicate recipient prevention
- Opt-out / DO_NOT_CONTACT guard
"""

import abc
import logging
from typing import Any, Dict, Optional
from uuid import UUID

from app.config.settings import Settings, get_settings
from app.database.repositories.linkedin_lead_repository import LinkedInLeadRepository
from app.outreach.linkedin.budget_guard import LinkedInBudgetGuard
from app.outreach.linkedin.email_generator import LinkedInEmailGenerator
from app.outreach.linkedin.models import (
    DraftStatus,
    LeadStatus,
    LinkedInLead,
    LinkedInLeadUpdate,
    LinkedInOutreachDraft,
)

logger = logging.getLogger(__name__)


class OutreachSafetyError(RuntimeError):
    """Raised when an action violates safety constraints (e.g. attempting to send while auto-send is disabled)."""
    pass


class LeadOptedOutError(RuntimeError):
    """Raised when attempting outreach to an opted-out or do-not-contact lead."""
    pass


class DuplicateOutreachError(RuntimeError):
    """Raised when an outreach draft already exists or email was previously contacted."""
    pass


class EmailProvider(abc.ABC):
    """Abstract interface for email transmission providers."""

    @abc.abstractmethod
    async def send_email(self, draft: LinkedInOutreachDraft) -> bool:
        """Transmit an outreach email."""
        raise NotImplementedError


class DraftOnlyEmailProvider(EmailProvider):
    """Safety-enforcing provider that operates strictly in draft mode."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    async def send_email(self, draft: LinkedInOutreachDraft) -> bool:
        """Safety Gate: blocks automated sending unless explicit authorized flags are configured."""
        auto_send = getattr(self.settings, "OUTREACH_AUTO_SEND", False)
        if not auto_send:
            err = (
                f"BLOCKED: OUTREACH_AUTO_SEND is False. Automated email sending is disabled in Phase 6.2. "
                f"Draft {draft.id} for {draft.recipient_email} remains in DRAFT status."
            )
            logger.warning(err)
            raise OutreachSafetyError(err)

        # In Phase 6.2, even if accidentally toggled, no real SMTP or third-party client is hooked up
        raise OutreachSafetyError("No live email transport is configured for Phase 6.2. Draft mode only.")


class LinkedInOutreachEmailService:
    """Manages the lifecycle of outreach drafts, reviews, and safety gates."""

    def __init__(
        self,
        repository: Optional[LinkedInLeadRepository] = None,
        generator: Optional[LinkedInEmailGenerator] = None,
        provider: Optional[EmailProvider] = None,
        budget_guard: Optional[LinkedInBudgetGuard] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repo = repository or LinkedInLeadRepository()
        self.generator = generator or LinkedInEmailGenerator(self.settings)
        self.provider = provider or DraftOnlyEmailProvider(self.settings)
        self.budget_guard = budget_guard or LinkedInBudgetGuard(self.settings)

    def prepare_draft_for_lead(self, lead_id: UUID) -> LinkedInOutreachDraft:
        """Create and persist an outreach email draft for a lead."""
        lead = self.repo.get_lead_by_id(lead_id)
        if not lead:
            raise ValueError(f"Lead with ID {lead_id} not found")

        # 1. Opt-out / Do Not Contact Check
        if lead.status == LeadStatus.DO_NOT_CONTACT:
            raise LeadOptedOutError(f"Lead {lead_id} is marked DO_NOT_CONTACT")

        # 2. Existing Draft Check
        existing_draft = self.repo.get_draft_by_lead_id(lead_id)
        if existing_draft:
            logger.info("Draft already exists for lead %s (draft_id=%s)", lead_id, existing_draft.id)
            return existing_draft

        # 3. Duplicate Recipient Check
        if lead.contact_email and self.repo.has_contacted_email(lead.contact_email):
            logger.warning("Recipient email %s has already been drafted or contacted", lead.contact_email)
            raise DuplicateOutreachError(f"Email {lead.contact_email} has already received outreach or has a draft")

        # 4. Generate Draft
        draft = self.generator.generate_draft(lead)

        # 5. Persist Draft and Update Lead Status
        saved_draft = self.repo.create_draft(draft)
        self.repo.update_lead(lead_id, LinkedInLeadUpdate(status=LeadStatus.DRAFTED))
        self.budget_guard.record_draft_created()

        logger.info("Successfully generated outreach draft %s for lead %s (%s)", saved_draft.id, lead_id, lead.contact_email)
        return saved_draft

    def get_draft_preview(self, lead_id: UUID) -> Optional[LinkedInOutreachDraft]:
        """Fetch the existing draft for preview."""
        return self.repo.get_draft_by_lead_id(lead_id)

    def mark_lead_skipped(self, lead_id: UUID, opt_out: bool = False) -> Optional[LinkedInLead]:
        """Mark a lead as skipped or permanently opted-out."""
        new_status = LeadStatus.DO_NOT_CONTACT if opt_out else LeadStatus.SKIPPED
        updated = self.repo.update_lead(lead_id, LinkedInLeadUpdate(status=new_status))
        logger.info("Marked lead %s as %s", lead_id, new_status.value)
        return updated
