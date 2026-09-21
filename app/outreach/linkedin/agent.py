"""LinkedIn Outreach Agent Coordinator (Phase 6.2).

Coordinates:
- LinkedIn post discovery via Apify
- Multi-factor relevance scoring against candidate profile
- Lead persistence and deduplication
- Outreach email drafting with resume attachment verification
- Strict safety enforcement (OUTREACH_AUTO_SEND=False)
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.config.settings import Settings, get_settings
from app.database.repositories.linkedin_lead_repository import LinkedInLeadRepository
from app.outreach.linkedin.budget_guard import LinkedInBudgetGuard
from app.outreach.linkedin.discovery_provider import LinkedInApifyDiscoveryProvider
from app.outreach.linkedin.email_service import LinkedInOutreachEmailService
from app.outreach.linkedin.models import (
    LeadStatus,
    LinkedInDiscoverRequest,
    LinkedInDiscoverResponse,
    LinkedInDiscoveryDiagnostics,
    LinkedInLead,
    LinkedInOutreachDraft,
)
from app.outreach.linkedin.relevance_engine import LinkedInLeadRelevanceEngine

logger = logging.getLogger(__name__)


class LinkedInOutreachAgent:
    """Central orchestrator for LinkedIn hiring post discovery and lead management."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        repository: Optional[LinkedInLeadRepository] = None,
        discovery_provider: Optional[LinkedInApifyDiscoveryProvider] = None,
        relevance_engine: Optional[LinkedInLeadRelevanceEngine] = None,
        email_service: Optional[LinkedInOutreachEmailService] = None,
        budget_guard: Optional[LinkedInBudgetGuard] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repo = repository or LinkedInLeadRepository()
        self.budget_guard = budget_guard or LinkedInBudgetGuard(self.settings)
        self.discovery_provider = discovery_provider or LinkedInApifyDiscoveryProvider(
            self.settings, self.budget_guard
        )
        self.relevance_engine = relevance_engine or LinkedInLeadRelevanceEngine()
        self.email_service = email_service or LinkedInOutreachEmailService(
            repository=self.repo,
            budget_guard=self.budget_guard,
            settings=self.settings,
        )

    async def discover_leads(self, request: LinkedInDiscoverRequest) -> LinkedInDiscoverResponse:
        """Execute a complete LinkedIn post discovery, scoring, and lead creation cycle."""
        max_posts = request.max_posts or getattr(self.settings, "LINKEDIN_MAX_POSTS_PER_RUN", 10)

        # 1. Discover Posts via Apify
        raw_leads, diag = await self.discovery_provider.run_post_search(
            queries=request.search_queries,
            max_posts=max_posts,
            date_posted=request.date_posted,
        )

        persisted_leads: List[LinkedInLead] = []
        qualified_count = 0

        # 2. Score, Deduplicate, and Persist Leads
        for lead_create in raw_leads:
            # Calculate relevance score
            score, details = self.relevance_engine.calculate_relevance(lead_create)
            lead_create.relevance_score = score
            lead_create.relevance_details = details

            is_qualified = score >= request.minimum_relevance_score
            if is_qualified:
                lead_create.status = LeadStatus.QUALIFIED
                qualified_count += 1
            else:
                lead_create.status = LeadStatus.NEW

            # Check existing post_url
            existing = self.repo.get_lead_by_post_url(lead_create.post_url)
            if existing:
                diag.posts_deduplicated += 1
                persisted_leads.append(existing)
                continue

            # Save lead
            saved_lead = self.repo.create_lead(lead_create)
            persisted_leads.append(saved_lead)

            # Optional immediate draft generation if qualified and verified email present
            if request.auto_generate_drafts and is_qualified and saved_lead.contact_email_verified:
                try:
                    self.email_service.prepare_draft_for_lead(saved_lead.id)
                except Exception as draft_exc:
                    logger.warning("Could not auto-generate draft for lead %s: %s", saved_lead.id, draft_exc)

        status_str = "success" if not diag.errors else ("partial_success" if persisted_leads else "failed")

        return LinkedInDiscoverResponse(
            status=status_str,
            leads_count=len(persisted_leads),
            qualified_count=qualified_count,
            diagnostics=diag,
            leads=persisted_leads,
        )

    def list_leads(
        self,
        status: Optional[LeadStatus] = None,
        min_relevance_score: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[LinkedInLead]:
        """Fetch stored leads matching filter criteria."""
        return self.repo.list_leads(
            status=status,
            min_relevance_score=min_relevance_score,
            limit=limit,
            offset=offset,
        )

    def get_lead(self, lead_id: UUID) -> Optional[LinkedInLead]:
        """Fetch a specific lead by ID."""
        return self.repo.get_lead_by_id(lead_id)

    def prepare_draft(self, lead_id: UUID) -> LinkedInOutreachDraft:
        """Create and persist an outreach email draft for a lead."""
        return self.email_service.prepare_draft_for_lead(lead_id)

    def get_draft(self, lead_id: UUID) -> Optional[LinkedInOutreachDraft]:
        """Retrieve the draft created for a lead."""
        return self.email_service.get_draft_preview(lead_id)

    def mark_lead_skipped(self, lead_id: UUID, opt_out: bool = False) -> Optional[LinkedInLead]:
        """Mark a lead as skipped or opted out."""
        return self.email_service.mark_lead_skipped(lead_id, opt_out=opt_out)

    async def run_diagnostics(self) -> Dict[str, Any]:
        """Perform a comprehensive health check on LinkedIn outreach configuration and dependencies."""
        token = self.discovery_provider._get_api_token()
        token_valid = False
        if token:
            token_valid = await self.discovery_provider.verify_token_validity(token)

        resume_resolved = None
        resume_exists = False
        try:
            resume_resolved = str(self.email_service.generator.validate_resume_path())
            resume_exists = True
        except Exception:
            raw_path = getattr(self.settings, "RESUME_PDF_PATH", "data/profile/resume.pdf")
            resume_resolved = str(self.settings.resolve_path(raw_path))

        budget_state = self.budget_guard.get_diagnostics()

        return {
            "status": "healthy" if (token_valid and resume_exists) else "warning",
            "outreach_enabled": getattr(self.settings, "LINKEDIN_OUTREACH_ENABLED", True),
            "outreach_auto_send": getattr(self.settings, "OUTREACH_AUTO_SEND", False),
            "apify_token_configured": bool(token),
            "apify_token_valid": token_valid,
            "resume_path": resume_resolved,
            "resume_exists": resume_exists,
            "actors": {
                "post_search": getattr(self.settings, "LINKEDIN_POST_SEARCH_ACTOR_ID", "harvestapi~linkedin-post-search"),
                "profile_posts": getattr(self.settings, "LINKEDIN_PROFILE_POSTS_ACTOR_ID", "harvestapi~linkedin-profile-posts"),
            },
            "budget": budget_state,
        }
