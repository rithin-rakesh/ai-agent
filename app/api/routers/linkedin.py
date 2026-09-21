"""FastAPI Router for LinkedIn Outreach Agent (Phase 6.2).

Exposes REST endpoints for:
- POST /linkedin/discover: Discover public posts, score relevance, extract emails, create leads
- GET /linkedin/leads: List and filter leads
- GET /linkedin/leads/{lead_id}: Retrieve individual lead details
- POST /linkedin/leads/{lead_id}/draft: Generate outreach email draft with resume attachment
- GET /linkedin/leads/{lead_id}/draft: Preview generated draft
- POST /linkedin/leads/{lead_id}/skip: Mark lead as skipped or do-not-contact
- POST /linkedin/test: Diagnostic health check (token, resume path, budget state)
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from app.outreach.linkedin.agent import LinkedInOutreachAgent
from app.outreach.linkedin.email_service import (
    DuplicateOutreachError,
    LeadOptedOutError,
    OutreachSafetyError,
)
from app.outreach.linkedin.email_generator import ResumeNotFoundError
from app.outreach.linkedin.models import (
    LeadStatus,
    LinkedInDiscoverRequest,
    LinkedInDiscoverResponse,
    LinkedInLead,
    LinkedInOutreachDraft,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/linkedin", tags=["LinkedIn Outreach"])

_agent_instance: Optional[LinkedInOutreachAgent] = None


def get_agent() -> LinkedInOutreachAgent:
    """Retrieve singleton instance of LinkedInOutreachAgent."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = LinkedInOutreachAgent()
    return _agent_instance


@router.post(
    "/discover",
    response_model=LinkedInDiscoverResponse,
    summary="Discover public LinkedIn hiring posts",
    description="Searches LinkedIn public posts via Apify actor, scores against candidate profile, and creates leads.",
)
async def discover_leads(request: LinkedInDiscoverRequest) -> LinkedInDiscoverResponse:
    """Run an autonomous LinkedIn post discovery cycle."""
    agent = get_agent()
    return await agent.discover_leads(request)


@router.get(
    "/leads",
    response_model=List[LinkedInLead],
    summary="List stored LinkedIn leads",
    description="Retrieve paginated LinkedIn leads with optional status and relevance score filters.",
)
async def list_leads(
    status_filter: Optional[LeadStatus] = Query(None, alias="status", description="Filter by lead status"),
    min_score: Optional[float] = Query(None, ge=0.0, le=100.0, description="Minimum relevance score filter"),
    limit: int = Query(50, ge=1, le=100, description="Maximum leads to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> List[LinkedInLead]:
    """List stored leads."""
    agent = get_agent()
    return agent.list_leads(
        status=status_filter,
        min_relevance_score=min_score,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/leads/{lead_id}",
    response_model=LinkedInLead,
    summary="Get single LinkedIn lead",
    description="Retrieve full details for a specific LinkedIn lead by UUID.",
)
async def get_lead(lead_id: UUID) -> LinkedInLead:
    """Fetch single lead by ID."""
    agent = get_agent()
    lead = agent.get_lead(lead_id)
    if not lead:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LinkedIn lead with ID {lead_id} was not found",
        )
    return lead


@router.post(
    "/leads/{lead_id}/draft",
    response_model=LinkedInOutreachDraft,
    summary="Generate personalized outreach email draft",
    description="Prepares a draft email referencing post details and validates the local resume PDF attachment.",
)
async def create_draft(lead_id: UUID) -> LinkedInOutreachDraft:
    """Generate an outreach email draft for a lead."""
    agent = get_agent()
    try:
        return agent.prepare_draft(lead_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except ResumeNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except LeadOptedOutError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except DuplicateOutreachError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        logger.error("Failed creating draft for lead %s: %s", lead_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Draft generation failed: {exc}",
        )


@router.get(
    "/leads/{lead_id}/draft",
    response_model=LinkedInOutreachDraft,
    summary="Preview outreach email draft",
    description="Retrieve the generated draft email for a lead.",
)
async def get_draft(lead_id: UUID) -> LinkedInOutreachDraft:
    """Preview draft for a lead."""
    agent = get_agent()
    draft = agent.get_draft(lead_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No draft found for lead {lead_id}. Call POST /linkedin/leads/{lead_id}/draft to generate one.",
        )
    return draft


@router.post(
    "/leads/{lead_id}/skip",
    response_model=LinkedInLead,
    summary="Skip or opt-out lead",
    description="Marks lead as SKIPPED or DO_NOT_CONTACT.",
)
async def skip_lead(
    lead_id: UUID,
    opt_out: bool = Query(False, description="If true, permanently flags lead as DO_NOT_CONTACT"),
) -> LinkedInLead:
    """Skip or opt-out a lead."""
    agent = get_agent()
    updated = agent.mark_lead_skipped(lead_id, opt_out=opt_out)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LinkedIn lead with ID {lead_id} was not found",
        )
    return updated


@router.post(
    "/test",
    summary="LinkedIn Outreach diagnostics and health check",
    description="Validates Apify token, checks resume attachment path, and returns budget statistics.",
)
async def test_diagnostics() -> Dict[str, Any]:
    """Run diagnostics and return component status."""
    agent = get_agent()
    return await agent.run_diagnostics()
