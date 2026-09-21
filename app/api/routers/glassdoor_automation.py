"""FastAPI Router for Glassdoor PyWinAuto UI Automation Endpoints (Phase 5.6)."""

import logging
from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from app.automation.glassdoor.apply_service import GlassdoorApplyService, _maybe_await
from app.automation.glassdoor.models import (
    GlassdoorAutomationApplyRequest,
    GlassdoorAutomationInspectRequest,
    GlassdoorAutomationNavigateRequest,
    GlassdoorAutomationResumeRequest,
    GlassdoorAutomationResult,
)
from app.automation.glassdoor.playwright_driver import GlassdoorPlaywrightDriver
from app.database.repositories.application_repository import ApplicationRepository
from app.profile.service import ProfileService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/automation/glassdoor", tags=["Glassdoor Automation"])


def get_glassdoor_playwright_driver() -> GlassdoorPlaywrightDriver:
    """Dependency provider for GlassdoorPlaywrightDriver."""
    return GlassdoorPlaywrightDriver()


def get_glassdoor_apply_service(
    playwright_driver: Optional[Any] = Depends(get_glassdoor_playwright_driver),
) -> GlassdoorApplyService:
    """Dependency provider for GlassdoorApplyService."""
    profile_service = ProfileService()
    app_repo = ApplicationRepository()
    return GlassdoorApplyService(
        profile_service=profile_service,
        application_repository=app_repo,
        playwright_driver=playwright_driver,
    )


@router.post(
    "/inspect",
    response_model=GlassdoorAutomationResult,
    summary="Inspect a Glassdoor job posting and verify Easy Apply availability without clicking.",
)
async def inspect_glassdoor_job(
    request: GlassdoorAutomationInspectRequest,
    service: GlassdoorApplyService = Depends(get_glassdoor_apply_service),
) -> GlassdoorAutomationResult:
    """Inspect a Glassdoor job posting URL on Windows via PyWinAuto.

    Verifies:
    1. Source is 'glassdoor' and domain belongs to legitimate Glassdoor domain.
    2. Active browser window is open/attached.
    3. Easy Apply button is discovered and verified without performing clicks.
    """
    try:
        return await _maybe_await(service.inspect_job_application(request))
    except Exception as exc:
        logger.error("Error inspecting Glassdoor job %s: %s", request.job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to inspect Glassdoor job: {str(exc)}",
        )


@router.post(
    "/navigate-to-submit",
    response_model=GlassdoorAutomationResult,
    summary="Enter Glassdoor Easy Apply modal, dynamically fill questions, and stop at SUBMISSION_READY.",
)
async def navigate_glassdoor_to_submit(
    request: GlassdoorAutomationNavigateRequest,
    service: GlassdoorApplyService = Depends(get_glassdoor_apply_service),
) -> GlassdoorAutomationResult:
    """Navigate through Glassdoor Easy Apply modal steps and stop strictly at SUBMISSION_READY."""
    try:
        return await _maybe_await(service.navigate_to_submit(request))
    except Exception as exc:
        logger.error("Error navigating Glassdoor job %s to submit: %s", request.job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed navigating Glassdoor application to submit: {str(exc)}",
        )


@router.post(
    "/resume",
    response_model=GlassdoorAutomationResult,
    summary="Resume a paused Glassdoor application after human input or with direct answers.",
)
async def resume_glassdoor_application(
    request: GlassdoorAutomationResumeRequest,
    service: GlassdoorApplyService = Depends(get_glassdoor_apply_service),
) -> GlassdoorAutomationResult:
    """Resume paused Glassdoor application flow without restarting job navigation or re-clicking Easy Apply."""
    try:
        return await _maybe_await(service.resume(request))
    except Exception as exc:
        logger.error("Error resuming Glassdoor application: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed resuming Glassdoor application: {str(exc)}",
        )


@router.post(
    "/apply",
    response_model=GlassdoorAutomationResult,
    summary="Complete full assisted application on Glassdoor with single submission.",
)
async def apply_to_glassdoor_job(
    request: GlassdoorAutomationApplyRequest,
    service: GlassdoorApplyService = Depends(get_glassdoor_apply_service),
) -> GlassdoorAutomationResult:
    """Execute verified single submission for a Glassdoor Easy Apply job."""
    try:
        return await _maybe_await(service.apply_to_job(request))
    except Exception as exc:
        logger.error("Error applying to Glassdoor job %s: %s", request.job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed applying to Glassdoor job: {str(exc)}",
        )


@router.get(
    "/sessions/{automation_session_id}",
    summary="Read-only debug endpoint to retrieve paused Glassdoor automation session metadata.",
)
async def get_glassdoor_session(
    automation_session_id: str,
    service: GlassdoorApplyService = Depends(get_glassdoor_apply_service),
) -> dict:
    """Retrieve stored session data for a paused application without touching the browser."""
    session_data = service.get_session(automation_session_id)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Automation session '{automation_session_id}' not found.",
        )
    return {
        "automation_session_id": automation_session_id,
        "job_url": session_data.get("job_url"),
        "job_id": session_data.get("job_id"),
        "job_listing_id": session_data.get("job_listing_id"),
        "source": session_data.get("source"),
        "application_page_url": session_data.get("application_page_url"),
        "application_page_title": session_data.get("application_page_title"),
        "application_host": session_data.get("application_host") or session_data.get("application_page_host"),
        "current_state": session_data.get("current_state") or session_data.get("pause_state"),
        "unresolved_questions": session_data.get("unresolved_questions", []),
        "unresolved_question_keys": session_data.get("unresolved_question_keys", []),
        "total_steps": session_data.get("total_steps", 0),
        "resume_attempt": session_data.get("resume_attempt", 0),
        "paused_at": session_data.get("paused_at"),
    }
