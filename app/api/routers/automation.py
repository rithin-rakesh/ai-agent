"""FastAPI router for browser automation, session management, and Indeed flow automation."""

import inspect
import logging
from typing import Any, Dict, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.automation.browser_profiles import BrowserProfileManager
from app.automation.indeed.apply_service import IndeedApplyService
from app.automation.indeed.models import (
    IndeedAutomationApplyRequest,
    IndeedAutomationInspectRequest,
    IndeedAutomationNavigateRequest,
    IndeedAutomationResumeRequest,
    IndeedAutomationResult,
)
from app.automation.session_manager import SessionStatus
from app.config.settings import Settings, get_settings
from app.platforms.indeed.inspector import (
    IndeedJobInspectionResult,
    IndeedJobInspector,
)
from app.platforms.indeed.session import IndeedSessionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/automation", tags=["Automation"])


class AutomationHealthResponse(BaseModel):
    """Health check schema for browser automation infrastructure."""

    playwright: Dict[str, Any] = Field(..., description="Playwright engine status")
    indeed: Dict[str, Any] = Field(..., description="Indeed browser profile configuration")


class IndeedOpenRequest(BaseModel):
    """Optional request payload when opening Indeed home or a specific job URL."""

    job_url: Optional[str] = Field(
        default=None,
        description="Optional Indeed job URL to inspect. If omitted, opens Indeed homepage.",
    )


class IndeedOpenResponse(BaseModel):
    """Response schema for opening Indeed homepage."""

    session: SessionStatus = Field(..., description="Resulting session status")
    screenshot_saved: bool = Field(default=False, description="Whether a screenshot was captured")
    screenshot_path: str = Field(default="", description="Relative path of saved screenshot")


class IndeedJobInspectionRequest(BaseModel):
    """Request payload for inspecting an Indeed job page and application flow."""

    job_url: str = Field(..., description="Indeed job posting URL (*.indeed.com)")


def get_indeed_session_manager(
    settings: Settings = Depends(get_settings),
) -> IndeedSessionManager:
    """Dependency provider for IndeedSessionManager."""
    return IndeedSessionManager(settings=settings)


def get_indeed_inspector(
    settings: Settings = Depends(get_settings),
) -> IndeedJobInspector:
    """Dependency provider for IndeedJobInspector."""
    return IndeedJobInspector(settings=settings)


def get_indeed_apply_service() -> IndeedApplyService:
    """Dependency provider for IndeedApplyService."""
    return IndeedApplyService()


@router.get(
    "/health",
    response_model=AutomationHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check browser automation health and profile status",
)
async def get_automation_health(
    settings: Settings = Depends(get_settings),
) -> AutomationHealthResponse:
    """Return health status of Playwright and the Indeed persistent browser profile."""
    profile_manager = BrowserProfileManager()
    profile_path = settings.get_indeed_profile_path()

    playwright_available = False
    try:
        import playwright  # noqa: F401
        playwright_available = True
    except ImportError:
        playwright_available = False

    profile_exists = profile_manager.profile_exists(profile_path)
    file_count = profile_manager.get_profile_file_count(profile_path)

    return AutomationHealthResponse(
        playwright={
            "available": playwright_available,
            "browser": settings.PLAYWRIGHT_BROWSER,
            "headless_default": settings.PLAYWRIGHT_HEADLESS,
            "timeout_ms": settings.PLAYWRIGHT_TIMEOUT_MS,
            "trace_enabled": settings.PLAYWRIGHT_TRACE,
        },
        indeed={
            "profile_configured": True,
            "profile_path": str(settings.INDEED_BROWSER_PROFILE_PATH),
            "profile_directory_exists": profile_exists,
            "stored_files_count": file_count,
        },
    )


@router.get(
    "/indeed/session",
    response_model=SessionStatus,
    status_code=status.HTTP_200_OK,
    summary="Inspect Indeed session authentication status",
)
async def get_indeed_session_status(
    manager: IndeedSessionManager = Depends(get_indeed_session_manager),
) -> SessionStatus:
    """Inspect persistent browser context and report whether an authenticated Indeed session is active."""
    return await manager.check_session_status()


@router.post(
    "/indeed/open",
    response_model=Union[IndeedOpenResponse, IndeedJobInspectionResult],
    status_code=status.HTTP_200_OK,
    summary="Open Indeed homepage or inspect a specific job URL",
)
async def open_indeed_page(
    payload: Optional[IndeedOpenRequest] = None,
    session_manager: IndeedSessionManager = Depends(get_indeed_session_manager),
    inspector: IndeedJobInspector = Depends(get_indeed_inspector),
) -> Union[IndeedOpenResponse, IndeedJobInspectionResult]:
    """Launch persistent browser and open Indeed home or inspect a supplied job URL."""
    try:
        if payload and payload.job_url:
            return await inspector.inspect_job(payload.job_url, save_screenshot=True)

        session_status, screenshot_path = await session_manager.open_indeed_home(save_screenshot=True)
        return IndeedOpenResponse(
            session=session_status,
            screenshot_saved=screenshot_path is not None and screenshot_path.exists(),
            screenshot_path=str(screenshot_path) if screenshot_path else "",
        )
    except Exception as exc:
        logger.error("Failed to open Indeed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to open Indeed: {str(exc)}",
        )


async def _maybe_await(val: Any) -> Any:
    if inspect.isawaitable(val):
        return await val
    return val


@router.post(
    "/indeed/inspect",
    response_model=Union[IndeedJobInspectionResult, IndeedAutomationResult],
    status_code=status.HTTP_200_OK,
    summary="Inspect Indeed job page and application controls",
)
async def inspect_indeed_job(
    request: Union[IndeedAutomationInspectRequest, IndeedJobInspectionRequest],
    inspector: IndeedJobInspector = Depends(get_indeed_inspector),
    apply_service: IndeedApplyService = Depends(get_indeed_apply_service),
) -> Union[IndeedJobInspectionResult, IndeedAutomationResult]:
    """Inspect an Indeed job URL, verify 'Apply with Indeed', and check accessible controls without clicking."""
    if isinstance(request, IndeedAutomationInspectRequest):
        return await _maybe_await(apply_service.inspect_job_application(request))
    return await inspector.inspect_job(request.job_url, save_screenshot=True)


@router.post(
    "/indeed/navigate-to-submit",
    response_model=IndeedAutomationResult,
    status_code=status.HTTP_200_OK,
    summary="Enter Indeed application flow, navigate 22 TABs, and stop at SUBMISSION_READY",
)
async def navigate_indeed_to_submit(
    request: IndeedAutomationNavigateRequest,
    apply_service: IndeedApplyService = Depends(get_indeed_apply_service),
) -> IndeedAutomationResult:
    """Click verified Apply with Indeed, wait for redirect, navigate toward Submit, and stop before final submission."""
    return await _maybe_await(apply_service.navigate_to_submit(request))


@router.post(
    "/indeed/apply",
    response_model=IndeedAutomationResult,
    status_code=status.HTTP_200_OK,
    summary="Execute full assisted Indeed application workflow with verified submission",
)
async def apply_to_indeed_job(
    request: IndeedAutomationApplyRequest,
    apply_service: IndeedApplyService = Depends(get_indeed_apply_service),
) -> IndeedAutomationResult:
    """Complete full assisted Indeed application: inspect, click Apply, navigate, verify Submit, and submit once."""
    return await _maybe_await(apply_service.apply_to_job(request))


@router.post(
    "/indeed/resume-submit",
    response_model=IndeedAutomationResult,
    status_code=status.HTTP_200_OK,
    summary="Resume final submission on active application window after user manually completes verification",
)
async def resume_indeed_submit(
    request: IndeedAutomationResumeRequest,
    apply_service: IndeedApplyService = Depends(get_indeed_apply_service),
) -> IndeedAutomationResult:
    """Resume submission without re-discovering or re-clicking Apply: checks challenges, verifies Submit, submits once."""
    return await _maybe_await(apply_service.resume_submit(request))

