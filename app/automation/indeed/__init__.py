"""Indeed PyWinAuto Automation Package."""

from app.automation.indeed.apply_service import IndeedApplyService
from app.automation.indeed.models import (
    AutomationState,
    IndeedAutomationApplyRequest,
    IndeedAutomationInspectRequest,
    IndeedAutomationNavigateRequest,
    IndeedAutomationResumeRequest,
    IndeedAutomationResult,
)
from app.automation.indeed.pywinauto_driver import PyWinAutoIndeedDriver, is_windows
from app.automation.indeed.state_detector import IndeedStateDetector
from app.automation.indeed.url_validator import is_indeed_domain, validate_indeed_request

__all__ = [
    "AutomationState",
    "IndeedApplyService",
    "IndeedAutomationApplyRequest",
    "IndeedAutomationInspectRequest",
    "IndeedAutomationNavigateRequest",
    "IndeedAutomationResumeRequest",
    "IndeedAutomationResult",
    "IndeedStateDetector",
    "PyWinAutoIndeedDriver",
    "is_indeed_domain",
    "is_windows",
    "validate_indeed_request",
]
