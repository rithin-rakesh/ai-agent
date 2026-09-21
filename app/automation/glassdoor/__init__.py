"""Glassdoor Automation Package.

Provides state-driven UI automation for Glassdoor single-job Easy Apply applications.
"""

from app.automation.glassdoor.apply_service import GlassdoorApplyService
from app.automation.glassdoor.confirmation_detector import GlassdoorConfirmationDetector
from app.automation.glassdoor.modal_inspector import GlassdoorModalInspector
from app.automation.glassdoor.models import (
    GlassdoorAutomationApplyRequest,
    GlassdoorAutomationInspectRequest,
    GlassdoorAutomationNavigateRequest,
    GlassdoorAutomationResult,
    GlassdoorAutomationState,
)
from app.automation.glassdoor.pywinauto_driver import PyWinAutoGlassdoorDriver
from app.automation.glassdoor.resume_handler import GlassdoorResumeHandler
from app.automation.glassdoor.state_detector import GlassdoorStateDetector
from app.automation.glassdoor.url_validator import (
    extract_glassdoor_job_id,
    is_glassdoor_domain,
    validate_glassdoor_request,
)

__all__ = [
    "GlassdoorAutomationState",
    "GlassdoorAutomationInspectRequest",
    "GlassdoorAutomationNavigateRequest",
    "GlassdoorAutomationApplyRequest",
    "GlassdoorAutomationResult",
    "GlassdoorApplyService",
    "PyWinAutoGlassdoorDriver",
    "GlassdoorStateDetector",
    "GlassdoorModalInspector",
    "GlassdoorResumeHandler",
    "GlassdoorConfirmationDetector",
    "is_glassdoor_domain",
    "validate_glassdoor_request",
    "extract_glassdoor_job_id",
]
