"""Data Models and State Machine Definitions for Indeed PyWinAuto Automation.

Defines the 30-state automation lifecycle, request payloads, and structured execution results.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class AutomationState(str, Enum):
    """Lifecycle states for Indeed PyWinAuto desktop automation."""

    INITIALIZING = "INITIALIZING"
    BROWSER_ATTACHED = "BROWSER_ATTACHED"
    JOB_PAGE = "JOB_PAGE"
    APPLY_WITH_INDEED_AVAILABLE = "APPLY_WITH_INDEED_AVAILABLE"
    APPLY_WITH_INDEED_NOT_VERIFIED = "APPLY_WITH_INDEED_NOT_VERIFIED"
    EXTERNAL_APPLY = "EXTERNAL_APPLY"
    LOGIN_REQUIRED = "LOGIN_REQUIRED"
    MFA_REQUIRED = "MFA_REQUIRED"
    CAPTCHA_OR_CHALLENGE = "CAPTCHA_OR_CHALLENGE"
    APPLICATION_LOADING = "APPLICATION_LOADING"
    APPLICATION_FORM = "APPLICATION_FORM"
    NEEDS_USER_INPUT = "NEEDS_USER_INPUT"
    REVIEW_PAGE = "REVIEW_PAGE"
    TAB_START_STATE_UNVERIFIED = "TAB_START_STATE_UNVERIFIED"
    SUBMIT_FOCUS_UNVERIFIED = "SUBMIT_FOCUS_UNVERIFIED"
    SUBMISSION_READY = "SUBMISSION_READY"
    SUBMITTING = "SUBMITTING"
    SUBMISSION_CONFIRMATION_PENDING = "SUBMISSION_CONFIRMATION_PENDING"
    APPLICATION_SUBMITTED = "APPLICATION_SUBMITTED"
    SUBMISSION_CONFIRMATION_UNVERIFIED = "SUBMISSION_CONFIRMATION_UNVERIFIED"
    ALREADY_APPLIED = "ALREADY_APPLIED"
    JOB_UNAVAILABLE = "JOB_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    UNSUPPORTED_AUTOMATION_SOURCE = "UNSUPPORTED_AUTOMATION_SOURCE"
    INVALID_JOB_URL = "INVALID_JOB_URL"
    UNSUPPORTED_PLATFORM = "UNSUPPORTED_PLATFORM"
    BROWSER_NOT_VERIFIED = "BROWSER_NOT_VERIFIED"
    COORDINATE_ENVIRONMENT_MISMATCH = "COORDINATE_ENVIRONMENT_MISMATCH"
    JOB_NAVIGATION_FAILED = "JOB_NAVIGATION_FAILED"
    UNKNOWN = "UNKNOWN"
    FAILED = "FAILED"


class IndeedAutomationInspectRequest(BaseModel):
    """Request schema for inspecting an Indeed job without performing clicks."""

    job_id: UUID = Field(..., description="Unique job identifier UUID")
    url: str = Field(..., description="Direct Indeed job posting URL")
    source: Optional[str] = Field(default="indeed", description="Job board source platform")
    profile_id: Optional[UUID] = Field(default=None, description="Optional candidate profile UUID")


class IndeedAutomationNavigateRequest(BaseModel):
    """Request schema for entering Indeed application flow and stopping at SUBMISSION_READY."""

    job_id: UUID = Field(..., description="Unique job identifier UUID")
    url: str = Field(..., description="Direct Indeed job posting URL")
    source: Optional[str] = Field(default="indeed", description="Job board source platform")
    profile_id: Optional[UUID] = Field(default=None, description="Optional candidate profile UUID")


class IndeedAutomationApplyRequest(BaseModel):
    """Request schema for complete assisted application up to verified submission."""

    job_id: UUID = Field(..., description="Unique job identifier UUID")
    url: str = Field(..., description="Direct Indeed job posting URL")
    source: Optional[str] = Field(default="indeed", description="Job board source platform")
    profile_id: Optional[UUID] = Field(default=None, description="Optional candidate profile UUID")


class IndeedAutomationResumeRequest(BaseModel):
    """Request schema for resuming submission after manual human verification."""

    job_id: UUID = Field(..., description="Unique job identifier UUID")
    url: str = Field(..., description="Direct Indeed job posting URL")
    source: Optional[str] = Field(default="indeed", description="Job board source platform")
    profile_id: Optional[UUID] = Field(default=None, description="Optional candidate profile UUID")


class IndeedAutomationResult(BaseModel):
    """Comprehensive result of Indeed application automation attempt."""

    status: str = Field(..., description="'success', 'blocked', 'manual_action_required', or 'error'")
    job_id: UUID = Field(..., description="Job primary key UUID")
    url: str = Field(..., description="Direct Indeed job listing URL")
    platform: str = Field(default="indeed", description="Target job platform")
    browser: Optional[str] = Field(default=None, description="Detected browser engine ('chrome' | 'edge')")
    apply_method: Optional[str] = Field(
        default=None,
        description="Method used to activate Apply ('uia_control' | 'coordinate_fallback')",
    )
    button_text: Optional[str] = Field(default=None, description="Verified accessible button text")
    button_verified: bool = Field(default=False, description="Whether 'Apply with Indeed' was positively confirmed")
    apply_coordinates: Optional[List[int]] = Field(default=None, description="Fallback coordinates [X, Y] if used")
    apply_clicked: bool = Field(default=False, description="Whether Apply button activation was executed")
    redirect_completed: bool = Field(default=False, description="Whether application page loaded successfully")
    redirect_time_ms: Optional[int] = Field(default=None, description="Time taken for application page to load in ms")
    tabs_sent: int = Field(default=0, description="Count of TAB keypresses sent in navigation fallback")
    submit_verified: bool = Field(default=False, description="Whether the final Submit control was positively verified")
    final_submit_clicked: bool = Field(default=False, description="Whether final submission Enter was sent")
    submission_confirmed: bool = Field(default=False, description="Whether post-submission confirmation was detected")
    current_state: AutomationState = Field(..., description="Final or current state machine state")
    manual_action_required: bool = Field(default=False, description="Whether human intervention is needed (CAPTCHA/MFA)")
    message: Optional[str] = Field(default=None, description="Human-readable outcome or failure explanation")
    diagnostics: Optional[Dict[str, Any]] = Field(default=None, description="Detailed diagnostic metrics and logs")
