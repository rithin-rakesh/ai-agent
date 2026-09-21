"""Data Models and State Machine Definitions for Glassdoor PyWinAuto Automation.

Defines the Glassdoor automation lifecycle, request payloads, and structured execution results.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class GlassdoorAutomationState(str, Enum):
    """Lifecycle states for Glassdoor desktop UI automation."""

    INITIALIZING = "INITIALIZING"
    BROWSER_ATTACHED = "BROWSER_ATTACHED"
    JOB_PAGE = "JOB_PAGE"

    EASY_APPLY_AVAILABLE = "EASY_APPLY_AVAILABLE"
    EASY_APPLY_NOT_VERIFIED = "EASY_APPLY_NOT_VERIFIED"
    EXTERNAL_APPLY = "EXTERNAL_APPLY"

    APPLICATION_MODAL = "APPLICATION_MODAL"
    RESUME_STEP = "RESUME_STEP"
    QUESTIONS_STEP = "QUESTIONS_STEP"
    REQUIREMENTS_WARNING = "REQUIREMENTS_WARNING"
    OPTIONAL_SURVEY_STEP = "OPTIONAL_SURVEY_STEP"
    SMARTAPPLY_INTERSTITIAL = "SMARTAPPLY_INTERSTITIAL"
    REVIEW_STEP = "REVIEW_STEP"

    NEXT_AVAILABLE = "NEXT_AVAILABLE"
    CONTINUE_AVAILABLE = "CONTINUE_AVAILABLE"
    REVIEW_AVAILABLE = "REVIEW_AVAILABLE"

    NEEDS_USER_INPUT = "NEEDS_USER_INPUT"
    FORM_VALIDATION_ERROR = "FORM_VALIDATION_ERROR"

    SUBMISSION_READY = "SUBMISSION_READY"
    SUBMITTING = "SUBMITTING"
    APPLICATION_SUBMITTED = "APPLICATION_SUBMITTED"

    CAPTCHA_OR_CHALLENGE = "CAPTCHA_OR_CHALLENGE"
    LOGIN_REQUIRED = "LOGIN_REQUIRED"
    MFA_REQUIRED = "MFA_REQUIRED"

    ALREADY_APPLIED = "ALREADY_APPLIED"
    JOB_UNAVAILABLE = "JOB_UNAVAILABLE"

    ACTION_CONTROL_NOT_VERIFIED = "ACTION_CONTROL_NOT_VERIFIED"
    REQUIREMENTS_ACTION_NOT_VERIFIED = "REQUIREMENTS_ACTION_NOT_VERIFIED"
    SUBMIT_CONTROL_NOT_VERIFIED = "SUBMIT_CONTROL_NOT_VERIFIED"
    SUBMISSION_CONFIRMATION_UNVERIFIED = "SUBMISSION_CONFIRMATION_UNVERIFIED"

    EASY_APPLY_CLICK_UNVERIFIED = "EASY_APPLY_CLICK_UNVERIFIED"
    SMART_APPLY_HOST_VERIFIED = "SMART_APPLY_HOST_VERIFIED"
    APPLICATION_PROGRESS_STALLED = "APPLICATION_PROGRESS_STALLED"

    JOB_NAVIGATION_FAILED = "JOB_NAVIGATION_FAILED"
    BROWSER_NOT_VERIFIED = "BROWSER_NOT_VERIFIED"
    BROWSER_SESSION_MISMATCH = "BROWSER_SESSION_MISMATCH"
    APPLICATION_PAGE_NOT_FOUND = "APPLICATION_PAGE_NOT_FOUND"
    PAUSED_APPLICATION_NOT_FOUND = "PAUSED_APPLICATION_NOT_FOUND"
    INVALID_JOB_URL = "INVALID_JOB_URL"
    UNSUPPORTED_AUTOMATION_SOURCE = "UNSUPPORTED_AUTOMATION_SOURCE"
    AUTOMATION_RUNTIME_ERROR = "AUTOMATION_RUNTIME_ERROR"
    UNKNOWN = "UNKNOWN"
    FAILED = "FAILED"


class GlassdoorAutomationInspectRequest(BaseModel):
    """Request schema for inspecting a Glassdoor job without performing clicks."""

    url: str = Field(..., description="Direct Glassdoor job posting URL")
    job_id: Optional[UUID] = Field(default=None, description="Optional unique job identifier UUID")
    source: Optional[str] = Field(default="glassdoor", description="Job board source platform")
    profile_id: Optional[UUID] = Field(default=None, description="Optional candidate profile UUID")


class GlassdoorAutomationNavigateRequest(BaseModel):
    """Request schema for entering Glassdoor application flow and stopping at SUBMISSION_READY."""

    url: str = Field(..., description="Direct Glassdoor job posting URL")
    job_id: Optional[UUID] = Field(default=None, description="Optional unique job identifier UUID")
    source: Optional[str] = Field(default="glassdoor", description="Job board source platform")
    profile_id: Optional[UUID] = Field(default=None, description="Optional candidate profile UUID")


class GlassdoorAutomationApplyRequest(BaseModel):
    """Request schema for executing full assisted application on Glassdoor with single submission."""

    job_id: UUID = Field(..., description="Unique job identifier UUID")
    url: str = Field(..., description="Direct Glassdoor job posting URL")
    source: Optional[str] = Field(default="glassdoor", description="Job board source platform")
    profile_id: Optional[UUID] = Field(default=None, description="Optional candidate profile UUID")


class GlassdoorAutomationResumeRequest(BaseModel):
    """Request schema for resuming a paused Glassdoor application after human input or with direct answers."""

    automation_session_id: Optional[str] = Field(default=None, description="Automation session identifier from previous pause")
    answers: Optional[Dict[str, Any]] = Field(default=None, description="Optional map of question keys/text to user-supplied answers")
    url: Optional[str] = Field(default=None, description="Optional job URL to resume")
    job_id: Optional[UUID] = Field(default=None, description="Optional unique job identifier UUID")
    source: Optional[str] = Field(default="glassdoor", description="Job board source platform")
    profile_id: Optional[UUID] = Field(default=None, description="Optional candidate profile UUID")


class GlassdoorAutomationResult(BaseModel):
    """Execution result for Glassdoor single-job automation."""

    status: str = Field(..., description="Status string: success, blocked, failed, manual_action_required")
    url: str = Field(..., description="Target job URL")
    job_id: Optional[UUID] = Field(default=None, description="Target job UUID if supplied")
    platform: str = Field(default="glassdoor", description="Target platform identifier")
    browser: Optional[str] = Field(default=None, description="Attached browser process name")

    easy_apply_verified: bool = Field(default=False, description="Whether Easy Apply button was positively verified")
    easy_apply_clicked: bool = Field(default=False, description="Whether Easy Apply button was clicked")

    steps_processed: int = Field(default=0, description="Total modal steps processed")
    questions_detected: int = Field(default=0, description="Count of form questions detected")
    questions_answered: int = Field(default=0, description="Count of form questions successfully answered")
    questions_unresolved: int = Field(default=0, description="Count of unresolved required questions")

    resume_state: Optional[str] = Field(default=None, description="Resume selection state: selected, default_selected, unverified, not_needed")

    current_state: GlassdoorAutomationState = Field(..., description="Current state machine terminal/lifecycle state")
    manual_action_required: bool = Field(default=False, description="True if human intervention is needed")

    submit_verified: bool = Field(default=False, description="True if Submit button was positively verified")
    final_submit_clicked: bool = Field(default=False, description="True if final Submit button was activated")
    submission_confirmed: bool = Field(default=False, description="True if post-submission confirmation was verified")

    message: Optional[str] = Field(default=None, description="Detailed diagnostic or error message")
    failure_reason: Optional[str] = Field(default=None, description="Specific failure reason if any")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Structured diagnostics and control details")
