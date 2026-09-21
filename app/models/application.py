"""Pydantic models for Application, ApplicationAnswer, and AutomationLog entities."""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


# ==============================================================================
# APPLICATION ANSWERS
# ==============================================================================
class ApplicationAnswerBase(BaseModel):
    """Base schema for question-answer records during application autofill."""

    question: str = Field(..., description="Application question text")
    answer: str = Field(..., description="Answer text provided")
    source: Optional[str] = Field(default=None, description="Source of the answer (e.g. profile, resume, llm, user)")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score (0.0 - 1.0)")


class ApplicationAnswerCreate(ApplicationAnswerBase):
    """Schema for creating a new application answer record."""

    application_id: UUID


class ApplicationAnswer(ApplicationAnswerBase):
    """Complete ApplicationAnswer model with identifiers and timestamps."""

    id: UUID
    application_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==============================================================================
# AUTOMATION LOGS
# ==============================================================================
class AutomationLogBase(BaseModel):
    """Base schema for logging automation events, clicks, submissions, and errors."""

    application_id: Optional[UUID] = Field(default=None, description="Associated application ID if applicable")
    platform: Optional[str] = Field(default=None, description="Target platform (e.g. linkedin, greenhouse, lever)")
    action: str = Field(..., description="Action taken (e.g. open_page, fill_field, upload_resume, submit)")
    status: str = Field(..., description="Action outcome: success, failure, pending, warning")
    details: Dict[str, Any] = Field(default_factory=dict, description="Metadata or structured diagnostics")
    screenshot_path: Optional[str] = Field(default=None, description="File path to captured screenshot if any")


class AutomationLogCreate(AutomationLogBase):
    """Schema for creating an automation log entry."""

    pass


class AutomationLog(AutomationLogBase):
    """Complete AutomationLog model with identifiers and timestamps."""

    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==============================================================================
# APPLICATIONS
# ==============================================================================
class ApplicationBase(BaseModel):
    """Base schema for a job application submission process."""

    job_id: UUID = Field(..., description="Referenced job ID")
    profile_id: UUID = Field(..., description="Referenced user profile ID")
    platform: str = Field(..., description="Application portal / platform")
    status: str = Field(default="pending", description="Status: pending, in_progress, submitted, failed, skipped")
    started_at: Optional[datetime] = Field(default=None, description="When application process started")
    submitted_at: Optional[datetime] = Field(default=None, description="When application was successfully submitted")
    failure_reason: Optional[str] = Field(default=None, description="Error message or reason if application failed")
    application_url: Optional[str] = Field(default=None, description="Direct URL of application form")
    confirmation_text: Optional[str] = Field(default=None, description="Confirmation response or confirmation number")


class ApplicationCreate(ApplicationBase):
    """Schema for creating an application record."""

    pass


class ApplicationUpdate(BaseModel):
    """Schema for updating an application record."""

    status: Optional[str] = None
    started_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    application_url: Optional[str] = None
    confirmation_text: Optional[str] = None


class Application(ApplicationBase):
    """Complete Application model with identifiers, timestamps, and nested records."""

    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
