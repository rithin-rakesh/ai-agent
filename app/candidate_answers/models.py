"""Data Models for Candidate Answer Bank (Phase 5.8).

Defines schemas for approved candidate answers, question categories,
answer types, and diagnostic test request/response structures.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QuestionCategory(str, Enum):
    """Categorization taxonomy for job application questions."""

    PERSONAL = "personal"
    EDUCATION = "education"
    EXPERIENCE = "experience"
    SKILLS = "skills"
    SALARY = "salary"
    NOTICE_PERIOD = "notice_period"
    LOCATION = "location"
    RELOCATION = "relocation"
    WORK_MODE = "work_mode"
    AUTHORIZATION = "authorization"
    CERTIFICATIONS = "certifications"
    PROJECTS = "projects"
    MOTIVATION = "motivation"
    FREE_TEXT = "free_text"
    PLATFORM_SURVEY = "platform_survey"
    UNKNOWN = "unknown"


class AnswerType(str, Enum):
    """Supported answer data and UI control types."""

    TEXT = "text"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SELECT = "select"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    CHECKBOX_MULTI = "checkbox_multi"
    FREE_TEXT = "free_text"
    NUMBER_OR_TEXT = "number_or_text"


class CandidateAnswer(BaseModel):
    """Approved candidate answer record for reusable automated question answering."""

    normalized_key: str = Field(..., description="Canonical snake_case key")
    category: QuestionCategory = Field(default=QuestionCategory.UNKNOWN, description="Domain category")
    answer_type: AnswerType = Field(default=AnswerType.TEXT, description="Expected control/data type")
    value: Any = Field(..., description="Approved answer payload (string, number, boolean, or list)")
    source: str = Field(
        default="candidate_approved",
        description="Source of truth: candidate_profile, candidate_approved, manual_user_input, etc.",
    )
    approved: bool = Field(default=True, description="Whether this answer has been approved by the candidate")
    scope: str = Field(default="global", description="Scope of application: 'global', 'platform', or 'employer'")
    platforms: List[str] = Field(
        default_factory=lambda: ["indeed", "glassdoor"],
        description="Platforms where this answer is valid",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    requires_user_approval: bool = Field(default=False, description="Whether user review is required before use")
    description: Optional[str] = Field(default=None, description="Human-readable description or rationale")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CandidateAnswerUpdateRequest(BaseModel):
    """Payload to create or update an approved candidate answer."""

    normalized_key: str = Field(..., description="Canonical key to set")
    value: Any = Field(..., description="Answer value")
    category: Optional[QuestionCategory] = Field(default=QuestionCategory.UNKNOWN)
    answer_type: Optional[AnswerType] = Field(default=AnswerType.TEXT)
    source: Optional[str] = Field(default="candidate_approved")
    approved: Optional[bool] = Field(default=True)
    scope: Optional[str] = Field(default="global")
    platforms: Optional[List[str]] = Field(default_factory=lambda: ["indeed", "glassdoor"])
    requires_user_approval: Optional[bool] = Field(default=False)
    description: Optional[str] = Field(default=None)


class CandidateAnswerTestRequest(BaseModel):
    """Diagnostic dry-run request to test question resolution against the answer bank."""

    question_text: str = Field(..., description="Raw question or field label text")
    options: Optional[List[str]] = Field(default_factory=list, description="Available selectable options if any")
    input_type: Optional[str] = Field(default="text", description="Detected UI control type")
    platform: Optional[str] = Field(default="glassdoor", description="Target platform (glassdoor or indeed)")
    host: Optional[str] = Field(default="glassdoor", description="Application host (glassdoor or indeed_smartapply)")


class CandidateAnswerTestResponse(BaseModel):
    """Structured resolution diagnostics for preview/debug mode."""

    question: str
    normalized_key: str
    category: str
    resolved_answer: Optional[Any] = None
    answer_source: Optional[str] = None
    confidence: float = 1.0
    control_type: str = "text"
    action: str = "fill"  # "fill", "skip", "leave_unanswered", "needs_user_input"
    is_resolved: bool = False
    reason: Optional[str] = None
