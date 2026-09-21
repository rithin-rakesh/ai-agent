"""Data Models for Generic Job Application Form Infrastructure.

Defines schemas for normalized application questions, input control types,
form inspection results, and answer resolution objects.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class QuestionInputType(str, Enum):
    """Supported input control types in job application forms."""

    TEXT = "text"
    NUMBER = "number"
    DROPDOWN = "dropdown"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    CHECKBOX_MULTI = "checkbox_multi"
    UNKNOWN = "unknown"


class ApplicationQuestion(BaseModel):
    """Normalized representation of a single question/field in a job application form."""

    text: str = Field(..., description="Visible question or label text")
    normalized_key: str = Field(..., description="Canonical snake_case key derived from question text")
    input_type: QuestionInputType = Field(default=QuestionInputType.TEXT, description="Detected input control type")
    required: bool = Field(default=False, description="Whether this field is mandatory/required")
    current_value: Optional[str] = Field(default=None, description="Current value or text populated in control")
    options: List[str] = Field(default_factory=list, description="Available selectable options for dropdown, radio, or checkbox")
    is_ambiguous: bool = Field(default=False, description="True if label-to-input association was ambiguous")
    field_name: Optional[str] = Field(default=None, description="Accessibility or DOM field name identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic and UI coordinate metadata")


class AnswerResolution(BaseModel):
    """Result of attempting to resolve a question using profile and memory data."""

    question: ApplicationQuestion
    is_resolved: bool = Field(default=False, description="Whether a reliable answer was resolved")
    resolved_value: Optional[Union[str, int, float, bool, List[str]]] = Field(
        default=None, description="Resolved answer payload"
    )
    source: Optional[str] = Field(
        default=None, description="Source of resolved answer (e.g. stored_answer, candidate_approved, profile, skill, unresolved)"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the resolved answer")
    reason: Optional[str] = Field(default=None, description="Explanation for resolution or failure")
    action: str = Field(default="fill", description="Action to take: fill, skip, leave_unanswered, needs_user_input")
    is_optional: bool = Field(default=False, description="Whether question is optional in UI")
    normalized_key: Optional[str] = Field(default=None, description="Canonical question key")


class FormInspectionResult(BaseModel):
    """Structured result of inspecting an active form container."""

    questions: List[ApplicationQuestion] = Field(default_factory=list, description="List of detected questions")
    unresolved_required_count: int = Field(default=0, description="Count of required questions without answers")
    detected_controls_count: int = Field(default=0, description="Total input controls observed in container")
    is_valid: bool = Field(default=True, description="Whether form inspection succeeded cleanly")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Metadata and raw control metrics")
