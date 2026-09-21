"""Session state models and definitions for browser automation."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class SessionState(str, Enum):
    """Lifecycle states for a platform browser session."""

    SESSION_UNKNOWN = "session_unknown"
    LOGIN_REQUIRED = "login_required"
    AUTHENTICATED = "authenticated"
    SESSION_ERROR = "session_error"


class SessionStatus(BaseModel):
    """Structured platform session report."""

    platform: str = Field(..., description="Target job platform (e.g. indeed)")
    browser: str = Field(default="chromium", description="Browser engine utilized")
    profile_path: Optional[str] = Field(default=None, description="Resolved persistent profile directory")
    authenticated: bool = Field(default=False, description="Whether the session is currently authenticated")
    status: SessionState = Field(default=SessionState.SESSION_UNKNOWN, description="Detailed session lifecycle status")
    detected_user: Optional[str] = Field(default=None, description="Non-sensitive user handle or indicator if found")
    message: Optional[str] = Field(default=None, description="Human-readable description of session state")
