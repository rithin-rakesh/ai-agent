"""Indeed platform automation, session management, and job flow inspection."""

from app.platforms.indeed.adapter import IndeedPlatformAdapter
from app.platforms.indeed.inspector import (
    FormFieldInfo,
    IndeedJobInspectionResult,
    IndeedJobInspector,
)
from app.platforms.indeed.session import IndeedSessionManager

__all__ = [
    "IndeedPlatformAdapter",
    "IndeedSessionManager",
    "IndeedJobInspector",
    "IndeedJobInspectionResult",
    "FormFieldInfo",
]
