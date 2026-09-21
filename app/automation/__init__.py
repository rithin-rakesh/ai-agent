"""Browser automation module for job platforms."""

from app.automation.browser_profiles import BrowserProfileManager
from app.automation.playwright_manager import PlaywrightManager
from app.automation.session_manager import SessionState, SessionStatus

__all__ = [
    "BrowserProfileManager",
    "PlaywrightManager",
    "SessionState",
    "SessionStatus",
]
