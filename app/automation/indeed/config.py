"""Configuration and machine-specific constants for Indeed PyWinAuto Automation.

Centralizes timeouts, fallback coordinate constants, and tab navigation parameters.
"""

from typing import Set

# Supported Indeed Hostname Domains
SUPPORTED_INDEED_DOMAINS: Set[str] = {
    "indeed.com",
    "www.indeed.com",
    "in.indeed.com",
    "uk.indeed.com",
    "ca.indeed.com",
    "au.indeed.com",
    "de.indeed.com",
    "fr.indeed.com",
    "sg.indeed.com",
}

# Machine-Specific Observed Coordinate Fallback
INDEED_APPLY_FALLBACK_ENABLED: bool = True
INDEED_APPLY_FALLBACK_X: int = 420
INDEED_APPLY_FALLBACK_Y: int = 563

# Expected Screen Environment for Coordinate Fallback
FALLBACK_EXPECTED_PRIMARY_WIDTH: int = 1920
FALLBACK_EXPECTED_PRIMARY_HEIGHT: int = 1080

# Keyboard Navigation
INDEED_TAB_COUNT_TO_SUBMIT: int = 22
KEYBOARD_INTER_KEY_DELAY_SECONDS: float = 0.05

# Timeouts in Seconds
BROWSER_ATTACH_TIMEOUT_SECONDS: int = 15
INITIAL_JOB_PAGE_TIMEOUT_SECONDS: int = 15
INDEED_APPLICATION_TIMEOUT_SECONDS: int = 60
CONTROL_LOOKUP_TIMEOUT_SECONDS: int = 10
CONFIRMATION_TIMEOUT_SECONDS: int = 15
POLL_INTERVAL_SECONDS: float = 1.0

# Supported Desktop Browsers
SUPPORTED_BROWSER_PROCESSES: Set[str] = {
    "chrome.exe",
    "msedge.exe",
}
SUPPORTED_BROWSER_CLASSES: Set[str] = {
    "Chrome_WidgetWin_1",
}
