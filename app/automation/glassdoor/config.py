"""Configuration constants for Glassdoor PyWinAuto Automation."""

# Supported Hostnames and Subdomains
SUPPORTED_GLASSDOOR_DOMAINS = [
    "glassdoor.com",
    "www.glassdoor.com",
    "glassdoor.co.in",
    "www.glassdoor.co.in",
    "glassdoor.ca",
    "www.glassdoor.ca",
    "glassdoor.co.uk",
    "www.glassdoor.co.uk",
    "glassdoor.com.au",
    "www.glassdoor.com.au",
]

# Timeouts (Seconds)
BROWSER_ATTACH_TIMEOUT_SECONDS = 15
GLASSDOOR_APPLICATION_TIMEOUT_SECONDS = 60
CONFIRMATION_TIMEOUT_SECONDS = 15
POST_EASY_APPLY_REDIRECT_WAIT_SECONDS = 10
POLL_INTERVAL_SECONDS = 1.0
CDP_CONNECT_TIMEOUT_SECONDS = 15

# CDP Connection Settings
import os


def is_live_cdp_allowed() -> bool:
    """Check if live CDP connection is permitted.

    In test environments (under pytest), defaults to False for test isolation unless explicitly overridden.
    Outside test environments (normal runtime / FastAPI), defaults to True.
    """
    env_val = os.getenv("ALLOW_LIVE_CDP")
    if env_val is not None:
        return env_val.lower() in ("true", "1", "yes")
    return "PYTEST_CURRENT_TEST" not in os.environ


ALLOW_LIVE_CDP = is_live_cdp_allowed()
CDP_HOST = "127.0.0.1"
CDP_REMOTE_DEBUGGING_PORT = 9222

# Traversal & Automation Limits
MAX_TAB_TRAVERSAL = 30
MAX_APPLICATION_STEPS = 20
MAX_MODAL_STEPS = 20
MAX_STALLED_SIGNATURE_COUNT = 3
KEYBOARD_INTER_KEY_DELAY_SECONDS = 0.05

# Coordinate Fallback Settings
GLASSDOOR_EASY_APPLY_FALLBACK_X = 270
GLASSDOOR_EASY_APPLY_FALLBACK_Y = 506
ALLOW_VERIFIED_GLASSDOOR_COORDINATE_FALLBACK = True

# SmartApply Hosts
SMARTAPPLY_HOST_DOMAINS = [
    "smartapply.indeed.com",
    "indeed.com",
]

# SPA Transition Settings
SPA_TRANSITION_TIMEOUT_SECONDS = 6.0
SPA_TRANSITION_POLL_INTERVAL_SECONDS = 0.2

# Requirements Warning Settings
CONTINUE_ON_REQUIREMENTS_WARNING = True

# Survey Interstitial Settings
SURVEY_HEADING_PATTERNS = [
    "help indeed learn more about why you're applying",
    "help indeed learn more about why you are applying",
    "why you're applying",
    "why you are applying",
]

SURVEY_SUBTEXT_PATTERNS = [
    "we won't share your response with the employer",
    "we will not share your response with the employer",
]

SURVEY_FIELD_IDENTIFIERS = [
    "reason for applying",
    "reason_for_applying",
]

# Button Name Lists
ALREADY_APPLIED_EXACT_NAMES = [
    "applied",
    "you applied",
    "already applied",
    "you've applied",
    "you ve applied",
    "you have applied",
    "application submitted",
]

EASY_APPLY_BUTTON_NAMES = [
    "easy apply",
    "apply now",
]

APPLY_ANYWAY_BUTTON_NAMES = [
    "apply anyway",
]

BLACKLISTED_PROGRESSION_BUTTON_NAMES = [
    "return to job search",
    "exit",
    "back",
    "cancel",
    "save and exit",
]

CONTINUE_BUTTON_NAMES = [
    "continue",
    "continue to application",
]

NEXT_BUTTON_NAMES = [
    "next",
    "next step",
]

REVIEW_BUTTON_NAMES = [
    "review",
    "review your application",
    "review application",
]

EXACT_SUBMIT_NAMES = {
    "submit application",
    "submit your application",
    "submit",
}
