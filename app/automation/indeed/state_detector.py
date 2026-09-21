"""State Detector for Indeed UI Automation.

Analyzes accessible UI controls, titles, and text trees to classify page states
(CAPTCHA, Login, External Apply, Apply with Indeed, Review, Submission Confirmation).
"""

import re
from typing import Any, List, Optional, Tuple
from app.automation.indeed.models import AutomationState


# Target Phrases and Patterns (Strict interactive challenge indicators; informational footers excluded)
CAPTCHA_PATTERNS = [
    r"\bi\s*'?\s*m\s+not\s+a\s+robot\b",
    r"\bverify\s+you\s+are\s+(?:a\s+)?human\b",
    r"\bsecurity\s+check\b",
    r"\bcloudflare\s+(?:turnstile|challenge|security)\b",
    r"\bcf-turnstile\b",
    r"\bunusual\s+traffic\b",
    r"\bhuman\s+verification\b",
    r"\badditional\s+verification\s+required\b",
    r"\brecaptcha\s+(?:challenge|checkbox|widget|interactive)\b",
    r"\bhcaptcha(?:\s+challenge|\s+checkbox)?\b",
    r"\bcomplete\s+the\s+security\s+check\b",
    r"\bpress\s+(?:&|and)\s+hold\b",
]

LOGIN_PATTERNS = [
    r"\bsign\s+in\s+to\s+indeed\b",
    r"\benter\s+your\s+password\b",
    r"\blog\s*in\b",
    r"\bsign\s+in\s+with\s+google\b",
    r"\bsign\s+in\s+with\s+apple\b",
]

MFA_PATTERNS = [
    r"\benter\s+the\s+code\b",
    r"\bverification\s+code\b",
    r"\b2-step\s+verification\b",
    r"\botp\b",
    r"\bverify\s+your\s+phone\b",
]

EXTERNAL_APPLY_PATTERNS = [
    r"\bapply\s+on\s+company\s+site\b",
    r"\bapply\s+on\s+employer\s+site\b",
    r"\bvisit\s+company\s+website\b",
    r"\bcompany\s+site\b",
    r"\bemployer\s+website\b",
]

ALREADY_APPLIED_PATTERNS = [
    r"\byou\s+already\s+applied\b",
    r"\byou\s+have\s+applied\b",
    r"\bapplication\s+submitted\b",
    r"\byou\s+applied\b",
]

JOB_UNAVAILABLE_PATTERNS = [
    r"\bjob\s+expired\b",
    r"\bjob\s+no\s+longer\s+available\b",
    r"\bposition\s+closed\b",
    r"\bthis\s+job\s+has\s+expired\b",
]

EXACT_SUBMIT_NAMES = {
    "submit",
    "submit application",
    "submit your application",
}

SUBMIT_BUTTON_NAMES = [
    "submit",
    "submit application",
    "submit your application",
]

CONFIRMATION_PATTERNS = [
    r"\byour\s+application\s+has\s+been\s+submitted\b",
    r"\bapplication\s+submitted\b",
    r"\byour\s+application\s+was\s+submitted\b",
    r"\bapplication\s+successfully\s+submitted\b",
    r"\byou\s+applied\b",
]


def normalize_submit_text(text: Optional[str]) -> str:
    """Normalize text for strict Submit name verification: lowercase, strip, collapse repeated whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def _normalize_text(text: Optional[str]) -> str:
    """Normalize text into clean lowercase alphanumeric string."""
    if not text:
        return ""
    t = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


class IndeedStateDetector:
    """Classifies Indeed application states from observed window and UIA controls."""

    @staticmethod
    def is_apply_with_indeed(text: Optional[str]) -> bool:
        """Verify whether an accessibility string matches 'Apply with Indeed' exactly."""
        norm = _normalize_text(text)
        return norm == "apply with indeed" or "apply with indeed" in norm

    @staticmethod
    def is_external_apply(text: Optional[str]) -> bool:
        """Check whether text indicates an external employer application."""
        norm = _normalize_text(text)
        return any(re.search(pat, norm) for pat in EXTERNAL_APPLY_PATTERNS)

    @staticmethod
    def is_exact_submit_name(name: Optional[str]) -> bool:
        """Check if an element name is EXACTLY 'submit', 'submit application', or 'submit your application'."""
        norm = normalize_submit_text(name)
        return norm in EXACT_SUBMIT_NAMES

    @staticmethod
    def is_submit_control(name: Optional[str], control_type: Optional[str] = None) -> bool:
        """Check if an element matches the final submission control."""
        return IndeedStateDetector.is_exact_submit_name(name)

    @staticmethod
    def is_submission_confirmation(text: Optional[str]) -> bool:
        """Check if text indicates application was successfully submitted."""
        norm = _normalize_text(text)
        return any(re.search(pat, norm) for pat in CONFIRMATION_PATTERNS)

    @staticmethod
    def is_stale_confirmation_page(page_text: Optional[str]) -> bool:
        """Check if page content indicates a stale submission confirmation page from a previous application."""
        if not page_text:
            return False
        norm = _normalize_text(page_text)
        return any(re.search(pat, norm) for pat in CONFIRMATION_PATTERNS)

    @staticmethod
    def detect_blocking_state(page_text: str) -> Optional[Tuple[AutomationState, str]]:
        """Inspect combined page text for CAPTCHA, login, MFA, already applied, or unavailable states.

        Returns:
            Tuple: (AutomationState, reason_message) or None if clean.
        """
        norm = _normalize_text(page_text)
        # Strip informational footer disclaimers before checking CAPTCHA patterns
        clean_norm = re.sub(r"this site is protected by recaptcha and the google privacy policy and terms of service apply", "", norm)
        clean_norm = re.sub(r"protected by recaptcha", "", clean_norm)

        # 1. CAPTCHA / Challenge Detection
        for pat in CAPTCHA_PATTERNS:
            if re.search(pat, clean_norm):
                return AutomationState.CAPTCHA_OR_CHALLENGE, "Security check or human verification challenge detected."

        # 2. MFA / OTP Detection
        for pat in MFA_PATTERNS:
            if re.search(pat, norm):
                return AutomationState.MFA_REQUIRED, "Multi-factor authentication (OTP/code) required."

        # 3. Login Detection
        for pat in LOGIN_PATTERNS:
            if re.search(pat, norm):
                return AutomationState.LOGIN_REQUIRED, "Manual login or password entry required."

        # 4. Already Applied Detection
        for pat in ALREADY_APPLIED_PATTERNS:
            if re.search(pat, norm):
                return AutomationState.ALREADY_APPLIED, "Candidate has already applied to this position."

        # 5. Job Unavailable Detection
        for pat in JOB_UNAVAILABLE_PATTERNS:
            if re.search(pat, norm):
                return AutomationState.JOB_UNAVAILABLE, "Job listing is closed or no longer available."

        return None
