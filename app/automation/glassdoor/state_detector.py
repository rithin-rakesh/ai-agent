"""State Detector for Glassdoor UI Automation.

Analyzes accessible UI controls, modal containers, and page text trees to classify
Glassdoor states (Easy Apply, External Apply, Modal Steps, Actions, Submit, Challenges).
"""

import re
from typing import Any, List, Optional, Tuple
from app.automation.glassdoor.config import (
    ALREADY_APPLIED_EXACT_NAMES,
    CONTINUE_BUTTON_NAMES,
    EASY_APPLY_BUTTON_NAMES,
    EXACT_SUBMIT_NAMES,
    NEXT_BUTTON_NAMES,
    REVIEW_BUTTON_NAMES,
)
from app.automation.glassdoor.models import GlassdoorAutomationState

# Blocking Patterns (Strict interactive challenge indicators; informational footers excluded)
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
    r"\bsign\s+in\s+to\s+glassdoor\b",
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
    r"\bapply\s+on\s+(?:the\s+)?(?:company|employer|external)(?:\s+(?:site|website))?\b",
    r"\bapply\s+(?:via|through)\s+(?:the\s+)?(?:company|employer|external)\b",
    r"\bapply\s+externally\b",
    r"\bredirect\s+to\s+company\s+(?:site|website)\b",
    r"\bapply\s+now\s+on\b",
    r"\bvisit\s+company\s+(?:site|website)\b",
]

ALREADY_APPLIED_PATTERNS = [
    r"^(?:status\s*:?\s*)?you\s+already\s+applied(?:\s+to\s+this\s+(?:job|position))?$",
    r"^(?:status\s*:?\s*)?you\s+have\s+applied(?:\s+to\s+this\s+(?:job|position))?$",
    r"^(?:status\s*:?\s*)?you\s+ve\s+applied(?:\s+to\s+this\s+(?:job|position))?$",
    r"^(?:status\s*:?\s*)?you\s+applied(?:\s+to\s+this\s+(?:job|position)|\s+on\s+[a-z0-9\s]+)?$",
    r"^(?:status\s*:?\s*)?already\s+applied$",
    r"^(?:status\s*:?\s*)?applied$",
    r"^(?:status\s*:?\s*)?application\s+submitted$",
]

JOB_UNAVAILABLE_PATTERNS = [
    r"\bjob\s+expired\b",
    r"\bjob\s+no\s+longer\s+available\b",
    r"\bposition\s+closed\b",
    r"\bthis\s+job\s+has\s+expired\b",
]


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    t = re.sub(r"[^\w\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", t).strip()


class GlassdoorStateDetector:
    """Classifies Glassdoor UI states and action buttons."""

    @staticmethod
    def is_easy_apply_button(text: Optional[str]) -> bool:
        """Verify whether an accessibility string corresponds to Glassdoor Easy Apply."""
        if not text:
            return False
        norm = _normalize(text)
        if not norm:
            return False
        if GlassdoorStateDetector.is_external_apply(norm):
            return False
        if any(w in norm for w in ["employer", "company site", "employer site", "external"]):
            return False
        return any(name in norm for name in EASY_APPLY_BUTTON_NAMES)

    @staticmethod
    def is_already_applied_control(text: Optional[str]) -> bool:
        """Check if an element label/name is an explicit Already Applied indicator."""
        if not text:
            return False
        norm = _normalize(text)
        if norm in ALREADY_APPLIED_EXACT_NAMES:
            return True
        return any(re.match(pat, norm) for pat in ALREADY_APPLIED_PATTERNS)

    @staticmethod
    def is_external_apply(text: Optional[str]) -> bool:
        """Check whether text indicates an external employer application."""
        norm = _normalize(text)
        return any(re.search(pat, norm) for pat in EXTERNAL_APPLY_PATTERNS)

    @staticmethod
    def is_exact_submit_name(name: Optional[str]) -> bool:
        """Check if an element name is EXACTLY an accepted Submit action."""
        if not name:
            return False
        norm = re.sub(r"\s+", " ", str(name).lower().strip())
        return norm in EXACT_SUBMIT_NAMES

    @staticmethod
    def is_continue_button(name: Optional[str]) -> bool:
        """Check if control name is a Continue action."""
        norm = _normalize(name)
        return any(n in norm for n in CONTINUE_BUTTON_NAMES)

    @staticmethod
    def is_next_button(name: Optional[str]) -> bool:
        """Check if control name is a Next action."""
        norm = _normalize(name)
        return any(n in norm for n in NEXT_BUTTON_NAMES)

    @staticmethod
    def is_review_button(name: Optional[str]) -> bool:
        """Check if control name is a Review action."""
        norm = _normalize(name)
        return any(n in norm for n in REVIEW_BUTTON_NAMES)

    @staticmethod
    def detect_blocking_state(page_text: str) -> Optional[Tuple[GlassdoorAutomationState, str]]:
        """Inspect combined page text for CAPTCHA, login, MFA, already applied, or unavailable states.

        Returns:
            Tuple: (GlassdoorAutomationState, reason_message) or None if clean.
        """
        norm = _normalize(page_text)

        # Strip known benign informational footer disclaimers before challenge evaluation
        clean_text = re.sub(
            r"this\s+site\s+is\s+protected\s+by\s+recaptcha.*?(?:terms\s+of\s+service\s+apply|privacy\s+policy)",
            " ",
            norm,
            flags=re.DOTALL,
        )
        clean_text = re.sub(r"protected\s+by\s+recaptcha", " ", clean_text)

        # 1. CAPTCHA / Challenge Detection
        for pat in CAPTCHA_PATTERNS:
            if re.search(pat, clean_text):
                return GlassdoorAutomationState.CAPTCHA_OR_CHALLENGE, "Security check or human verification challenge detected."

        # 2. MFA / OTP Detection
        for pat in MFA_PATTERNS:
            if re.search(pat, norm):
                return GlassdoorAutomationState.MFA_REQUIRED, "Multi-factor authentication (OTP/code) required."

        # 3. Login Detection
        for pat in LOGIN_PATTERNS:
            if re.search(pat, norm):
                return GlassdoorAutomationState.LOGIN_REQUIRED, "Manual login or password entry required."

        # 4. Already Applied Detection (Line-by-line / anchored match only; ignores sidebars/recommendations)
        if isinstance(page_text, str):
            for line in page_text.splitlines():
                line_norm = _normalize(line)
                if line_norm and any(re.match(pat, line_norm) for pat in ALREADY_APPLIED_PATTERNS):
                    return GlassdoorAutomationState.ALREADY_APPLIED, "Candidate has already applied to this position on Glassdoor."

        # 5. Job Unavailable Detection
        for pat in JOB_UNAVAILABLE_PATTERNS:
            if re.search(pat, norm):
                return GlassdoorAutomationState.JOB_UNAVAILABLE, "Job listing is closed or no longer available."

        return None
