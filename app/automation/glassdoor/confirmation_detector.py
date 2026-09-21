"""Confirmation Detector for Glassdoor Applications.

Detects post-submission confirmation patterns and confirmation dialogs on Glassdoor.
"""

import re
from typing import Optional

CONFIRMATION_PATTERNS = [
    r"\bapplication\s+submitted\b",
    r"\byour\s+application\s+(?:was|has\s+been)\s+submitted\b",
    r"\bapplication\s+sent\b",
    r"\bthank\s+you\s+for\s+applying\b",
    r"\byour\s+application\s+has\s+been\s+received\b",
    r"\byou\s+applied\b",
    r"\byou\s+have\s+applied\b",
    r"\bapplied\b",
]


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    t = re.sub(r"[^\w\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", t).strip()


class GlassdoorConfirmationDetector:
    """Detects Glassdoor submission confirmation states."""

    @staticmethod
    def is_submission_confirmation(text: Optional[str]) -> bool:
        """Check if visible text contains a positive application submission confirmation."""
        norm = normalize_text(text)
        return any(re.search(pat, norm) for pat in CONFIRMATION_PATTERNS)

    @staticmethod
    def is_stale_confirmation_page(text: Optional[str]) -> bool:
        """Check if page shows a stale confirmation page from a previous application."""
        return GlassdoorConfirmationDetector.is_submission_confirmation(text)
