"""Modal Inspector for Glassdoor Applications.

Discovers and scopes UI inspection to the active Glassdoor Easy Apply modal window/container.
"""

import logging
import re
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    t = re.sub(r"[^\w\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", t).strip()


class GlassdoorModalInspector:
    """Finds and inspects the active Glassdoor Easy Apply application modal."""

    @staticmethod
    def find_application_modal(window: Any) -> Optional[Any]:
        """Locate the active Glassdoor application modal within the browser window."""
        if not window:
            return None

        try:
            # 1. Search for a Dialog control or Group with modal indicators
            dialogs = window.descendants(control_type="Dialog") if hasattr(window, "descendants") else []
            for d in dialogs:
                d_name = getattr(d.element_info, "name", "")
                if getattr(d.element_info, "is_visible", True):
                    return d

            # 2. Search for Group / Pane containers with "apply", "application", or "review"
            panes = window.descendants(control_type="Pane") if hasattr(window, "descendants") else []
            for p in panes:
                p_name = _normalize(getattr(p.element_info, "name", ""))
                if "application" in p_name or "easy apply" in p_name:
                    return p

            # 3. Fallback: if no isolated modal sub-dialog found, return the top window
            return window

        except Exception as exc:
            logger.debug("Modal discovery encountered warning: %s", exc)
            return window

    @staticmethod
    def get_modal_title_and_step(modal: Any) -> Tuple[str, str]:
        """Extract title and current step name from modal text."""
        if not modal:
            return "Application", "unknown"

        try:
            text = ""
            if hasattr(modal, "window_text"):
                text = modal.window_text() or ""

            norm = _normalize(text)
            if "resume" in norm or "cv" in norm:
                return text[:60], "resume"
            if "review" in norm:
                return text[:60], "review"
            if "question" in norm or "qualifications" in norm:
                return text[:60], "questions"

            return text[:60] if text else "Glassdoor Application", "form"
        except Exception:
            return "Glassdoor Application", "form"
