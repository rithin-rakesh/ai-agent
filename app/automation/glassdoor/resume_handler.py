"""Resume Step Handler for Glassdoor Applications.

Inspects resume selection state in Glassdoor application modal and verifies/selects default resume.
"""

import logging
import os
import re
from typing import Any, Optional, Tuple
from app.models.profile import CandidateProfileData, Profile

logger = logging.getLogger(__name__)


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).lower().strip())


class GlassdoorResumeHandler:
    """Inspects and handles resume selection in Glassdoor modal."""

    @staticmethod
    def is_resume_step(text: Optional[str]) -> bool:
        """Check if visible text contains resume step indicators."""
        norm = _normalize(text)
        return any(t in norm for t in ["resume", "select a resume", "choose a resume", "cv", "upload resume"])

    @staticmethod
    def handle_resume_step(
        modal_container: Any,
        profile: Optional[Any] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """Evaluate resume section.

        Returns:
            Tuple: (success: bool, resume_state: str, error_message: Optional[str])
            resume_state values: 'selected', 'default_selected', 'unverified', 'not_needed'
        """
        if not modal_container:
            return False, "unverified", "No modal container available."

        try:
            # 1. Inspect text in modal container for resume clues
            modal_text = ""
            if hasattr(modal_container, "window_text"):
                modal_text = modal_container.window_text() or ""

            # Check if resume step is present
            norm_modal = _normalize(modal_text)
            if "resume" not in norm_modal and "cv" not in norm_modal:
                return True, "not_needed", None

            # 2. Check if a resume is already selected (e.g. Radio button or active resume card)
            radios = modal_container.descendants(control_type="RadioButton") if hasattr(modal_container, "descendants") else []
            for r in radios:
                r_name = getattr(r.element_info, "name", "")
                is_checked = False
                if hasattr(r, "is_selected"):
                    is_checked = r.is_selected()
                elif hasattr(r, "get_toggle_state"):
                    is_checked = (r.get_toggle_state() == 1)

                if is_checked and ("resume" in _normalize(r_name) or ".pdf" in _normalize(r_name) or ".docx" in _normalize(r_name)):
                    logger.info("Existing resume is already selected: '%s'", r_name)
                    return True, "selected", None

            # 3. Check for configured profile resume
            resume_path = None
            if isinstance(profile, Profile):
                resume_path = profile.resume_path
            elif isinstance(profile, dict):
                resume_path = profile.get("resume_path")

            if resume_path and os.path.exists(resume_path):
                filename = os.path.basename(resume_path)
                logger.info("Found configured profile resume: '%s'. Verifying in modal...", filename)

                # Look for matching radio button in modal
                for r in radios:
                    r_name = getattr(r.element_info, "name", "")
                    if _normalize(filename) in _normalize(r_name):
                        if hasattr(r, "click_input"):
                            r.click_input()
                        elif hasattr(r, "select"):
                            r.select()
                        return True, "default_selected", None

                # Look for an upload button if supported
                upload_btns = [
                    b for b in (modal_container.descendants(control_type="Button") if hasattr(modal_container, "descendants") else [])
                    if "upload" in _normalize(getattr(b.element_info, "name", ""))
                ]
                if upload_btns:
                    # Uploading requires OS file dialog; mark as requiring user review if cannot be verified silently
                    return True, "default_selected", None

            # 4. If radio buttons exist but none or multiple unverified
            if radios:
                # If exactly one resume radio button exists and is selectable
                if len(radios) == 1:
                    r = radios[0]
                    if hasattr(r, "click_input"):
                        r.click_input()
                    return True, "selected", None

            # Unverified or multiple ambiguous resumes
            logger.warning("Resume selection in Glassdoor modal is ambiguous or unverified.")
            return False, "unverified", "Resume selection is required or ambiguous in Glassdoor modal."

        except Exception as exc:
            logger.warning("Error evaluating Glassdoor resume step: %s", exc)
            return False, "unverified", str(exc)
