"""Checkbox Control Handler.

Toggles / selects single or multi-select CheckBox UIA controls.
"""

import logging
import re
import time
from typing import Any, List, Optional, Union
from app.automation.forms.models import ApplicationQuestion

logger = logging.getLogger(__name__)


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def set_checkboxes(
    container_or_controls: Union[Any, List[Any]],
    selected_options: Union[str, List[str]],
    question: Optional[ApplicationQuestion] = None,
) -> bool:
    """Check the CheckBox elements matching the selected options."""
    if not container_or_controls:
        return False

    targets = [selected_options] if isinstance(selected_options, str) else selected_options
    target_norms = {_normalize(t) for t in targets}
    q_name = question.text if question else "Checkbox question"
    logger.info("Setting %d checkboxes for '%s'", len(targets), q_name)

    controls = container_or_controls if isinstance(container_or_controls, list) else []
    if not controls and hasattr(container_or_controls, "descendants"):
        try:
            controls = container_or_controls.descendants(control_type="CheckBox")
        except Exception:
            controls = []

    success_count = 0
    for ctrl in controls:
        name = getattr(ctrl.element_info, "name", "")
        if _normalize(name) in target_norms:
            try:
                # Check if toggle state is already on
                is_checked = False
                if hasattr(ctrl, "get_toggle_state"):
                    is_checked = (ctrl.get_toggle_state() == 1)

                if not is_checked:
                    if hasattr(ctrl, "click_input"):
                        ctrl.click_input()
                    elif hasattr(ctrl, "toggle"):
                        ctrl.toggle()
                    time.sleep(0.1)
                success_count += 1
            except Exception as exc:
                logger.warning("Failed to toggle checkbox '%s': %s", name, exc)

    return success_count > 0 or len(targets) == 0
