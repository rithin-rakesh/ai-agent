"""Radio Button Group Control Handler.

Selects an exact matching RadioButton among available options in a group.
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


def select_radio_option(
    container_or_controls: Union[Any, List[Any]],
    option_text: str,
    question: Optional[ApplicationQuestion] = None,
) -> bool:
    """Select the RadioButton element whose label matches option_text."""
    if not container_or_controls or not option_text:
        return False

    q_name = question.text if question else "Radio question"
    target_norm = _normalize(option_text)
    logger.info("Selecting radio option '%s' for '%s'", option_text, q_name)

    # Convert container or list into elements
    controls = container_or_controls if isinstance(container_or_controls, list) else []
    if not controls and hasattr(container_or_controls, "descendants"):
        try:
            controls = container_or_controls.descendants(control_type="RadioButton")
        except Exception:
            controls = []

    for ctrl in controls:
        name = getattr(ctrl.element_info, "name", "")
        if _normalize(name) == target_norm:
            try:
                if hasattr(ctrl, "click_input"):
                    ctrl.click_input()
                elif hasattr(ctrl, "select"):
                    ctrl.select()
                time.sleep(0.2)
                return True
            except Exception as exc:
                logger.warning("Failed clicking radio button '%s': %s", name, exc)

    return False
