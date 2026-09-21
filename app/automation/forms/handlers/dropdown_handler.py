"""Dropdown / ComboBox Control Handler.

Selects exact matching option in a ComboBox or Dropdown UIA control.
"""

import logging
import re
import time
from typing import Any, List, Optional
from app.automation.forms.models import ApplicationQuestion

logger = logging.getLogger(__name__)


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def select_dropdown_option(
    control: Any,
    option_text: str,
    question: Optional[ApplicationQuestion] = None,
) -> bool:
    """Select a specific option within a ComboBox UIA control."""
    if not control or not option_text:
        return False

    q_name = question.text if question else "Dropdown"
    target_norm = _normalize(option_text)
    logger.info("Selecting dropdown option '%s' for '%s'", option_text, q_name)

    try:
        # Strategy 1: Expand and search child ListItems
        if hasattr(control, "expand"):
            try:
                control.expand()
                time.sleep(0.3)
            except Exception:
                pass

        # Search children for matching ListItem
        children = control.children() if hasattr(control, "children") else []
        for child in children:
            child_name = getattr(child.element_info, "name", "")
            if _normalize(child_name) == target_norm:
                if hasattr(child, "click_input"):
                    child.click_input()
                elif hasattr(child, "select"):
                    child.select()
                time.sleep(0.2)
                return True

        # Strategy 2: SelectPattern
        if hasattr(control, "select"):
            try:
                control.select(option_text)
                time.sleep(0.2)
                return True
            except Exception:
                pass

        # Strategy 3: Type keys into ComboBox
        if hasattr(control, "type_keys"):
            control.type_keys(option_text + "{ENTER}", with_spaces=True)
            time.sleep(0.2)
            return True

        return False
    except Exception as exc:
        logger.warning("Failed to select dropdown option '%s' for '%s': %s", option_text, q_name, exc)
        return False
