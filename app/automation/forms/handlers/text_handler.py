"""Text Input Control Handler.

Fills single-line and multi-line text input fields using PyWinAuto UIA controls.
"""

import logging
import time
from typing import Any, Optional
from app.automation.forms.models import ApplicationQuestion

logger = logging.getLogger(__name__)


def fill_text_field(
    control: Any,
    value: str,
    question: Optional[ApplicationQuestion] = None,
) -> bool:
    """Populate an Edit or Textbox UIA control with the specified string value."""
    if not control:
        return False

    q_name = question.text if question else "Text field"
    logger.info("Filling text field '%s' with value length=%d", q_name, len(value))

    try:
        # Strategy 1: UIA ValuePattern or set_edit_text
        if hasattr(control, "set_edit_text"):
            control.set_edit_text(value)
            time.sleep(0.2)
            return True

        if hasattr(control, "iface_value") and control.iface_value:
            control.iface_value.SetValue(value)
            time.sleep(0.2)
            return True

        # Strategy 2: Focus and type_keys
        if hasattr(control, "set_focus"):
            control.set_focus()
            time.sleep(0.1)
        if hasattr(control, "type_keys"):
            control.type_keys("^a{BACKSPACE}", with_spaces=True)
            control.type_keys(value, with_spaces=True)
            time.sleep(0.2)
            return True

        return False
    except Exception as exc:
        logger.warning("Failed to fill text field '%s': %s", q_name, exc)
        return False
