"""Number Input Control Handler.

Fills numeric fields (e.g. years of experience, salary) using PyWinAuto UIA controls.
"""

import logging
from typing import Any, Optional, Union
from app.automation.forms.handlers.text_handler import fill_text_field
from app.automation.forms.models import ApplicationQuestion

logger = logging.getLogger(__name__)


def fill_number_field(
    control: Any,
    value: Union[int, float, str],
    question: Optional[ApplicationQuestion] = None,
) -> bool:
    """Populate a numeric Edit UIA control with the string representation of the number."""
    if not control:
        return False

    val_str = str(value).strip()
    return fill_text_field(control, val_str, question)
