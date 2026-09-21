"""Handlers package for UI input controls."""

from app.automation.forms.handlers.checkbox_handler import set_checkboxes
from app.automation.forms.handlers.dropdown_handler import select_dropdown_option
from app.automation.forms.handlers.number_handler import fill_number_field
from app.automation.forms.handlers.radio_handler import select_radio_option
from app.automation.forms.handlers.text_handler import fill_text_field

__all__ = [
    "fill_text_field",
    "fill_number_field",
    "select_dropdown_option",
    "select_radio_option",
    "set_checkboxes",
]
