"""Generic Form Inspector for UI Automation Containers.

Scans an active modal or form container, discovers input controls (Edit, ComboBox,
RadioButton, CheckBox), associates them with accessible labels / nearby text,
and generates structured ApplicationQuestion models.
"""

import logging
from typing import Any, Dict, List, Optional
from app.automation.forms.models import ApplicationQuestion, FormInspectionResult, QuestionInputType
from app.automation.forms.question_normalizer import (
    classify_input_type_from_control,
    is_question_required,
    normalize_question_key,
)

logger = logging.getLogger(__name__)


class FormInspector:
    """Inspects form elements within an active container."""

    @staticmethod
    def inspect_container(container_element: Any) -> FormInspectionResult:
        """Inspect visible UI controls inside container_element and construct ApplicationQuestion items."""
        if not container_element:
            return FormInspectionResult(
                questions=[],
                unresolved_required_count=0,
                detected_controls_count=0,
                is_valid=False,
                diagnostics={"error": "No container element provided"},
            )

        questions: List[ApplicationQuestion] = []
        raw_controls_count = 0

        try:
            # 1. Discover all descendants in the container
            descendants = container_element.descendants() if hasattr(container_element, "descendants") else []
            raw_controls_count = len(descendants)

            # Group controls by control type
            edits = []
            combos = []
            radios = []
            checkboxes = []
            text_elements = []

            for d in descendants:
                try:
                    ctype = getattr(d.element_info, "control_type", "")
                    if ctype == "Edit" or "edit" in ctype.lower():
                        edits.append(d)
                    elif ctype == "ComboBox" or "combobox" in ctype.lower():
                        combos.append(d)
                    elif ctype == "RadioButton" or "radio" in ctype.lower():
                        radios.append(d)
                    elif ctype == "CheckBox" or "checkbox" in ctype.lower():
                        checkboxes.append(d)
                    elif ctype == "Text" or "text" in ctype.lower():
                        text_elements.append(d)
                except Exception:
                    continue

            # 2. Process Edit / Text controls
            for edit in edits:
                label_text, is_req, is_ambig = FormInspector._resolve_control_label(edit, text_elements)
                if not label_text:
                    label_text = getattr(edit.element_info, "name", "") or "Text field"

                cur_val = ""
                try:
                    if hasattr(edit, "window_text"):
                        cur_val = edit.window_text() or ""
                except Exception:
                    pass

                q_key = normalize_question_key(label_text)
                input_t = classify_input_type_from_control("Edit", text_hint=label_text)

                questions.append(
                    ApplicationQuestion(
                        text=label_text,
                        normalized_key=q_key,
                        input_type=input_t,
                        required=is_req or is_question_required(label_text),
                        current_value=cur_val,
                        is_ambiguous=is_ambig,
                        field_name=getattr(edit.element_info, "name", None),
                        metadata={"control_type": "Edit"},
                    )
                )

            # 3. Process ComboBox / Dropdowns
            for combo in combos:
                label_text, is_req, is_ambig = FormInspector._resolve_control_label(combo, text_elements)
                if not label_text:
                    label_text = getattr(combo.element_info, "name", "") or "Dropdown selection"

                options = []
                try:
                    # Enumerate list items if available
                    items = combo.children(control_type="ListItem") if hasattr(combo, "children") else []
                    for it in items:
                        it_name = getattr(it.element_info, "name", "")
                        if it_name:
                            options.append(it_name)
                except Exception:
                    pass

                cur_val = getattr(combo.element_info, "name", "")
                q_key = normalize_question_key(label_text)

                questions.append(
                    ApplicationQuestion(
                        text=label_text,
                        normalized_key=q_key,
                        input_type=QuestionInputType.DROPDOWN,
                        required=is_req or is_question_required(label_text),
                        current_value=cur_val,
                        options=options,
                        is_ambiguous=is_ambig,
                        metadata={"control_type": "ComboBox"},
                    )
                )

            # 4. Process Radio Button Groups
            if radios:
                # Group radio buttons by parent or proximity
                radio_options = [getattr(r.element_info, "name", "") for r in radios if getattr(r.element_info, "name", "")]
                parent_label, is_req, is_ambig = FormInspector._resolve_group_label(radios[0], text_elements)
                if not parent_label:
                    parent_label = "Selection choice"

                questions.append(
                    ApplicationQuestion(
                        text=parent_label,
                        normalized_key=normalize_question_key(parent_label),
                        input_type=QuestionInputType.RADIO,
                        required=is_req or is_question_required(parent_label),
                        options=radio_options,
                        is_ambiguous=is_ambig,
                        metadata={"control_type": "RadioButtonGroup", "radio_count": len(radios)},
                    )
                )

            # 5. Process Checkboxes
            if checkboxes:
                cb_options = [getattr(c.element_info, "name", "") for c in checkboxes if getattr(c.element_info, "name", "")]
                if len(checkboxes) == 1:
                    cb = checkboxes[0]
                    cb_name = getattr(cb.element_info, "name", "") or "Checkbox"
                    questions.append(
                        ApplicationQuestion(
                            text=cb_name,
                            normalized_key=normalize_question_key(cb_name),
                            input_type=QuestionInputType.CHECKBOX,
                            required=is_question_required(cb_name),
                            options=[cb_name],
                            metadata={"control_type": "CheckBox"},
                        )
                    )
                else:
                    group_label, is_req, is_ambig = FormInspector._resolve_group_label(checkboxes[0], text_elements)
                    if not group_label:
                        group_label = "Select technologies"
                    questions.append(
                        ApplicationQuestion(
                            text=group_label,
                            normalized_key=normalize_question_key(group_label),
                            input_type=QuestionInputType.CHECKBOX_MULTI,
                            required=is_req or is_question_required(group_label),
                            options=cb_options,
                            is_ambiguous=is_ambig,
                            metadata={"control_type": "CheckBoxGroup", "checkbox_count": len(checkboxes)},
                        )
                    )

            unresolved_req = sum(1 for q in questions if q.required and not q.current_value)

            return FormInspectionResult(
                questions=questions,
                unresolved_required_count=unresolved_req,
                detected_controls_count=raw_controls_count,
                is_valid=True,
                diagnostics={"questions_count": len(questions), "raw_controls": raw_controls_count},
            )

        except Exception as exc:
            logger.warning("Form inspection encountered error: %s", exc)
            return FormInspectionResult(
                questions=[],
                unresolved_required_count=0,
                detected_controls_count=raw_controls_count,
                is_valid=False,
                diagnostics={"error": str(exc)},
            )

    @staticmethod
    def _resolve_control_label(control: Any, text_elements: List[Any]) -> tuple:
        """Resolve the accessible label or nearby preceding Text element for an input control."""
        # Check direct accessible name
        direct_name = getattr(control.element_info, "name", "")
        if direct_name and len(direct_name.strip()) > 1 and not direct_name.strip().isdigit():
            return direct_name.strip(), is_question_required(direct_name), False

        # Proximity search: find the Text element immediately preceding or above this control
        try:
            c_rect = getattr(control.element_info, "rectangle", None)
            if c_rect and text_elements:
                best_label = ""
                min_dist = 999999
                for t in text_elements:
                    t_rect = getattr(t.element_info, "rectangle", None)
                    t_text = getattr(t.element_info, "name", "")
                    if t_rect and t_text and t_rect.bottom <= c_rect.top:
                        dist = c_rect.top - t_rect.bottom
                        if 0 <= dist < min_dist:
                            min_dist = dist
                            best_label = t_text
                if best_label and min_dist < 100:
                    return best_label.strip(), is_question_required(best_label), False
        except Exception:
            pass

        return direct_name, is_question_required(direct_name), (not direct_name)

    @staticmethod
    def _resolve_group_label(first_control: Any, text_elements: List[Any]) -> tuple:
        """Resolve the heading / label for a group of radio buttons or checkboxes."""
        return FormInspector._resolve_control_label(first_control, text_elements)
