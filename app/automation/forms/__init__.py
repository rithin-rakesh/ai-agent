"""Reusable Form Infrastructure Package.

Provides generic question normalization, answer resolution, form inspection,
and input control handlers for UI application automations.
"""

from app.automation.forms.answer_resolver import FormAnswerResolver
from app.automation.forms.form_inspector import FormInspector
from app.automation.forms.models import (
    AnswerResolution,
    ApplicationQuestion,
    FormInspectionResult,
    QuestionInputType,
)
from app.automation.forms.question_normalizer import (
    QuestionNormalizer,
    classify_input_type_from_control,
    classify_question_category,
    is_platform_survey_question,
    is_question_required,
    normalize_question_key,
    parse_skill_and_duration_from_text,
)

__all__ = [
    "ApplicationQuestion",
    "QuestionInputType",
    "AnswerResolution",
    "FormInspectionResult",
    "FormAnswerResolver",
    "FormInspector",
    "QuestionNormalizer",
    "normalize_question_key",
    "classify_question_category",
    "is_platform_survey_question",
    "is_question_required",
    "parse_skill_and_duration_from_text",
    "classify_input_type_from_control",
]
