"""Question Normalizer and Domain Classifier for Form Fields (Phase 5.8).

Normalizes raw UI labels into canonical keys, classifies question intent
into domain categories (SALARY, NOTICE_PERIOD, RELOCATION, WORK_MODE, etc.),
and isolates specific skills/domains (e.g. teaching experience vs python experience).
"""

from enum import Enum
import re
from typing import Optional, Tuple
from app.automation.forms.models import QuestionInputType
from app.candidate_answers.models import QuestionCategory


def normalize_question_key(text: Optional[str]) -> str:
    """Convert raw question/label text into a clean canonical snake_case key.

    Supports semantic variations:
        "What is your current CTC?" -> "current_ctc_lpa"
        "Current salary package (in LPA)?" -> "current_ctc_lpa"
        "Expected CTC?" -> "expected_ctc_lpa"
        "Notice period (in days):" -> "notice_period_days"
        "Are you willing to relocate?" -> "willing_to_relocate"
        "Preferred location(s):" -> "preferred_locations"
        "Are you comfortable working hybrid?" -> "work_preference"
        "How many years of Teaching experience do you have?" -> "teaching_experience_years"
    """
    if not text:
        return "unknown_field"

    raw_lower = str(text).lower().strip()

    # 1. Platform survey questions
    if is_platform_survey_question(raw_lower):
        if "reason" in raw_lower or "why" in raw_lower:
            return "platform_survey_reason_for_applying"
        if "hear" in raw_lower or "source" in raw_lower:
            return "platform_survey_referral_source"
        return "platform_survey_question"

    # 2. Specific Experience questions (Skill isolation)
    if "teaching" in raw_lower and ("experience" in raw_lower or "year" in raw_lower):
        return "teaching_experience_years"

    skill, is_years = parse_skill_and_duration_from_text(text)
    if is_years and skill:
        clean_skill = re.sub(r"[^\w\s]", "", skill).strip()
        clean_skill = re.sub(r"\s+", "_", clean_skill).lower()
        if clean_skill in ("teaching", "teach"):
            return "teaching_experience_years"
        return f"years_experience_{clean_skill}"

    # 3. Salary & CTC variations
    if ("current" in raw_lower or "present" in raw_lower) and (
        "ctc" in raw_lower or "salary" in raw_lower or "compensation" in raw_lower or "package" in raw_lower or "pay" in raw_lower
    ):
        return "current_ctc_lpa"
    if ("expected" in raw_lower or "desired" in raw_lower or "demanded" in raw_lower or "expectation" in raw_lower) and (
        "ctc" in raw_lower or "salary" in raw_lower or "compensation" in raw_lower or "package" in raw_lower or "pay" in raw_lower
    ):
        return "expected_ctc_lpa"
    if "current ctc" in raw_lower:
        return "current_ctc_lpa"
    if "expected ctc" in raw_lower:
        return "expected_ctc_lpa"

    # 4. Notice Period variations
    if "notice" in raw_lower or "availability to join" in raw_lower or "how soon can you join" in raw_lower or "joining time" in raw_lower:
        return "notice_period"

    # 5. Relocation variations
    if "relocate" in raw_lower or "relocation" in raw_lower:
        return "willing_to_relocate"

    # 6. Work Preference & Mode (Hybrid, Remote, On-site)
    if "hybrid" in raw_lower or "work mode" in raw_lower or "work preference" in raw_lower or "work environment" in raw_lower:
        return "work_preference"
    if ("remote" in raw_lower or "onsite" in raw_lower or "on-site" in raw_lower) and ("preference" in raw_lower or "comfortable" in raw_lower or "willing" in raw_lower):
        return "work_preference"

    # 7. Location Preferences
    if ("preferred" in raw_lower or "preference" in raw_lower or "desired" in raw_lower or "choice" in raw_lower) and (
        "location" in raw_lower or "city" in raw_lower or "place" in raw_lower
    ):
        return "preferred_locations"

    # 8. Motivation & Summary
    if ("why should we hire you" in raw_lower or "why are you interested" in raw_lower or "tell us about yourself" in raw_lower or
        "cover letter" in raw_lower or "summary pitch" in raw_lower or "statement of purpose" in raw_lower or "universal motivation" in raw_lower):
        return "universal_motivation"

    # 9. Work Authorization & Sponsorship
    if "authorized" in raw_lower or "authorization" in raw_lower:
        return "work_authorization"
    if "sponsorship" in raw_lower or "sponsor" in raw_lower:
        return "visa_sponsorship"

    # 10. Education, Internship & Graduation
    if "university" in raw_lower or "college" in raw_lower or "institution" in raw_lower or "school" in raw_lower:
        if "attend" in raw_lower or "name" in raw_lower or "which" in raw_lower or "what" in raw_lower:
            return "education_institution"
    if "degree" in raw_lower or "qualification" in raw_lower or "highest level of education" in raw_lower:
        return "education_degree"
    if "major" in raw_lower or "field of study" in raw_lower or "branch" in raw_lower or "specialization" in raw_lower:
        return "education_field"
    if "internship" in raw_lower and ("experience" in raw_lower or "month" in raw_lower or "duration" in raw_lower or "year" in raw_lower):
        return "internship_experience_months"
    if "graduation year" in raw_lower or "year of passing" in raw_lower or "pass out year" in raw_lower or "passing year" in raw_lower:
        return "graduation_year"

    # General cleanup
    cleaned = re.sub(r"[\*\(\)\[\]:?\"',]", " ", raw_lower)
    cleaned = re.sub(r"\b(please|enter|select|provide|your|choose|what|is|the|do|you|have|with|in)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if "salary" in cleaned or "compensation" in cleaned:
        return "expected_ctc_lpa"

    key = re.sub(r"[^\w\s]", "", cleaned)
    key = re.sub(r"\s+", "_", key).strip("_")
    return key[:64] if key else "field"


def classify_question_category(text: Optional[str], key: Optional[str] = None) -> QuestionCategory:
    """Classify a question or normalized key into a domain QuestionCategory."""
    k = (key or normalize_question_key(text)).lower()
    t = str(text or "").lower()

    if is_platform_survey_question(t) or k.startswith("platform_survey"):
        return QuestionCategory.PLATFORM_SURVEY

    if k in ("current_ctc_lpa", "expected_ctc_lpa", "salary", "compensation") or "salary" in t or "ctc" in t:
        return QuestionCategory.SALARY

    if k in ("notice_period_days", "notice_period", "notice") or "notice" in t or "joining" in t:
        return QuestionCategory.NOTICE_PERIOD

    if k == "willing_to_relocate" or "relocate" in t or "relocation" in t:
        return QuestionCategory.RELOCATION

    if k == "work_preference" or "hybrid" in t or "remote" in t or "work mode" in t:
        return QuestionCategory.WORK_MODE

    if k == "preferred_locations" or ("location" in t and "current" not in t):
        return QuestionCategory.LOCATION

    if k in ("teaching_experience_years", "years_experience_teaching", "total_experience", "years_experience") or k.startswith("years_experience_") or "experience" in t:
        return QuestionCategory.EXPERIENCE

    if "education" in t or "degree" in t or "university" in t or "college" in t or "gpa" in t or k.startswith("education_"):
        return QuestionCategory.EDUCATION

    if k == "universal_motivation" or "motivation" in t or "why" in t or "tell us" in t:
        return QuestionCategory.MOTIVATION

    if "authorized" in t or "sponsorship" in t or "visa" in t or k in ("work_authorization", "visa_sponsorship"):
        return QuestionCategory.AUTHORIZATION

    if k in ("full_name", "email", "phone", "first_name", "last_name", "contact_number"):
        return QuestionCategory.PERSONAL

    return QuestionCategory.UNKNOWN


def is_platform_survey_question(text: Optional[str]) -> bool:
    """Determine whether a question is a platform-specific survey question (e.g. Indeed/SmartApply survey)."""
    if not text:
        return False
    lower = text.lower()
    survey_patterns = [
        "reason for applying",
        "how did you hear about",
        "where did you hear about",
        "what prompted you to apply",
        "why are you interested in this position on indeed",
        "how did you find this job",
        "referral source",
    ]
    return any(p in lower for p in survey_patterns)


def is_question_required(text: Optional[str], control_metadata: Optional[dict] = None) -> bool:
    """Determine whether a form question is marked as required."""
    if control_metadata and control_metadata.get("is_required"):
        return True
    if not text:
        return False
    t = text.strip()
    if t.startswith("*") or t.endswith("*") or " *" in t or "* " in t:
        return True
    lower = t.lower()
    if "(required)" in lower or "[required]" in lower or "mandatory" in lower or "required" in lower:
        if "(optional)" not in lower and "[optional]" not in lower:
            return True
    return False


def parse_skill_and_duration_from_text(text: str) -> Tuple[Optional[str], bool]:
    """Parse if a question asks for years of experience with a specific skill or domain.

    Examples:
        "How many years of Teaching experience do you have? *" -> ("Teaching", True)
        "How many years of experience do you have with Python?" -> ("Python", True)
        "Years of experience in Java:" -> ("Java", True)

    Returns:
        Tuple: (skill_name: Optional[str], is_years_question: bool)
    """
    lower = text.lower().strip()
    if "how many years" in lower or "years of experience" in lower or "years experience" in lower or "experience in years" in lower:
        # Pattern 1: "years of [skill] experience"
        m1 = re.search(r"(?:how\s+many\s+years\s+of|years\s+of)\s+([a-zA-Z0-9\+\#\.\s]+?)\s+experience", text, re.IGNORECASE)
        if m1:
            raw_skill = m1.group(1).strip("?:. *")
            cleaned = re.split(r"\b(do you have|have you|in years|total|overall)\b", raw_skill, flags=re.IGNORECASE)[0].strip()
            if cleaned and cleaned.lower() not in ("relevant", "total", "work", "professional", "overall"):
                return cleaned, True

        # Pattern 2: "experience with/in [skill]"
        m2 = re.search(
            r"(?:experience\s+with|experience\s+in|using|with)\s+([a-zA-Z0-9\+\#\.\s]+)",
            text,
            re.IGNORECASE,
        )
        if m2:
            raw_skill = m2.group(1).strip("?:. *")
            cleaned = re.split(r"\b(do you have|have you|in years)\b", raw_skill, flags=re.IGNORECASE)[0].strip()
            if cleaned:
                return cleaned, True

        # Pattern 3: "[skill] experience do you have" or "[skill] experience:"
        m3 = re.search(r"([a-zA-Z0-9\+\#\.\s]+?)\s+experience\s+(?:do\s+you\s+have|in\s+years)", text, re.IGNORECASE)
        if m3:
            raw_skill = m3.group(1).strip("?:. *")
            cleaned = re.split(r"\b(how many years of|years of|total|overall)\b", raw_skill, flags=re.IGNORECASE)[-1].strip()
            if cleaned and cleaned.lower() not in ("relevant", "total", "work", "professional", "overall"):
                return cleaned, True

        return None, True
    return None, False


def classify_input_type_from_control(
    control_type: Optional[str],
    options_count: int = 0,
    text_hint: Optional[str] = None,
) -> QuestionInputType:
    """Map UI Automation control type string and attributes to QuestionInputType."""
    if not control_type:
        return QuestionInputType.TEXT

    ctype = control_type.lower()
    if "combobox" in ctype or "dropdown" in ctype:
        return QuestionInputType.DROPDOWN
    if "radio" in ctype:
        return QuestionInputType.RADIO
    if "checkbox" in ctype:
        return QuestionInputType.CHECKBOX_MULTI if options_count > 1 else QuestionInputType.CHECKBOX
    if "edit" in ctype or "textbox" in ctype or "document" in ctype:
        if text_hint and ("how many" in text_hint.lower() or "number" in text_hint.lower() or "years" in text_hint.lower()):
            return QuestionInputType.NUMBER
        return QuestionInputType.TEXT

    return QuestionInputType.TEXT


class QuestionNormalizer:
    """Class wrapper for question normalization and category classification utilities."""

    @staticmethod
    def normalize(text: Optional[str]) -> str:
        return normalize_question_key(text)

    @staticmethod
    def normalize_question_key(text: Optional[str]) -> str:
        return normalize_question_key(text)

    @staticmethod
    def classify_question_category(text: Optional[str]) -> QuestionCategory:
        return classify_question_category(text)

    @staticmethod
    def is_platform_survey_question(text: Optional[str]) -> bool:
        return is_platform_survey_question(text)

    @staticmethod
    def classify_input_type(
        control_type: Optional[str],
        options_count: int = 0,
        text_hint: Optional[str] = None,
    ) -> QuestionInputType:
        return classify_input_type_from_control(control_type, options_count, text_hint)

