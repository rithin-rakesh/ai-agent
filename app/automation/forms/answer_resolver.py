"""Truthful Answer Resolver for Job Application Questions (Phase 5.8).

Resolves form questions using strict 6-tier priority:
1. Explicit application-specific approved answer (application_specific)
2. Candidate Answer Bank (candidate_approved)
3. Candidate profile explicit values (candidate_profile)
4. Resume-derived facts in candidate profile (resume_facts / skill)
5. Previously user-approved application_answers memory (stored_answer)
6. Free-text adaptation using only approved facts (candidate_approved)

Enforces:
- never_infer_unverified_experience = True
- Experience question safety & skill isolation (teaching vs python vs ml)
- Semantic option mapping for dropdowns, radios, and checkboxes
- Unknown question policy (optional skipped, required governed by allow_unanswered_questions)
- Platform survey memory isolation
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union

from app.automation.forms.models import AnswerResolution, ApplicationQuestion, QuestionInputType
from app.automation.forms.question_normalizer import (
    classify_question_category,
    is_platform_survey_question,
    normalize_question_key,
    parse_skill_and_duration_from_text,
)
from app.candidate_answers.bank import CandidateAnswerBank, UNIVERSAL_MOTIVATION
from app.candidate_answers.models import QuestionCategory
from app.config.settings import get_settings
from app.models.profile import CandidateProfileData, Profile

logger = logging.getLogger(__name__)

try:
    ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS = getattr(get_settings(), "ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS", False)
except Exception:
    ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS = False


def _normalize_string(val: Optional[Any]) -> str:
    if val is None:
        return ""
    return re.sub(r"\s+", " ", str(val).lower().strip())


_UNSET = object()


class FormAnswerResolver:
    """Resolves application form questions against Candidate Answer Bank and candidate profile facts."""

    def __init__(
        self,
        answer_bank: Any = _UNSET,
        stored_answers: Optional[Dict[str, str]] = None,
        profile: Optional[Union[Profile, CandidateProfileData, dict]] = None,
        profile_service: Optional[Any] = None,
        memory: Optional[Any] = None,
        platform: str = "glassdoor",
        host: str = "glassdoor",
        allow_unanswered_questions: bool = False,
        never_infer_unverified_experience: bool = True,
        candidate_answer_bank: Any = _UNSET,
    ) -> None:
        if candidate_answer_bank is not _UNSET:
            self.answer_bank = candidate_answer_bank
        elif answer_bank is not _UNSET:
            self.answer_bank = answer_bank
        else:
            self.answer_bank = CandidateAnswerBank()
        self.stored_answers = stored_answers or {}
        if memory is not None:
            if hasattr(memory, "answers") and isinstance(memory.answers, dict):
                self.stored_answers.update(memory.answers)
            elif isinstance(memory, dict):
                self.stored_answers.update(memory)
        self.profile = profile
        self.profile_service = profile_service
        if self.profile is None and self.profile_service is not None:
            if hasattr(self.profile_service, "get_active_profile_data"):
                self.profile = self.profile_service.get_active_profile_data()
            if self.profile is None and hasattr(self.profile_service, "get_profile"):
                self.profile = self.profile_service.get_profile()
        self.platform = platform
        self.host = host
        self.allow_unanswered_questions = allow_unanswered_questions
        self.never_infer_unverified_experience = never_infer_unverified_experience

    def resolve(
        self,
        question: ApplicationQuestion,
        platform: Optional[str] = None,
        host: Optional[str] = None,
    ) -> AnswerResolution:
        """Alias for resolve_question supporting platform/host overrides."""
        if platform:
            self.platform = platform
        if host:
            self.host = host
        return self.resolve_question(question)

    def resolve_question(self, question: ApplicationQuestion) -> AnswerResolution:
        """Resolve an answer for an ApplicationQuestion following the strict 6-tier hierarchy."""
        if question.is_ambiguous:
            return AnswerResolution(
                question=question,
                is_resolved=False,
                action="needs_user_input",
                is_optional=not question.required,
                normalized_key=question.normalized_key or "unknown_ambiguous",
                reason="Question association was ambiguous in UI form container.",
            )

        q_text = question.text.strip()
        q_norm = _normalize_string(q_text)
        q_key = question.normalized_key or normalize_question_key(q_text)
        category = classify_question_category(q_text, q_key)

        # ----------------------------------------------------------------------
        # Priority 1: Explicit application-specific or passed stored answer
        # ----------------------------------------------------------------------
        app_specific_key = f"app_specific:{q_key}"
        if app_specific_key in self.stored_answers:
            val = self.stored_answers[app_specific_key]
            return self._match_and_format_answer(question, val, source="application_specific", key=q_key)
        if q_key in self.stored_answers and not q_key.startswith("platform_survey"):
            val = self.stored_answers[q_key]
            return self._match_and_format_answer(question, val, source="stored_answer", key=q_key)

        # ----------------------------------------------------------------------
        # Priority 2: Candidate Answer Bank
        # ----------------------------------------------------------------------
        bank_ans = self.answer_bank.get_answer(q_key, platform=self.platform) if self.answer_bank is not None else None
        if bank_ans and bank_ans.approved:
            is_teaching_ans = bank_ans.normalized_key in ("teaching_experience_years", "years_experience_teaching")
            is_teaching_q = q_key in ("teaching_experience_years", "years_experience_teaching")

            # Invariant: Never infer approved 1-year teaching experience for non-teaching keys
            if is_teaching_ans and not is_teaching_q:
                pass
            else:
                return self._match_and_format_answer(
                    question, bank_ans.value, source=bank_ans.source or "candidate_approved", key=q_key
                )

        # ----------------------------------------------------------------------
        # Priority 3: Candidate profile explicit values
        # ----------------------------------------------------------------------
        profile_match = self._resolve_from_profile(question, q_key, q_norm)
        if profile_match is not None:
            return profile_match

        # ----------------------------------------------------------------------
        # Priority 4: Resume-derived facts in candidate profile
        # ----------------------------------------------------------------------
        resume_match = self._resolve_resume_facts(question, q_key, q_text, q_norm)
        if resume_match is not None:
            return resume_match

        # ----------------------------------------------------------------------
        # Priority 5: Existing application_answers memory
        # ----------------------------------------------------------------------
        # Platform survey isolation: survey questions must not match employer questions
        is_survey = category == QuestionCategory.PLATFORM_SURVEY or is_platform_survey_question(q_text)
        if not is_survey:
            lookup_keys = [q_norm, q_key]
            if q_key.endswith("_experience_years"):
                base = q_key[:-len("_experience_years")]
                lookup_keys.extend([f"years_experience_{base}", f"{base}_experience", base])
            elif q_key.startswith("years_experience_"):
                base = q_key[len("years_experience_"):]
                lookup_keys.extend([f"{base}_experience_years", f"{base}_experience", base])

            for k in lookup_keys:
                if k in self.stored_answers and not k.startswith("platform_survey"):
                    val = self.stored_answers[k]
                    return self._match_and_format_answer(question, val, source="stored_answer", key=q_key)

        # ----------------------------------------------------------------------
        # Priority 6: Free-text adaptation using only approved facts
        # ----------------------------------------------------------------------
        if category in (QuestionCategory.MOTIVATION, QuestionCategory.FREE_TEXT) or q_key == "universal_motivation":
            return self._match_and_format_answer(
                question, UNIVERSAL_MOTIVATION, source="candidate_approved", key="universal_motivation"
            )

        # ----------------------------------------------------------------------
        # Unknown Question Policy Handling (Sections 14 & 16)
        # ----------------------------------------------------------------------
        if not question.required:
            logger.info("Unknown optional question '%s' (%s) skipped per policy.", q_text, q_key)
            return AnswerResolution(
                question=question,
                is_resolved=False,
                action="skip",
                is_optional=True,
                normalized_key=q_key,
                reason=f"Unknown optional question '{q_text}' skipped per policy.",
            )

        # Required unknown question
        allow_unanswered = self.allow_unanswered_questions or ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS
        if allow_unanswered:
            logger.info(
                "Required unknown question '%s' (%s) left unanswered per ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS.",
                q_text,
                q_key,
            )
            return AnswerResolution(
                question=question,
                is_resolved=False,
                action="leave_unanswered",
                is_optional=False,
                normalized_key=q_key,
                reason=f"Required unknown question '{q_text}' left unanswered per policy for platform validation.",
            )

        logger.warning("Required question '%s' (%s) has no approved answer. Triggering NEEDS_USER_INPUT.", q_text, q_key)
        return AnswerResolution(
            question=question,
            is_resolved=False,
            action="needs_user_input",
            is_optional=False,
            normalized_key=q_key,
            reason=f"No approved answer or explicit profile fact available for required question '{q_text}'.",
        )

    def _resolve_from_profile(
        self,
        question: ApplicationQuestion,
        key: str,
        norm_text: str,
    ) -> Optional[AnswerResolution]:
        """Extract explicit fields from CandidateProfileData or Profile model."""
        if not self.profile:
            return None

        p_name = getattr(self.profile, "name", None) or (self.profile.get("name") if isinstance(self.profile, dict) else None)
        p_email = getattr(self.profile, "email", None) or (self.profile.get("email") if isinstance(self.profile, dict) else None)
        p_phone = getattr(self.profile, "phone", None) or (self.profile.get("phone") if isinstance(self.profile, dict) else None)
        p_loc = getattr(self.profile, "location", None) or (self.profile.get("location") if isinstance(self.profile, dict) else None)
        p_exp = getattr(self.profile, "experience_years", None) or (self.profile.get("experience_years") if isinstance(self.profile, dict) else None)

        if isinstance(self.profile, CandidateProfileData):
            p_name = self.profile.personal.name or p_name
            p_email = self.profile.personal.email or p_email
            p_phone = self.profile.personal.phone or p_phone
            p_loc = self.profile.personal.location or p_loc
            p_exp = self.profile.career.experience_years or p_exp

        # Contact Details
        if key in ("full_name", "name", "first_and_last_name", "candidate_name") or "full name" in norm_text:
            if p_name:
                return self._match_and_format_answer(question, p_name, source="candidate_profile", key=key)
        if key in ("email", "email_address") or "email address" in norm_text:
            if p_email:
                return self._match_and_format_answer(question, p_email, source="candidate_profile", key=key)
        if key in ("phone", "phone_number", "mobile", "contact_number") or "phone number" in norm_text:
            if p_phone:
                return self._match_and_format_answer(question, p_phone, source="candidate_profile", key=key)
        if key in ("location", "city", "current_location", "address") or "current location" in norm_text:
            if p_loc:
                return self._match_and_format_answer(question, p_loc, source="candidate_profile", key=key)

        # Total Experience Years (explicit profile fact)
        if key in ("total_experience", "years_experience", "total_years_of_experience") or (
            "total" in norm_text and "experience" in norm_text and "years" in norm_text
        ):
            if p_exp is not None:
                exp_val = int(p_exp) if float(p_exp).is_integer() else float(p_exp)
                return self._match_and_format_answer(question, str(exp_val), source="candidate_profile", key=key)

        return None

    def _resolve_resume_facts(
        self,
        question: ApplicationQuestion,
        key: str,
        text: str,
        norm_text: str,
    ) -> Optional[AnswerResolution]:
        """Resolve specific skill durations, education, or internships from verified resume facts."""
        if not self.profile:
            return None

        # 1. Education Facts
        edu_list = []
        if isinstance(self.profile, CandidateProfileData):
            edu_list = self.profile.education or []
        elif isinstance(self.profile, dict):
            edu_list = self.profile.get("education") or []

        if edu_list:
            top_edu = edu_list[0]
            deg = getattr(top_edu, "degree", None) or (top_edu.get("degree") if isinstance(top_edu, dict) else None)
            field = getattr(top_edu, "field", None) or (top_edu.get("field") if isinstance(top_edu, dict) else None)
            inst = getattr(top_edu, "institution", None) or (top_edu.get("institution") if isinstance(top_edu, dict) else None)

            if key in ("education_degree", "degree", "highest_degree") or "highest level of education" in norm_text or "degree" in norm_text:
                if deg:
                    return self._match_and_format_answer(question, deg, source="resume_facts", key=key)
            if key in ("education_field", "major", "field_of_study") or "major" in norm_text or "field of study" in norm_text:
                if field:
                    return self._match_and_format_answer(question, field, source="resume_facts", key=key)
            if key in ("education_institution", "university", "college", "school") or "university" in norm_text or "institution" in norm_text:
                if inst:
                    return self._match_and_format_answer(question, inst, source="resume_facts", key=key)
        else:
            c_deg = getattr(self.profile, "education_degree", None)
            c_field = getattr(self.profile, "education_field", None)
            if isinstance(self.profile, CandidateProfileData) and self.profile.career:
                c_deg = self.profile.career.education_degree or c_deg
                c_field = self.profile.career.education_field or c_field
            if c_deg and (key in ("education_degree", "degree", "highest_degree") or "highest level of education" in norm_text or "degree" in norm_text):
                return self._match_and_format_answer(question, c_deg, source="resume_facts", key=key)
            if c_field and (key in ("education_field", "major", "field_of_study") or "major" in norm_text or "field of study" in norm_text):
                return self._match_and_format_answer(question, c_field, source="resume_facts", key=key)

        # 2. Skill-Specific Numeric Duration Facts (Skill Isolation Guaranteed)
        skill_name, is_years = parse_skill_and_duration_from_text(text)
        if is_years and skill_name:
            norm_target = _normalize_string(skill_name)
            # Never infer 1 year for unverified skills
            skills_list = []
            if isinstance(self.profile, (Profile, CandidateProfileData)):
                skills_list = self.profile.skills or []
            elif isinstance(self.profile, dict):
                skills_list = self.profile.get("skills") or []

            for s in skills_list:
                s_name = getattr(s, "skill", None) or getattr(s, "name", None) or (s.get("skill") or s.get("name") if isinstance(s, dict) else "")
                s_years = getattr(s, "years_experience", None) or (s.get("years_experience") if isinstance(s, dict) else None)
                if _normalize_string(s_name) == norm_target:
                    if s_years is not None and s_years > 0:
                        val = int(s_years) if float(s_years).is_integer() else float(s_years)
                        return self._match_and_format_answer(question, str(val), source="skill", key=key)
                    else:
                        return AnswerResolution(
                            question=question,
                            is_resolved=False,
                            action="needs_user_input" if question.required else "skip",
                            is_optional=not question.required,
                            normalized_key=key,
                            reason=f"Skill '{s_name}' found in profile but explicit years_experience is not specified.",
                        )

            # Skill not found in verified profile -> do NOT invent duration
            return AnswerResolution(
                question=question,
                is_resolved=False,
                action="needs_user_input" if question.required else "skip",
                is_optional=not question.required,
                normalized_key=key,
                reason=f"No approved answer found: Candidate resume facts have no verified experience for skill '{skill_name}'.",
            )

        # 3. Multi-Select Checkbox Skills
        if question.input_type in (QuestionInputType.CHECKBOX, QuestionInputType.CHECKBOX_MULTI) and question.options:
            skills_list = []
            if isinstance(self.profile, (Profile, CandidateProfileData)):
                skills_list = self.profile.skills or []
            elif isinstance(self.profile, dict):
                skills_list = self.profile.get("skills") or []

            candidate_skill_names = {
                _normalize_string(getattr(s, "skill", None) or (s.get("skill") if isinstance(s, dict) else ""))
                for s in skills_list
            }
            matched_options = []
            for opt in question.options:
                if _normalize_string(opt) in candidate_skill_names:
                    matched_options.append(opt)

            if matched_options:
                return AnswerResolution(
                    question=question,
                    is_resolved=True,
                    resolved_value=matched_options,
                    source="resume_facts",
                    action="fill",
                    is_optional=not question.required,
                    normalized_key=key,
                    reason=f"Selected {len(matched_options)} explicitly supported skills.",
                )

        return None

    def _match_and_format_answer(
        self,
        question: ApplicationQuestion,
        raw_val: Any,
        source: str,
        key: Optional[str] = None,
    ) -> AnswerResolution:
        """Format and validate resolved value against question input type (dropdown options, radio, etc.)."""
        # Dropdown validation: semantic matching
        if question.input_type == QuestionInputType.DROPDOWN:
            if not question.options:
                return AnswerResolution(
                    question=question,
                    is_resolved=True,
                    resolved_value=str(raw_val),
                    source=source,
                    action="fill",
                    is_optional=not question.required,
                    normalized_key=key,
                )

            matched_opt = self._find_best_option_match(raw_val, question.options)
            if matched_opt:
                return AnswerResolution(
                    question=question,
                    is_resolved=True,
                    resolved_value=matched_opt,
                    source=source,
                    action="fill",
                    is_optional=not question.required,
                    normalized_key=key,
                )

            return AnswerResolution(
                question=question,
                is_resolved=False,
                action="needs_user_input" if question.required else "skip",
                is_optional=not question.required,
                normalized_key=key,
                reason=f"Answer '{raw_val}' does not match available dropdown options: {question.options}",
            )

        # Radio button validation: semantic matching
        if question.input_type == QuestionInputType.RADIO:
            if not question.options:
                return AnswerResolution(
                    question=question,
                    is_resolved=True,
                    resolved_value=str(raw_val),
                    source=source,
                    action="fill",
                    is_optional=not question.required,
                    normalized_key=key,
                )

            matched_opt = self._find_best_option_match(raw_val, question.options)
            if matched_opt:
                return AnswerResolution(
                    question=question,
                    is_resolved=True,
                    resolved_value=matched_opt,
                    source=source,
                    action="fill",
                    is_optional=not question.required,
                    normalized_key=key,
                )

            return AnswerResolution(
                question=question,
                is_resolved=False,
                action="needs_user_input" if question.required else "skip",
                is_optional=not question.required,
                normalized_key=key,
                reason=f"Radio value '{raw_val}' does not match available choices: {question.options}",
            )

        # Number validation
        if question.input_type == QuestionInputType.NUMBER:
            try:
                # Handle range strings like "4-5"
                raw_str = str(raw_val).strip()
                if "-" in raw_str and not raw_str.startswith("-"):
                    parts = raw_str.split("-")
                    num = float(parts[0].strip())
                else:
                    num = float(re.sub(r"[^\d\.]", "", raw_str) or 0)
                val_str = str(int(num)) if num.is_integer() else str(num)
                return AnswerResolution(
                    question=question,
                    is_resolved=True,
                    resolved_value=val_str,
                    source=source,
                    action="fill",
                    is_optional=not question.required,
                    normalized_key=key,
                )
            except (ValueError, TypeError):
                return AnswerResolution(
                    question=question,
                    is_resolved=False,
                    action="needs_user_input" if question.required else "skip",
                    is_optional=not question.required,
                    normalized_key=key,
                    reason=f"Value '{raw_val}' cannot be converted to a number for question '{question.text}'.",
                )

        # Checkbox validation
        if question.input_type == QuestionInputType.CHECKBOX:
            bool_val = bool(raw_val) if not isinstance(raw_val, str) else raw_val.lower() in ("true", "yes", "1")
            return AnswerResolution(
                question=question,
                is_resolved=True,
                resolved_value=bool_val,
                source=source,
                action="fill",
                is_optional=not question.required,
                normalized_key=key,
            )

        # Multi-Checkbox validation
        if question.input_type == QuestionInputType.CHECKBOX_MULTI:
            if isinstance(raw_val, list):
                selected = []
                for opt in question.options:
                    if any(_normalize_string(item) in _normalize_string(opt) for item in raw_val):
                        selected.append(opt)
                return AnswerResolution(
                    question=question,
                    is_resolved=bool(selected),
                    resolved_value=selected,
                    source=source,
                    action="fill" if selected else ("needs_user_input" if question.required else "skip"),
                    is_optional=not question.required,
                    normalized_key=key,
                    reason=f"Selected {len(selected)} matching checkbox options.",
                )

        # Text & Textarea types
        return AnswerResolution(
            question=question,
            is_resolved=True,
            resolved_value=str(raw_val),
            source=source,
            action="fill",
            is_optional=not question.required,
            normalized_key=key,
        )

    def _find_best_option_match(self, raw_val: Any, options: List[str]) -> Optional[str]:
        """Find matching option from available list using semantic mapping."""
        val_str = _normalize_string(raw_val)

        # 1. Exact normalized match
        for opt in options:
            if _normalize_string(opt) == val_str:
                return opt

        # 2. Boolean Yes / No mapping
        if val_str in ("true", "1", "yes"):
            for opt in options:
                norm_opt = _normalize_string(opt)
                if norm_opt in ("yes", "y", "true", "i agree", "agree") or norm_opt.startswith("yes") or "willing" in norm_opt:
                    return opt
        elif val_str in ("false", "0", "no"):
            for opt in options:
                norm_opt = _normalize_string(opt)
                if norm_opt in ("no", "n", "false", "disagree") or norm_opt.startswith("no") or "not willing" in norm_opt:
                    return opt

        # 3. Work preference mapping (Hybrid)
        if "hybrid" in val_str:
            for opt in options:
                norm_opt = _normalize_string(opt)
                if "hybrid" in norm_opt:
                    return opt

        # 4. Notice period mapping (e.g. 15 days)
        if "15" in val_str:
            for opt in options:
                norm_opt = _normalize_string(opt)
                if "15" in norm_opt:
                    return opt

        # 5. Substring match
        for opt in options:
            norm_opt = _normalize_string(opt)
            if val_str and val_str in norm_opt:
                return opt

        return None
