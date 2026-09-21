"""Candidate Answer Bank Repository & Cache (Phase 5.8).

Maintains persistent candidate-approved answers, handles alias normalization,
enforces experience isolation, and guarantees never_infer_unverified_experience = True.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.candidate_answers.models import AnswerType, CandidateAnswer, QuestionCategory

logger = logging.getLogger(__name__)

DEFAULT_ANSWERS_FILE = "data/candidate_answers.json"

# Default candidate-approved universal motivation template
UNIVERSAL_MOTIVATION = (
    "I am an aspiring AI/ML engineer with hands-on experience in Python, machine learning, deep learning, "
    "data analysis, Generative AI, and Agentic AI. I have worked on AI-driven applications and intelligent "
    "automation projects and enjoy solving real-world problems using data and modern AI technologies. "
    "I am looking for an opportunity where I can apply my technical skills, continue learning, and contribute "
    "to a team building practical AI solutions."
)

DEFAULT_APPROVED_ANSWERS: List[Dict[str, Any]] = [
    {
        "normalized_key": "current_ctc_lpa",
        "category": QuestionCategory.SALARY.value,
        "answer_type": AnswerType.NUMBER_OR_TEXT.value,
        "value": "0",
        "source": "candidate_approved",
        "approved": True,
        "scope": "global",
        "platforms": ["indeed", "glassdoor"],
        "confidence": 1.0,
        "requires_user_approval": False,
        "description": "Current CTC (0 LPA)",
    },
    {
        "normalized_key": "expected_ctc_lpa",
        "category": QuestionCategory.SALARY.value,
        "answer_type": AnswerType.NUMBER_OR_TEXT.value,
        "value": "4-5",
        "source": "candidate_approved",
        "approved": True,
        "scope": "global",
        "platforms": ["indeed", "glassdoor"],
        "confidence": 1.0,
        "requires_user_approval": False,
        "description": "Expected CTC (4-5 LPA)",
    },
    {
        "normalized_key": "notice_period_days",
        "category": QuestionCategory.NOTICE_PERIOD.value,
        "answer_type": AnswerType.NUMBER_OR_TEXT.value,
        "value": "15",
        "source": "candidate_approved",
        "approved": True,
        "scope": "global",
        "platforms": ["indeed", "glassdoor"],
        "confidence": 1.0,
        "requires_user_approval": False,
        "description": "Notice period in days (15 days)",
    },
    {
        "normalized_key": "preferred_locations",
        "category": QuestionCategory.LOCATION.value,
        "answer_type": AnswerType.CHECKBOX_MULTI.value,
        "value": ["Chennai", "Bangalore", "Kochi", "Kozhikode", "Trivandrum"],
        "source": "candidate_approved",
        "approved": True,
        "scope": "global",
        "platforms": ["indeed", "glassdoor"],
        "confidence": 1.0,
        "requires_user_approval": False,
        "description": "Target job locations in India",
    },
    {
        "normalized_key": "willing_to_relocate",
        "category": QuestionCategory.RELOCATION.value,
        "answer_type": AnswerType.BOOLEAN.value,
        "value": True,
        "source": "candidate_approved",
        "approved": True,
        "scope": "global",
        "platforms": ["indeed", "glassdoor"],
        "confidence": 1.0,
        "requires_user_approval": False,
        "description": "Open and willing to relocate for the role",
    },
    {
        "normalized_key": "work_preference",
        "category": QuestionCategory.WORK_MODE.value,
        "answer_type": AnswerType.SELECT.value,
        "value": "Hybrid",
        "source": "candidate_approved",
        "approved": True,
        "scope": "global",
        "platforms": ["indeed", "glassdoor"],
        "confidence": 1.0,
        "requires_user_approval": False,
        "description": "Preferred work mode (Hybrid)",
    },
    {
        "normalized_key": "teaching_experience_years",
        "category": QuestionCategory.EXPERIENCE.value,
        "answer_type": AnswerType.NUMBER.value,
        "value": "1",
        "source": "candidate_approved",
        "approved": True,
        "scope": "global",
        "platforms": ["indeed", "glassdoor"],
        "confidence": 1.0,
        "requires_user_approval": False,
        "description": "Explicit candidate-approved teaching experience (1 year)",
    },
    {
        "normalized_key": "years_experience_teaching",
        "category": QuestionCategory.EXPERIENCE.value,
        "answer_type": AnswerType.NUMBER.value,
        "value": "1",
        "source": "candidate_approved",
        "approved": True,
        "scope": "global",
        "platforms": ["indeed", "glassdoor"],
        "confidence": 1.0,
        "requires_user_approval": False,
        "description": "Explicit candidate-approved teaching experience (1 year)",
    },
    {
        "normalized_key": "universal_motivation",
        "category": QuestionCategory.MOTIVATION.value,
        "answer_type": AnswerType.FREE_TEXT.value,
        "value": UNIVERSAL_MOTIVATION,
        "source": "candidate_approved",
        "approved": True,
        "scope": "global",
        "platforms": ["indeed", "glassdoor"],
        "confidence": 1.0,
        "requires_user_approval": False,
        "description": "Approved universal motivation template",
    },
    {
        "normalized_key": "education_degree",
        "category": QuestionCategory.EDUCATION.value,
        "answer_type": AnswerType.TEXT.value,
        "value": "B.Tech",
        "source": "candidate_profile",
        "approved": True,
        "scope": "global",
        "platforms": ["indeed", "glassdoor"],
        "confidence": 1.0,
        "requires_user_approval": False,
        "description": "Degree qualification",
    },
    {
        "normalized_key": "education_field",
        "category": QuestionCategory.EDUCATION.value,
        "answer_type": AnswerType.TEXT.value,
        "value": "Computer Science and Engineering",
        "source": "candidate_profile",
        "approved": True,
        "scope": "global",
        "platforms": ["indeed", "glassdoor"],
        "confidence": 1.0,
        "requires_user_approval": False,
        "description": "Field of study",
    },
    {
        "normalized_key": "education_institution",
        "category": QuestionCategory.EDUCATION.value,
        "answer_type": AnswerType.TEXT.value,
        "value": "SRM Institute of Science and Technology",
        "source": "candidate_profile",
        "approved": True,
        "scope": "global",
        "platforms": ["indeed", "glassdoor"],
        "confidence": 1.0,
        "requires_user_approval": False,
        "description": "Undergraduate University",
    },
]

ALIAS_MAP: Dict[str, List[str]] = {
    "current_ctc_lpa": ["current_ctc", "current_salary", "current_compensation", "current_package", "ctc_current"],
    "expected_ctc_lpa": ["expected_ctc", "expected_salary", "expected_compensation", "expected_package", "ctc_expected"],
    "notice_period_days": ["notice_period", "notice", "availability_to_join", "availability_period", "how_soon_can_you_join"],
    "notice_period": ["notice_period_days", "notice", "availability_to_join", "availability_period", "how_soon_can_you_join"],
    "willing_to_relocate": ["relocation", "open_to_relocation", "relocate", "willing_relocate"],
    "work_preference": ["work_mode", "hybrid_preference", "work_type", "work_environment", "preferred_work_mode"],
    "teaching_experience_years": ["years_experience_teaching", "teaching_experience"],
    "years_experience_teaching": ["teaching_experience_years", "teaching_experience"],
    "universal_motivation": ["motivation", "why_hire_you", "about_yourself", "summary_pitch", "cover_letter"],
    "education_institution": ["university", "college", "school", "institution", "undergraduate_college", "university_or_college_did_attend"],
    "education_degree": ["degree", "highest_degree", "qualification"],
    "education_field": ["major", "field_of_study", "specialization", "branch"],
    "internship_experience_months": ["internship_experience", "internships", "internship_duration", "how_many_months_of_internship_experience"],
    "graduation_year": ["year_of_passing", "passing_year", "pass_out_year"],
}


class CandidateAnswerBank:
    """Manages persistent storage and retrieval of approved candidate answers."""

    def __init__(self, file_path: Optional[str] = None) -> None:
        self.file_path = file_path or DEFAULT_ANSWERS_FILE
        self.never_infer_unverified_experience: bool = True
        self._answers: Dict[str, CandidateAnswer] = {}
        self._load()

    def _load(self) -> None:
        """Load answers from JSON file or initialize with seed defaults."""
        path = Path(self.file_path)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                if isinstance(raw_data, list):
                    for item in raw_data:
                        ans = CandidateAnswer.model_validate(item)
                        self._answers[ans.normalized_key] = ans
                elif isinstance(raw_data, dict):
                    for k, item in raw_data.items():
                        ans = CandidateAnswer.model_validate(item)
                        self._answers[k] = ans
                logger.debug("Loaded %d candidate answers from %s", len(self._answers), self.file_path)
                return
            except Exception as exc:
                logger.warning("Failed to load %s (%s). Re-seeding defaults.", self.file_path, exc)

        # Initialize defaults
        self._seed_defaults()
        self._save()

    def _seed_defaults(self) -> None:
        """Populate initial candidate-approved facts."""
        for item in DEFAULT_APPROVED_ANSWERS:
            ans = CandidateAnswer.model_validate(item)
            self._answers[ans.normalized_key] = ans

    def _save(self) -> None:
        """Persist answers to disk."""
        try:
            path = Path(self.file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            serializable = [ans.model_dump(mode="json") for ans in self._answers.values()]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, default=str)
        except Exception as exc:
            logger.error("Failed to persist candidate answers to %s: %s", self.file_path, exc)

    def get_answer(self, key: str, platform: Optional[str] = None) -> Optional[CandidateAnswer]:
        """Lookup an approved answer by exact key or registered alias."""
        clean_key = key.strip().lower()

        # 1. Exact match
        ans = self._answers.get(clean_key)
        if ans and ans.approved:
            if not platform or ans.scope == "global" or platform in ans.platforms:
                return ans

        # 2. Alias lookup
        for canonical, aliases in ALIAS_MAP.items():
            if clean_key == canonical or clean_key in aliases:
                canonical_ans = self._answers.get(canonical)
                if canonical_ans and canonical_ans.approved:
                    if not platform or canonical_ans.scope == "global" or platform in canonical_ans.platforms:
                        return canonical_ans
                for alt in aliases:
                    alt_ans = self._answers.get(alt)
                    if alt_ans and alt_ans.approved:
                        if not platform or alt_ans.scope == "global" or platform in alt_ans.platforms:
                            return alt_ans

        return None

    def get_answer_for_key(self, key: str, platform: Optional[str] = None) -> Optional[CandidateAnswer]:
        """Alias for get_answer."""
        return self.get_answer(key, platform=platform)

    def set_answer(
        self,
        normalized_key: Optional[str] = None,
        value: Any = None,
        category: Optional[QuestionCategory] = None,
        answer_type: Optional[AnswerType] = None,
        source: str = "candidate_approved",
        approved: bool = True,
        scope: str = "global",
        platforms: Optional[List[str]] = None,
        requires_user_approval: bool = False,
        description: Optional[str] = None,
        answer: Optional[CandidateAnswer] = None,
    ) -> CandidateAnswer:
        """Add or update an approved candidate answer. Supports passing a CandidateAnswer instance or fields."""
        if answer is not None:
            ans = answer
            ans.updated_at = datetime.now(timezone.utc)
        elif isinstance(normalized_key, CandidateAnswer):
            ans = normalized_key
            ans.updated_at = datetime.now(timezone.utc)
        else:
            key = str(normalized_key).strip().lower()
            ans = CandidateAnswer(
                normalized_key=key,
                value=value,
                category=category or QuestionCategory.UNKNOWN,
                answer_type=answer_type or AnswerType.TEXT,
                source=source,
                approved=approved,
                scope=scope,
                platforms=platforms or ["indeed", "glassdoor"],
                requires_user_approval=requires_user_approval,
                description=description,
                updated_at=datetime.now(timezone.utc),
            )
        self._answers[ans.normalized_key] = ans
        self._save()
        return ans

    def delete_answer(self, key: str) -> bool:
        """Remove an answer from the bank."""
        clean_key = key.strip().lower()
        if clean_key in self._answers:
            del self._answers[clean_key]
            self._save()
            return True
        return False

    def list_answers(self) -> List[CandidateAnswer]:
        """Return all answers currently in the bank."""
        return list(self._answers.values())


_candidate_answer_bank_instance: Optional[CandidateAnswerBank] = None


def get_candidate_answer_bank(file_path: Optional[str] = None) -> CandidateAnswerBank:
    """Dependency injection helper and singleton accessor for CandidateAnswerBank."""
    global _candidate_answer_bank_instance
    if _candidate_answer_bank_instance is None or file_path is not None:
        _candidate_answer_bank_instance = CandidateAnswerBank(file_path=file_path)
    return _candidate_answer_bank_instance

