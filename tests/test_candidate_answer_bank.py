"""Comprehensive Test Suite for Phase 5.8: Candidate Answer Bank & Automated Form Resolution.

Covers all 42 criteria:
1. current_ctc_lpa resolves to 0
2. expected_ctc_lpa resolves to 4-5
3. notice_period_days resolves to 15
4. preferred_locations resolves to ['Chennai', 'Bangalore', 'Kochi', 'Kozhikode', 'Trivandrum']
5. willing_to_relocate resolves to True
6. work_preference resolves to "Hybrid"
7. teaching_experience_years resolves to 1
8. years_experience_teaching resolves to 1
9. python_experience_years does NOT resolve to 1 (isolated to resume: 0.8)
10. machine_learning_experience_years does NOT resolve to 1 (isolated: 0.8)
11. total_experience_years does NOT resolve to 1 (isolated: 0.8)
12. education_degree resolves to B.Tech
13. education_field resolves to Computer Science and Engineering
14. education_institution resolves to SRM Institute of Science and Technology
15. College/graduation year resolves accurately
16. Internship experience resolves accurately
17. Text input resolves accurately
18. Number input resolves accurately
19. Dropdown exact match resolves accurately
20. Dropdown semantic match resolves accurately
21. Radio exact match resolves accurately
22. Radio semantic match resolves accurately
23. Single checkbox resolves accurately
24. Multi-checkbox resolves accurately
25. Automatic question filling when answer is known
26. Read-back verification confirms selected radio
27. Read-back verification confirms checked checkbox
28. Read-back mismatch triggers one safe retry
29. Read-back failure logs diagnostic and pauses or handles cleanly
30. Optional question skipped without pausing
31. Unknown required question pauses when ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS = False
32. Unknown required question left unanswered when ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS = True
33. Platform allows progress when optional questions skipped
34. Platform blocks on missing required question
35. Unanswered question leaves field empty for platform validation
36. Survey / metadata question handled appropriately
37. Candidate Answer Bank persistent file read/write
38. Candidate Answer Bank alias matching
39. Candidate Answer Bank CRUD API endpoints
40. Dry-run question testing API endpoint
41. Submit safety policy: AUTO_SUBMIT_ENABLED = False stops before final submit
42. Submit safety policy: AUTO_SUBMIT_ENABLED = True allows final submit
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.automation.forms.answer_resolver import FormAnswerResolver
from app.automation.forms.models import ApplicationQuestion, QuestionInputType
from app.automation.forms.question_normalizer import (
    QuestionNormalizer,
    classify_question_category,
    is_platform_survey_question,
)
from app.automation.glassdoor.apply_service import GlassdoorApplyService
from app.automation.glassdoor.models import (
    GlassdoorAutomationApplyRequest,
    GlassdoorAutomationNavigateRequest,
    GlassdoorAutomationResult,
    GlassdoorAutomationState,
)
from app.automation.glassdoor.playwright_driver import GlassdoorPlaywrightDriver
from app.automation.indeed.apply_service import IndeedApplyService
from app.automation.indeed.models import (
    AutomationState,
    IndeedAutomationApplyRequest,
    IndeedAutomationResult,
)
from app.candidate_answers.bank import CandidateAnswerBank, get_candidate_answer_bank
from app.candidate_answers.models import (
    AnswerType,
    CandidateAnswer,
    CandidateAnswerUpdateRequest,
    QuestionCategory,
)
from app.config.settings import Settings, get_settings
from app.models.profile import (
    CandidateProfileData,
    CareerPreferences,
    PersonalDetails,
    Profile,
    SkillItem,
)
from app.profile.service import ProfileService


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def mock_profile_service() -> ProfileService:
    service = MagicMock(spec=ProfileService)
    profile_data = CandidateProfileData(
        personal=PersonalDetails(
            name="Rithin Rakesh",
            email="rithinrakesh2002@gmail.com",
            phone="+91 81118 55550",
            location="Kannur, Kerala, India",
        ),
        career=CareerPreferences(
            experience_years=0.8,
            preferred_roles=["AI Engineer", "Machine Learning Engineer"],
            preferred_locations=["Chennai", "Bangalore", "Kochi", "Kozhikode", "Trivandrum"],
            remote_preference="any",
            job_type="full-time",
            education_degree="B.Tech",
            education_field="Computer Science and Engineering",
        ),
        skills=[
            SkillItem(skill="Python", proficiency="intermediate", years_experience=0.8),
            SkillItem(skill="Machine Learning", proficiency="intermediate", years_experience=0.8),
        ],
    )
    now = datetime.now(timezone.utc)
    profile = Profile(
        id=uuid4(),
        created_at=now,
        updated_at=now,
        name="Rithin Rakesh",
        email="rithinrakesh2002@gmail.com",
        phone="+91 81118 55550",
        location="Kannur, Kerala, India",
        experience_years=0.8,
        preferred_locations=["Chennai", "Bangalore", "Kochi", "Kozhikode", "Trivandrum"],
        preferred_roles=["AI Engineer", "Machine Learning Engineer"],
        skills=[],
    )
    service.get_profile.return_value = profile
    service.get_active_profile_data.return_value = profile_data
    return service


@pytest.fixture
def answer_bank(tmp_path) -> CandidateAnswerBank:
    """Fixture providing an isolated answer bank with standard candidate answers."""
    file_path = str(tmp_path / "test_candidate_answers.json")
    bank = CandidateAnswerBank(file_path=file_path)
    return bank


@pytest.fixture
def resolver(answer_bank, mock_profile_service) -> FormAnswerResolver:
    return FormAnswerResolver(
        answer_bank=answer_bank,
        profile_service=mock_profile_service,
        memory=None,
        never_infer_unverified_experience=True,
    )


# ==============================================================================
# SECTION 1: Candidate Answer Bank & Truthful Resolution Tests (Criteria 1-16)
# ==============================================================================

class TestCandidateAnswerBankCriteria:

    def test_01_current_ctc_lpa_resolves_to_0(self, resolver):
        q = ApplicationQuestion(
            text="What is your current CTC in LPA?",
            normalized_key=QuestionNormalizer.normalize("What is your current CTC in LPA?"),
            input_type=QuestionInputType.TEXT,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert str(res.resolved_value) == "0"
        assert res.source == "candidate_approved"

    def test_02_expected_ctc_lpa_resolves_to_4_to_5(self, resolver):
        q = ApplicationQuestion(
            text="What is your expected salary/CTC?",
            normalized_key=QuestionNormalizer.normalize("What is your expected salary/CTC?"),
            input_type=QuestionInputType.TEXT,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert res.resolved_value in ("4-5", "4 - 5 LPA", "400000")
        assert res.source == "candidate_approved"

    def test_03_notice_period_days_resolves_to_15(self, resolver):
        q = ApplicationQuestion(
            text="What is your notice period (in days)?",
            normalized_key=QuestionNormalizer.normalize("What is your notice period (in days)?"),
            input_type=QuestionInputType.NUMBER,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert str(res.resolved_value) == "15"

    def test_04_preferred_locations_resolves_accurately(self, resolver):
        q = ApplicationQuestion(
            text="Which preferred locations are you open to work in?",
            normalized_key=QuestionNormalizer.normalize("Which preferred locations are you open to work in?"),
            input_type=QuestionInputType.CHECKBOX_MULTI,
            required=True,
            options=["Bangalore", "Chennai", "Delhi", "Kochi", "Kolkata"],
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert isinstance(res.resolved_value, list)
        assert "Bangalore" in res.resolved_value
        assert "Chennai" in res.resolved_value
        assert "Kochi" in res.resolved_value
        assert "Delhi" not in res.resolved_value

    def test_05_willing_to_relocate_resolves_to_true(self, resolver):
        q = ApplicationQuestion(
            text="Are you willing to relocate?",
            normalized_key=QuestionNormalizer.normalize("Are you willing to relocate?"),
            input_type=QuestionInputType.RADIO,
            required=True,
            options=["Yes", "No"],
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert res.resolved_value in ("Yes", True)

    def test_06_work_preference_resolves_to_hybrid(self, resolver):
        q = ApplicationQuestion(
            text="What is your work mode preference?",
            normalized_key=QuestionNormalizer.normalize("What is your work mode preference?"),
            input_type=QuestionInputType.DROPDOWN,
            required=True,
            options=["Remote", "Hybrid", "On-site"],
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert res.resolved_value == "Hybrid"

    def test_07_teaching_experience_years_resolves_to_1_from_bank(self, resolver):
        q = ApplicationQuestion(
            text="How many years of teaching experience do you have?",
            normalized_key=QuestionNormalizer.normalize("How many years of teaching experience do you have?"),
            input_type=QuestionInputType.NUMBER,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert str(res.resolved_value) == "1"
        assert res.source == "candidate_approved"

    def test_08_years_experience_teaching_resolves_to_1_from_bank(self, resolver):
        q = ApplicationQuestion(
            text="Years of experience in teaching",
            normalized_key=QuestionNormalizer.normalize("Years of experience in teaching"),
            input_type=QuestionInputType.NUMBER,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert str(res.resolved_value) == "1"
        assert res.source == "candidate_approved"

    def test_09_python_experience_years_does_not_resolve_to_1(self, resolver):
        """Must NOT infer 1 year for python experience; falls back to resume 0.8."""
        q = ApplicationQuestion(
            text="How many years of work experience do you have with Python?",
            normalized_key=QuestionNormalizer.normalize("How many years of work experience do you have with Python?"),
            input_type=QuestionInputType.NUMBER,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.resolved_value != 1 and res.resolved_value != "1"
        if res.is_resolved:
            assert res.resolved_value in (0.8, "0.8")
            assert res.source in ("skill", "resume_facts")

    def test_10_machine_learning_experience_years_does_not_resolve_to_1(self, resolver):
        """Must NOT infer 1 year for ML experience."""
        q = ApplicationQuestion(
            text="How many years of experience do you have in Machine Learning?",
            normalized_key=QuestionNormalizer.normalize("How many years of experience do you have in Machine Learning?"),
            input_type=QuestionInputType.NUMBER,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.resolved_value != 1 and res.resolved_value != "1"
        if res.is_resolved:
            assert res.resolved_value in (0.8, "0.8")

    def test_11_total_experience_years_does_not_resolve_to_1(self, resolver):
        """Total overall experience must reflect profile (0.8), never 1."""
        q = ApplicationQuestion(
            text="Total years of professional experience?",
            normalized_key=QuestionNormalizer.normalize("Total years of professional experience?"),
            input_type=QuestionInputType.NUMBER,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.resolved_value != 1 and res.resolved_value != "1"
        if res.is_resolved:
            assert res.resolved_value in (0.8, "0.8")

    def test_12_education_degree_resolves_to_btech(self, resolver):
        q = ApplicationQuestion(
            text="What is your highest degree?",
            normalized_key=QuestionNormalizer.normalize("What is your highest degree?"),
            input_type=QuestionInputType.TEXT,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert res.resolved_value == "B.Tech"

    def test_13_education_field_resolves_to_cse(self, resolver):
        q = ApplicationQuestion(
            text="What is your field of study or major?",
            normalized_key=QuestionNormalizer.normalize("What is your field of study or major?"),
            input_type=QuestionInputType.TEXT,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert "Computer Science" in str(res.resolved_value)

    def test_14_education_institution_resolves_to_srm(self, resolver):
        q = ApplicationQuestion(
            text="What university or college did you attend?",
            normalized_key=QuestionNormalizer.normalize("What university or college did you attend?"),
            input_type=QuestionInputType.TEXT,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert "SRM Institute" in str(res.resolved_value)

    def test_15_graduation_year_resolves_accurately(self, resolver, answer_bank):
        answer_bank.set_answer("graduation_year", "2024", QuestionCategory.EDUCATION, AnswerType.NUMBER)
        q = ApplicationQuestion(
            text="Graduation Year",
            normalized_key=QuestionNormalizer.normalize("Graduation Year"),
            input_type=QuestionInputType.NUMBER,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert str(res.resolved_value) == "2024"

    def test_16_internship_experience_resolves_accurately(self, resolver, answer_bank):
        answer_bank.set_answer("internship_experience_months", "6", QuestionCategory.EXPERIENCE, AnswerType.NUMBER)
        q = ApplicationQuestion(
            text="How many months of internship experience do you have?",
            normalized_key=QuestionNormalizer.normalize("How many months of internship experience do you have?"),
            input_type=QuestionInputType.NUMBER,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert str(res.resolved_value) == "6"


# ==============================================================================
# SECTION 2: Input Controls & Option Matching Tests (Criteria 17-24, 36)
# ==============================================================================

class TestInputControlsAndOptions:

    def test_17_text_input_resolves_accurately(self, resolver):
        q = ApplicationQuestion(
            text="Current CTC",
            normalized_key=QuestionNormalizer.normalize("Current CTC"),
            input_type=QuestionInputType.TEXT,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert res.resolved_value == "0"

    def test_18_number_input_resolves_accurately(self, resolver):
        q = ApplicationQuestion(
            text="Notice Period in days",
            normalized_key=QuestionNormalizer.normalize("Notice Period in days"),
            input_type=QuestionInputType.NUMBER,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert str(res.resolved_value) == "15"

    def test_19_dropdown_exact_match(self, resolver):
        q = ApplicationQuestion(
            text="Select your work preference:",
            normalized_key=QuestionNormalizer.normalize("work_preference"),
            input_type=QuestionInputType.DROPDOWN,
            options=["Remote", "Hybrid", "In-office"],
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert res.resolved_value == "Hybrid"

    def test_20_dropdown_semantic_match(self, resolver):
        q = ApplicationQuestion(
            text="Select your work arrangement:",
            normalized_key=QuestionNormalizer.normalize("work_preference"),
            input_type=QuestionInputType.DROPDOWN,
            options=["Fully Remote", "Hybrid (2-3 days office)", "100% On-site"],
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert "Hybrid" in res.resolved_value

    def test_21_radio_exact_match(self, resolver):
        q = ApplicationQuestion(
            text="Willing to relocate?",
            normalized_key=QuestionNormalizer.normalize("willing_to_relocate"),
            input_type=QuestionInputType.RADIO,
            options=["Yes", "No"],
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert res.resolved_value == "Yes"

    def test_22_radio_semantic_match(self, resolver):
        q = ApplicationQuestion(
            text="Are you open to relocation?",
            normalized_key=QuestionNormalizer.normalize("willing_to_relocate"),
            input_type=QuestionInputType.RADIO,
            options=["Yes, willing to relocate for this position", "No, local candidates only"],
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert "Yes" in res.resolved_value

    def test_23_single_checkbox_resolves_accurately(self, resolver):
        q = ApplicationQuestion(
            text="I agree that I am willing to relocate if required",
            normalized_key=QuestionNormalizer.normalize("willing_to_relocate"),
            input_type=QuestionInputType.CHECKBOX,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert res.resolved_value is True

    def test_24_multi_checkbox_resolves_accurately(self, resolver):
        q = ApplicationQuestion(
            text="Select preferred locations:",
            normalized_key=QuestionNormalizer.normalize("preferred_locations"),
            input_type=QuestionInputType.CHECKBOX_MULTI,
            options=["Bangalore", "Mumbai", "Pune", "Kochi", "Gurgaon"],
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is True
        assert isinstance(res.resolved_value, list)
        assert set(res.resolved_value).issubset({"Bangalore", "Kochi", "Chennai", "Kozhikode", "Trivandrum"})

    def test_36_survey_metadata_question_handled_appropriately(self):
        survey_q = "How did you hear about this position at our company?"
        assert is_platform_survey_question(survey_q) is True
        cat = classify_question_category(survey_q)
        assert cat == QuestionCategory.PLATFORM_SURVEY


# ==============================================================================
# SECTION 3: Filling, Read-Back Verification & Safe Retries (Criteria 25-29)
# ==============================================================================

class TestQuestionFillingAndReadBack:

    @pytest.mark.asyncio
    async def test_25_automatic_question_filling_when_answer_known(self):
        pw_driver = GlassdoorPlaywrightDriver()
        mock_page = MagicMock()
        mock_control = MagicMock()
        mock_control.count = AsyncMock(return_value=1)
        mock_control.first = mock_control
        mock_control.fill = AsyncMock()
        mock_control.scroll_into_view_if_needed = AsyncMock()
        mock_control.focus = AsyncMock()
        mock_control.input_value = AsyncMock(return_value="0")
        mock_page.get_by_label.return_value = mock_control
        mock_page.locator.return_value = mock_control

        q = ApplicationQuestion(
            text="Current CTC",
            normalized_key="current_ctc_lpa",
            input_type=QuestionInputType.TEXT,
            required=True,
        )
        ok = await pw_driver.fill_question(mock_page, q, "0")
        assert ok is True
        mock_control.fill.assert_awaited_once_with("0")

    @pytest.mark.asyncio
    async def test_26_read_back_verification_confirms_selected_radio(self):
        pw_driver = GlassdoorPlaywrightDriver()
        mock_page = MagicMock()
        mock_radio = MagicMock()
        mock_radio.count = AsyncMock(return_value=1)
        mock_radio.first = mock_radio
        mock_radio.click = AsyncMock()
        mock_radio.check = AsyncMock()
        mock_radio.scroll_into_view_if_needed = AsyncMock()
        mock_radio.is_checked = AsyncMock(return_value=True)
        mock_page.get_by_role.return_value = mock_radio
        mock_page.get_by_label.return_value = mock_radio
        mock_page.locator.return_value = mock_radio

        q = ApplicationQuestion(
            text="Willing to relocate?",
            normalized_key="willing_to_relocate",
            input_type=QuestionInputType.RADIO,
            options=["Yes", "No"],
            required=True,
        )
        ok = await pw_driver.fill_question(mock_page, q, "Yes")
        assert ok is True
        mock_radio.is_checked.assert_awaited()

    @pytest.mark.asyncio
    async def test_27_read_back_verification_confirms_checked_checkbox(self):
        pw_driver = GlassdoorPlaywrightDriver()
        mock_page = MagicMock()
        mock_cb = MagicMock()
        mock_cb.count = AsyncMock(return_value=1)
        mock_cb.first = mock_cb
        mock_cb.check = AsyncMock()
        mock_cb.click = AsyncMock()
        mock_cb.scroll_into_view_if_needed = AsyncMock()
        mock_cb.is_checked = AsyncMock(return_value=True)
        mock_page.get_by_label.return_value = mock_cb
        mock_page.locator.return_value = mock_cb

        q = ApplicationQuestion(
            text="I agree to terms",
            normalized_key="agree_terms",
            input_type=QuestionInputType.CHECKBOX,
            required=True,
        )
        ok = await pw_driver.fill_question(mock_page, q, True)
        assert ok is True
        mock_cb.is_checked.assert_awaited()

    @pytest.mark.asyncio
    async def test_28_read_back_mismatch_triggers_one_safe_retry(self):
        pw_driver = GlassdoorPlaywrightDriver()
        mock_page = MagicMock()
        mock_radio = MagicMock()
        mock_radio.count = AsyncMock(return_value=1)
        mock_radio.first = mock_radio
        mock_radio.click = AsyncMock()
        mock_radio.check = AsyncMock()
        mock_radio.scroll_into_view_if_needed = AsyncMock()
        # First check fails (False), second check after retry succeeds (True)
        mock_radio.is_checked = AsyncMock(side_effect=[False, True])
        mock_page.get_by_role.return_value = mock_radio
        mock_page.get_by_label.return_value = mock_radio
        mock_page.locator.return_value = mock_radio

        q = ApplicationQuestion(
            text="Willing to relocate?",
            normalized_key="willing_to_relocate",
            input_type=QuestionInputType.RADIO,
            options=["Yes", "No"],
            required=True,
        )
        ok = await pw_driver.fill_question(mock_page, q, "Yes")
        # Verified that element was attempted twice (initial check + 1 retry click)
        assert mock_radio.check.await_count + mock_radio.click.await_count == 2

    @pytest.mark.asyncio
    async def test_29_read_back_failure_handles_cleanly(self):
        pw_driver = GlassdoorPlaywrightDriver()
        mock_page = MagicMock()
        mock_radio = MagicMock()
        mock_radio.count = AsyncMock(return_value=1)
        mock_radio.first = mock_radio
        mock_radio.click = AsyncMock()
        mock_radio.check = AsyncMock()
        mock_radio.scroll_into_view_if_needed = AsyncMock()
        # Consistently fails readback
        mock_radio.is_checked = AsyncMock(return_value=False)
        mock_page.get_by_role.return_value = mock_radio
        mock_page.get_by_label.return_value = mock_radio
        mock_page.locator.return_value = mock_radio

        q = ApplicationQuestion(
            text="Willing to relocate?",
            normalized_key="willing_to_relocate",
            input_type=QuestionInputType.RADIO,
            options=["Yes", "No"],
            required=True,
        )
        ok = await pw_driver.fill_question(mock_page, q, "Yes")
        assert ok is False


# ==============================================================================
# SECTION 4: Unknown Questions Policy (Criteria 30-35)
# ==============================================================================

class TestUnknownQuestionsPolicy:

    def test_30_optional_question_skipped_without_pausing(self, resolver):
        q = ApplicationQuestion(
            text="Optional: Please share your GitHub profile link",
            normalized_key="github_link",
            input_type=QuestionInputType.TEXT,
            required=False,
        )
        res = resolver.resolve(q)
        assert res.action == "skip"
        assert res.is_optional is True

    def test_31_unknown_required_question_pauses_when_submission_not_allowed(self, resolver, monkeypatch):
        monkeypatch.setattr("app.automation.forms.answer_resolver.ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS", False)
        q = ApplicationQuestion(
            text="What was the budget of your most recent enterprise project in USD?",
            normalized_key="project_budget_usd",
            input_type=QuestionInputType.NUMBER,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is False
        assert res.action == "needs_user_input"

    def test_32_unknown_required_question_leaves_unanswered_when_allowed(self, resolver, monkeypatch):
        monkeypatch.setattr("app.automation.forms.answer_resolver.ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS", True)
        q = ApplicationQuestion(
            text="What was the budget of your most recent enterprise project in USD?",
            normalized_key="project_budget_usd",
            input_type=QuestionInputType.NUMBER,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.is_resolved is False
        assert res.action == "leave_unanswered"

    def test_33_platform_allows_progress_when_optional_questions_skipped(self, resolver):
        q = ApplicationQuestion(
            text="Additional portfolio link (optional)",
            normalized_key="additional_portfolio_link",
            input_type=QuestionInputType.TEXT,
            required=False,
        )
        res = resolver.resolve(q)
        assert res.action == "skip"

    def test_34_platform_blocks_on_missing_required_question(self, resolver, monkeypatch):
        monkeypatch.setattr("app.automation.forms.answer_resolver.ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS", False)
        q = ApplicationQuestion(
            text="Do you have a valid secret clearance?",
            normalized_key="secret_clearance",
            input_type=QuestionInputType.RADIO,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.action == "needs_user_input"

    def test_35_unanswered_question_leaves_field_empty(self, resolver, monkeypatch):
        monkeypatch.setattr("app.automation.forms.answer_resolver.ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS", True)
        q = ApplicationQuestion(
            text="Special Security clearance identifier",
            normalized_key="security_clearance_id",
            input_type=QuestionInputType.TEXT,
            required=True,
        )
        res = resolver.resolve(q)
        assert res.action == "leave_unanswered"
        assert res.resolved_value is None


# ==============================================================================
# SECTION 5: Persistence & Aliases (Criteria 37-38)
# ==============================================================================

class TestAnswerBankPersistenceAndAliases:

    def test_37_candidate_answer_bank_persistent_file_read_write(self, tmp_path):
        test_file = str(tmp_path / "custom_answers.json")
        bank1 = CandidateAnswerBank(file_path=test_file)
        bank1.set_answer("custom_skill_cert", "AWS Certified Developer", QuestionCategory.CERTIFICATIONS, AnswerType.TEXT)

        # Read back with a separate bank instance pointing to the same file
        bank2 = CandidateAnswerBank(file_path=test_file)
        ans = bank2.get_answer_for_key("custom_skill_cert")
        assert ans is not None
        assert ans.value == "AWS Certified Developer"
        assert ans.category == QuestionCategory.CERTIFICATIONS

    def test_38_candidate_answer_bank_alias_matching(self, answer_bank):
        # Notice period aliases
        ans_np = answer_bank.get_answer_for_key("notice_period")
        assert ans_np is not None
        assert str(ans_np.value) == "15"

        ans_np2 = answer_bank.get_answer_for_key("how_soon_can_you_join")
        assert ans_np2 is not None
        assert str(ans_np2.value) == "15"

        # CTC aliases
        ans_ctc = answer_bank.get_answer_for_key("current_salary")
        assert ans_ctc is not None
        assert str(ans_ctc.value) == "0"


# ==============================================================================
# SECTION 6: Candidate Answer Bank REST API (Criteria 39-40)
# ==============================================================================

class TestCandidateAnswerBankAPI:

    def test_39_api_crud_endpoints(self, client):
        # GET /candidate/answers
        res = client.get("/candidate/answers")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["total_answers"] >= 10
        assert any(a["normalized_key"] == "current_ctc_lpa" for a in data["answers"])

        # PUT /candidate/answers
        update_payload = {
            "normalized_key": "custom_github_username",
            "value": "rithin-rakesh",
            "category": "personal",
            "answer_type": "text",
            "description": "Candidate GitHub profile",
        }
        res = client.put("/candidate/answers", json=update_payload)
        assert res.status_code == 200
        put_data = res.json()
        assert put_data["status"] == "success"
        assert put_data["answer"]["value"] == "rithin-rakesh"

        # Verify it now exists in GET
        res = client.get("/candidate/answers")
        data = res.json()
        assert any(a["normalized_key"] == "custom_github_username" for a in data["answers"])

    def test_40_api_dry_run_testing(self, client):
        # Dry-run test endpoint
        test_payload = {
            "question_text": "What is your current CTC in LPA?",
            "input_type": "text",
            "platform": "glassdoor",
        }
        res = client.post("/candidate/answers/test", json=test_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["is_resolved"] is True
        assert str(data["resolved_answer"]) == "0"
        assert data["action"] == "fill"
        assert data["category"] == "salary"


# ==============================================================================
# SECTION 7: Final Submit Safety Policy (Criteria 41-42)
# ==============================================================================

class TestSubmitSafetyPolicy:

    @pytest.mark.asyncio
    async def test_41_glassdoor_auto_submit_disabled_stops_before_final_submit(self):
        """When AUTO_SUBMIT_ENABLED is False, Glassdoor apply stops safely at SUBMISSION_READY."""
        service = GlassdoorApplyService()
        service.settings.AUTO_SUBMIT_ENABLED = False
        test_job_id = uuid4()

        # Mock navigate_to_submit reaching SUBMISSION_READY
        ready_result = GlassdoorAutomationResult(
            status="submission_ready",
            job_id=test_job_id,
            url="https://www.glassdoor.com/job-listing/test?jobListingId=123",
            current_state=GlassdoorAutomationState.SUBMISSION_READY,
            submit_verified=True,
            final_submit_clicked=False,
            submission_confirmed=False,
        )
        service.navigate_to_submit = AsyncMock(return_value=ready_result)
        service.driver.get_window_text_content = MagicMock(return_value="Review your application. Submit")

        req = GlassdoorAutomationApplyRequest(
            job_id=test_job_id,
            url="https://www.glassdoor.com/job-listing/test?jobListingId=123",
            source="glassdoor",
        )
        result = await service.apply_to_job(req)
        assert result.current_state == GlassdoorAutomationState.SUBMISSION_READY
        assert result.final_submit_clicked is False
        assert result.submit_verified is True
        assert "AUTO_SUBMIT_ENABLED=False" in result.message

    @pytest.mark.asyncio
    async def test_42_glassdoor_auto_submit_enabled_allows_final_submit(self):
        """When AUTO_SUBMIT_ENABLED is True, Glassdoor apply proceeds to click final submit."""
        service = GlassdoorApplyService()
        service.settings.AUTO_SUBMIT_ENABLED = True
        test_job_id = uuid4()

        ready_result = GlassdoorAutomationResult(
            status="submission_ready",
            job_id=test_job_id,
            url="https://www.glassdoor.com/job-listing/test?jobListingId=123",
            current_state=GlassdoorAutomationState.SUBMISSION_READY,
            submit_verified=True,
            final_submit_clicked=False,
            submission_confirmed=False,
        )
        service.navigate_to_submit = AsyncMock(return_value=ready_result)
        service.driver.get_window_text_content = MagicMock(return_value="Review your application. Submit")
        service.driver.detect_submission_confirmation = MagicMock(return_value=True)

        # Mock playwright driver click submit
        service.playwright_driver.page = MagicMock()
        service.playwright_driver.click_final_submit = AsyncMock(return_value=(True, None))

        req = GlassdoorAutomationApplyRequest(
            job_id=test_job_id,
            url="https://www.glassdoor.com/job-listing/test?jobListingId=123",
            source="glassdoor",
        )
        result = await service.apply_to_job(req)
        assert result.final_submit_clicked is True
        assert result.current_state == GlassdoorAutomationState.APPLICATION_SUBMITTED
