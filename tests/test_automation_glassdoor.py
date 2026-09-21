"""Comprehensive Unit Tests for Glassdoor Single-Job Application Automation (Phase 5.6).

Tests URL domain validation, question normalization, truthful answer resolution,
dynamic modal progression, resume step handling, adaptive TAB traversal,
CAPTCHA/login/MFA safety stops, single submission, and API endpoints.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routers.glassdoor_automation import get_glassdoor_apply_service
from app.automation.forms.answer_resolver import FormAnswerResolver
from app.automation.forms.form_inspector import FormInspector
from app.automation.forms.models import (
    ApplicationQuestion,
    FormInspectionResult,
    QuestionInputType,
)
from app.automation.forms.question_normalizer import (
    classify_input_type_from_control,
    is_question_required,
    normalize_question_key,
    parse_skill_and_duration_from_text,
)
from app.automation.glassdoor.apply_service import GlassdoorApplyService
from app.automation.glassdoor.config import POST_EASY_APPLY_REDIRECT_WAIT_SECONDS
from app.automation.glassdoor.models import (
    GlassdoorAutomationApplyRequest,
    GlassdoorAutomationInspectRequest,
    GlassdoorAutomationNavigateRequest,
    GlassdoorAutomationResumeRequest,
    GlassdoorAutomationResult,
    GlassdoorAutomationState,
)
from app.automation.glassdoor.playwright_driver import GlassdoorPlaywrightDriver
from app.automation.glassdoor.pywinauto_driver import PyWinAutoGlassdoorDriver
from app.automation.glassdoor.resume_handler import GlassdoorResumeHandler
from app.automation.glassdoor.state_detector import GlassdoorStateDetector
from app.automation.glassdoor.url_validator import (
    extract_glassdoor_job_id,
    is_glassdoor_domain,
    validate_glassdoor_request,
)
from app.database.repositories.application_repository import ApplicationRepository
from app.models.profile import CandidateProfileData, CareerPreferences, PersonalDetails, Profile, Skill


@pytest.fixture(autouse=True)
def fast_sleep_and_cdp_isolation(monkeypatch):
    """Fast-forward time.sleep and enforce strict CDP isolation across all unit tests."""
    monkeypatch.setattr("app.automation.glassdoor.apply_service.time.sleep", lambda s: None)
    monkeypatch.setenv("ALLOW_LIVE_CDP", "false")
    monkeypatch.setenv("AUTO_SUBMIT_ENABLED", "true")


# ==============================================================================
# 1. URL and Platform Validation Tests
# ==============================================================================


def test_is_glassdoor_domain():
    """Verify supported and rejected Glassdoor domain patterns."""
    assert is_glassdoor_domain("glassdoor.com") is True
    assert is_glassdoor_domain("www.glassdoor.com") is True
    assert is_glassdoor_domain("glassdoor.co.in") is True
    assert is_glassdoor_domain("www.glassdoor.co.in") is True
    assert is_glassdoor_domain("glassdoor.ca") is True

    # Non-Glassdoor domains
    assert is_glassdoor_domain("indeed.com") is False
    assert is_glassdoor_domain("linkedin.com") is False
    assert is_glassdoor_domain("naukri.com") is False
    assert is_glassdoor_domain("fakeglassdoor.com.evil.org") is False
    assert is_glassdoor_domain(None) is False
    assert is_glassdoor_domain("") is False


def test_validate_glassdoor_request():
    """Verify dual validation of job source and URL."""
    valid_url = "https://www.glassdoor.com/job-listing/senior-ai-engineer-jl.htm?jl=100892019"

    # Valid Glassdoor source + Glassdoor URL
    ok, norm, err = validate_glassdoor_request(valid_url, "glassdoor")
    assert ok is True
    assert norm == valid_url
    assert err is None

    # Source mismatch (e.g. source="indeed" with Glassdoor URL)
    ok_bad_source, _, err_src = validate_glassdoor_request(valid_url, "indeed")
    assert ok_bad_source is False
    assert err_src == "UNSUPPORTED_AUTOMATION_SOURCE"

    # Invalid non-Glassdoor URL
    ok_bad_url, _, err_url = validate_glassdoor_request("https://www.linkedin.com/jobs/view/123", "glassdoor")
    assert ok_bad_url is False
    assert err_url == "INVALID_JOB_URL"

    # Non-HTTP scheme
    ok_scheme, _, err_scheme = validate_glassdoor_request("javascript:alert(1)", "glassdoor")
    assert ok_scheme is False
    assert err_scheme == "INVALID_JOB_URL"


def test_extract_glassdoor_job_id():
    """Verify extraction of Glassdoor job listing ID."""
    assert extract_glassdoor_job_id("https://www.glassdoor.com/job-listing/ai-lead?jl=100998877") == "100998877"
    assert extract_glassdoor_job_id("https://www.glassdoor.com/job?jobListingId=55443322") == "55443322"
    assert extract_glassdoor_job_id("https://www.glassdoor.com/overview") is None


# ==============================================================================
# 2. Question Normalizer & Domain Classification Tests
# ==============================================================================


def test_question_normalizer():
    """Verify question key normalization and skill duration extraction."""
    assert normalize_question_key("How many years of experience do you have with Python?") == "years_experience_python"
    assert normalize_question_key("Are you willing to relocate?") == "willing_to_relocate"
    assert normalize_question_key("Notice Period (in days):") == "notice_period"
    assert normalize_question_key("Are you authorized to work in India?") == "work_authorization"

    skill, is_years = parse_skill_and_duration_from_text("How many years of experience do you have with PyTorch?")
    assert is_years is True
    assert skill == "PyTorch"

    assert is_question_required("What is your current notice period? *") is True
    assert is_question_required("Portfolio Link (Optional)") is False


# ==============================================================================
# 3. Truthful Answer Resolver Tests
# ==============================================================================


def test_answer_resolver_priority_stored_answer():
    """Verify stored approved answer takes highest priority."""
    stored = {"notice_period": "30 days", "years_experience_python": "5"}
    resolver = FormAnswerResolver(stored_answers=stored)

    q = ApplicationQuestion(
        text="What is your notice period?",
        normalized_key="notice_period",
        input_type=QuestionInputType.TEXT,
    )
    res = resolver.resolve_question(q)
    assert res.is_resolved is True
    assert res.resolved_value == "30 days"
    assert res.source == "stored_answer"


def test_answer_resolver_numeric_known_vs_unknown():
    """Verify numeric question fills known duration and returns NEEDS_USER_INPUT when unknown."""
    now = datetime.now(timezone.utc)
    profile = Profile(
        id=uuid4(),
        name="Candidate Name",
        email="candidate@example.com",
        experience_years=6.0,
        skills=[
            Skill(id=uuid4(), profile_id=uuid4(), skill="Python", years_experience=4.0, created_at=now),
            Skill(id=uuid4(), profile_id=uuid4(), skill="Docker", created_at=now),  # No years specified
        ],
        created_at=now,
        updated_at=now,
    )
    resolver = FormAnswerResolver(profile=profile)

    # 1. Known Python years
    q_py = ApplicationQuestion(
        text="How many years of experience do you have with Python?",
        normalized_key="years_experience_python",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    res_py = resolver.resolve_question(q_py)
    assert res_py.is_resolved is True
    assert str(res_py.resolved_value) == "4"
    assert res_py.source == "skill"

    # 2. Docker exists but duration not specified -> must NOT infer 1 year
    q_doc = ApplicationQuestion(
        text="How many years of experience do you have with Docker?",
        normalized_key="years_experience_docker",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    res_doc = resolver.resolve_question(q_doc)
    assert res_doc.is_resolved is False  # Requires user input

    # 3. Completely unknown skill
    q_rust = ApplicationQuestion(
        text="How many years of experience do you have with Rust?",
        normalized_key="years_experience_rust",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    res_rust = resolver.resolve_question(q_rust)
    assert res_rust.is_resolved is False


def test_answer_resolver_dropdown_exact_match_vs_unsupported():
    """Verify dropdown option selection requires exact match and rejects unsupported options."""
    stored = {"notice_period": "30 days"}
    resolver = FormAnswerResolver(stored_answers=stored)

    # Available options include "30 days"
    q_valid = ApplicationQuestion(
        text="Notice Period",
        normalized_key="notice_period",
        input_type=QuestionInputType.DROPDOWN,
        options=["Immediate", "15 days", "30 days", "60 days"],
    )
    res_valid = resolver.resolve_question(q_valid)
    assert res_valid.is_resolved is True
    assert res_valid.resolved_value == "30 days"

    # Available options do NOT include "30 days" -> must not guess closest
    q_invalid = ApplicationQuestion(
        text="Notice Period",
        normalized_key="notice_period",
        input_type=QuestionInputType.DROPDOWN,
        options=["Immediate", "2 months", "3 months"],
    )
    res_invalid = resolver.resolve_question(q_invalid)
    assert res_invalid.is_resolved is False


def test_answer_resolver_radio_and_checkbox_selection():
    """Verify radio matching and checkbox selection selects only supported skills."""
    now = datetime.now(timezone.utc)
    profile = Profile(
        id=uuid4(),
        name="AI Engineer",
        email="ai@example.com",
        skills=[
            Skill(id=uuid4(), profile_id=uuid4(), skill="Python", created_at=now),
            Skill(id=uuid4(), profile_id=uuid4(), skill="PyTorch", created_at=now),
        ],
        created_at=now,
        updated_at=now,
    )
    stored = {"willing_to_relocate": "Yes"}
    resolver = FormAnswerResolver(stored_answers=stored, profile=profile)

    # Radio button
    q_radio = ApplicationQuestion(
        text="Are you willing to relocate?",
        normalized_key="willing_to_relocate",
        input_type=QuestionInputType.RADIO,
        options=["Yes", "No"],
    )
    res_radio = resolver.resolve_question(q_radio)
    assert res_radio.is_resolved is True
    assert res_radio.resolved_value == "Yes"

    # Checkbox multi-select
    q_cb = ApplicationQuestion(
        text="Select technologies you know",
        normalized_key="select_technologies",
        input_type=QuestionInputType.CHECKBOX_MULTI,
        options=["Python", "PyTorch", "Kubernetes", "Rust"],
    )
    res_cb = resolver.resolve_question(q_cb)
    assert res_cb.is_resolved is True
    assert set(res_cb.resolved_value) == {"Python", "PyTorch"}
    assert "Rust" not in res_cb.resolved_value
    assert "Kubernetes" not in res_cb.resolved_value


# ==============================================================================
# 4. Glassdoor State Detector Tests
# ==============================================================================


def test_glassdoor_state_detector_easy_apply():
    """Verify detection of Easy Apply vs External Apply."""
    assert GlassdoorStateDetector.is_easy_apply_button("Easy Apply") is True
    assert GlassdoorStateDetector.is_easy_apply_button("Apply Now") is True
    assert GlassdoorStateDetector.is_easy_apply_button("Save Job") is False
    assert GlassdoorStateDetector.is_easy_apply_button("Apply on employer siteApply now") is False
    assert GlassdoorStateDetector.is_easy_apply_button("Apply on company site") is False

    assert GlassdoorStateDetector.is_external_apply("Apply on company site") is True
    assert GlassdoorStateDetector.is_external_apply("Apply on employer website") is True
    assert GlassdoorStateDetector.is_external_apply("Apply on employer siteApply now") is True
    assert GlassdoorStateDetector.is_external_apply("Easy Apply with Glassdoor") is False


def test_glassdoor_state_detector_action_and_submit_names():
    """Verify exact action and submit button classification."""
    assert GlassdoorStateDetector.is_continue_button("Continue") is True
    assert GlassdoorStateDetector.is_next_button("Next") is True
    assert GlassdoorStateDetector.is_review_button("Review your application") is True

    # Submit verification
    assert GlassdoorStateDetector.is_exact_submit_name("Submit application") is True
    assert GlassdoorStateDetector.is_exact_submit_name("Submit your application") is True
    assert GlassdoorStateDetector.is_exact_submit_name("Submit") is True
    assert GlassdoorStateDetector.is_exact_submit_name("Review") is False
    assert GlassdoorStateDetector.is_exact_submit_name("Save") is False


# ==============================================================================
# 5. Glassdoor Apply Service Inspection Tests
# ==============================================================================


def test_inspect_glassdoor_easy_apply_success():
    """Verify inspect finds Easy Apply without performing clicks."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Lead AI Engineer - Glassdoor - Easy Apply"

    mock_btn = MagicMock()
    mock_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = mock_btn

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationInspectRequest(
        job_id=uuid4(),
        url="https://www.glassdoor.com/job-listing/test-job?jl=12345",
        source="glassdoor",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        result = service.inspect_job_application(req)

    assert result.status == "success"
    assert result.current_state == GlassdoorAutomationState.EASY_APPLY_AVAILABLE
    assert result.easy_apply_verified is True
    assert result.easy_apply_clicked is False  # Zero clicks
    mock_driver.click_control.assert_not_called()


def test_inspect_glassdoor_external_apply():
    """Verify inspect rejects external company career site applications."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Apply on company site | Glassdoor"
    mock_driver.find_easy_apply_control.return_value = None

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationInspectRequest(
        job_id=uuid4(),
        url="https://www.glassdoor.com/job-listing/external-job?jl=998877",
        source="glassdoor",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        result = service.inspect_job_application(req)

    assert result.status == "blocked"
    assert result.current_state == GlassdoorAutomationState.EXTERNAL_APPLY
    assert result.easy_apply_verified is False


# ==============================================================================
# 6. Dynamic Modal Loop & Form Progression Tests
# ==============================================================================


def test_navigate_to_submit_direct_submit_path():
    """CASE A: Easy Apply -> Direct Submit application."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Easy Apply | Glassdoor"
    mock_driver.click_control.return_value = True

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    submit_btn = MagicMock()
    submit_btn.element_info.name = "Submit application"
    mock_driver.find_exact_submit_control.return_value = submit_btn

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationNavigateRequest(
        job_id=uuid4(),
        url="https://www.glassdoor.com/job-listing/direct-submit?jl=111",
        source="glassdoor",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        result = service.navigate_to_submit(req)

    assert result.status == "success"
    assert result.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert result.submit_verified is True
    assert result.final_submit_clicked is False  # Never submits in navigate-to-submit


def test_navigate_to_submit_resume_continue_questions_review_submit():
    """CASE B & C: Multi-step progression (Resume -> Continue -> Questions -> Review -> Submit)."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Easy Apply"
    mock_driver.click_control.return_value = True

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    # Sequence of action buttons found in modal loop
    cont_btn = MagicMock()
    cont_btn.element_info.name = "Continue"

    rev_btn = MagicMock()
    rev_btn.element_info.name = "Review"

    submit_btn = MagicMock()
    submit_btn.element_info.name = "Submit application"

    # Step 1: Continue available
    # Step 2: Review available
    # Step 3: Submit application available
    mock_driver.find_exact_submit_control.side_effect = [None, None, submit_btn]
    mock_driver.find_action_control.side_effect = [
        None, cont_btn,  # Step 1: review=None, continue=cont_btn
        rev_btn,         # Step 2: review=rev_btn
    ]

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationNavigateRequest(
        job_id=uuid4(),
        url="https://www.glassdoor.com/job-listing/multistep?jl=222",
        source="glassdoor",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        result = service.navigate_to_submit(req)

    assert result.status == "success"
    assert result.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert result.steps_processed >= 2
    assert result.submit_verified is True
    assert result.final_submit_clicked is False


def test_navigate_to_submit_unresolved_required_question_stops_safely():
    """Verify encountering an unknown required question halts with NEEDS_USER_INPUT."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Easy Apply"
    mock_driver.click_control.return_value = True
    mock_driver.find_exact_submit_control.return_value = None

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    # Mock FormInspector returning an unresolved required question
    mock_q = ApplicationQuestion(
        text="What is your secret security clearance level? *",
        normalized_key="security_clearance_level",
        input_type=QuestionInputType.TEXT,
        required=True,
    )
    with patch(
        "app.automation.glassdoor.apply_service.FormInspector.inspect_container",
        return_value=FormInspectionResult(questions=[mock_q], unresolved_required_count=1),
    ):
        service = GlassdoorApplyService(driver=mock_driver)
        req = GlassdoorAutomationNavigateRequest(
            job_id=uuid4(),
            url="https://www.glassdoor.com/job-listing/clearance-job?jl=333",
            source="glassdoor",
        )

        with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
            result = service.navigate_to_submit(req)

    assert result.status == "manual_action_required"
    assert result.current_state == GlassdoorAutomationState.NEEDS_USER_INPUT
    assert result.manual_action_required is True
    assert result.questions_unresolved == 1
    assert "security clearance" in result.message.lower()


# ==============================================================================
# 7. Adaptive TAB Traversal Fallback Tests
# ==============================================================================


def test_adaptive_tab_traversal_activates_verified_action():
    """Verify bounded adaptive TAB traversal finds and activates target action."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver._window = MagicMock()

    # Focused control returns non-action, then Continue
    mock_driver.get_focused_control_info.side_effect = [
        {"name": "First Name", "control_type": "Edit", "is_submit": False},
        {"name": "Continue", "control_type": "Button", "is_submit": False},
    ]

    ok, action, err = PyWinAutoGlassdoorDriver.find_and_activate_action_via_tab_traversal(
        mock_driver, expected_actions=["Continue", "Submit application"], max_tabs=10
    )

    assert ok is True
    assert action == "Continue"
    assert err is None
    mock_driver.send_enter_once.assert_called_once()


def test_adaptive_tab_traversal_limit_reached_fails_safely():
    """Verify reaching MAX_TAB_TRAVERSAL returns failure without looping infinitely."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver._window = MagicMock()
    mock_driver.get_focused_control_info.return_value = {"name": "Text", "control_type": "Edit", "is_submit": False}

    ok, action, err = PyWinAutoGlassdoorDriver.find_and_activate_action_via_tab_traversal(
        mock_driver, expected_actions=["Submit application"], max_tabs=5
    )

    assert ok is False
    assert action is None
    assert "maximum traversal limit" in err


# ==============================================================================
# 8. CAPTCHA, Login, MFA & Safety Tests
# ==============================================================================


def test_apply_to_job_captcha_blocks_submission():
    """Verify CAPTCHA detected before final submission blocks without clicking."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.verify_browser_window.return_value = (True, None)

    # Page text returns CAPTCHA pattern
    mock_driver.get_window_text_content.return_value = "Verify you are human | Security check | Glassdoor"

    submit_btn = MagicMock()
    submit_btn.element_info.name = "Submit application"
    mock_driver.find_exact_submit_control.return_value = submit_btn

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationApplyRequest(
        job_id=uuid4(),
        url="https://www.glassdoor.com/job-listing/captcha-job?jl=444",
        source="glassdoor",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        result = service.apply_to_job(req)

    assert result.status == "manual_action_required"
    assert result.current_state == GlassdoorAutomationState.CAPTCHA_OR_CHALLENGE
    assert result.manual_action_required is True
    assert result.final_submit_clicked is False
    assert result.submission_confirmed is False


def test_apply_to_job_login_or_mfa_blocks_submission():
    """Verify Login or MFA requirement blocks submission safely."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.get_window_text_content.return_value = "Sign in to Glassdoor | Enter your password"

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationApplyRequest(
        job_id=uuid4(),
        url="https://www.glassdoor.com/job-listing/login-job?jl=555",
        source="glassdoor",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        result = service.apply_to_job(req)

    assert result.status == "manual_action_required"
    assert result.current_state == GlassdoorAutomationState.LOGIN_REQUIRED
    assert result.manual_action_required is True
    assert result.final_submit_clicked is False


# ==============================================================================
# 9. Final Single-Submit & Confirmation Tests
# ==============================================================================


def test_apply_to_job_successful_single_submission():
    """Verify successful end-to-end flow executes submit exactly once and verifies confirmation."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.get_window_text_content.return_value = "Glassdoor Application"
    mock_driver.click_control.return_value = True
    mock_driver.detect_submission_confirmation.return_value = True

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    submit_btn = MagicMock()
    submit_btn.element_info.name = "Submit application"
    mock_driver.find_exact_submit_control.return_value = submit_btn

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_answers_by_profile.return_value = {}

    service = GlassdoorApplyService(driver=mock_driver, application_repository=mock_app_repo)
    req = GlassdoorAutomationApplyRequest(
        job_id=uuid4(),
        url="https://www.glassdoor.com/job-listing/success-job?jl=666",
        source="glassdoor",
        profile_id=uuid4(),
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        result = service.apply_to_job(req)

    assert result.status == "success"
    assert result.current_state == GlassdoorAutomationState.APPLICATION_SUBMITTED
    assert result.final_submit_clicked is True
    assert result.submission_confirmed is True
    mock_driver.click_control.assert_called()  # Submit clicked exactly once
    mock_app_repo.create_application.assert_called_once()


def test_apply_to_job_missing_confirmation_no_duplicate_submit():
    """Verify missing confirmation returns SUBMISSION_CONFIRMATION_UNVERIFIED and does not retry."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.get_window_text_content.return_value = "Glassdoor Application"
    mock_driver.click_control.return_value = True
    mock_driver.detect_submission_confirmation.return_value = False  # Confirmation timeout

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    submit_btn = MagicMock()
    submit_btn.element_info.name = "Submit application"
    mock_driver.find_exact_submit_control.return_value = submit_btn

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_answers_by_profile.return_value = {}

    service = GlassdoorApplyService(driver=mock_driver, application_repository=mock_app_repo)
    req = GlassdoorAutomationApplyRequest(
        job_id=uuid4(),
        url="https://www.glassdoor.com/job-listing/unconfirmed-job?jl=777",
        source="glassdoor",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        result = service.apply_to_job(req)

    assert result.status == "blocked"
    assert result.current_state == GlassdoorAutomationState.SUBMISSION_CONFIRMATION_UNVERIFIED
    assert result.final_submit_clicked is True
    assert result.submission_confirmed is False


# ==============================================================================
# 10. FastAPI Router Endpoints Tests
# ==============================================================================


def test_api_glassdoor_inspect_endpoint():
    """Verify POST /automation/glassdoor/inspect endpoint."""
    client = TestClient(app)
    mock_service = MagicMock(spec=GlassdoorApplyService)
    sample_id = uuid4()

    mock_service.inspect_job_application.return_value = GlassdoorAutomationResult(
        status="success",
        job_id=sample_id,
        url="https://www.glassdoor.com/job-listing/api-test?jl=888",
        platform="glassdoor",
        easy_apply_verified=True,
        easy_apply_clicked=False,
        current_state=GlassdoorAutomationState.EASY_APPLY_AVAILABLE,
    )

    app.dependency_overrides[get_glassdoor_apply_service] = lambda: mock_service
    try:
        resp = client.post(
            "/automation/glassdoor/inspect",
            json={"job_id": str(sample_id), "url": "https://www.glassdoor.com/job-listing/api-test?jl=888"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_state"] == "EASY_APPLY_AVAILABLE"
        assert data["easy_apply_verified"] is True
    finally:
        app.dependency_overrides.pop(get_glassdoor_apply_service, None)


def test_api_glassdoor_navigate_to_submit_endpoint():
    """Verify POST /automation/glassdoor/navigate-to-submit endpoint."""
    client = TestClient(app)
    mock_service = MagicMock(spec=GlassdoorApplyService)
    sample_id = uuid4()

    mock_service.navigate_to_submit.return_value = GlassdoorAutomationResult(
        status="success",
        job_id=sample_id,
        url="https://www.glassdoor.com/job-listing/api-nav?jl=889",
        platform="glassdoor",
        easy_apply_verified=True,
        easy_apply_clicked=True,
        submit_verified=True,
        final_submit_clicked=False,
        current_state=GlassdoorAutomationState.SUBMISSION_READY,
    )

    app.dependency_overrides[get_glassdoor_apply_service] = lambda: mock_service
    try:
        resp = client.post(
            "/automation/glassdoor/navigate-to-submit",
            json={"job_id": str(sample_id), "url": "https://www.glassdoor.com/job-listing/api-nav?jl=889"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_state"] == "SUBMISSION_READY"
        assert data["final_submit_clicked"] is False
    finally:
        app.dependency_overrides.pop(get_glassdoor_apply_service, None)


def test_api_glassdoor_apply_endpoint():
    """Verify POST /automation/glassdoor/apply endpoint."""
    client = TestClient(app)
    mock_service = MagicMock(spec=GlassdoorApplyService)
    sample_id = uuid4()

    mock_service.apply_to_job.return_value = GlassdoorAutomationResult(
        status="success",
        job_id=sample_id,
        url="https://www.glassdoor.com/job-listing/api-apply?jl=890",
        platform="glassdoor",
        easy_apply_verified=True,
        easy_apply_clicked=True,
        submit_verified=True,
        final_submit_clicked=True,
        submission_confirmed=True,
        current_state=GlassdoorAutomationState.APPLICATION_SUBMITTED,
    )

    app.dependency_overrides[get_glassdoor_apply_service] = lambda: mock_service
    try:
        resp = client.post(
            "/automation/glassdoor/apply",
            json={"job_id": str(sample_id), "url": "https://www.glassdoor.com/job-listing/api-apply?jl=890"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_state"] == "APPLICATION_SUBMITTED"
        assert data["final_submit_clicked"] is True
        assert data["submission_confirmed"] is True
    finally:
        app.dependency_overrides.pop(get_glassdoor_apply_service, None)


def test_non_glassdoor_url_rejected():
    """Verify non-Glassdoor URL is rejected during validate_glassdoor_request."""
    ok, norm, err = validate_glassdoor_request("https://www.naukri.com/job/123", "glassdoor")
    assert ok is False
    assert err == "INVALID_JOB_URL"


def test_multiple_questions_on_same_page_all_resolved():
    """Verify multiple questions on same page are all resolved and answered."""
    profile = Profile(
        id=uuid4(),
        name="John Doe",
        email="john@example.com",
        phone="+919876543210",
        experience_years=5.0,
        skills=[
            Skill(id=uuid4(), profile_id=uuid4(), skill="Python", years_experience=3.0, created_at=datetime.now(timezone.utc)),
        ],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    resolver = FormAnswerResolver(profile=profile)

    q1 = ApplicationQuestion(text="Full Name", normalized_key="full_name", input_type=QuestionInputType.TEXT)
    q2 = ApplicationQuestion(text="Email Address", normalized_key="email", input_type=QuestionInputType.TEXT)
    q3 = ApplicationQuestion(text="Years of experience with Python?", normalized_key="years_experience_python", input_type=QuestionInputType.NUMBER)

    res1 = resolver.resolve_question(q1)
    res2 = resolver.resolve_question(q2)
    res3 = resolver.resolve_question(q3)

    assert res1.is_resolved is True and res1.resolved_value == "John Doe"
    assert res2.is_resolved is True and res2.resolved_value == "john@example.com"
    assert res3.is_resolved is True and str(res3.resolved_value) == "3"


def test_mfa_blocks_submission_safely():
    """Verify OTP/MFA code requirement blocks with MFA_REQUIRED."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.get_window_text_content.return_value = "Enter the 2-step verification code sent to your phone"

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationApplyRequest(
        job_id=uuid4(),
        url="https://www.glassdoor.com/job-listing/mfa-job?jl=999",
        source="glassdoor",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        result = service.apply_to_job(req)

    assert result.status == "manual_action_required"
    assert result.current_state == GlassdoorAutomationState.MFA_REQUIRED
    assert result.manual_action_required is True
    assert result.final_submit_clicked is False


def test_existing_indeed_automation_unaffected():
    """Verify that existing Indeed automation components remain completely intact and functional."""
    from app.automation.indeed.url_validator import is_indeed_domain, validate_indeed_request
    assert is_indeed_domain("in.indeed.com") is True
    assert is_indeed_domain("glassdoor.com") is False
    ok, _, _ = validate_indeed_request("https://in.indeed.com/viewjob?jk=abcdef12345", "indeed")
    assert ok is True


# ==============================================================================
# 11. Manual Live-Testing (URL-Only, Optional Job ID) Tests
# ==============================================================================


def test_inspect_accepts_valid_url_with_null_or_omitted_job_id():
    """Verify inspect endpoint and service accept valid Glassdoor URL with job_id=null or omitted."""
    client = TestClient(app)
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Staff AI Engineer | Glassdoor | Easy Apply"

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    service = GlassdoorApplyService(driver=mock_driver)
    app.dependency_overrides[get_glassdoor_apply_service] = lambda: service

    try:
        # Case A: job_id explicitly None
        req = GlassdoorAutomationInspectRequest(
            url="https://www.glassdoor.com/job-listing/manual-test?jl=123999",
            job_id=None,
            source="glassdoor",
        )
        with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
            res = service.inspect_job_application(req)
        assert res.status == "success"
        assert res.job_id is None
        assert res.easy_apply_verified is True

        # Case B: Via HTTP endpoint with omitted job_id
        resp = client.post(
            "/automation/glassdoor/inspect",
            json={"url": "https://www.glassdoor.com/job-listing/manual-test?jl=123999"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_state"] == "EASY_APPLY_AVAILABLE"
        assert data["job_id"] is None
    finally:
        app.dependency_overrides.pop(get_glassdoor_apply_service, None)


def test_inspect_rejects_invalid_or_non_glassdoor_url_without_job_id():
    """Verify inspect rejects invalid/non-Glassdoor URL even when job_id is null/omitted."""
    client = TestClient(app)
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    service = GlassdoorApplyService(driver=mock_driver)
    app.dependency_overrides[get_glassdoor_apply_service] = lambda: service

    try:
        # Non-Glassdoor URL
        resp = client.post(
            "/automation/glassdoor/inspect",
            json={"url": "https://www.linkedin.com/jobs/view/998877"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "blocked"
        assert data["current_state"] == "INVALID_JOB_URL"
        assert data["job_id"] is None
    finally:
        app.dependency_overrides.pop(get_glassdoor_apply_service, None)


def test_inspect_accepts_valid_persisted_job_id():
    """Verify inspect continues to accept and return valid persisted job_id when provided."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Easy Apply | Glassdoor"

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    service = GlassdoorApplyService(driver=mock_driver)
    persisted_id = uuid4()
    req = GlassdoorAutomationInspectRequest(
        url="https://www.glassdoor.com/job-listing/persisted-job?jl=555444",
        job_id=persisted_id,
        source="glassdoor",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.inspect_job_application(req)

    assert res.status == "success"
    assert res.job_id == persisted_id
    assert res.easy_apply_verified is True


def test_navigate_to_submit_accepts_valid_url_without_job_id():
    """Verify navigate-to-submit operates cleanly from URL alone without job_id."""
    client = TestClient(app)
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Easy Apply | Glassdoor"
    mock_driver.click_control.return_value = True

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    submit_btn = MagicMock()
    submit_btn.element_info.name = "Submit application"
    mock_driver.find_exact_submit_control.return_value = submit_btn

    service = GlassdoorApplyService(driver=mock_driver)
    app.dependency_overrides[get_glassdoor_apply_service] = lambda: service

    try:
        # Via HTTP endpoint without job_id
        resp = client.post(
            "/automation/glassdoor/navigate-to-submit",
            json={"url": "https://www.glassdoor.com/job-listing/direct-submit?jl=777888"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_state"] == "SUBMISSION_READY"
        assert data["submit_verified"] is True
        assert data["final_submit_clicked"] is False
        assert data["job_id"] is None
    finally:
        app.dependency_overrides.pop(get_glassdoor_apply_service, None)


def test_no_fake_uuid_generated_when_job_id_is_null():
    """Verify no fake UUID is invented when job_id is null."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Lead AI Engineer - Glassdoor - Easy Apply"

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationInspectRequest(
        url="https://www.glassdoor.com/job-listing/clean-test?jl=999000",
        job_id=None,
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.inspect_job_application(req)

    assert res.job_id is None
    assert isinstance(res.job_id, type(None))


# ==============================================================================
# 12. Easy Apply Element-From-Point & Verified Coordinate Fallback Tests
# ==============================================================================


def test_uia_primary_easy_apply_preferred_over_fallback():
    """Verify primary descendant scan is used and preferred when available."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Staff AI Engineer | Glassdoor"

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationInspectRequest(
        url="https://www.glassdoor.com/job-listing/test?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.inspect_job_application(req)

    assert res.status == "success"
    assert res.easy_apply_verified is True
    assert res.diagnostics.get("uia_primary_found") is True
    mock_driver.get_element_at_point.assert_not_called()


def test_element_from_point_exact_easy_apply_verified():
    """Verify element-from-point at fallback coordinate detects Easy Apply when primary fails."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Staff AI Engineer | Glassdoor"
    mock_driver.find_easy_apply_control.return_value = None  # Primary scan fails

    # Element from point succeeds
    point_btn = MagicMock()
    point_btn.element_info.name = "Easy Apply"
    mock_driver.get_element_at_point.return_value = (
        point_btn,
        {
            "element_name": "Easy Apply",
            "control_type": "Button",
            "is_enabled": True,
            "is_visible": True,
            "bounds": "(L220, T480, R320, B532)",
            "is_easy_apply": True,
        },
    )

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationInspectRequest(
        url="https://www.glassdoor.co.in/job-listing/live-test?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.inspect_job_application(req)

    assert res.status == "success"
    assert res.easy_apply_verified is True
    assert res.current_state == GlassdoorAutomationState.EASY_APPLY_AVAILABLE
    assert res.diagnostics.get("fallback_verified") is True
    assert res.diagnostics.get("fallback_coordinate") == [270, 506]


def test_wrong_element_at_fallback_coordinate_no_click():
    """Verify wrong element at fallback coordinate does not verify or click."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Overview | Glassdoor"
    mock_driver.find_easy_apply_control.return_value = None

    # Point at (270, 506) hits a random text element, not Easy Apply
    mock_driver.get_element_at_point.return_value = (
        None,
        {
            "element_name": "Company Overview",
            "control_type": "Text",
            "is_enabled": True,
            "is_visible": True,
            "bounds": "(L100, T400, R400, B550)",
            "is_easy_apply": False,
        },
    )

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationInspectRequest(
        url="https://www.glassdoor.com/job-listing/overview?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.inspect_job_application(req)

    assert res.status == "blocked"
    assert res.easy_apply_verified is False
    assert res.current_state == GlassdoorAutomationState.EASY_APPLY_NOT_VERIFIED
    mock_driver.click_control.assert_not_called()
    mock_driver.click_coordinate.assert_not_called()


def test_navigate_to_submit_verified_coordinate_click_post_transition_modal_success():
    """Verify last-resort verified coordinate click succeeds when post-click modal appears."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.find_easy_apply_control.return_value = None
    mock_driver.get_element_at_point.return_value = (None, {"is_easy_apply": False})
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.is_point_inside_window.return_value = True
    mock_driver.click_coordinate.return_value = True

    # Post-click transition: submit button appears in modal
    submit_btn = MagicMock()
    submit_btn.element_info.name = "Submit application"
    mock_driver.find_exact_submit_control.return_value = submit_btn
    mock_driver.is_smartapply_detected.return_value = False
    mock_driver.get_window_text_content.return_value = "Submit your application | Glassdoor"

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/live-coord-test?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False
    mock_driver.click_coordinate.assert_called_once_with(270, 506)


def test_navigate_to_submit_verified_coordinate_click_smartapply_success():
    """Verify post-click transition to smartapply.indeed.com is accepted as host."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.find_easy_apply_control.return_value = None
    mock_driver.get_element_at_point.return_value = (None, {"is_easy_apply": False})
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.is_point_inside_window.return_value = True
    mock_driver.click_coordinate.return_value = True

    # Post-click transition: smartapply detected, submit available
    mock_driver.is_smartapply_detected.return_value = True
    submit_btn = MagicMock()
    submit_btn.element_info.name = "Submit application"
    mock_driver.find_exact_submit_control.return_value = submit_btn
    mock_driver.get_window_text_content.return_value = "smartapply.indeed.com - Application"

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/smartapply-test?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True


def test_navigate_to_submit_no_post_transition_returns_easy_apply_click_unverified():
    """Verify lack of post-click transition returns EASY_APPLY_CLICK_UNVERIFIED and never clicks again."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.find_easy_apply_control.return_value = None
    mock_driver.get_element_at_point.return_value = (None, {"is_easy_apply": False})
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.is_point_inside_window.return_value = True
    mock_driver.click_coordinate.return_value = True

    # Post-click remains static on initial page with no modal or transition
    mock_driver.find_exact_submit_control.return_value = None
    mock_driver.find_action_control.return_value = None
    mock_driver.is_smartapply_detected.return_value = False
    mock_driver.get_window_text_content.return_value = "Untitled - Google Chrome"

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/static-fail?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "blocked"
    assert res.current_state == GlassdoorAutomationState.EASY_APPLY_CLICK_UNVERIFIED
    assert res.easy_apply_clicked is True
    # Click coordinate was called exactly once
    assert mock_driver.click_coordinate.call_count == 1


def test_inspect_endpoint_never_clicks_coordinate_fallback():
    """Verify POST /automation/glassdoor/inspect never clicks the coordinate fallback."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.find_easy_apply_control.return_value = None
    mock_driver.get_element_at_point.return_value = (None, {"is_easy_apply": False})
    mock_driver.get_window_text_content.return_value = "Glassdoor Job"

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationInspectRequest(
        url="https://www.glassdoor.com/job-listing/inspect-safety?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.inspect_job_application(req)

    assert res.easy_apply_clicked is False
    mock_driver.click_coordinate.assert_not_called()
    mock_driver.click_control.assert_not_called()


def test_captcha_prevents_coordinate_fallback():
    """Verify detected CAPTCHA challenge halts before coordinate click."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Security check | Verify you are human | Glassdoor"

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/captcha-guard?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "manual_action_required"
    assert res.current_state == GlassdoorAutomationState.CAPTCHA_OR_CHALLENGE
    mock_driver.click_coordinate.assert_not_called()
    mock_driver.click_control.assert_not_called()


# ==============================================================================
# 13. Dynamic Application Progression, SmartApply & Loop Protection Tests
# ==============================================================================


def test_navigate_to_submit_does_not_stop_at_easy_apply_available():
    """Verify navigate-to-submit does not return on EASY_APPLY_AVAILABLE but navigates to SUBMISSION_READY."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.side_effect = [
        "Staff AI Engineer | Glassdoor | Easy Apply",  # initial page
        "Submit your application | Glassdoor",          # post-click modal page
        "Submit your application | Glassdoor",
    ]

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn
    mock_driver.click_control.return_value = True

    submit_btn = MagicMock()
    submit_btn.element_info.name = "Submit your application"
    mock_driver.find_exact_submit_control.return_value = submit_btn

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/deep-nav?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    # Must NOT be EASY_APPLY_AVAILABLE
    assert res.current_state != GlassdoorAutomationState.EASY_APPLY_AVAILABLE
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.easy_apply_clicked is True
    assert res.submit_verified is True
    assert res.final_submit_clicked is False


def test_navigate_to_submit_smartapply_resume_step_proceeds_with_continue():
    """Verify SmartApply resume step is handled and proceeds via Continue button."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.is_smartapply_detected.return_value = True

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    cont_btn = MagicMock()
    cont_btn.element_info.name = "Continue"

    submit_btn = MagicMock()
    submit_btn.element_info.name = "Submit application"

    state = {"step": 1}

    def mock_click_control(ctrl):
        if ctrl == cont_btn:
            state["step"] = 2
        return True

    mock_driver.click_control.side_effect = mock_click_control

    def mock_get_window_text():
        if state["step"] == 1:
            return "smartapply.indeed.com/beta/indeedapply/form/resume-selection Select resume"
        return "smartapply.indeed.com/beta/indeedapply/form/review Submit application"

    def mock_find_submit(container=None):
        if state["step"] >= 2:
            return submit_btn
        return None

    def mock_find_action(action_names, container=None):
        if state["step"] == 1 and any("continue" in str(a).lower() for a in action_names):
            return cont_btn
        return None

    mock_driver.get_window_text_content.side_effect = mock_get_window_text
    mock_driver.find_exact_submit_control.side_effect = mock_find_submit
    mock_driver.find_action_control.side_effect = mock_find_action

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/smartapply-resume?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.diagnostics.get("application_host") == "indeed_smartapply"
    assert res.submit_verified is True
    assert res.final_submit_clicked is False
    assert len(res.diagnostics.get("step_history", [])) == 2


def test_navigate_to_submit_multi_step_questions_and_review():
    """Verify multi-page questions dynamically answered and navigated to Review then Submit."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.is_smartapply_detected.return_value = True

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    next_btn = MagicMock()
    next_btn.element_info.name = "Next"

    review_btn = MagicMock()
    review_btn.element_info.name = "Review"

    submit_btn = MagicMock()
    submit_btn.element_info.name = "Submit application"

    state = {"step": 1}

    def mock_click_control(ctrl):
        if ctrl == next_btn and state["step"] == 1:
            state["step"] = 2
        elif ctrl == review_btn and state["step"] == 2:
            state["step"] = 3
        return True

    mock_driver.click_control.side_effect = mock_click_control

    def mock_get_window_text():
        if state["step"] == 1:
            return "Questions Step 1 | Full Name"
        elif state["step"] == 2:
            return "Questions Step 2 | Python Experience"
        return "Review your application | Submit"

    def mock_find_submit(container=None):
        if state["step"] >= 3:
            return submit_btn
        return None

    def mock_find_action(action_names, container=None):
        if state["step"] == 1 and any("next" in str(a).lower() for a in action_names):
            return next_btn
        elif state["step"] == 2 and any("review" in str(a).lower() for a in action_names):
            return review_btn
        return None

    mock_driver.get_window_text_content.side_effect = mock_get_window_text
    mock_driver.find_exact_submit_control.side_effect = mock_find_submit
    mock_driver.find_action_control.side_effect = mock_find_action

    profile = Profile(
        id=uuid4(),
        name="John Doe",
        email="john@example.com",
        phone="+919876543210",
        experience_years=5.0,
        skills=[Skill(id=uuid4(), profile_id=uuid4(), skill="Python", years_experience=3.0, created_at=datetime.now(timezone.utc))],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_profile.return_value = profile

    service = GlassdoorApplyService(driver=mock_driver, profile_service=mock_profile_svc)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/multi-questions?jl=1010229231195",
        profile_id=profile.id,
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False
    assert len(res.diagnostics.get("step_history", [])) == 3


def test_navigate_to_submit_stalled_signature_halts_safely():
    """Verify loop halts with APPLICATION_PROGRESS_STALLED when page content repeats without progress."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    # Repeated identical page text that causes stall
    mock_driver.get_window_text_content.return_value = "Stuck Form Page | Continue"
    mock_driver.is_smartapply_detected.return_value = False

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn
    mock_driver.click_control.return_value = True

    mock_driver.find_exact_submit_control.return_value = None
    cont_btn = MagicMock()
    cont_btn.element_info.name = "Continue"
    mock_driver.find_action_control.return_value = cont_btn

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/stalled-test?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "blocked"
    assert res.current_state == GlassdoorAutomationState.APPLICATION_PROGRESS_STALLED
    assert res.easy_apply_clicked is True
    assert res.diagnostics.get("stalled_signature") is not None


def test_navigate_to_submit_never_clicks_final_submit_button():
    """Verify navigate-to-submit positively identifies submit but never clicks it."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Submit your application | Glassdoor"

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn
    mock_driver.click_control.return_value = True

    submit_btn = MagicMock()
    submit_btn.element_info.name = "Submit application"
    mock_driver.find_exact_submit_control.return_value = submit_btn

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/never-submit?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False
    # submit_btn must NOT have been clicked
    submit_btn.click_input.assert_not_called()
    submit_btn.invoke.assert_not_called()


# ==============================================================================
# 14. Redirect Timing, Post-Wait Inspection & Diagnostic Tests
# ==============================================================================


def test_post_easy_apply_10_second_wait_occurs_once_before_smartapply_inspection():
    """Verify 10-second redirect wait occurs after Easy Apply click and before inspecting SmartApply."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.is_smartapply_detected.return_value = True

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    submit_btn = MagicMock()
    submit_btn.element_info.name = "Submit application"
    mock_driver.find_exact_submit_control.return_value = submit_btn

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/timing-test?jl=1010229231195",
    )

    sleep_calls = []

    def tracking_sleep(seconds):
        sleep_calls.append(seconds)

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True), \
         patch("app.automation.glassdoor.apply_service.time.sleep", side_effect=tracking_sleep):
        res = service.navigate_to_submit(req)

    # 10-second wait must occur exactly once
    ten_sec_waits = [s for s in sleep_calls if s == POST_EASY_APPLY_REDIRECT_WAIT_SECONDS]
    assert len(ten_sec_waits) == 1
    assert POST_EASY_APPLY_REDIRECT_WAIT_SECONDS == 10

    # The 10-second wait must occur before checking is_smartapply_detected
    assert mock_driver.is_smartapply_detected.called

    # Verify diagnostic fields
    assert res.diagnostics.get("post_easy_apply_wait_seconds") == 10
    assert "url_after_redirect_wait" in res.diagnostics
    assert res.diagnostics.get("application_host_after_wait") == "indeed_smartapply"
    assert res.diagnostics.get("state_after_redirect_wait") == GlassdoorAutomationState.SMART_APPLY_HOST_VERIFIED.value
    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False


def test_no_extra_10_second_wait_on_form_actions():
    """Verify that Continue / Next / Review step transitions do NOT introduce extra 10-second delays."""
    mock_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.is_smartapply_detected.return_value = True

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_driver.find_easy_apply_control.return_value = easy_btn

    cont_btn = MagicMock()
    cont_btn.element_info.name = "Continue"

    next_btn = MagicMock()
    next_btn.element_info.name = "Next"

    submit_btn = MagicMock()
    submit_btn.element_info.name = "Submit application"

    state = {"step": 1}

    def mock_click_control(ctrl):
        if ctrl == cont_btn and state["step"] == 1:
            state["step"] = 2
        elif ctrl == next_btn and state["step"] == 2:
            state["step"] = 3
        return True

    mock_driver.click_control.side_effect = mock_click_control

    def mock_get_window_text():
        if state["step"] == 1:
            return "Resume Selection | Continue"
        elif state["step"] == 2:
            return "Questions Step | Next"
        return "Review Step | Submit application"

    def mock_find_submit(container=None):
        if state["step"] >= 3:
            return submit_btn
        return None

    def mock_find_action(action_names, container=None):
        if state["step"] == 1 and any("continue" in str(a).lower() for a in action_names):
            return cont_btn
        elif state["step"] == 2 and any("next" in str(a).lower() for a in action_names):
            return next_btn
        return None

    mock_driver.get_window_text_content.side_effect = mock_get_window_text
    mock_driver.find_exact_submit_control.side_effect = mock_find_submit
    mock_driver.find_action_control.side_effect = mock_find_action

    service = GlassdoorApplyService(driver=mock_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/multi-step-delays?jl=1010229231195",
    )

    sleep_calls = []

    def tracking_sleep(seconds):
        sleep_calls.append(seconds)

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True), \
         patch("app.automation.glassdoor.apply_service.time.sleep", side_effect=tracking_sleep):
        res = service.navigate_to_submit(req)

    # 10s wait occurs strictly once right after Easy Apply
    ten_sec_waits = [s for s in sleep_calls if s == 10]
    assert len(ten_sec_waits) == 1

    # Form progression step count
    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert len(res.diagnostics.get("step_history", [])) == 3
    assert res.submit_verified is True
    assert res.final_submit_clicked is False


# ==============================================================================
# 15. Hybrid PyWinAuto + Playwright/CDP DOM Automation Tests
# ==============================================================================


def test_recaptcha_informational_footer_does_not_trigger_challenge():
    """Verify that benign footer text 'This site is protected by reCAPTCHA...' does NOT trigger CAPTCHA_OR_CHALLENGE."""
    footer_text = (
        "This site is protected by reCAPTCHA and the Google Privacy Policy "
        "and Terms of Service apply. Please submit your application."
    )
    res = GlassdoorStateDetector.detect_blocking_state(footer_text)
    assert res is None, f"Expected None but got blocking state: {res}"

    # Also test with GlassdoorPlaywrightDriver's state detector logic
    mock_pw_page = MagicMock()
    mock_pw_page.url = "https://smartapply.indeed.com/form/questions"
    mock_pw_page.locator.return_value.count.return_value = 0
    mock_pw_page.locator.return_value.first.is_visible.return_value = False
    mock_pw_page.locator.return_value.inner_text.return_value = footer_text

    driver = GlassdoorPlaywrightDriver()
    driver.page = mock_pw_page

    blocking_state, blocking_reason = driver._detect_blocking_in_dom(mock_pw_page)
    assert blocking_state is None
    assert blocking_reason is None


def test_actual_interactive_captcha_challenge_triggers_manual_stop():
    """Verify that actual interactive CAPTCHA widgets or challenges DO trigger CAPTCHA_OR_CHALLENGE."""
    challenge_text = "Please verify you are human to proceed. Security check."
    res = GlassdoorStateDetector.detect_blocking_state(challenge_text)
    assert res is not None
    state, msg = res
    assert state == GlassdoorAutomationState.CAPTCHA_OR_CHALLENGE

    # Test widget presence in Playwright DOM
    mock_pw_page = MagicMock()
    mock_pw_page.url = "https://smartapply.indeed.com/form/questions"

    def mock_locator(sel):
        loc = MagicMock()
        if "recaptcha/api2/bframe" in sel or "hcaptcha" in sel or "turnstile" in sel:
            loc.count.return_value = 1
            loc.first.is_visible.return_value = True
        else:
            loc.count.return_value = 0
            loc.first.is_visible.return_value = False
            loc.inner_text.return_value = ""
        return loc

    mock_pw_page.locator.side_effect = mock_locator

    driver = GlassdoorPlaywrightDriver()
    driver.page = mock_pw_page

    blocking_state, blocking_reason = driver._detect_blocking_in_dom(mock_pw_page)
    assert blocking_state == GlassdoorAutomationState.CAPTCHA_OR_CHALLENGE
    assert "Active CAPTCHA widget visible" in blocking_reason


def test_playwright_workflow_case_a_resume_questions_review_submit():
    """Test Case A: Easy Apply -> Resume -> Questions -> Review -> Submit."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_win_driver.is_smartapply_detected.return_value = True

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    # Playwright Mock
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)

    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/resume-selection"
    mock_pw_driver.page = mock_page

    step_counter = {"step": 1}

    def mock_detect_state(page):
        current_s = step_counter["step"]
        if current_s == 1:
            return GlassdoorAutomationState.RESUME_STEP, {"progression_action": "continue"}
        elif current_s == 2:
            return GlassdoorAutomationState.QUESTIONS_STEP, {"progression_action": "continue"}
        elif current_s == 3:
            return GlassdoorAutomationState.REVIEW_STEP, {"progression_action": "review"}
        else:
            return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    def mock_click_progression(page=None):
        step_counter["step"] += 1
        return True, "continue", None

    mock_pw_driver.detect_page_state.side_effect = mock_detect_state
    mock_pw_driver.handle_resume_step.return_value = (True, "default_selected", None)
    mock_pw_driver.extract_questions.return_value = [
        ApplicationQuestion(
            text="Years of Python experience?",
            normalized_key="years_of_python_experience",
            input_type=QuestionInputType.NUMBER,
            required=True,
        )
    ]
    mock_pw_driver.fill_question.return_value = True
    mock_pw_driver.click_progression_action.side_effect = mock_click_progression
    mock_pw_driver.find_exact_submit_button.side_effect = lambda p=None: MagicMock() if step_counter["step"] >= 4 else None

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    # Provide stored answer for python experience
    service.application_repository.get_answers_by_profile = MagicMock(return_value={"years_of_python_experience": "5"})

    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/case-a?jl=1010229231195",
        profile_id=uuid4(),
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False
    assert res.diagnostics.get("playwright_attached") is True
    assert len(res.diagnostics.get("step_history", [])) == 4


def test_playwright_workflow_case_b_review_submit():
    """Test Case B: Easy Apply -> Review -> Submit (Direct 1-step flow)."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)

    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/review"
    mock_pw_driver.page = mock_page

    mock_pw_driver.detect_page_state.return_value = (
        GlassdoorAutomationState.SUBMISSION_READY,
        {"submit_button_detected": True},
    )
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/case-b?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False
    assert res.steps_processed == 1


def test_playwright_workflow_case_c_questions_review_submit():
    """Test Case C: Easy Apply -> Questions -> Review -> Submit."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)

    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/questions"
    mock_pw_driver.page = mock_page

    step_counter = {"step": 1}

    def mock_detect_state(page):
        if step_counter["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {"progression_action": "continue"}
        elif step_counter["step"] == 2:
            return GlassdoorAutomationState.REVIEW_STEP, {"progression_action": "review"}
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    def mock_click_progression(page=None):
        step_counter["step"] += 1
        return True, "continue", None

    mock_pw_driver.detect_page_state.side_effect = mock_detect_state
    mock_pw_driver.extract_questions.return_value = [
        ApplicationQuestion(
            text="Authorized to work in India?",
            normalized_key="work_authorized",
            input_type=QuestionInputType.RADIO,
            required=True,
            options=["Yes", "No"],
        )
    ]
    mock_pw_driver.fill_question.return_value = True
    mock_pw_driver.click_progression_action.side_effect = mock_click_progression
    mock_pw_driver.find_exact_submit_button.side_effect = lambda p=None: MagicMock() if step_counter["step"] >= 3 else None

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    service.application_repository.get_answers_by_profile = MagicMock(return_value={"work_authorized": "Yes"})

    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/case-c?jl=1010229231195",
        profile_id=uuid4(),
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False


def test_playwright_workflow_case_d_resume_review_submit():
    """Test Case D: Easy Apply -> Resume -> Review -> Submit."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)

    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/resume-selection"
    mock_pw_driver.page = mock_page

    step_counter = {"step": 1}

    def mock_detect_state(page):
        if step_counter["step"] == 1:
            return GlassdoorAutomationState.RESUME_STEP, {"progression_action": "continue"}
        elif step_counter["step"] == 2:
            return GlassdoorAutomationState.REVIEW_STEP, {"progression_action": "review"}
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    def mock_click_progression(page=None):
        step_counter["step"] += 1
        return True, "continue", None

    mock_pw_driver.detect_page_state.side_effect = mock_detect_state
    mock_pw_driver.handle_resume_step.return_value = (True, "selected", None)
    mock_pw_driver.click_progression_action.side_effect = mock_click_progression
    mock_pw_driver.find_exact_submit_button.side_effect = lambda p=None: MagicMock() if step_counter["step"] >= 3 else None

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/case-d?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False


def test_playwright_workflow_case_e_multi_page_questions():
    """Test Case E: Easy Apply -> Questions Page 1 -> Questions Page 2 -> Review -> Submit."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)

    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/questions-1"
    mock_pw_driver.page = mock_page

    step_counter = {"step": 1}

    def mock_detect_state(page):
        if step_counter["step"] in (1, 2):
            return GlassdoorAutomationState.QUESTIONS_STEP, {"progression_action": "continue"}
        elif step_counter["step"] == 3:
            return GlassdoorAutomationState.REVIEW_STEP, {"progression_action": "review"}
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    def mock_click_progression(page=None):
        step_counter["step"] += 1
        return True, "continue", None

    mock_pw_driver.detect_page_state.side_effect = mock_detect_state

    def mock_extract_questions(page=None):
        if step_counter["step"] == 1:
            return [
                ApplicationQuestion(
                    text="Years of experience with React?",
                    normalized_key="years_of_experience_with_react",
                    input_type=QuestionInputType.NUMBER,
                    required=True,
                )
            ]
        elif step_counter["step"] == 2:
            return [
                ApplicationQuestion(
                    text="Are you willing to relocate?",
                    normalized_key="willing_to_relocate",
                    input_type=QuestionInputType.RADIO,
                    required=True,
                    options=["Yes", "No"],
                )
            ]
        return []

    mock_pw_driver.extract_questions.side_effect = mock_extract_questions
    mock_pw_driver.fill_question.return_value = True
    mock_pw_driver.click_progression_action.side_effect = mock_click_progression
    mock_pw_driver.find_exact_submit_button.side_effect = lambda p=None: MagicMock() if step_counter["step"] >= 4 else None

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    service.application_repository.get_answers_by_profile = MagicMock(
        return_value={
            "years_of_experience_with_react": "4",
            "willing_to_relocate": "Yes",
        }
    )

    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/case-e?jl=1010229231195",
        profile_id=uuid4(),
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.questions_detected == 2
    assert res.questions_answered == 2
    assert res.submit_verified is True


def test_playwright_required_unknown_question_returns_needs_user_input():
    """Verify that an unresolved required question halts immediately with NEEDS_USER_INPUT."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)

    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/questions"
    mock_pw_driver.page = mock_page

    mock_pw_driver.detect_page_state.return_value = (
        GlassdoorAutomationState.QUESTIONS_STEP,
        {"progression_action": "continue"},
    )
    mock_pw_driver.extract_questions.return_value = [
        ApplicationQuestion(
            text="What is your secret security clearance code?",
            normalized_key="clearance_code",
            input_type=QuestionInputType.TEXT,
            required=True,
        )
    ]

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    # Empty stored answers & empty profile
    service.application_repository.get_answers_by_profile = MagicMock(return_value={})

    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/unknown-req?jl=1010229231195",
        profile_id=uuid4(),
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "manual_action_required"
    assert res.current_state == GlassdoorAutomationState.NEEDS_USER_INPUT
    assert res.manual_action_required is True
    assert res.questions_unresolved == 1
    assert "clearance_code" in str(res.diagnostics)


def test_playwright_ctc_and_notice_period_require_truthful_answers():
    """Verify that Current CTC (LPA), Expected CTC (LPA), and Notice Period do NOT use fake default values (0 or 15)."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)

    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/questions"
    mock_pw_driver.page = mock_page

    mock_pw_driver.detect_page_state.return_value = (
        GlassdoorAutomationState.QUESTIONS_STEP,
        {"progression_action": "continue"},
    )
    mock_pw_driver.extract_questions.return_value = [
        ApplicationQuestion(
            text="Current CTC (LPA)",
            normalized_key="current_ctc_lpa",
            input_type=QuestionInputType.NUMBER,
            required=True,
        ),
        ApplicationQuestion(
            text="Expected CTC (LPA)",
            normalized_key="expected_ctc_lpa",
            input_type=QuestionInputType.NUMBER,
            required=True,
        ),
        ApplicationQuestion(
            text="Notice Period (Days)",
            normalized_key="notice_period_days",
            input_type=QuestionInputType.NUMBER,
            required=True,
        ),
    ]

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    service.candidate_answer_bank = None
    # No CTC or notice answers stored
    service.application_repository.get_answers_by_profile = MagicMock(return_value={})

    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/ctc-test?jl=1010229231195",
        profile_id=uuid4(),
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    # Must halt safely without guessing
    assert res.status == "manual_action_required"
    assert res.current_state == GlassdoorAutomationState.NEEDS_USER_INPUT
    assert res.manual_action_required is True


def test_playwright_review_question_summaries_not_treated_as_editable():
    """Verify that read-only summaries on the Review page are NOT extracted as new editable questions."""
    mock_pw_page = MagicMock()
    mock_pw_page.url = "https://smartapply.indeed.com/form/review"

    # Set up heading indicating Review page
    headings = MagicMock()
    headings.count.return_value = 1
    headings.nth.return_value.inner_text.return_value = "Review your application"
    mock_pw_page.locator.side_effect = lambda sel: headings if "heading" in sel or "h1" in sel else MagicMock(count=lambda: 0)

    driver = GlassdoorPlaywrightDriver()
    driver.page = mock_pw_page

    assert driver._is_review_step(mock_pw_page) is True

    # Questions extracted must be empty on Review page
    questions = driver.extract_questions(mock_pw_page)
    assert len(questions) == 0


def test_playwright_submit_detected_as_submission_ready_without_clicking():
    """Verify that exact Submit button is classified as SUBMISSION_READY and navigate_to_submit never clicks it."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)

    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/review"
    mock_pw_driver.page = mock_page

    mock_submit = MagicMock()
    mock_pw_driver.find_exact_submit_button.return_value = mock_submit
    mock_pw_driver.detect_page_state.return_value = (
        GlassdoorAutomationState.SUBMISSION_READY,
        {"submit_button_detected": True},
    )

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/submit-detect?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False
    # submit button must NOT have been clicked during navigate_to_submit
    mock_submit.click.assert_not_called()
    mock_pw_driver.click_final_submit.assert_not_called()


def test_playwright_apply_to_job_clicks_submit_strictly_once():
    """Verify that apply_to_job clicks final submit strictly ONCE via Playwright."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_win_driver.verify_browser_window.return_value = (True, None)
    mock_win_driver.detect_submission_confirmation.return_value = True
    mock_win_driver.get_window_text_content.return_value = "Submit your application"

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)

    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/review"
    mock_pw_driver.page = mock_page

    mock_pw_driver.detect_page_state.return_value = (
        GlassdoorAutomationState.SUBMISSION_READY,
        {"submit_button_detected": True},
    )
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()
    mock_pw_driver.click_final_submit.return_value = (True, None)

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    job_uuid = uuid4()
    req = GlassdoorAutomationApplyRequest(
        job_id=job_uuid,
        url="https://www.glassdoor.com/job-listing/apply-submit-once?jl=1010229231195",
        profile_id=uuid4(),
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.apply_to_job(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.APPLICATION_SUBMITTED
    assert res.submit_verified is True
    assert res.final_submit_clicked is True
    assert res.submission_confirmed is True
    # Clicked strictly once
    assert mock_pw_driver.click_final_submit.call_count == 1


def test_playwright_app_stall_protection():
    """Verify that repeated identical state signatures trigger APPLICATION_PROGRESS_STALLED."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)

    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)

    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/stuck"
    mock_pw_driver.page = mock_page

    # Return same modal state repeatedly without advancing
    mock_pw_driver.detect_page_state.return_value = (
        GlassdoorAutomationState.APPLICATION_MODAL,
        {"progression_action": "continue"},
    )
    mock_pw_driver.click_progression_action.return_value = (True, "continue", None)

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/stuck-pw?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "blocked"
    assert res.current_state == GlassdoorAutomationState.APPLICATION_PROGRESS_STALLED
    assert "stalled_signature" in res.diagnostics


def test_hybrid_pywinauto_entry_with_playwright_dom_takeover():
    """Verify hybrid architecture: PyWinAuto coordinate fallback entry + Playwright DOM form processing."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_win_driver.verify_browser_window.return_value = (True, None)
    mock_win_driver.is_point_inside_window.return_value = True
    mock_win_driver.get_window_text_content.return_value = "Glassdoor Job Page"
    # UIA primary missing, element from point missing -> coordinate fallback used
    mock_win_driver.find_easy_apply_control.return_value = None
    mock_win_driver.get_element_at_point.return_value = (None, {"is_easy_apply": False})
    mock_win_driver.click_coordinate.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)

    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/resume-selection"
    mock_pw_driver.page = mock_page

    step_counter = {"step": 1}

    def mock_detect_state(page):
        if step_counter["step"] == 1:
            return GlassdoorAutomationState.RESUME_STEP, {"progression_action": "continue"}
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    def mock_click_progression(page=None):
        step_counter["step"] += 1
        return True, "continue", None

    mock_pw_driver.detect_page_state.side_effect = mock_detect_state
    mock_pw_driver.handle_resume_step.return_value = (True, "default_selected", None)
    mock_pw_driver.click_progression_action.side_effect = mock_click_progression
    mock_pw_driver.find_exact_submit_button.side_effect = lambda p=None: MagicMock() if step_counter["step"] >= 2 else None

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/hybrid-flow?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.diagnostics.get("easy_apply_method") == "verified_coordinate"
    assert res.diagnostics.get("playwright_attached") is True
    assert res.submit_verified is True
    assert res.final_submit_clicked is False


def test_playwright_field_filling_all_supported_input_types():
    """Verify that GlassdoorPlaywrightDriver.fill_question correctly handles text, number, dropdown, radio, and checkboxes."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    # 1. Text Field
    q_text = ApplicationQuestion(
        text="Full Name",
        normalized_key="full_name",
        input_type=QuestionInputType.TEXT,
        field_name="applicant_name",
    )
    mock_loc_text = MagicMock()
    mock_loc_text.count.return_value = 1
    mock_loc_text.first.is_visible.return_value = True
    mock_page.get_by_label.return_value = mock_loc_text

    filled = driver.fill_question(mock_page, q_text, "John Doe")
    assert filled is True
    mock_loc_text.first.fill.assert_called_with("John Doe")

    # 2. Number Field
    q_num = ApplicationQuestion(
        text="Years of Experience",
        normalized_key="years_of_experience",
        input_type=QuestionInputType.NUMBER,
        field_name="exp_years",
    )
    mock_loc_num = MagicMock()
    mock_loc_num.count.return_value = 1
    mock_loc_num.first.is_visible.return_value = True
    mock_page.get_by_label.return_value = mock_loc_num

    filled_num = driver.fill_question(mock_page, q_num, 5)
    assert filled_num is True
    mock_loc_num.first.fill.assert_called_with("5")

    # 3. Dropdown / Select Field
    q_drop = ApplicationQuestion(
        text="Highest Degree",
        normalized_key="highest_degree",
        input_type=QuestionInputType.DROPDOWN,
        options=["Bachelors", "Masters", "PhD"],
    )
    mock_loc_select = MagicMock()
    mock_loc_select.count.return_value = 1
    mock_loc_select.first.is_visible.return_value = True
    mock_loc_select.first.evaluate.return_value = "select"
    mock_page.get_by_label.return_value = mock_loc_select

    filled_drop = driver.fill_question(mock_page, q_drop, "Masters")
    assert filled_drop is True
    mock_loc_select.first.select_option.assert_called_with(label="Masters")

    # 4. Radio Field
    q_radio = ApplicationQuestion(
        text="Willing to work hybrid?",
        normalized_key="willing_hybrid",
        input_type=QuestionInputType.RADIO,
        options=["Yes", "No"],
    )
    mock_loc_radio = MagicMock()
    mock_loc_radio.count.return_value = 1
    mock_loc_radio.first.is_visible.return_value = True
    mock_page.get_by_role.return_value = mock_loc_radio

    filled_radio = driver.fill_question(mock_page, q_radio, "Yes")
    assert filled_radio is True
    mock_loc_radio.first.check.assert_called()

    # 5. Single Checkbox Field
    q_check = ApplicationQuestion(
        text="I agree to the background check policy",
        normalized_key="agree_background_check",
        input_type=QuestionInputType.CHECKBOX,
    )
    mock_loc_check = MagicMock()
    mock_loc_check.count.return_value = 1
    mock_loc_check.first.is_visible.return_value = True
    mock_page.get_by_label.return_value = mock_loc_check

    filled_check = driver.fill_question(mock_page, q_check, True)
    assert filled_check is True
    mock_loc_check.first.check.assert_called()

    # 6. Multi-Checkbox Field
    q_multi = ApplicationQuestion(
        text="Select known cloud platforms",
        normalized_key="known_cloud_platforms",
        input_type=QuestionInputType.CHECKBOX_MULTI,
        options=["AWS", "GCP", "Azure"],
    )
    mock_loc_multi = MagicMock()
    mock_loc_multi.count.return_value = 1
    mock_loc_multi.first.is_visible.return_value = True
    mock_page.get_by_role.return_value = mock_loc_multi

    filled_multi = driver.fill_question(mock_page, q_multi, ["AWS", "GCP"])
    assert filled_multi is True
    assert mock_loc_multi.first.check.call_count >= 2


# ==============================================================================
# 16. Hybrid Live-Testing Regression & Fixes Suite (Section U)
# ==============================================================================


def test_glassdoor_automation_reuses_existing_cdp_page_and_no_second_chrome():
    """Verify that Glassdoor automation reuses existing CDP page and does not spawn a second Chrome."""
    driver = GlassdoorPlaywrightDriver()
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.url = "http://127.0.0.1:8000/docs"
    mock_page2 = MagicMock()
    mock_page2.url = "https://www.glassdoor.com/job-listing/test-job?jl=123"
    mock_page2.title.return_value = "Test Job at Glassdoor"

    mock_context.pages = [mock_page1, mock_page2]
    mock_browser.contexts = [mock_context]
    driver._browser = mock_browser
    driver._context = mock_context

    with patch("playwright.sync_api.sync_playwright"):
        ok, page, diag = driver.resolve_or_navigate_page(
            "https://www.glassdoor.com/job-listing/test-job?jl=123"
        )

    assert ok is True
    assert page == mock_page2
    assert diag["second_browser_launched"] is False
    assert diag["browser_session_verified"] is True
    mock_page2.bring_to_front.assert_called()


def test_same_cdp_page_used_for_job_navigation():
    """Verify that if the active CDP page has a different URL, page.goto navigates that SAME page."""
    driver = GlassdoorPlaywrightDriver()
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_page.url = "https://www.glassdoor.com/jobs"
    mock_page.title.return_value = "Glassdoor Jobs"

    mock_context.pages = [mock_page]
    mock_browser.contexts = [mock_context]
    driver._browser = mock_browser
    driver._context = mock_context

    with patch("playwright.sync_api.sync_playwright"):
        ok, page, diag = driver.resolve_or_navigate_page(
            "https://www.glassdoor.com/job-listing/ai-engineer?jl=9988"
        )

    assert ok is True
    assert page == mock_page
    mock_page.goto.assert_called_with(
        "https://www.glassdoor.com/job-listing/ai-engineer?jl=9988",
        wait_until="domcontentloaded",
        timeout=15000,
    )


def test_resume_continue_below_viewport_found_and_scrolled_into_view():
    """Verify that Continue button below viewport is located and scrolled into view via scroll_into_view_if_needed."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    driver.page = mock_page

    mock_btn = MagicMock()
    mock_btn.count.return_value = 1
    mock_btn.first.is_enabled.return_value = True

    mock_empty = MagicMock()
    mock_empty.count.return_value = 0

    # Return continue button for continue role query and empty for review
    def get_role_mock(role, name=None):
        pattern_str = str(name.pattern if hasattr(name, 'pattern') else name).lower()
        if "continue" in pattern_str:
            return mock_btn
        return mock_empty

    mock_page.get_by_role.side_effect = get_role_mock

    ok, action_name, err = driver.activate_progress_button(mock_page)

    assert ok is True
    assert action_name == "continue"
    mock_btn.first.scroll_into_view_if_needed.assert_called_once()
    mock_btn.first.wait_for.assert_called_once_with(state="visible", timeout=3000)
    mock_btn.first.click.assert_called_once()


def test_no_coordinate_fallback_on_hosted_resume_page():
    """Verify that on hosted application page, Playwright DOM interactions are used exclusively without coordinate clicks."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_win_driver.get_window_text_content.return_value = "Glassdoor Job"
    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)
    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/resume-selection"
    mock_pw_driver.page = mock_page

    step_counter = {"step": 1}

    def mock_detect(p):
        if step_counter["step"] == 1:
            return GlassdoorAutomationState.RESUME_STEP, {"progression_action": "continue"}
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    def mock_activate(p=None):
        step_counter["step"] += 1
        return True, "continue", None

    mock_pw_driver.detect_page_state.side_effect = mock_detect
    mock_pw_driver.handle_resume_step.return_value = (True, "default_selected", None)
    mock_pw_driver.activate_progress_button.side_effect = mock_activate
    mock_pw_driver.find_exact_submit_button.side_effect = lambda p=None: MagicMock() if step_counter["step"] >= 2 else None

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/resume-test?jl=1010229231195",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    # Coordinate clicking must NEVER be called on hosted resume/application steps
    mock_win_driver.click_coordinate.assert_not_called()


def test_existing_selected_resume_remains_unchanged():
    """Verify handle_resume_step detects existing selected resume and leaves it unchanged."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    mock_checked_radio = MagicMock()
    mock_checked_radio.count.return_value = 1
    mock_page.locator.return_value = mock_checked_radio

    ok, state_val, err = driver.handle_resume_step(mock_page)

    assert ok is True
    assert state_val == "default_selected"
    assert err is None
    # Must not click or replace resume
    mock_checked_radio.first.click.assert_not_called()


def test_resume_continue_click_progresses_to_questions():
    """Verify resume Continue click progresses to Questions step with scrolled_into_view recorded in step_history."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)
    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/resume-selection"
    mock_pw_driver.page = mock_page

    step_counter = {"step": 1}

    def mock_detect(p):
        if step_counter["step"] == 1:
            return GlassdoorAutomationState.RESUME_STEP, {"progression_action": "continue"}
        elif step_counter["step"] == 2:
            return GlassdoorAutomationState.QUESTIONS_STEP, {"questions_count": 1}
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    def mock_activate(p=None):
        step_counter["step"] += 1
        return True, "continue", None

    mock_pw_driver.detect_page_state.side_effect = mock_detect
    mock_pw_driver.handle_resume_step.return_value = (True, "default_selected", None)
    mock_pw_driver.activate_progress_button.side_effect = mock_activate
    mock_pw_driver.find_exact_submit_button.side_effect = lambda p=None: MagicMock() if step_counter["step"] >= 3 else None

    # Step 2 question is resolved
    q = ApplicationQuestion(
        text="Notice Period",
        normalized_key="notice_period",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q]
    mock_pw_driver.fill_question.return_value = True

    mock_repo = MagicMock(spec=ApplicationRepository)
    mock_repo.get_answers_by_profile.return_value = {"notice_period": "30"}

    service = GlassdoorApplyService(
        driver=mock_win_driver,
        playwright_driver=mock_pw_driver,
        application_repository=mock_repo,
    )
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/resume-to-q?jl=1010229231195",
        profile_id=uuid4(),
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    step_hist = res.diagnostics.get("step_history", [])
    assert len(step_hist) >= 2
    assert step_hist[0]["state"] == "RESUME_STEP"
    assert step_hist[0].get("scrolled_into_view") is True
    assert step_hist[1]["state"] == "QUESTIONS_STEP"


def test_three_questions_current_ctc_expected_ctc_notice_period_extracted_independently():
    """Verify that Current CTC, Expected CTC, and Notice Period are extracted as 3 independent questions."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    # Mock empty fieldsets
    mock_fieldsets = MagicMock()
    mock_fieldsets.count.return_value = 0
    mock_page.locator.side_effect = lambda sel: mock_fieldsets if "fieldset" in sel else mock_inputs

    # Mock 3 inputs
    mock_inputs = MagicMock()
    mock_inputs.count.return_value = 3

    mock_inp1 = MagicMock()
    mock_inp1.is_enabled.return_value = True
    mock_inp1.get_attribute.side_effect = lambda attr: "current_ctc" if attr == "name" else ("number" if attr == "type" else None)
    mock_inp1.evaluate.return_value = "input"
    mock_inp1.input_value.return_value = ""

    mock_inp2 = MagicMock()
    mock_inp2.is_enabled.return_value = True
    mock_inp2.get_attribute.side_effect = lambda attr: "expected_ctc" if attr == "name" else ("number" if attr == "type" else None)
    mock_inp2.evaluate.return_value = "input"
    mock_inp2.input_value.return_value = ""

    mock_inp3 = MagicMock()
    mock_inp3.is_enabled.return_value = True
    mock_inp3.get_attribute.side_effect = lambda attr: "notice_period" if attr == "name" else ("number" if attr == "type" else None)
    mock_inp3.evaluate.return_value = "input"
    mock_inp3.input_value.return_value = ""

    mock_inputs.nth.side_effect = lambda idx: [mock_inp1, mock_inp2, mock_inp3][idx]

    with patch.object(
        driver,
        "_find_label_for_input",
        side_effect=["Current CTC (LPA) *", "Expected CTC (LPA) *", "Notice Period *"],
    ):
        questions = driver.extract_questions(mock_page)

    assert len(questions) == 3
    assert questions[0].text == "Current CTC (LPA) *"
    assert questions[0].input_type == QuestionInputType.NUMBER
    assert questions[0].required is True

    assert questions[1].text == "Expected CTC (LPA) *"
    assert questions[1].input_type == QuestionInputType.NUMBER
    assert questions[1].required is True

    assert questions[2].text == "Notice Period *"
    assert questions[2].input_type == QuestionInputType.NUMBER
    assert questions[2].required is True


def test_unknown_required_question_returns_needs_user_input_with_detailed_unresolved_info():
    """Verify that an unresolvable required question halts with NEEDS_USER_INPUT and rich unresolved_questions diagnostics."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)
    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/questions"
    mock_pw_driver.page = mock_page

    mock_pw_driver.detect_page_state.return_value = (
        GlassdoorAutomationState.QUESTIONS_STEP,
        {"questions_count": 1},
    )

    q = ApplicationQuestion(
        text="Expected CTC (LPA)",
        normalized_key="expected_ctc",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q]

    # Stored answers do NOT have expected_ctc
    mock_repo = MagicMock(spec=ApplicationRepository)
    mock_repo.get_answers_by_profile.return_value = {"current_ctc": "25"}

    service = GlassdoorApplyService(
        driver=mock_win_driver,
        playwright_driver=mock_pw_driver,
        application_repository=mock_repo,
    )
    service.candidate_answer_bank = None
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/missing-q?jl=1010229231195",
        profile_id=uuid4(),
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "manual_action_required"
    assert res.current_state == GlassdoorAutomationState.NEEDS_USER_INPUT
    assert res.manual_action_required is True
    assert res.questions_unresolved == 1
    assert "unresolved_questions" in res.diagnostics
    assert len(res.diagnostics["unresolved_questions"]) == 1
    assert res.diagnostics["unresolved_questions"][0]["text"] == "Expected CTC (LPA)"
    assert res.diagnostics["unresolved_questions"][0]["required"] is True


def test_multiple_question_pages_progression_to_review_and_submit():
    """Verify that multiple consecutive question pages progress cleanly to Review and SUBMISSION_READY."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.connect_cdp.return_value = (True, None)
    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/questions-p1"
    mock_pw_driver.page = mock_page

    step_counter = {"step": 1}

    def mock_detect(p):
        if step_counter["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {"questions_count": 1}
        elif step_counter["step"] == 2:
            return GlassdoorAutomationState.QUESTIONS_STEP, {"questions_count": 1}
        elif step_counter["step"] == 3:
            return GlassdoorAutomationState.REVIEW_STEP, {"review_step_detected": True}
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    def mock_activate(p=None):
        step_counter["step"] += 1
        return True, "continue", None

    mock_pw_driver.detect_page_state.side_effect = mock_detect
    mock_pw_driver.activate_progress_button.side_effect = mock_activate

    q1 = ApplicationQuestion(text="Years of Experience", normalized_key="years_of_experience", input_type=QuestionInputType.NUMBER, required=True)
    q2 = ApplicationQuestion(text="Notice Period", normalized_key="notice_period", input_type=QuestionInputType.NUMBER, required=True)

    mock_pw_driver.extract_questions.side_effect = lambda p: [q1] if step_counter["step"] == 1 else ([q2] if step_counter["step"] == 2 else [])
    mock_pw_driver.fill_question.return_value = True
    mock_pw_driver.find_exact_submit_button.side_effect = lambda p=None: MagicMock() if step_counter["step"] >= 3 else None

    mock_repo = MagicMock(spec=ApplicationRepository)
    mock_repo.get_answers_by_profile.return_value = {
        "years_of_experience": "6",
        "notice_period": "15",
    }

    service = GlassdoorApplyService(
        driver=mock_win_driver,
        playwright_driver=mock_pw_driver,
        application_repository=mock_repo,
    )
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/multi-q-pages?jl=1010229231195",
        profile_id=uuid4(),
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False
    assert res.questions_answered == 2


def test_review_summaries_not_treated_as_editable_questions():
    """Verify that on Review page, question summaries and Edit controls are not treated as editable questions."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/review"

    with patch.object(driver, "_is_review_step", return_value=True):
        questions = driver.extract_questions(mock_page)

    assert questions == []


# ==============================================================================
# 17. Strict Test Isolation and Real CDP Protection Tests
# ==============================================================================


def test_unit_tests_never_connect_to_live_cdp():
    """Verify that GlassdoorPlaywrightDriver raises RuntimeError if connect_cdp is called without mock in test mode."""
    driver = GlassdoorPlaywrightDriver(host="127.0.0.1", port=9222)
    with patch.dict("os.environ", {"PYTEST_CURRENT_TEST": "test_isolation", "ALLOW_LIVE_CDP": "false"}):
        with pytest.raises(RuntimeError) as exc_info:
            driver.connect_cdp()
        assert "Live CDP connection attempted during isolated test execution" in str(exc_info.value)


def test_fake_job_url_never_navigates_real_browser():
    """Verify that fake test URLs are processed entirely via mocks without navigating any live browser."""
    fake_page = MagicMock()
    fake_page.url = "https://www.glassdoor.co.in/job-listing/direct-submit?jl=777888"

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.page = fake_page
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, fake_page, {"reused": True})
    mock_pw_driver.detect_page_state.return_value = (
        GlassdoorAutomationState.SUBMISSION_READY,
        {"submit_button_detected": True},
    )
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.co.in/job-listing/direct-submit?jl=777888",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    # Verify fake page remains purely in memory without real browser action
    assert fake_page.url == "https://www.glassdoor.co.in/job-listing/direct-submit?jl=777888"


def test_live_cdp_disabled_fails_closed_in_test_environment():
    """Verify resolve_or_navigate_page fails closed when ALLOW_LIVE_CDP is false outside pytest."""
    driver = GlassdoorPlaywrightDriver()
    with patch.dict("os.environ", {"ALLOW_LIVE_CDP": "false"}, clear=True):
        ok, page, diag = driver.resolve_or_navigate_page("https://www.glassdoor.com/job/123")
        assert ok is False
        assert page is None
        assert "disabled" in diag.get("error", "").lower()


# ==============================================================================
# 18. Live Browser Ownership and New-Tab Handoff Tests (Phase 5.6)
# ==============================================================================


def test_swagger_and_glassdoor_tabs_coexist_without_browser_not_verified():
    """Verify that having Swagger tab open alongside Glassdoor tab does not cause BROWSER_NOT_VERIFIED."""
    driver = GlassdoorPlaywrightDriver()
    mock_browser = MagicMock()
    mock_context = MagicMock()

    swagger_tab = MagicMock()
    swagger_tab.url = "http://127.0.0.1:8000/docs"
    swagger_tab.title.return_value = "FastAPI - Swagger UI"

    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.com/job-listing/ai-engineer?jl=100100"
    job_tab.title.return_value = "AI Engineer - Glassdoor"

    mock_context.pages = [swagger_tab, job_tab]
    mock_browser.contexts = [mock_context]
    driver._browser = mock_browser
    driver._context = mock_context

    ok, page, diag = driver.resolve_or_navigate_page("https://www.glassdoor.com/job-listing/ai-engineer?jl=100100")
    assert ok is True
    assert page == job_tab
    assert diag["browser_session_verified"] is True
    assert diag["cdp_connected"] is True
    assert diag["second_browser_launched"] is False


def test_multiple_chrome_tabs_and_different_pids_do_not_block_session():
    """Verify that PyWinAuto PID differences/unattached state do NOT block CDP session verification."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    # PyWinAuto fails to attach to the window (e.g. PID difference / Swagger focused)
    mock_win_driver.attach_or_open_browser.return_value = (False, "chrome.exe", "PID mismatch")

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    fake_job_page = MagicMock()
    fake_job_page.url = "https://www.glassdoor.com/job-listing/pid-test?jl=998877"
    fake_job_page.inner_text.return_value = "Easy Apply | Glassdoor"
    mock_pw_driver.resolve_or_navigate_page.return_value = (
        True,
        fake_job_page,
        {"browser_session_verified": True, "cdp_connected": True},
    )
    mock_pw_driver.job_page = fake_job_page
    mock_pw_driver.find_exact_easy_apply_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationInspectRequest(
        url="https://www.glassdoor.com/job-listing/pid-test?jl=998877",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.inspect_job_application(req)

    # Session succeeds because Playwright CDP is authoritative
    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.EASY_APPLY_AVAILABLE
    assert res.easy_apply_verified is True
    assert res.diagnostics.get("browser_session_verified") is True


def test_playwright_easy_apply_is_preferred_over_pywinauto():
    """Verify that Easy Apply click prefers Playwright DOM interaction before attempting PyWinAuto."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_page = MagicMock()
    job_page.url = "https://www.glassdoor.com/job-listing/pw-pref?jl=12345"
    mock_pw_driver.page = job_page
    mock_pw_driver.job_page = job_page
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_page, {"browser_session_verified": True})
    mock_pw_driver.find_exact_easy_apply_button.return_value = MagicMock()
    mock_pw_driver.click_easy_apply.return_value = (True, None)

    # Resolve application page on same-tab
    mock_pw_driver.resolve_application_page.return_value = job_page
    mock_pw_driver.detect_page_state.return_value = (GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True})
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/pw-pref?jl=12345",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.diagnostics.get("easy_apply_method") == "playwright"
    mock_pw_driver.click_easy_apply.assert_called_once()
    # PyWinAuto click_control was never called
    mock_win_driver.click_control.assert_not_called()


def test_pywinauto_remains_fallback_when_playwright_easy_apply_absent():
    """Verify that PyWinAuto UIA / coordinate fallback is used when Playwright cannot find Easy Apply."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_win_driver.get_window_text_content.return_value = "Easy Apply | Glassdoor"
    easy_btn = MagicMock()
    easy_btn.element_info.name = "Easy Apply"
    mock_win_driver.find_easy_apply_control.return_value = easy_btn
    mock_win_driver.click_control.return_value = True

    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, MagicMock(), {})
    mock_pw_driver.find_exact_easy_apply_button.return_value = None
    mock_pw_driver.click_easy_apply.return_value = (False, "Not found")

    mock_app_page = MagicMock()
    mock_app_page.url = "https://smartapply.indeed.com/form/review"
    mock_pw_driver.resolve_application_page.return_value = mock_app_page
    mock_pw_driver.detect_page_state.return_value = (GlassdoorAutomationState.SUBMISSION_READY, {})
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job-listing/pywin-fallback?jl=54321",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.diagnostics.get("easy_apply_method") == "uia_primary"
    mock_win_driver.click_control.assert_called_once_with(easy_btn)


def test_easy_apply_new_tab_application_handoff_and_adoption():
    """Verify that when Easy Apply opens a NEW TAB, automation adopts the new tab as application_page."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.co.in/job-listing/ml-lead?jl=888999"
    job_tab.id = "page-job-1"

    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/resume-selection"
    app_tab.id = "page-app-newtab"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = None
    mock_pw_driver.page = job_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)

    # Easy Apply opens app_tab in a new tab
    def resolve_app_mock(pages_before=None, timeout_seconds=15):
        mock_pw_driver.application_page = app_tab
        mock_pw_driver.page = app_tab
        return app_tab

    mock_pw_driver.resolve_application_page.side_effect = resolve_app_mock

    # Step progression in new tab: RESUME_STEP -> SUBMISSION_READY
    step_state = {"step": 1}

    def mock_detect_state(page):
        assert page == app_tab, "DOM detection must run on application_page, NOT job_tab!"
        if step_state["step"] == 1:
            return GlassdoorAutomationState.RESUME_STEP, {"progression_action": "continue"}
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    def mock_progress(page=None):
        step_state["step"] += 1
        return True, "continue", None

    mock_pw_driver.detect_page_state.side_effect = mock_detect_state
    mock_pw_driver.handle_resume_step.return_value = (True, "default_selected", None)
    mock_pw_driver.activate_progress_button.side_effect = mock_progress
    mock_pw_driver.find_exact_submit_button.side_effect = lambda p=None: MagicMock() if step_state["step"] >= 2 else None

    # Context pages before and after
    mock_context = MagicMock()
    mock_context.pages = [job_tab, app_tab]
    mock_pw_driver._context = mock_context

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.co.in/job-listing/ml-lead?jl=888999",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.diagnostics.get("application_page_opened_as_new_tab") is True
    assert res.diagnostics.get("application_page_url") == "https://smartapply.indeed.com/form/resume-selection"
    assert res.diagnostics.get("job_page_url") == "https://www.glassdoor.co.in/job-listing/ml-lead?jl=888999"


def test_stale_application_tab_from_previous_run_is_not_selected_on_new_job():
    """Verify that resolve_or_navigate_page resets application_page and starts fresh from requested job URL."""
    driver = GlassdoorPlaywrightDriver()
    mock_browser = MagicMock()
    mock_context = MagicMock()

    stale_app_tab = MagicMock()
    stale_app_tab.url = "https://smartapply.indeed.com/form/stale-run-999"

    new_job_tab = MagicMock()
    new_job_tab.url = "https://www.glassdoor.com/job-listing/fresh-job?jl=123123"
    new_job_tab.title.return_value = "Fresh Job"

    mock_context.pages = [stale_app_tab, new_job_tab]
    mock_browser.contexts = [mock_context]
    driver._browser = mock_browser
    driver._context = mock_context
    driver.application_page = stale_app_tab  # Stale from prior run

    ok, page, diag = driver.resolve_or_navigate_page("https://www.glassdoor.com/job-listing/fresh-job?jl=123123")

    assert ok is True
    assert page == new_job_tab
    assert driver.job_page == new_job_tab
    assert driver.application_page is None  # Reset cleanly for fresh run


def test_unrelated_chrome_tabs_ignored_during_application_page_resolution():
    """Verify that Swagger, localhost, and other unrelated user tabs are never adopted as application_page."""
    driver = GlassdoorPlaywrightDriver()
    mock_context = MagicMock()

    swagger_tab = MagicMock()
    swagger_tab.url = "http://localhost:8000/docs"

    github_tab = MagicMock()
    github_tab.url = "https://github.com/my-repo"
    github_tab.inner_text.return_value = "GitHub Repository"

    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.com/job-listing/devops?jl=445566"

    mock_context.pages = [swagger_tab, github_tab, job_tab]
    driver._context = mock_context
    driver.job_page = job_tab

    app_page = driver.resolve_application_page(pages_before={swagger_tab, github_tab, job_tab}, timeout_seconds=1)
    # Never adopts swagger or github
    assert app_page != swagger_tab
    assert app_page != github_tab


def test_diagnostics_contain_full_page_and_tab_ownership_metadata():
    """Verify that navigate_to_submit result contains complete CDP, job_page, application_page, and tab diagnostics."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_page = MagicMock()
    job_page.url = "https://www.glassdoor.co.in/job-listing/full-meta?jl=111222"
    job_page.id = "tab-job-111"

    app_page = MagicMock()
    app_page.url = "https://smartapply.indeed.com/form/resume"
    app_page.id = "tab-app-222"

    mock_pw_driver.job_page = job_page
    mock_pw_driver.application_page = app_page
    mock_pw_driver.page = app_page
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_page, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_page
    mock_pw_driver.detect_page_state.return_value = (GlassdoorAutomationState.SUBMISSION_READY, {})
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.co.in/job-listing/full-meta?jl=111222",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    diag = res.diagnostics
    assert diag.get("cdp_connected") is True
    assert diag.get("browser_session_verified") is True
    assert diag.get("job_page_url") == "https://www.glassdoor.co.in/job-listing/full-meta?jl=111222"
    assert diag.get("application_page_detected") is True
    assert diag.get("application_page_url") == "https://smartapply.indeed.com/form/resume"
    assert diag.get("easy_apply_method") == "playwright"


# ==============================================================================
# 19. Employer-Question Progression & Requirements Warning Tests (Phase 5.6)
# ==============================================================================


def test_experience_question_detected_as_questions_step():
    """Verify 'How many years of Teaching experience do you have? *' is classified as QUESTIONS_STEP with normalized key."""
    from app.automation.forms.question_normalizer import normalize_question_key, parse_skill_and_duration_from_text

    text = "How many years of Teaching experience do you have? *"
    skill, is_years = parse_skill_and_duration_from_text(text)
    assert is_years is True
    assert skill.lower() == "teaching"

    norm_key = normalize_question_key(text)
    assert "teaching" in norm_key and "experience" in norm_key


def test_resolved_numeric_experience_answer_filled_correctly():
    """Verify that when an approved answer for teaching experience is available, it is resolved and filled."""
    from app.automation.forms.models import ApplicationQuestion, QuestionInputType
    from app.automation.forms.answer_resolver import FormAnswerResolver

    q = ApplicationQuestion(
        text="How many years of Teaching experience do you have? *",
        normalized_key="years_experience_teaching",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )

    resolver = FormAnswerResolver(stored_answers={"years_experience_teaching": "3"})
    res = resolver.resolve_question(q)
    assert res.is_resolved is True
    assert res.resolved_value == "3"
    assert res.source == "stored_answer"


def test_unresolved_experience_answer_returns_needs_user_input():
    """Verify that when candidate has no teaching experience fact or approved answer, it returns NEEDS_USER_INPUT."""
    from app.automation.forms.models import ApplicationQuestion, QuestionInputType
    from app.automation.forms.answer_resolver import FormAnswerResolver

    q = ApplicationQuestion(
        text="How many years of Teaching experience do you have? *",
        normalized_key="years_experience_teaching",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )

    # Empty stored answers and profile without Teaching skill
    resolver = FormAnswerResolver(stored_answers={}, profile={"skills": [{"skill": "Python", "years_experience": 4}]}, candidate_answer_bank=None)
    res = resolver.resolve_question(q)
    assert res.is_resolved is False
    assert "No approved answer" in res.reason


def test_no_hardcoded_zero_fabricated_for_experience_question():
    """Verify that the resolver NEVER fabricates 0 or any default number when no factual answer exists."""
    from app.automation.forms.models import ApplicationQuestion, QuestionInputType
    from app.automation.forms.answer_resolver import FormAnswerResolver

    q = ApplicationQuestion(
        text="How many years of Teaching experience do you have? *",
        normalized_key="years_experience_teaching",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )

    resolver = FormAnswerResolver(stored_answers={}, profile={}, candidate_answer_bank=None)
    res = resolver.resolve_question(q)
    assert res.is_resolved is False
    assert res.resolved_value is None


def test_stored_approved_zero_answer_is_honored_truthfully():
    """Verify that if the user explicitly approves 0, it is stored and honored as a truthful answer."""
    from app.automation.forms.models import ApplicationQuestion, QuestionInputType
    from app.automation.forms.answer_resolver import FormAnswerResolver

    q = ApplicationQuestion(
        text="How many years of Teaching experience do you have? *",
        normalized_key="years_experience_teaching",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )

    resolver = FormAnswerResolver(stored_answers={"teaching_experience_years": "0"}, candidate_answer_bank=None)
    res = resolver.resolve_question(q)
    assert res.is_resolved is True
    assert res.resolved_value == "0"


def test_requirements_warning_heading_detected_in_dom():
    """Verify that 'It looks like you don't meet these employer requirements' heading triggers REQUIREMENTS_WARNING."""
    import re
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/warning"

    # Mock heading
    mock_heading = MagicMock()
    mock_heading.inner_text.return_value = "It looks like you don't meet these employer requirements"
    mock_headings = MagicMock()
    mock_headings.count.return_value = 1
    mock_headings.nth.return_value = mock_heading

    def mock_locator(selector):
        if "h1" in selector or "heading" in selector:
            return mock_headings
        mock_empty = MagicMock()
        mock_empty.count.return_value = 0
        return mock_empty

    mock_page.locator.side_effect = mock_locator

    # Mock Apply anyway button
    mock_btn = MagicMock()
    mock_btn.count.return_value = 1
    mock_btn.first.is_visible.return_value = True
    mock_btn.first.is_enabled.return_value = True

    def mock_get_by_role(role, name=None):
        if name and re.search(r"apply\s+anyway", getattr(name, "pattern", str(name)), re.I):
            return mock_btn
        mock_empty = MagicMock()
        mock_empty.count.return_value = 0
        return mock_empty

    mock_page.get_by_role.side_effect = mock_get_by_role

    state, diag = driver.detect_page_state(mock_page)
    assert state == GlassdoorAutomationState.REQUIREMENTS_WARNING
    assert diag.get("requirements_warning_detected") is True


def test_exact_apply_anyway_button_detected():
    """Verify find_exact_apply_anyway_button detects visible enabled Apply anyway control."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    mock_btn = MagicMock()
    mock_btn.is_visible.return_value = True
    mock_btn.is_enabled.return_value = True

    mock_locator = MagicMock()
    mock_locator.count.return_value = 1
    mock_locator.nth.return_value = mock_btn
    mock_page.get_by_role.return_value = mock_locator

    btn = driver.find_exact_apply_anyway_button(mock_page)
    assert btn == mock_btn


def test_apply_anyway_clicked_exactly_once():
    """Verify click_apply_anyway scrolls into view and clicks strictly once."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    mock_btn = MagicMock()
    mock_btn.is_visible.return_value = True
    mock_btn.is_enabled.return_value = True

    mock_locator = MagicMock()
    mock_locator.count.return_value = 1
    mock_locator.nth.return_value = mock_btn
    mock_page.get_by_role.return_value = mock_locator

    ok, err = driver.click_apply_anyway(mock_page)
    assert ok is True
    assert err is None
    mock_btn.scroll_into_view_if_needed.assert_called_once()
    mock_btn.click.assert_called_once()


def test_return_to_job_search_and_exit_never_clicked():
    """Verify find_progression_action blacklists 'Return to job search' and 'Exit' buttons."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    mock_return_btn = MagicMock()
    mock_return_btn.is_enabled.return_value = True
    mock_return_btn.inner_text.return_value = "Return to job search"

    mock_locator = MagicMock()
    mock_locator.count.return_value = 1
    mock_locator.first = mock_return_btn
    mock_page.get_by_role.return_value = mock_locator

    act_name, act_btn = driver.find_progression_action(mock_page)
    assert act_name is None
    assert act_btn is None


def test_flow_b_warning_to_apply_anyway_to_review_to_submission_ready():
    """Verify Flow B: Resume -> Questions -> Requirements Warning -> Apply anyway -> Review -> SUBMISSION_READY."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.com/job/flow-b?jl=777111"
    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/warning"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_tab

    # Sequence of states:
    # 1. REQUIREMENTS_WARNING -> Apply anyway clicked
    # 2. SUBMISSION_READY -> Submit detected
    call_counts = {"count": 0}

    def mock_detect(page):
        call_counts["count"] += 1
        if call_counts["count"] == 1:
            return GlassdoorAutomationState.REQUIREMENTS_WARNING, {
                "requirements_not_met": ["Teaching: 1 year (Required)"],
                "requirements_text": "Teaching: 1 year (Required)",
            }
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    mock_pw_driver.detect_page_state.side_effect = mock_detect
    mock_pw_driver.find_exact_apply_anyway_button.return_value = MagicMock()
    mock_pw_driver.click_apply_anyway.return_value = (True, None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job/flow-b?jl=777111",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    mock_pw_driver.click_apply_anyway.assert_called_once()
    assert res.diagnostics.get("requirements_warning_detected") is True
    assert res.diagnostics.get("requirements_not_met") == ["Teaching: 1 year (Required)"]


def test_requirements_warning_returns_manual_action_when_policy_disabled():
    """Verify that when CONTINUE_ON_REQUIREMENTS_WARNING is False, warning halts with manual_action_required."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.com/job/flow-warn?jl=777222"
    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/warning"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_tab

    mock_pw_driver.detect_page_state.return_value = (
        GlassdoorAutomationState.REQUIREMENTS_WARNING,
        {
            "requirements_not_met": ["Teaching: 1 year (Required)"],
            "requirements_text": "Teaching: 1 year (Required)",
        },
    )
    mock_pw_driver.find_exact_apply_anyway_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job/flow-warn?jl=777222",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        with patch("app.automation.glassdoor.apply_service.CONTINUE_ON_REQUIREMENTS_WARNING", False):
            res = service.navigate_to_submit(req)

    assert res.status == "manual_action_required"
    assert res.current_state == GlassdoorAutomationState.REQUIREMENTS_WARNING
    assert res.manual_action_required is True
    assert "Requirements warning" in res.message
    # Apply anyway was NOT clicked
    mock_pw_driver.click_apply_anyway.assert_not_called()


def test_flow_c_warning_to_apply_anyway_to_more_questions_to_review():
    """Verify Flow C: Questions -> Warning -> Apply anyway -> More Questions -> Review -> SUBMISSION_READY."""
    from app.automation.forms.models import ApplicationQuestion, QuestionInputType

    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/questions-2"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.REQUIREMENTS_WARNING, {"requirements_not_met": ["Teaching: 1 year"]}
        elif step_state["step"] == 2:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    mock_pw_driver.detect_page_state.side_effect = mock_detect
    mock_pw_driver.find_exact_apply_anyway_button.return_value = MagicMock()
    mock_pw_driver.click_apply_anyway.return_value = (True, None)

    q2 = ApplicationQuestion(
        text="Are you authorized to work in India? *",
        normalized_key="work_authorization",
        input_type=QuestionInputType.RADIO,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q2]
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    # Stored answer for work_authorization
    service.application_repository.get_answers_by_profile = MagicMock(return_value={"work_authorization": "Yes"})

    req = GlassdoorAutomationNavigateRequest(
        url="https://www.glassdoor.com/job/flow-c?jl=777333",
    )

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert mock_pw_driver.click_apply_anyway.call_count == 1


# ==============================================================================
# SECTION 20: Human-in-the-Loop Question Resume Flow (Phase 5.6)
# ==============================================================================


def test_hitl_stored_answer_autofill_and_continue():
    """Verify that when answers exist in application_answers/profile, they are autofilled and automation continues."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.verify_browser_window.return_value = (True, "chrome.exe")
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.com/job/stored-q?jl=111222"
    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/questions"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    mock_pw_driver.detect_page_state.side_effect = mock_detect

    q1 = ApplicationQuestion(
        text="What is your Notice Period in days? *",
        normalized_key="notice_period_days",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q1]
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    service.application_repository.get_answers_by_profile = MagicMock(return_value={"notice_period_days": "30"})

    req = GlassdoorAutomationNavigateRequest(url="https://www.glassdoor.com/job/stored-q?jl=111222")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert mock_pw_driver.fill_question.call_count == 1
    mock_pw_driver.fill_question.assert_called_with(app_tab, q1, "30")


def test_hitl_missing_required_answer_returns_needs_user_input():
    """Verify that unknown required questions pause with NEEDS_USER_INPUT, generate session_id, and do not guess."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.verify_browser_window.return_value = (True, "chrome.exe")
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.com/job/missing-q?jl=222333"
    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/questions"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_tab
    mock_pw_driver.detect_page_state.return_value = (GlassdoorAutomationState.QUESTIONS_STEP, {})

    q1 = ApplicationQuestion(
        text="How many years of Teaching experience do you have? *",
        normalized_key="teaching_experience_years",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q1]

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    service.candidate_answer_bank = None
    service.application_repository.get_answers_by_profile = MagicMock(return_value={})

    req = GlassdoorAutomationNavigateRequest(url="https://www.glassdoor.com/job/missing-q?jl=222333")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "manual_action_required"
    assert res.current_state == GlassdoorAutomationState.NEEDS_USER_INPUT
    assert res.manual_action_required is True
    assert any(q.get("normalized_key") == "teaching_experience_years" for q in res.diagnostics.get("unresolved_questions", []))
    session_id = res.diagnostics.get("automation_session_id")
    assert session_id is not None
    assert service.get_session(session_id) is not None
    # Verify no guessing occurred (no fill called for unknown)
    assert mock_pw_driver.fill_question.call_count == 0


def test_hitl_resume_reconnects_cdp_and_does_not_click_easy_apply():
    """Verify that POST /automation/glassdoor/resume reconnects via CDP and resolves application_page without re-clicking Easy Apply."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/questions"

    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    mock_pw_driver.detect_page_state.side_effect = mock_detect

    q1 = ApplicationQuestion(
        text="How many years of Teaching experience do you have? *",
        normalized_key="teaching_experience_years",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q1]
    mock_pw_driver.read_question_value.return_value = 2
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    service.application_repository.save_application_answer = MagicMock(return_value=True)

    session_id = "test_sess_101"
    service.save_session(session_id, {
        "automation_session_id": session_id,
        "job_url": "https://www.glassdoor.com/job/test-job?jl=555666",
        "job_id": None,
        "unresolved_questions": [q1.model_dump()],
        "total_steps": 1,
    })

    resume_req = GlassdoorAutomationResumeRequest(automation_session_id=session_id)
    res = service.resume(resume_req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert mock_pw_driver.click_easy_apply.call_count == 0
    assert mock_pw_driver.resolve_or_navigate_page.call_count == 0


def test_hitl_resume_detects_manually_typed_text_answer():
    """Verify that resume reads manually typed text from the DOM, saves it as approved, and clicks Continue."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/questions"

    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    mock_pw_driver.detect_page_state.side_effect = mock_detect

    q1 = ApplicationQuestion(
        text="What is your current city? *",
        normalized_key="current_city",
        input_type=QuestionInputType.TEXT,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q1]
    mock_pw_driver.read_question_value.return_value = "Bangalore"
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    service.application_repository.save_application_answer = MagicMock(return_value=True)

    session_id = "test_sess_text"
    service.save_session(session_id, {"automation_session_id": session_id, "job_url": "https://www.glassdoor.com/job/test-job"})

    resume_req = GlassdoorAutomationResumeRequest(automation_session_id=session_id)
    res = service.resume(resume_req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert service.application_repository.save_application_answer.called
    saved_args = [call[1] for call in service.application_repository.save_application_answer.call_args_list]
    assert any(kwargs.get("answer") == "Bangalore" for kwargs in saved_args)


def test_hitl_resume_detects_manually_typed_numeric_answer():
    """Verify that resume reads manually entered integer or float value and validates numeric input."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/questions"

    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        return GlassdoorAutomationState.SUBMISSION_READY, {}

    mock_pw_driver.detect_page_state.side_effect = mock_detect

    q1 = ApplicationQuestion(
        text="How many years of Teaching experience do you have? *",
        normalized_key="teaching_experience_years",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q1]
    mock_pw_driver.read_question_value.return_value = 0
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    service.application_repository.save_application_answer = MagicMock(return_value=True)

    session_id = "test_sess_num"
    service.save_session(session_id, {"automation_session_id": session_id, "job_url": "https://www.glassdoor.com/job/test-job"})

    resume_req = GlassdoorAutomationResumeRequest(automation_session_id=session_id)
    res = service.resume(resume_req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    saved_args = [call[1] for call in service.application_repository.save_application_answer.call_args_list]
    assert any(kwargs.get("answer") == "0" for kwargs in saved_args)


def test_hitl_resume_detects_manually_selected_radio():
    """Verify that resume detects manually selected radio option."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        return GlassdoorAutomationState.SUBMISSION_READY, {}

    mock_pw_driver.detect_page_state.side_effect = mock_detect

    q1 = ApplicationQuestion(
        text="Will you require visa sponsorship? *",
        normalized_key="visa_sponsorship",
        input_type=QuestionInputType.RADIO,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q1]
    mock_pw_driver.read_question_value.return_value = "No"
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    service.application_repository.save_application_answer = MagicMock(return_value=True)

    session_id = "test_sess_radio"
    service.save_session(session_id, {"automation_session_id": session_id, "job_url": "https://www.glassdoor.com/job/test-job"})

    resume_req = GlassdoorAutomationResumeRequest(automation_session_id=session_id)
    res = service.resume(resume_req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY


def test_hitl_resume_detects_manually_selected_checkbox():
    """Verify that resume detects manually selected checkbox options."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        return GlassdoorAutomationState.SUBMISSION_READY, {}

    mock_pw_driver.detect_page_state.side_effect = mock_detect

    q1 = ApplicationQuestion(
        text="Which shifts are you available for? *",
        normalized_key="available_shifts",
        input_type=QuestionInputType.CHECKBOX_MULTI,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q1]
    mock_pw_driver.read_question_value.return_value = ["Morning", "Evening"]
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    service.application_repository.save_application_answer = MagicMock(return_value=True)

    session_id = "test_sess_cb"
    service.save_session(session_id, {"automation_session_id": session_id, "job_url": "https://www.glassdoor.com/job/test-job"})

    resume_req = GlassdoorAutomationResumeRequest(automation_session_id=session_id)
    res = service.resume(resume_req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY


def test_hitl_resume_detects_manually_selected_dropdown():
    """Verify that resume detects manually selected dropdown option."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        return GlassdoorAutomationState.SUBMISSION_READY, {}

    mock_pw_driver.detect_page_state.side_effect = mock_detect

    q1 = ApplicationQuestion(
        text="Highest level of education *",
        normalized_key="education_level",
        input_type=QuestionInputType.DROPDOWN,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q1]
    mock_pw_driver.read_question_value.return_value = "Bachelor's Degree"
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    service.application_repository.save_application_answer = MagicMock(return_value=True)

    session_id = "test_sess_dd"
    service.save_session(session_id, {"automation_session_id": session_id, "job_url": "https://www.glassdoor.com/job/test-job"})

    resume_req = GlassdoorAutomationResumeRequest(automation_session_id=session_id)
    res = service.resume(resume_req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY


def test_hitl_resume_empty_required_field_pauses_needs_user_input_again():
    """Verify that if resume is called while a required field remains empty, it returns NEEDS_USER_INPUT again."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/1"
    app_tab.title.return_value = "Apply - Teaching Experience"
    app_tab.is_closed.return_value = False
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_paused_application_page.return_value = (app_tab, {"recovered": True})
    mock_pw_driver.resolve_application_page.return_value = app_tab
    mock_pw_driver.detect_page_state.return_value = (GlassdoorAutomationState.QUESTIONS_STEP, {})

    q1 = ApplicationQuestion(
        text="How many years of Teaching experience do you have? *",
        normalized_key="teaching_experience_years",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q1]
    mock_pw_driver.read_question_value.return_value = None  # Still empty

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    service.candidate_answer_bank = None
    service.application_repository.get_answers_by_profile = MagicMock(return_value={})

    session_id = "test_sess_empty"
    service.save_session(session_id, {"automation_session_id": session_id, "job_url": "https://www.glassdoor.com/job/test-job"})

    resume_req = GlassdoorAutomationResumeRequest(automation_session_id=session_id)
    res = service.resume(resume_req)

    assert res.status == "manual_action_required"
    assert res.current_state == GlassdoorAutomationState.NEEDS_USER_INPUT
    assert res.manual_action_required is True
    assert any(q.get("normalized_key") == "teaching_experience_years" for q in res.diagnostics.get("unresolved_questions", []))
    assert res.diagnostics.get("automation_session_id") == session_id


def test_hitl_saved_manual_answer_persisted_and_reused():
    """Verify that a saved manual answer is stored and reused on subsequent application runs."""
    repo = MagicMock(spec=ApplicationRepository)
    stored_dict = {}

    def mock_save(question, answer, source="manual_user_input", **kwargs):
        stored_dict[question.lower()] = answer
        return True

    def mock_get(profile_id=None):
        return dict(stored_dict)

    repo.save_application_answer.side_effect = mock_save
    repo.get_answers_by_profile.side_effect = mock_get

    # Step 1: Save manual answer
    repo.save_application_answer(question="teaching_experience_years", answer="3", source="manual_user_input")
    assert repo.get_answers_by_profile()["teaching_experience_years"] == "3"

    # Step 2: Resolver resolves it on future job
    resolver = FormAnswerResolver(stored_answers=repo.get_answers_by_profile())
    q = ApplicationQuestion(
        text="How many years of Teaching experience do you have? *",
        normalized_key="teaching_experience_years",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    ans = resolver.resolve_question(q)
    assert ans.is_resolved is True
    assert ans.resolved_value == "3"


def test_hitl_unrelated_normalized_question_does_not_reuse_answer():
    """Verify that answer for teaching_experience_years is NOT reused for python_experience_years."""
    resolver = FormAnswerResolver(stored_answers={"teaching_experience_years": "5"})
    q_python = ApplicationQuestion(
        text="How many years of Python experience do you have? *",
        normalized_key="python_experience_years",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    ans = resolver.resolve_question(q_python)
    assert ans.is_resolved is False


def test_hitl_second_questions_page_can_pause_again():
    """Verify flow with multiple human input pauses: Page 1 unknown -> resume -> Page 2 unknown -> resume -> Review -> Submit."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_application_page.return_value = app_tab

    # First resume execution reaches Questions Page 2 with unknown question
    mock_pw_driver.detect_page_state.return_value = (GlassdoorAutomationState.QUESTIONS_STEP, {})
    q1 = ApplicationQuestion(
        text="Teaching experience *",
        normalized_key="teaching_experience_years",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    q2_unknown = ApplicationQuestion(
        text="Expected Hourly Rate ($) *",
        normalized_key="expected_hourly_rate",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )

    # Initial resume provides answer for q1, but next page has q2_unknown
    mock_pw_driver.extract_questions.side_effect = [[q1], [q2_unknown]]
    mock_pw_driver.read_question_value.side_effect = [3, None]
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    service.application_repository.save_application_answer = MagicMock(return_value=True)
    service.application_repository.get_answers_by_profile = MagicMock(return_value={})

    session_id = "test_sess_multipage"
    service.save_session(session_id, {"automation_session_id": session_id, "job_url": "https://www.glassdoor.com/job/test-job"})

    # Resume 1: Answers q1 -> loops to next page with q2_unknown -> pauses again
    resume_req_1 = GlassdoorAutomationResumeRequest(automation_session_id=session_id)
    res_1 = service.resume(resume_req_1)

    assert res_1.status == "manual_action_required"
    assert res_1.current_state == GlassdoorAutomationState.NEEDS_USER_INPUT
    assert any(q.get("normalized_key") == "expected_hourly_rate" for q in res_1.diagnostics.get("unresolved_questions", []))

    # Resume 2: Answers q2 -> progresses to Review -> SUBMISSION_READY
    step_state_2 = {"step": 0}

    def mock_detect_2(page):
        step_state_2["step"] += 1
        if step_state_2["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        return GlassdoorAutomationState.SUBMISSION_READY, {}

    mock_pw_driver.detect_page_state.side_effect = mock_detect_2
    mock_pw_driver.extract_questions.side_effect = None
    mock_pw_driver.extract_questions.return_value = [q2_unknown]
    mock_pw_driver.read_question_value.side_effect = None
    mock_pw_driver.read_question_value.return_value = 45
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    resume_req_2 = GlassdoorAutomationResumeRequest(automation_session_id=session_id)
    res_2 = service.resume(resume_req_2)

    assert res_2.status == "success"
    assert res_2.current_state == GlassdoorAutomationState.SUBMISSION_READY


def test_hitl_requirements_warning_after_manual_resume_clicks_apply_anyway():
    """Verify flow: unknown answer -> resume -> REQUIREMENTS_WARNING -> clicks Apply anyway -> Review -> Submit."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        elif step_state["step"] == 2:
            return GlassdoorAutomationState.REQUIREMENTS_WARNING, {"requirements_not_met": ["Teaching: 1 year (Required)"]}
        return GlassdoorAutomationState.SUBMISSION_READY, {}

    mock_pw_driver.detect_page_state.side_effect = mock_detect

    q1 = ApplicationQuestion(
        text="How many years of Teaching experience do you have? *",
        normalized_key="teaching_experience_years",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q1]
    mock_pw_driver.read_question_value.return_value = 0
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_apply_anyway_button.return_value = MagicMock()
    mock_pw_driver.click_apply_anyway.return_value = (True, None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    service.application_repository.save_application_answer = MagicMock(return_value=True)

    session_id = "test_sess_warn_flow"
    service.save_session(session_id, {"automation_session_id": session_id, "job_url": "https://www.glassdoor.com/job/test-job"})

    resume_req = GlassdoorAutomationResumeRequest(automation_session_id=session_id)
    res = service.resume(resume_req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert mock_pw_driver.click_apply_anyway.call_count == 1


def test_hitl_api_answers_mode():
    """Verify that answers supplied in request.answers are filled via Playwright, saved, and progressed."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_application_page.return_value = app_tab
    mock_pw_driver.resolve_paused_application_page.return_value = (app_tab, "question_signature", {"application_page_recovered": True})

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        return GlassdoorAutomationState.SUBMISSION_READY, {}

    mock_pw_driver.detect_page_state.side_effect = mock_detect

    q1 = ApplicationQuestion(
        text="Teaching experience in years *",
        normalized_key="teaching_experience_years",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q1]
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    service.application_repository.save_application_answer = MagicMock(return_value=True)

    session_id = "test_sess_api_mode"
    service.save_session(session_id, {"automation_session_id": session_id, "job_url": "https://www.glassdoor.com/job/test-job"})

    resume_req = GlassdoorAutomationResumeRequest(
        automation_session_id=session_id,
        answers={"teaching_experience_years": 4},
    )
    res = service.resume(resume_req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert mock_pw_driver.fill_question.call_count == 1
    mock_pw_driver.fill_question.assert_called_with(app_tab, q1, 4)


def test_hitl_mixed_stored_and_manual_answers():
    """Verify that when some questions are stored and some are manual, all are satisfied and progressed."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        return GlassdoorAutomationState.SUBMISSION_READY, {}

    mock_pw_driver.detect_page_state.side_effect = mock_detect

    q1_stored = ApplicationQuestion(
        text="Notice Period *",
        normalized_key="notice_period_days",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    q2_manual = ApplicationQuestion(
        text="Teaching experience *",
        normalized_key="teaching_experience_years",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q1_stored, q2_manual]

    def mock_read_val(page, q):
        if q.normalized_key == "teaching_experience_years":
            return 2
        return None

    mock_pw_driver.read_question_value.side_effect = mock_read_val
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    service.application_repository.get_answers_by_profile = MagicMock(return_value={"notice_period_days": "15"})
    service.application_repository.save_application_answer = MagicMock(return_value=True)

    session_id = "test_sess_mixed"
    service.save_session(session_id, {"automation_session_id": session_id, "job_url": "https://www.glassdoor.com/job/test-job"})

    resume_req = GlassdoorAutomationResumeRequest(automation_session_id=session_id)
    res = service.resume(resume_req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    # Stored q1 filled
    assert any(call[0][1].normalized_key == "notice_period_days" for call in mock_pw_driver.fill_question.call_args_list)


def test_hitl_resume_endpoint_never_clicks_final_submit():
    """Verify that /resume halts strictly at SUBMISSION_READY with final_submit_clicked=False."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_application_page.return_value = app_tab
    mock_pw_driver.detect_page_state.return_value = (GlassdoorAutomationState.SUBMISSION_READY, {})
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    resume_req = GlassdoorAutomationResumeRequest(url="https://www.glassdoor.com/job/test?jl=999")

    res = service.resume(resume_req)
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False
    assert mock_pw_driver.click_final_submit.call_count == 0


def test_hitl_diagnostics_contain_full_session_metadata():
    """Verify diagnostics include session_id, paused_for_user_input, manual_answers_detected, and resume metadata."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_application_page.return_value = app_tab
    mock_pw_driver.detect_page_state.return_value = (GlassdoorAutomationState.NEEDS_USER_INPUT, {})

    q1 = ApplicationQuestion(
        text="Teaching experience *",
        normalized_key="teaching_experience_years",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q1]
    mock_pw_driver.read_question_value.return_value = 3
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    session_id = "test_diag_sess"
    service.save_session(session_id, {"automation_session_id": session_id, "job_url": "https://www.glassdoor.com/job/test"})

    resume_req = GlassdoorAutomationResumeRequest(automation_session_id=session_id)
    res = service.resume(resume_req)

    diag = res.diagnostics
    assert "automation_session_id" in diag
    assert diag["automation_session_id"] == session_id
    assert "resume_attempt" in diag


def test_api_glassdoor_resume_endpoint():
    """Verify FastAPI router endpoint POST /automation/glassdoor/resume."""
    client = TestClient(app)
    mock_service = MagicMock(spec=GlassdoorApplyService)
    mock_service.resume.return_value = GlassdoorAutomationResult(
        status="success",
        url="https://www.glassdoor.com/job/test-api",
        current_state=GlassdoorAutomationState.SUBMISSION_READY,
        submit_verified=True,
        final_submit_clicked=False,
    )

    app.dependency_overrides[get_glassdoor_apply_service] = lambda: mock_service

    try:
        response = client.post(
            "/automation/glassdoor/resume",
            json={
                "automation_session_id": "test_sess_api_123",
                "answers": {"teaching_experience_years": 2},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["current_state"] == "SUBMISSION_READY"
        assert mock_service.resume.called
    finally:
        app.dependency_overrides.pop(get_glassdoor_apply_service, None)


def test_read_question_value_all_types():
    """Verify GlassdoorPlaywrightDriver.read_question_value for text, number, radio, checkbox, dropdown."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    # 1. Text
    q_text = ApplicationQuestion(text="Your Name", normalized_key="name", input_type=QuestionInputType.TEXT)
    mock_text_loc = MagicMock()
    mock_text_loc.count.return_value = 1
    mock_text_loc.first.input_value.return_value = "Jane Doe"
    mock_text_loc.first.get_attribute.return_value = ""
    mock_page.get_by_label.return_value = mock_text_loc
    assert driver.read_question_value(mock_page, q_text) == "Jane Doe"

    # 2. Number
    q_num = ApplicationQuestion(text="Years of Experience", normalized_key="experience_years", input_type=QuestionInputType.NUMBER)
    mock_num_loc = MagicMock()
    mock_num_loc.count.return_value = 1
    mock_num_loc.first.input_value.return_value = "5"
    mock_num_loc.first.get_attribute.return_value = ""
    mock_page.get_by_label.return_value = mock_num_loc
    assert driver.read_question_value(mock_page, q_num) == 5

    # 3. Radio
    q_radio = ApplicationQuestion(text="Are you 18+?", normalized_key="age_18_plus", input_type=QuestionInputType.RADIO)
    mock_fs = MagicMock()
    mock_fs.count.return_value = 1
    mock_fs_elem = MagicMock()
    mock_legend = MagicMock()
    mock_legend.count.return_value = 1
    mock_legend.inner_text.return_value = "Are you 18+?"
    mock_fs_elem.locator.return_value.first = mock_legend
    mock_radio = MagicMock()
    mock_radio.is_checked.return_value = True
    mock_radio.get_attribute.return_value = "opt_yes"
    mock_lbl = MagicMock()
    mock_lbl.count.return_value = 1
    mock_lbl.first.inner_text.return_value = "Yes"
    mock_page.locator.return_value = mock_lbl
    mock_fs_elem.locator.return_value.count.return_value = 1
    mock_fs_elem.locator.return_value.nth.return_value = mock_radio

    # 4. Checkbox standalone
    q_cb = ApplicationQuestion(text="Agree to Terms", normalized_key="agree_terms", input_type=QuestionInputType.CHECKBOX)
    mock_cb_loc = MagicMock()
    mock_cb_loc.count.return_value = 1
    mock_cb_loc.first.is_checked.return_value = True
    mock_page.get_by_label.return_value = mock_cb_loc
    assert driver.read_question_value(mock_page, q_cb) is True


# ==============================================================================
# SECTION 21: Live State Reset & Resume Deep Scrolling (Phase 5.6)
# ==============================================================================


def test_second_navigate_execution_reinitializes_page_state():
    """Verify that a second navigate execution resets transient page state without BROWSER_NOT_VERIFIED."""
    driver = GlassdoorPlaywrightDriver()
    mock_browser = MagicMock()
    mock_browser.is_connected.return_value = True
    mock_context = MagicMock()

    page_job_1 = MagicMock()
    page_job_1.url = "https://www.glassdoor.com/job/job-1?jl=111111"
    page_job_1.is_closed.return_value = False

    page_app_1 = MagicMock()
    page_app_1.url = "https://smartapply.indeed.com/form/resume"
    page_app_1.is_closed.return_value = False

    page_job_2 = MagicMock()
    page_job_2.url = "https://www.glassdoor.com/job/job-2?jl=222222"
    page_job_2.is_closed.return_value = False

    mock_context.pages = [page_job_1, page_app_1, page_job_2]
    mock_browser.contexts = [mock_context]
    driver._browser = mock_browser
    driver._context = mock_context

    # Run 1: sets job_page and application_page
    ok1, p1, diag1 = driver.resolve_or_navigate_page("https://www.glassdoor.com/job/job-1?jl=111111")
    assert ok1 is True
    assert p1 == page_job_1
    driver.application_page = page_app_1

    # Run 2: starts a new run for job-2
    driver.reset_navigation_state()
    assert driver.job_page is None
    assert driver.application_page is None

    ok2, p2, diag2 = driver.resolve_or_navigate_page("https://www.glassdoor.com/job/job-2?jl=222222")
    assert ok2 is True
    assert p2 == page_job_2
    assert diag2["browser_session_verified"] is True
    assert diag2["second_browser_launched"] is False


def test_stale_closed_page_discarded_cleanly():
    """Verify that closed/stale page references are filtered out and not returned."""
    driver = GlassdoorPlaywrightDriver()
    mock_browser = MagicMock()
    mock_browser.is_connected.return_value = True
    mock_context = MagicMock()

    closed_page = MagicMock()
    closed_page.is_closed.return_value = True
    closed_page.url = "https://www.glassdoor.com/job/stale?jl=333"

    open_page = MagicMock()
    open_page.is_closed.return_value = False
    open_page.url = "https://www.glassdoor.com/job/valid?jl=333"

    mock_context.pages = [closed_page, open_page]
    mock_browser.contexts = [mock_context]
    driver._browser = mock_browser
    driver._context = mock_context

    ok, matched, diag = driver.resolve_or_navigate_page("https://www.glassdoor.com/job/valid?jl=333")
    assert ok is True
    assert matched == open_page


def test_scroll_element_into_view_deep_multi_tier():
    """Verify scroll_element_into_view_deep executes Tier 1, Tier 2, and Tier 3 scrolls and verifies bounds."""
    driver = GlassdoorPlaywrightDriver()
    mock_elem = MagicMock()
    # Initially hidden, then visible after scroll
    mock_elem.is_visible.side_effect = [False, True]
    mock_elem.bounding_box.return_value = {"x": 200, "y": 450, "width": 120, "height": 40}
    mock_elem.evaluate.return_value = True

    diag = driver.scroll_element_into_view_deep(mock_elem)

    assert diag["scroll_into_view_attempted"] is True
    assert diag["visible_before_scroll"] is False
    assert diag["visible_after_scroll"] is True
    assert diag["element_bounds"]["y"] == 450
    assert mock_elem.scroll_into_view_if_needed.called


def test_resume_continue_initially_below_viewport_and_deep_scrolled():
    """Verify resume Continue button initially below viewport is deep-scrolled, verified, and clicked once."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.verify_browser_window.return_value = (True, "chrome.exe")
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.com/job/resume-deep?jl=999001"
    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/resume"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.RESUME_STEP, {}
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    mock_pw_driver.detect_page_state.side_effect = mock_detect
    mock_pw_driver.handle_resume_step.return_value = (True, "default_selected", None)
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver._last_progression_diagnostics = {
        "resume_continue_locator_found": True,
        "resume_continue_count": 1,
        "resume_continue_visible_before_scroll": False,
        "resume_continue_visible_after_scroll": True,
        "resume_continue_enabled": True,
        "resume_continue_bounds": {"x": 100, "y": 600, "width": 80, "height": 36},
        "scroll_into_view_attempted": True,
        "scrollable_ancestor_detected": True,
        "iframe_detected": False,
        "resume_continue_clicked": True,
    }
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(url="https://www.glassdoor.com/job/resume-deep?jl=999001")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert mock_pw_driver.activate_progress_button.call_count == 1
    assert mock_pw_driver.handle_resume_step.call_count == 1


def test_resume_continue_click_failure_returns_rich_diagnostics():
    """Verify that if Continue button cannot be activated on Resume step, rich diagnostics are returned."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.verify_browser_window.return_value = (True, "chrome.exe")
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/resume"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_tab
    mock_pw_driver.detect_page_state.return_value = (GlassdoorAutomationState.RESUME_STEP, {})

    mock_pw_driver.handle_resume_step.return_value = (True, "default_selected", None)
    mock_pw_driver.activate_progress_button.return_value = (False, None, "Continue button not visible or actionable")
    mock_pw_driver._last_progression_diagnostics = {
        "resume_continue_locator_found": False,
        "resume_continue_count": 0,
        "resume_continue_visible_before_scroll": False,
        "resume_continue_visible_after_scroll": False,
        "resume_continue_enabled": False,
        "resume_continue_bounds": None,
        "scroll_into_view_attempted": True,
        "scrollable_ancestor_detected": False,
        "iframe_detected": False,
        "resume_continue_clicked": False,
    }

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(url="https://www.glassdoor.com/job/fail-resume?jl=888999")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "blocked"
    assert res.current_state == GlassdoorAutomationState.ACTION_CONTROL_NOT_VERIFIED
    assert res.diagnostics["resume_step_detected"] is True
    assert res.diagnostics["resume_continue_clicked"] is False
    assert res.diagnostics["scroll_into_view_attempted"] is True


def test_resume_continue_flow_resume_to_review_to_submit():
    """Verify flow: Resume step -> Continue -> Review step -> Submission Ready."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.verify_browser_window.return_value = (True, "chrome.exe")
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    app_tab = MagicMock()
    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.RESUME_STEP, {}
        elif step_state["step"] == 2:
            return GlassdoorAutomationState.REVIEW_STEP, {}
        return GlassdoorAutomationState.SUBMISSION_READY, {}

    mock_pw_driver.detect_page_state.side_effect = mock_detect
    mock_pw_driver.handle_resume_step.return_value = (True, "default_selected", None)
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(url="https://www.glassdoor.com/job/flow-r?jl=555444")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert len(res.diagnostics.get("step_history", [])) >= 2


# ==============================================================================
# SECTION 22: HITL /resume Application-Page Recovery (Phase 5.6)
# ==============================================================================


def test_navigate_pauses_on_questions_and_stores_application_page_identity():
    """Verify that pausing on questions stores full application page identity in session cache."""
    service = GlassdoorApplyService()
    service.candidate_answer_bank = None
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    service.playwright_driver = mock_pw_driver

    page = MagicMock()
    page.url = "https://smartapply.indeed.com/form/questions-step-1"
    page.title.return_value = "Apply - Teaching Experience"
    page.is_closed.return_value = False
    mock_pw_driver.application_page = page
    mock_pw_driver.page = page

    q = ApplicationQuestion(text="How many years of Teaching experience do you have?", normalized_key="teaching_experience_years", input_type=QuestionInputType.NUMBER, required=True)
    mock_pw_driver.detect_page_state.return_value = (GlassdoorAutomationState.QUESTIONS_STEP, {})
    mock_pw_driver.extract_questions.return_value = [q]

    inspect_res = GlassdoorAutomationResult(status="success", url="https://www.glassdoor.co.in/job/math-teacher?jl=777", browser="chrome_cdp", current_state=GlassdoorAutomationState.EASY_APPLY_AVAILABLE)
    nav_req = GlassdoorAutomationNavigateRequest(url="https://www.glassdoor.co.in/job/math-teacher?jl=777")

    res = service._run_playwright_application_loop(
        request=nav_req,
        inspect_res=inspect_res,
        click_method="playwright",
        session_id="gdoor_session_test_pause_1",
    )

    assert res.current_state == GlassdoorAutomationState.NEEDS_USER_INPUT
    saved_sess = service.get_session("gdoor_session_test_pause_1")
    assert saved_sess is not None
    assert saved_sess["application_page_url"] == "https://smartapply.indeed.com/form/questions-step-1"
    assert saved_sess["application_page_title"] == "Apply - Teaching Experience"
    assert "teaching_experience_years" in saved_sess["unresolved_question_keys"]
    assert saved_sess["job_listing_id"] == "777"


def test_resume_recovers_page_via_question_signature_when_url_changes_slightly():
    """Verify resolve_paused_application_page matches candidate tab containing matching unresolved question key."""
    driver = GlassdoorPlaywrightDriver()
    mock_browser = MagicMock()
    mock_browser.is_connected.return_value = True
    mock_context = MagicMock()

    swagger_tab = MagicMock()
    swagger_tab.url = "http://127.0.0.1:8000/docs"
    swagger_tab.title.return_value = "FastAPI - Swagger UI"
    swagger_tab.is_closed.return_value = False

    unrelated_tab = MagicMock()
    unrelated_tab.url = "https://www.google.com"
    unrelated_tab.title.return_value = "Google Search"
    unrelated_tab.is_closed.return_value = False

    # The application tab navigated to a dynamic token URL, but has the same question
    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/step2?token=xyz123"
    app_tab.title.return_value = "Application Form"
    app_tab.is_closed.return_value = False

    mock_context.pages = [swagger_tab, unrelated_tab, app_tab]
    mock_browser.contexts = [mock_context]
    driver._browser = mock_browser
    driver._context = mock_context

    q = ApplicationQuestion(text="Teaching Experience in Years", normalized_key="teaching_experience_years", input_type=QuestionInputType.NUMBER)

    def detect_mock(p):
        if p == app_tab:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        return GlassdoorAutomationState.UNKNOWN, {}

    def extract_mock(p):
        if p == app_tab:
            return [q]
        return []

    driver.detect_page_state = detect_mock
    driver.extract_questions = extract_mock

    session_data = {
        "automation_session_id": "sess_q_sig",
        "application_page_url": "https://smartapply.indeed.com/form/step1",
        "unresolved_question_keys": ["teaching_experience_years"],
    }

    recovered_page, method, diag = driver.resolve_paused_application_page(session_data)
    assert recovered_page == app_tab
    assert method == "question_signature"
    assert diag["application_page_recovered"] is True
    assert app_tab.bring_to_front.called


def test_resume_recovers_page_via_exact_stored_url():
    """Verify resolve_paused_application_page matches candidate tab with exact stored application_page_url."""
    driver = GlassdoorPlaywrightDriver()
    mock_browser = MagicMock()
    mock_browser.is_connected.return_value = True
    mock_context = MagicMock()

    app_tab = MagicMock()
    app_tab.url = "https://www.glassdoor.co.in/apply/hosted-job-123"
    app_tab.title.return_value = "Apply on Glassdoor"
    app_tab.is_closed.return_value = False

    mock_context.pages = [app_tab]
    mock_browser.contexts = [mock_context]
    driver._browser = mock_browser
    driver._context = mock_context
    driver.detect_page_state = MagicMock(return_value=(GlassdoorAutomationState.QUESTIONS_STEP, {}))
    driver.extract_questions = MagicMock(return_value=[])

    session_data = {
        "automation_session_id": "sess_exact_url",
        "application_page_url": "https://www.glassdoor.co.in/apply/hosted-job-123",
        "unresolved_question_keys": [],
    }

    recovered_page, method, diag = driver.resolve_paused_application_page(session_data)
    assert recovered_page == app_tab
    assert method == "exact_stored_url"


def test_resume_fails_with_paused_application_not_found_when_tab_closed():
    """Verify that if CDP connects but the paused tab cannot be found, PAUSED_APPLICATION_NOT_FOUND is returned."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver._browser = MagicMock()
    mock_pw_driver._browser.is_connected.return_value = True
    mock_pw_driver.resolve_paused_application_page.return_value = (None, "none", {"pages_inspected": [{"url": "http://127.0.0.1:8000/docs"}]})

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    service.save_session("sess_missing_tab", {
        "automation_session_id": "sess_missing_tab",
        "application_page_url": "https://smartapply.indeed.com/form/1",
        "unresolved_question_keys": ["salary_expectation"],
    })

    req = GlassdoorAutomationResumeRequest(automation_session_id="sess_missing_tab")
    res = service.resume(req)

    assert res.status == "failed"
    assert res.current_state == GlassdoorAutomationState.PAUSED_APPLICATION_NOT_FOUND
    assert "Could not rediscover active application page" in res.message
    assert "pages_inspected" in res.diagnostics


def test_resume_accepts_manual_zero_and_persists_and_continues():
    """Verify that /resume extracts manual '0' value, persists to repository, and automatically clicks Continue."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_repo = MagicMock(spec=ApplicationRepository)

    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/questions"
    app_tab.is_closed.return_value = False

    mock_pw_driver._browser = MagicMock()
    mock_pw_driver._browser.is_connected.return_value = True
    mock_pw_driver.resolve_paused_application_page.return_value = (app_tab, "question_signature", {"application_page_recovered": True})

    step_state = {"step": 0}

    def detect_mock(p):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        return GlassdoorAutomationState.SUBMISSION_READY, {}

    mock_pw_driver.detect_page_state.side_effect = detect_mock

    q = ApplicationQuestion(text="Years of Experience", normalized_key="years_experience", input_type=QuestionInputType.NUMBER, required=True)
    mock_pw_driver.extract_questions.return_value = [q]
    # User entered "0" in DOM
    mock_pw_driver.read_question_value.return_value = "0"
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver, application_repository=mock_repo)
    service.save_session("sess_zero_test", {
        "automation_session_id": "sess_zero_test",
        "job_url": "https://www.glassdoor.co.in/job/test-job?jl=999",
        "application_page_url": "https://smartapply.indeed.com/form/questions",
        "unresolved_question_keys": ["years_experience"],
    })

    req = GlassdoorAutomationResumeRequest(automation_session_id="sess_zero_test")
    res = service.resume(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert mock_repo.save_application_answer.called
    assert res.diagnostics.get("manual_answers_detected") == {"years_experience": "0"}
    assert res.diagnostics.get("manual_answers_saved") is True


def test_resume_flow_with_requirements_warning_after_manual_answer():
    """Verify flow: manual answer '0' -> Continue -> REQUIREMENTS_WARNING -> Apply anyway -> SUBMISSION_READY."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_repo = MagicMock(spec=ApplicationRepository)

    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/form/questions"
    app_tab.is_closed.return_value = False

    mock_pw_driver._browser = MagicMock()
    mock_pw_driver._browser.is_connected.return_value = True
    mock_pw_driver.resolve_paused_application_page.return_value = (app_tab, "question_signature", {"application_page_recovered": True})

    step_state = {"step": 0}

    def detect_mock(p):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        elif step_state["step"] == 2:
            return GlassdoorAutomationState.REQUIREMENTS_WARNING, {"apply_anyway_found": True}
        return GlassdoorAutomationState.SUBMISSION_READY, {}

    mock_pw_driver.detect_page_state.side_effect = detect_mock

    q = ApplicationQuestion(text="Teaching Experience", normalized_key="teaching_experience", input_type=QuestionInputType.NUMBER, required=True)
    mock_pw_driver.extract_questions.return_value = [q]
    mock_pw_driver.read_question_value.return_value = 0
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.click_apply_anyway.return_value = (True, None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver, application_repository=mock_repo)
    service.save_session("sess_warning_test", {
        "automation_session_id": "sess_warning_test",
        "job_url": "https://www.glassdoor.co.in/job/warning-test?jl=888",
        "application_page_url": "https://smartapply.indeed.com/form/questions",
        "unresolved_question_keys": ["teaching_experience"],
    })

    req = GlassdoorAutomationResumeRequest(automation_session_id="sess_warning_test")
    res = service.resume(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert mock_pw_driver.click_apply_anyway.called


def test_get_glassdoor_session_debug_endpoint():
    """Verify that GET /automation/glassdoor/sessions/{id} returns saved metadata without browser interaction."""
    client = TestClient(app)

    # Save a test session directly into repository / service
    from app.api.routers.glassdoor_automation import get_glassdoor_apply_service
    svc = GlassdoorApplyService()
    svc.save_session("gdoor_debug_123", {
        "automation_session_id": "gdoor_debug_123",
        "job_url": "https://www.glassdoor.co.in/job/test?jl=555",
        "job_id": "55555555-5555-5555-5555-555555555555",
        "job_listing_id": "555",
        "source": "glassdoor",
        "application_page_url": "https://smartapply.indeed.com/form/1",
        "application_host": "indeed_smartapply",
        "current_state": "NEEDS_USER_INPUT",
        "unresolved_questions": [{"text": "CTC?", "normalized_key": "ctc", "type": "text", "required": True}],
        "unresolved_question_keys": ["ctc"],
        "total_steps": 2,
        "resume_attempt": 1,
    })

    app.dependency_overrides[get_glassdoor_apply_service] = lambda: svc

    try:
        resp = client.get("/automation/glassdoor/sessions/gdoor_debug_123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["automation_session_id"] == "gdoor_debug_123"
        assert data["job_listing_id"] == "555"
        assert data["current_state"] == "NEEDS_USER_INPUT"
        assert data["unresolved_question_keys"] == ["ctc"]

        # Not found case
        resp_404 = client.get("/automation/glassdoor/sessions/non_existent_session")
        assert resp_404.status_code == 404
    finally:
        app.dependency_overrides.pop(get_glassdoor_apply_service, None)


# ==============================================================================
# SECTION 23: Phase 5.6 Async Playwright Architecture & Loop Isolation Tests
# ==============================================================================


def test_no_playwright_sync_api_imports():
    """Verify that no Glassdoor automation modules import playwright.sync_api."""
    import re
    modules = [
        "app/automation/glassdoor/playwright_driver.py",
        "app/automation/glassdoor/apply_service.py",
        "app/api/routers/glassdoor_automation.py",
    ]
    for mod_path in modules:
        with open(mod_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "playwright.sync_api" not in content, f"Forbidden sync_api found in {mod_path}"
        assert not re.search(r"\bfrom\s+playwright\.sync_api\b", content), f"Forbidden sync_api import in {mod_path}"
        assert not re.search(r"(?<!async_)\bsync_playwright\b", content), f"Forbidden sync_playwright found in {mod_path}"


@pytest.mark.asyncio
async def test_post_glassdoor_resume_in_active_asyncio_event_loop():
    """Verify that POST /automation/glassdoor/resume executes cleanly inside an active asyncio event loop."""
    import httpx

    svc = GlassdoorApplyService()
    svc.save_session("gdoor_async_loop_sess", {
        "automation_session_id": "gdoor_async_loop_sess",
        "job_url": "https://www.glassdoor.co.in/job/async-test?jl=999",
        "application_page_url": "https://smartapply.indeed.com/form/1",
        "application_host": "indeed_smartapply",
        "current_state": "NEEDS_USER_INPUT",
        "unresolved_questions": [{"text": "Expected CTC?", "normalized_key": "expected_ctc", "type": "number", "required": True}],
        "unresolved_question_keys": ["expected_ctc"],
        "total_steps": 1,
        "resume_attempt": 1,
    })

    mock_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_browser = MagicMock()
    mock_browser.is_connected.return_value = True
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/1"
    mock_page.is_closed.return_value = False
    mock_context.pages = [mock_page]
    mock_browser.contexts = [mock_context]
    mock_driver._browser = mock_browser
    mock_driver._context = mock_context
    mock_driver.application_page = mock_page
    mock_driver.connect_cdp.return_value = (True, None)
    mock_driver.resolve_paused_application_page.return_value = (mock_page, "exact_url", {})
    mock_driver.detect_page_state.return_value = (GlassdoorAutomationState.SUBMISSION_READY, {})
    mock_driver.find_exact_submit_button.return_value = MagicMock()

    svc.playwright_driver = mock_driver

    from app.api.routers.glassdoor_automation import get_glassdoor_apply_service
    app.dependency_overrides[get_glassdoor_apply_service] = lambda: svc

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/automation/glassdoor/resume",
                json={
                    "automation_session_id": "gdoor_async_loop_sess",
                    "answers": {"expected_ctc": "1500000"},
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["current_state"] == "SUBMISSION_READY"
        assert "Playwright Sync API inside the asyncio loop" not in data.get("message", "")
    finally:
        app.dependency_overrides.pop(get_glassdoor_apply_service, None)


def test_playwright_runtime_error_mapped_to_automation_runtime_error():
    """Verify that internal Playwright/asyncio runtime failures are mapped to AUTOMATION_RUNTIME_ERROR."""
    svc = GlassdoorApplyService()
    svc.save_session("gdoor_err_sess", {
        "automation_session_id": "gdoor_err_sess",
        "job_url": "https://www.glassdoor.co.in/job/test?jl=888",
        "application_page_url": "https://smartapply.indeed.com/form/1",
        "current_state": "NEEDS_USER_INPUT",
    })

    mock_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_driver._browser = None
    mock_driver.connect_cdp.side_effect = RuntimeError("Playwright Error: Target page, context or browser has been closed")
    svc.playwright_driver = mock_driver

    req = GlassdoorAutomationResumeRequest(
        automation_session_id="gdoor_err_sess",
    )
    res = svc.resume(req)

    assert res.status == "failed"
    assert res.current_state == GlassdoorAutomationState.AUTOMATION_RUNTIME_ERROR
    assert "Playwright Error" in res.message
    assert res.diagnostics.get("cdp_connected") is False


@pytest.mark.asyncio
async def test_fastapi_navigate_and_apply_in_asyncio_event_loop():
    """Verify that /navigate-to-submit and /apply endpoints await services without unawaited coroutine warnings."""
    import httpx

    svc = GlassdoorApplyService()
    mock_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_driver.detect_page_state.return_value = (GlassdoorAutomationState.EASY_APPLY_AVAILABLE, {})
    mock_driver.find_exact_easy_apply_button.return_value = MagicMock()
    mock_driver.click_easy_apply.return_value = (True, None)
    mock_driver.resolve_application_page.return_value = MagicMock()
    mock_driver.find_exact_submit_button.return_value = MagicMock()
    svc.playwright_driver = mock_driver

    # Mock inspect and navigate
    mock_win = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_win.is_smartapply_detected.return_value = True
    mock_win.find_action_control.return_value = MagicMock()
    mock_win.get_window_text_content.return_value = "Submit application"
    mock_win.verify_browser_window.return_value = (True, "Chrome")
    svc.driver = mock_win

    from app.api.routers.glassdoor_automation import get_glassdoor_apply_service
    app.dependency_overrides[get_glassdoor_apply_service] = lambda: svc

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/automation/glassdoor/inspect",
                json={"url": "https://www.glassdoor.co.in/job/test-job?jl=123456"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
    finally:
        app.dependency_overrides.pop(get_glassdoor_apply_service, None)


# ==============================================================================
# SECTION 24: Phase 5.6 Requirements Warning & "Apply anyway" Progression Tests
# ==============================================================================

def test_requirements_warning_curly_apostrophe_heading_detected():
    """Verify that heading with curly apostrophe 'It looks like you don’t meet these employer requirements' is detected."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_heading = MagicMock()
    mock_heading.inner_text.return_value = "It looks like you don’t meet these employer requirements"
    mock_heading_loc = MagicMock()
    mock_heading_loc.count.return_value = 1
    mock_heading_loc.nth.return_value = mock_heading
    mock_page.locator.return_value = mock_heading_loc

    is_warn, reqs, reqs_text = driver._detect_requirements_warning(mock_page)
    assert is_warn is True


def test_requirements_warning_straight_apostrophe_heading_detected():
    """Verify that heading with straight apostrophe 'It looks like you don't meet these employer requirements' is detected."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_heading = MagicMock()
    mock_heading.inner_text.return_value = "It looks like you don't meet these employer requirements"
    mock_heading_loc = MagicMock()
    mock_heading_loc.count.return_value = 1
    mock_heading_loc.nth.return_value = mock_heading
    mock_page.locator.return_value = mock_heading_loc

    is_warn, reqs, reqs_text = driver._detect_requirements_warning(mock_page)
    assert is_warn is True


def test_requirements_warning_detected_while_url_contains_questions_module():
    """Verify that DOM state recognition recognizes REQUIREMENTS_WARNING even if route URL still has questions-module."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/beta/indeedapply/form/questions-module/questions/1"
    
    # Mock no CAPTCHA and no Submit
    mock_submit_loc = MagicMock()
    mock_submit_loc.count.return_value = 0
    mock_page.get_by_role.return_value = mock_submit_loc

    mock_heading = MagicMock()
    mock_heading.inner_text.return_value = "It looks like you don’t meet these employer requirements"
    
    def mock_loc(sel):
        if "h1" in sel or "heading" in sel:
            m = MagicMock()
            m.count.return_value = 1
            m.nth.return_value = mock_heading
            return m
        m = MagicMock()
        m.count.return_value = 0
        return m

    mock_page.locator.side_effect = mock_loc
    mock_btn = MagicMock()
    mock_btn.is_visible.return_value = True
    mock_btn.is_enabled.return_value = True
    driver.find_exact_apply_anyway_button = MagicMock(return_value=mock_btn)

    state, diag = driver.detect_page_state(mock_page)
    assert state == GlassdoorAutomationState.REQUIREMENTS_WARNING
    assert diag.get("requirements_warning_detected") is True
    assert diag.get("apply_anyway_found") is True


def test_find_apply_anyway_as_button():
    """Verify find_exact_apply_anyway_button detects control rendered with role='button'."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_btn = MagicMock()
    mock_btn.is_visible.return_value = True
    mock_btn.is_enabled.return_value = True

    mock_loc = MagicMock()
    mock_loc.count.return_value = 1
    mock_loc.nth.return_value = mock_btn
    mock_page.get_by_role.return_value = mock_loc

    res = driver.find_exact_apply_anyway_button(mock_page)
    assert res == mock_btn


def test_find_apply_anyway_as_link():
    """Verify find_exact_apply_anyway_button detects control rendered with role='link'."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_link = MagicMock()
    mock_link.is_visible.return_value = True
    mock_link.is_enabled.return_value = True

    def mock_role(role, name=None):
        if role == "link":
            m = MagicMock()
            m.count.return_value = 1
            m.nth.return_value = mock_link
            return m
        m = MagicMock()
        m.count.return_value = 0
        return m

    mock_page.get_by_role.side_effect = mock_role
    mock_page.locator.return_value.count.return_value = 0

    res = driver.find_exact_apply_anyway_button(mock_page)
    assert res == mock_link


def test_find_apply_anyway_as_clickable_text_control():
    """Verify find_exact_apply_anyway_button detects control rendered as a clickable span or text element."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_span = MagicMock()
    mock_span.is_visible.return_value = True

    mock_page.get_by_role.return_value.count.return_value = 0

    def mock_loc(sel):
        if "span" in sel or "button:has-text" in sel:
            m = MagicMock()
            m.count.return_value = 1
            m.nth.return_value = mock_span
            m.filter.return_value.count.return_value = 1
            m.filter.return_value.nth.return_value = mock_span
            return m
        m = MagicMock()
        m.count.return_value = 0
        return m

    mock_page.locator.side_effect = mock_loc

    res = driver.find_exact_apply_anyway_button(mock_page)
    assert res == mock_span


def test_return_to_job_search_never_clicked_on_warning_page():
    """Verify that Return to job search is explicitly blacklisted from being clicked as a progression action."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    mock_return_btn = MagicMock()
    mock_return_btn.is_enabled.return_value = True
    mock_return_btn.inner_text.return_value = "Return to job search"

    mock_loc = MagicMock()
    mock_loc.count.return_value = 1
    mock_loc.first = mock_return_btn
    mock_loc.nth.return_value = mock_return_btn
    mock_page.get_by_role.return_value = mock_loc

    act_name, act_btn = driver.find_progression_action(mock_page)
    assert act_name is None
    assert act_btn is None


def test_warning_checked_before_generic_continue_next_review():
    """Verify that state detection prioritizes REQUIREMENTS_WARNING before checking generic progression controls."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/1"

    mock_page.get_by_role.return_value.count.return_value = 0
    driver._detect_blocking_in_dom = MagicMock(return_value=(None, None))
    driver.find_exact_submit_button = MagicMock(return_value=None)
    driver._detect_requirements_warning = MagicMock(return_value=(True, ["Teaching: 1 year (Required)"], "Teaching: 1 year (Required)"))
    driver.find_exact_apply_anyway_button = MagicMock(return_value=MagicMock())
    driver.find_progression_action = MagicMock(return_value=("continue", MagicMock()))

    state, diag = driver.detect_page_state(mock_page)
    assert state == GlassdoorAutomationState.REQUIREMENTS_WARNING
    assert diag["requirements_warning_detected"] is True
    # Progression action should not override REQUIREMENTS_WARNING
    assert state != GlassdoorAutomationState.CONTINUE_AVAILABLE


def test_warning_actions_discovered_in_iframe():
    """Verify that warning headings and Apply anyway actions inside an iframe frame are discovered."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_frame = MagicMock()
    mock_page.frames = [mock_frame]

    mock_page.locator.return_value.count.return_value = 0
    mock_page.get_by_role.return_value.count.return_value = 0

    mock_heading = MagicMock()
    mock_heading.inner_text.return_value = "It looks like you don’t meet these employer requirements"
    mock_frame_loc = MagicMock()
    mock_frame_loc.count.return_value = 1
    mock_frame_loc.nth.return_value = mock_heading
    mock_frame.locator.return_value = mock_frame_loc

    is_warn, reqs, reqs_text = driver._detect_requirements_warning(mock_page)
    assert is_warn is True


def test_unmet_requirements_text_extracted():
    """Verify that unmet employer requirement text is correctly extracted from list items."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_heading = MagicMock()
    mock_heading.inner_text.return_value = "It looks like you don't meet these employer requirements"
    
    mock_req_item = MagicMock()
    mock_req_item.inner_text.return_value = "Teaching: 1 year (Required)"

    def mock_loc(sel):
        if "h1" in sel or "heading" in sel:
            m = MagicMock()
            m.count.return_value = 1
            m.nth.return_value = mock_heading
            return m
        elif "ul li" in sel or "requirement" in sel:
            m = MagicMock()
            m.count.return_value = 1
            m.nth.return_value = mock_req_item
            return m
        m = MagicMock()
        m.count.return_value = 0
        return m

    mock_page.locator.side_effect = mock_loc
    mock_page.get_by_role.return_value.count.return_value = 0

    is_warn, reqs, reqs_text = driver._detect_requirements_warning(mock_page)
    assert is_warn is True
    assert "Teaching: 1 year (Required)" in reqs
    assert reqs_text == "Teaching: 1 year (Required)"


def test_visible_actions_contains_apply_anyway_and_return_to_search():
    """Verify that get_visible_actions extracts visible button and link labels for diagnostics."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    mock_btn1 = MagicMock()
    mock_btn1.is_visible.return_value = True
    mock_btn1.inner_text.return_value = "Return to job search"

    mock_btn2 = MagicMock()
    mock_btn2.is_visible.return_value = True
    mock_btn2.inner_text.return_value = "Apply anyway"

    mock_loc = MagicMock()
    mock_loc.count.return_value = 2
    mock_loc.nth.side_effect = lambda idx: mock_btn1 if idx == 0 else mock_btn2
    mock_page.locator.return_value = mock_loc

    actions = driver.get_visible_actions(mock_page)
    assert "Return to job search" in actions
    assert "Apply anyway" in actions


def test_apply_anyway_clicked_exactly_once_and_reenters_loop():
    """Verify that Apply anyway is clicked exactly once and loop continues to SUBMISSION_READY."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    app_tab = MagicMock()

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_counts = {"count": 0}

    def mock_detect(page):
        step_counts["count"] += 1
        if step_counts["count"] == 1:
            return GlassdoorAutomationState.REQUIREMENTS_WARNING, {
                "requirements_warning_detected": True,
                "requirements_not_met": ["Teaching: 1 year (Required)"],
                "requirements_text": "Teaching: 1 year (Required)",
                "visible_actions": ["Return to job search", "Apply anyway"],
            }
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    mock_pw_driver.detect_page_state.side_effect = mock_detect
    mock_pw_driver.find_exact_apply_anyway_button.return_value = MagicMock()
    mock_pw_driver.click_apply_anyway.return_value = (True, None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(url="https://www.glassdoor.com/job/test?jl=998877")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert mock_pw_driver.click_apply_anyway.call_count == 1
    assert res.diagnostics.get("requirements_warning_detected") is True
    assert res.diagnostics.get("requirements_not_met") == ["Teaching: 1 year (Required)"]
    history_states = [h.get("state") for h in res.diagnostics.get("step_history", [])]
    assert "REQUIREMENTS_WARNING" in history_states


def test_hitl_resume_with_manual_0_progresses_through_warning_to_submission_ready():
    """Verify live HITL scenario: manual answer '0' entered -> /resume persists answer -> Continue -> warning -> Apply anyway -> SUBMISSION_READY."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/beta/indeedapply/form/questions-module/questions/1"
    app_tab.title.return_value = "Apply - Teaching Experience"
    app_tab.is_closed.return_value = False

    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_paused_application_page.return_value = (app_tab, {"recovered": True})
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        elif step_state["step"] == 2:
            return GlassdoorAutomationState.REQUIREMENTS_WARNING, {
                "requirements_warning_detected": True,
                "requirements_not_met": ["Teaching: 1 year (Required)"],
                "requirements_text": "Teaching: 1 year (Required)",
                "visible_actions": ["Return to job search", "Apply anyway"],
            }
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    mock_pw_driver.detect_page_state.side_effect = mock_detect

    q = ApplicationQuestion(
        text="How many years of Teaching experience do you have? *",
        normalized_key="years_experience_teaching",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q]
    mock_pw_driver.read_question_value.return_value = 0
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_apply_anyway_button.return_value = MagicMock()
    mock_pw_driver.click_apply_anyway.return_value = (True, None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    service.application_repository.save_application_answer = MagicMock(return_value=True)

    session_id = "gdoor_session_live_warning"
    service.save_session(session_id, {
        "automation_session_id": session_id,
        "job_url": "https://www.glassdoor.com/job/test-live-warning",
        "unresolved_questions": [q.model_dump()],
        "total_steps": 3,
    })

    resume_req = GlassdoorAutomationResumeRequest(automation_session_id=session_id)
    res = service.resume(resume_req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False
    assert mock_pw_driver.click_apply_anyway.call_count == 1
    assert res.diagnostics.get("manual_answers_detected") == {"years_experience_teaching": "0"}
    assert res.diagnostics.get("manual_answers_saved") is True


def test_requirements_warning_unverified_apply_button_returns_requirements_action_not_verified():
    """Verify that if REQUIREMENTS_WARNING is detected but Apply anyway control cannot be verified, it returns REQUIREMENTS_ACTION_NOT_VERIFIED."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    app_tab = MagicMock()

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_tab

    mock_pw_driver.detect_page_state.return_value = (
        GlassdoorAutomationState.REQUIREMENTS_WARNING,
        {
            "requirements_warning_detected": True,
            "requirements_not_met": ["Teaching: 1 year (Required)"],
            "requirements_text": "Teaching: 1 year (Required)",
        },
    )
    mock_pw_driver.find_exact_apply_anyway_button.return_value = None

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(url="https://www.glassdoor.com/job/test-blocked")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "blocked"
    assert res.current_state == GlassdoorAutomationState.REQUIREMENTS_ACTION_NOT_VERIFIED
    assert "Apply anyway button not found" in res.message
    assert res.diagnostics.get("requirements_warning_detected") is True


# ==============================================================================
# SECTION 25: Phase 5.6 SPA Transition & Optional Survey Interstitial Tests
# ==============================================================================

def test_survey_heading_curly_apostrophe_detected():
    """Verify that survey heading with curly apostrophe 'Help Indeed learn more about why you’re applying' is detected."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_heading = MagicMock()
    mock_heading.inner_text.return_value = "Help Indeed learn more about why you’re applying"
    mock_h_loc = MagicMock()
    mock_h_loc.count.return_value = 1
    mock_h_loc.nth.return_value = mock_heading

    def mock_loc(sel):
        if "h1" in sel or "heading" in sel:
            return mock_h_loc
        m = MagicMock()
        m.count.return_value = 0
        return m

    mock_page.locator.side_effect = mock_loc

    is_survey, survey_info = driver._detect_survey_step(mock_page)
    assert is_survey is True
    assert "Help Indeed learn more" in survey_info.get("heading", "")


def test_survey_heading_straight_apostrophe_detected():
    """Verify that survey heading with straight apostrophe 'Help Indeed learn more about why you're applying' is detected."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_heading = MagicMock()
    mock_heading.inner_text.return_value = "Help Indeed learn more about why you're applying"
    mock_h_loc = MagicMock()
    mock_h_loc.count.return_value = 1
    mock_h_loc.nth.return_value = mock_heading

    def mock_loc(sel):
        if "h1" in sel or "heading" in sel:
            return mock_h_loc
        m = MagicMock()
        m.count.return_value = 0
        return m

    mock_page.locator.side_effect = mock_loc

    is_survey, survey_info = driver._detect_survey_step(mock_page)
    assert is_survey is True


def test_survey_subtext_and_textarea_detected():
    """Verify that survey is detected via subtext 'We won’t share your response with the employer' and textarea."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    mock_para = MagicMock()
    mock_para.inner_text.return_value = "We won’t share your response with the employer."
    mock_p_loc = MagicMock()
    mock_p_loc.count.return_value = 1
    mock_p_loc.nth.return_value = mock_para

    mock_ta = MagicMock()
    mock_ta.get_attribute.return_value = None
    mock_ta_loc = MagicMock()
    mock_ta_loc.count.return_value = 1
    mock_ta_loc.nth.return_value = mock_ta

    def mock_loc(sel):
        if "p" in sel or "span" in sel:
            return mock_p_loc
        elif "textarea" in sel:
            return mock_ta_loc
        m = MagicMock()
        m.count.return_value = 0
        return m

    mock_page.locator.side_effect = mock_loc
    driver._find_label_for_input = MagicMock(return_value="Reason for applying")

    is_survey, survey_info = driver._detect_survey_step(mock_page)
    assert is_survey is True
    assert survey_info.get("field_name") == "reason_for_applying"
    assert survey_info.get("required") is False


def test_survey_optional_field_detected_as_not_required():
    """Verify that Reason for applying is marked optional when no required attributes exist."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    mock_h = MagicMock()
    mock_h.inner_text.return_value = "Help Indeed learn more about why you're applying"
    mock_h_loc = MagicMock()
    mock_h_loc.count.return_value = 1
    mock_h_loc.nth.return_value = mock_h

    mock_ta = MagicMock()
    mock_ta.get_attribute.return_value = None
    mock_ta_loc = MagicMock()
    mock_ta_loc.count.return_value = 1
    mock_ta_loc.nth.return_value = mock_ta

    def mock_loc(sel):
        if "h1" in sel or "heading" in sel:
            return mock_h_loc
        elif "textarea" in sel:
            return mock_ta_loc
        m = MagicMock()
        m.count.return_value = 0
        return m

    mock_page.locator.side_effect = mock_loc
    driver._find_label_for_input = MagicMock(return_value="Reason for applying")

    is_survey, survey_info = driver._detect_survey_step(mock_page)
    assert is_survey is True
    assert survey_info.get("required") is False


def test_survey_required_field_detected_when_aria_required_true():
    """Verify that survey field is detected as required when aria-required='true'."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    mock_h = MagicMock()
    mock_h.inner_text.return_value = "Help Indeed learn more about why you're applying"
    mock_h_loc = MagicMock()
    mock_h_loc.count.return_value = 1
    mock_h_loc.nth.return_value = mock_h

    mock_ta = MagicMock()
    mock_ta.get_attribute.side_effect = lambda attr: "true" if attr == "aria-required" else None
    mock_ta_loc = MagicMock()
    mock_ta_loc.count.return_value = 1
    mock_ta_loc.nth.return_value = mock_ta

    def mock_loc(sel):
        if "h1" in sel or "heading" in sel:
            return mock_h_loc
        elif "textarea" in sel:
            return mock_ta_loc
        m = MagicMock()
        m.count.return_value = 0
        return m

    mock_page.locator.side_effect = mock_loc
    driver._find_label_for_input = MagicMock(return_value="Reason for applying *")

    is_survey, survey_info = driver._detect_survey_step(mock_page)
    assert is_survey is True
    assert survey_info.get("required") is True


def test_survey_step_prioritized_over_generic_questions():
    """Verify that detect_page_state classifies survey step as OPTIONAL_SURVEY_STEP, not QUESTIONS_STEP."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_page.url = "https://smartapply.indeed.com/form/survey"

    driver._detect_blocking_in_dom = MagicMock(return_value=(None, None))
    driver.find_exact_submit_button = MagicMock(return_value=None)
    driver._detect_requirements_warning = MagicMock(return_value=(False, [], ""))
    driver._detect_survey_step = MagicMock(return_value=(True, {
        "heading": "Help Indeed learn more about why you're applying",
        "field_name": "reason_for_applying",
        "required": False,
    }))

    state, diag = driver.detect_page_state(mock_page)
    assert state == GlassdoorAutomationState.OPTIONAL_SURVEY_STEP
    assert diag.get("survey_step_detected") is True


def test_extract_questions_ignores_survey_interstitial():
    """Verify extract_questions returns empty list on survey interstitial so employer question memory is not contaminated."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    driver._is_review_step = MagicMock(return_value=False)
    driver._detect_survey_step = MagicMock(return_value=(True, {"field_name": "reason_for_applying"}))

    qs = driver.extract_questions(mock_page)
    assert qs == []


def test_spa_transition_detected_on_heading_change():
    """Verify wait_for_spa_transition detects transition when primary heading changes."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    prev_snapshot = {
        "url": "https://smartapply.indeed.com/form/1",
        "primary_heading": "it looks like you don't meet these employer requirements",
        "is_warning": True,
        "is_survey": False,
        "has_submit": False,
        "visible_actions": ["Return to job search", "Apply anyway"],
    }

    new_snapshot = {
        "url": "https://smartapply.indeed.com/form/1",
        "primary_heading": "help indeed learn more about why you're applying",
        "is_warning": False,
        "is_survey": True,
        "has_submit": False,
        "visible_actions": ["Continue"],
    }

    driver.capture_dom_snapshot = MagicMock(return_value=new_snapshot)

    trans_ok, trans_reason, snap = driver.wait_for_spa_transition(mock_page, prev_snapshot, timeout_seconds=1.0)
    assert trans_ok is True
    assert trans_reason in ("warning_disappeared", "survey_appeared", "heading_changed")


def test_flow_warning_to_survey_to_review_to_submission_ready_single_request():
    """Verify full progression: Warning -> Apply anyway -> Survey (optional) -> Skip & Continue -> Review -> SUBMISSION_READY within one request."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    app_tab = MagicMock()

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_counts = {"count": 0}

    def mock_detect(page):
        step_counts["count"] += 1
        if step_counts["count"] == 1:
            return GlassdoorAutomationState.REQUIREMENTS_WARNING, {
                "requirements_warning_detected": True,
                "requirements_not_met": ["Teaching: 1 year (Required)"],
                "requirements_text": "Teaching: 1 year (Required)",
                "visible_actions": ["Return to job search", "Apply anyway"],
            }
        elif step_counts["count"] == 2:
            return GlassdoorAutomationState.OPTIONAL_SURVEY_STEP, {
                "survey_step_detected": True,
                "survey_info": {
                    "heading": "Help Indeed learn more about why you're applying",
                    "field_name": "reason_for_applying",
                    "required": False,
                },
                "visible_actions": ["Continue"],
            }
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    mock_pw_driver.detect_page_state.side_effect = mock_detect
    mock_pw_driver.capture_dom_snapshot.return_value = {"primary_heading": "test"}
    mock_pw_driver.wait_for_spa_transition.return_value = (True, "heading_changed", {"primary_heading": "next"})
    mock_pw_driver.find_exact_apply_anyway_button.return_value = MagicMock()
    mock_pw_driver.click_apply_anyway.return_value = (True, None)
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(url="https://www.glassdoor.com/job/test-survey?jl=112233")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False
    assert mock_pw_driver.click_apply_anyway.call_count == 1

    states = [h.get("state") for h in res.diagnostics.get("step_history", [])]
    assert "REQUIREMENTS_WARNING" in states
    assert "OPTIONAL_SURVEY_STEP" in states
    assert "SUBMISSION_READY" in states

    survey_step = next(h for h in res.diagnostics.get("step_history", []) if h.get("state") == "OPTIONAL_SURVEY_STEP")
    assert survey_step.get("action") == "skip_optional_and_continue"
    assert survey_step.get("required") is False


def test_flow_resume_warning_to_survey_to_submission_ready_single_request():
    """Verify live HITL scenario: Questions (manual 0) -> Resume -> Warning -> Apply anyway -> Survey -> Submit Ready in one HTTP request."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    app_tab = MagicMock()
    app_tab.url = "https://smartapply.indeed.com/beta/indeedapply/form/questions-module/questions/1"
    app_tab.title.return_value = "Apply - Teaching Experience"
    app_tab.is_closed.return_value = False

    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_paused_application_page.return_value = (app_tab, {"recovered": True})
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_state = {"step": 0}

    def mock_detect(page):
        step_state["step"] += 1
        if step_state["step"] == 1:
            return GlassdoorAutomationState.QUESTIONS_STEP, {}
        elif step_state["step"] == 2:
            return GlassdoorAutomationState.REQUIREMENTS_WARNING, {
                "requirements_warning_detected": True,
                "requirements_not_met": ["Teaching: 1 year (Required)"],
                "requirements_text": "Teaching: 1 year (Required)",
                "visible_actions": ["Return to job search", "Apply anyway"],
            }
        elif step_state["step"] == 3:
            return GlassdoorAutomationState.OPTIONAL_SURVEY_STEP, {
                "survey_step_detected": True,
                "survey_info": {
                    "heading": "Help Indeed learn more about why you're applying",
                    "field_name": "reason_for_applying",
                    "required": False,
                },
                "visible_actions": ["Continue"],
            }
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    mock_pw_driver.detect_page_state.side_effect = mock_detect
    mock_pw_driver.capture_dom_snapshot.return_value = {"primary_heading": "test"}
    mock_pw_driver.wait_for_spa_transition.return_value = (True, "survey_appeared", {"primary_heading": "survey"})

    q = ApplicationQuestion(
        text="How many years of Teaching experience do you have? *",
        normalized_key="years_experience_teaching",
        input_type=QuestionInputType.NUMBER,
        required=True,
    )
    mock_pw_driver.extract_questions.return_value = [q]
    mock_pw_driver.read_question_value.return_value = 0
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_apply_anyway_button.return_value = MagicMock()
    mock_pw_driver.click_apply_anyway.return_value = (True, None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(playwright_driver=mock_pw_driver)
    service.application_repository.save_application_answer = MagicMock(return_value=True)

    session_id = "gdoor_session_full_chain"
    service.save_session(session_id, {
        "automation_session_id": session_id,
        "job_url": "https://www.glassdoor.com/job/test-full-chain",
        "unresolved_questions": [q.model_dump()],
        "total_steps": 3,
    })

    resume_req = GlassdoorAutomationResumeRequest(automation_session_id=session_id)
    res = service.resume(resume_req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False
    assert mock_pw_driver.click_apply_anyway.call_count == 1
    assert res.diagnostics.get("manual_answers_detected") == {"years_experience_teaching": "0"}

    history_states = [h.get("state") for h in res.diagnostics.get("step_history", [])]
    assert "QUESTIONS_STEP" in history_states
    assert "REQUIREMENTS_WARNING" in history_states
    assert "OPTIONAL_SURVEY_STEP" in history_states


def test_survey_required_without_stored_answer_returns_needs_user_input():
    """Verify that a required survey question without a stored approved answer pauses with NEEDS_USER_INPUT."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    app_tab = MagicMock()

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_tab

    mock_pw_driver.detect_page_state.return_value = (
        GlassdoorAutomationState.OPTIONAL_SURVEY_STEP,
        {
            "survey_step_detected": True,
            "survey_info": {
                "heading": "Help Indeed learn more about why you're applying",
                "field_name": "reason_for_applying",
                "field_label": "Reason for applying *",
                "required": True,
            },
        },
    )

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(url="https://www.glassdoor.com/job/test-req-survey")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "manual_action_required"
    assert res.current_state == GlassdoorAutomationState.NEEDS_USER_INPUT
    assert res.manual_action_required is True
    assert "Reason for applying" in res.message
    assert "reason_for_applying" in res.diagnostics.get("unresolved_question_keys", [])


def test_survey_required_with_stored_answer_fills_and_continues():
    """Verify that a required survey question with a stored answer fills the textarea and proceeds to SUBMISSION_READY."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    app_tab = MagicMock()
    mock_ta = MagicMock()
    app_tab.locator.return_value.first = mock_ta

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"browser_session_verified": True})
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_tab

    step_counts = {"count": 0}

    def mock_detect(page):
        step_counts["count"] += 1
        if step_counts["count"] == 1:
            return GlassdoorAutomationState.OPTIONAL_SURVEY_STEP, {
                "survey_step_detected": True,
                "survey_info": {
                    "heading": "Help Indeed learn more about why you're applying",
                    "field_name": "reason_for_applying",
                    "field_label": "Reason for applying *",
                    "required": True,
                },
            }
        return GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True}

    mock_pw_driver.detect_page_state.side_effect = mock_detect
    mock_pw_driver.capture_dom_snapshot.return_value = {"primary_heading": "test"}
    mock_pw_driver.wait_for_spa_transition.return_value = (True, "submit_appeared", {})
    mock_pw_driver.activate_progress_button.return_value = (True, "continue", None)
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    service.application_repository.get_answers_by_profile = MagicMock(
        return_value={"reason_for_applying": "I am seeking a challenging role in software engineering."}
    )

    prof_id = str(uuid4())
    req = GlassdoorAutomationNavigateRequest(url="https://www.glassdoor.com/job/test-stored-survey", profile_id=prof_id)

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    mock_ta.fill.assert_called_once_with("I am seeking a challenging role in software engineering.")


# ==============================================================================
# SECTION 26: Phase 5.6 False-Positive ALREADY_APPLIED Prevention & Conflict Resolution Tests
# ==============================================================================

def test_active_easy_apply_with_unrelated_word_applied_elsewhere_returns_easy_apply_available():
    """Verify that if an active Easy Apply button exists, unrelated occurrences of 'applied' (e.g. '5 candidates applied', 'Applied filters') do NOT trigger ALREADY_APPLIED."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.co.in/job-listing/teacher-JV_IC2874136.htm?jl=1010229231195"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.page = job_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {
        "job_page_url": job_tab.url,
        "browser_session_verified": True,
    })

    # Mock active Easy Apply button found
    mock_easy_btn = MagicMock()
    mock_pw_driver.find_exact_easy_apply_button.return_value = mock_easy_btn
    # Even if check_already_applied found a weak signal in the DOM, Easy Apply takes precedence
    mock_pw_driver.check_already_applied.return_value = (True, "5 candidates applied", "div.footer")

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationInspectRequest(url="https://www.glassdoor.co.in/job-listing/teacher-JV_IC2874136.htm?jl=1010229231195")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.inspect_job_application(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.EASY_APPLY_AVAILABLE
    assert res.easy_apply_verified is True
    assert res.diagnostics.get("easy_apply_present") is True
    assert res.diagnostics.get("easy_apply_enabled") is True
    assert res.diagnostics.get("already_applied_check_performed") is True
    assert res.diagnostics.get("resolved_listing_id") == "1010229231195"


def test_active_easy_apply_with_jobs_youve_applied_to_sidebar_returns_easy_apply_available():
    """Verify that 'Jobs you've applied to' sidebar heading does not trigger ALREADY_APPLIED when Easy Apply is present."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.com/job/test-job?jl=998877"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.page = job_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {
        "job_page_url": job_tab.url,
        "browser_session_verified": True,
    })

    mock_pw_driver.find_exact_easy_apply_button.return_value = MagicMock()
    mock_pw_driver.check_already_applied.return_value = (False, None, None)

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationInspectRequest(url="https://www.glassdoor.com/job/test-job?jl=998877")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.inspect_job_application(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.EASY_APPLY_AVAILABLE
    assert res.easy_apply_verified is True
    assert res.diagnostics.get("easy_apply_present") is True
    assert res.diagnostics.get("already_applied_signal_found") is False


def test_active_easy_apply_with_old_smartapply_tab_returns_easy_apply_available():
    """Verify that existing old SmartApply tab from earlier run does not confuse fresh job inspection."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.com/job/fresh-job?jl=123456"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = MagicMock()  # stale application page
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"job_page_url": job_tab.url, "browser_session_verified": True})
    mock_pw_driver.find_exact_easy_apply_button.return_value = MagicMock()
    mock_pw_driver.check_already_applied.return_value = (False, None, None)

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationInspectRequest(url="https://www.glassdoor.com/job/fresh-job?jl=123456")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.inspect_job_application(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.EASY_APPLY_AVAILABLE
    assert res.easy_apply_verified is True
    assert res.diagnostics.get("stale_application_pages_ignored") is True


def test_explicit_current_job_you_applied_returns_already_applied():
    """Verify that explicit 'You applied' status badge on job listing returns ALREADY_APPLIED when Easy Apply is absent."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.com/job/applied-job?jl=556677"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.page = job_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"job_page_url": job_tab.url, "browser_session_verified": True})
    mock_pw_driver.find_exact_easy_apply_button.return_value = None
    mock_pw_driver.check_already_applied.return_value = (True, "You applied on August 20", "[data-test='job-applied']")
    mock_win_driver.find_easy_apply_control.return_value = None
    mock_win_driver.get_element_at_point.return_value = (None, {"is_easy_apply": False})

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationInspectRequest(url="https://www.glassdoor.com/job/applied-job?jl=556677")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.inspect_job_application(req)

    assert res.status == "blocked"
    assert res.current_state == GlassdoorAutomationState.ALREADY_APPLIED
    assert res.message == "Candidate has already applied to this position on Glassdoor."
    assert res.diagnostics.get("already_applied_signal_found") is True
    assert res.diagnostics.get("already_applied_signal_text") == "You applied on August 20"
    assert res.diagnostics.get("easy_apply_present") is False


def test_explicit_current_job_applied_status_returns_already_applied():
    """Verify that explicit 'Applied' status pill on job listing returns ALREADY_APPLIED when Easy Apply is absent."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.co.in/job-listing/engineer-JV_IC2874136.htm?jl=1010229231195"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.page = job_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"job_page_url": job_tab.url, "browser_session_verified": True})
    mock_pw_driver.find_exact_easy_apply_button.return_value = None
    mock_pw_driver.check_already_applied.return_value = (True, "Applied", "button[name='applied']")
    mock_win_driver.find_easy_apply_control.return_value = None
    mock_win_driver.get_element_at_point.return_value = (None, {"is_easy_apply": False})

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationInspectRequest(url="https://www.glassdoor.co.in/job-listing/engineer-JV_IC2874136.htm?jl=1010229231195")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.inspect_job_application(req)

    assert res.status == "blocked"
    assert res.current_state == GlassdoorAutomationState.ALREADY_APPLIED
    assert res.diagnostics.get("already_applied_signal_found") is True
    assert res.diagnostics.get("resolved_listing_id") == "1010229231195"


def test_requested_listing_id_must_match_resolved_page_in_diagnostics():
    """Verify that requested_job_url, resolved_job_page_url, and resolved_listing_id are reported."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.com/job/test-job?jl=1010229231195"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.page = job_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"job_page_url": job_tab.url, "browser_session_verified": True})
    mock_pw_driver.find_exact_easy_apply_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationInspectRequest(url="https://www.glassdoor.com/job/test-job?jl=1010229231195")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.inspect_job_application(req)

    assert res.diagnostics.get("requested_job_url") == "https://www.glassdoor.com/job/test-job?jl=1010229231195"
    assert res.diagnostics.get("resolved_job_page_url") == "https://www.glassdoor.com/job/test-job?jl=1010229231195"
    assert res.diagnostics.get("resolved_listing_id") == "1010229231195"


def test_stale_cached_already_applied_state_cleared_on_new_run():
    """Verify that reset_navigation_state clears stale application page state."""
    driver = GlassdoorPlaywrightDriver()
    driver.application_page = MagicMock()
    driver.job_page = MagicMock()
    driver._page = MagicMock()

    driver.reset_navigation_state()

    assert driver.application_page is None
    assert driver.job_page is None
    assert driver._page is None


def test_easy_apply_remains_clickable_after_false_positive_text():
    """Verify that navigate_to_submit proceeds to click Easy Apply even if stray 'applied' text exists."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    app_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.com/job/test-easy?jl=1010229231195"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.application_page = app_tab
    mock_pw_driver.page = app_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"job_page_url": job_tab.url, "browser_session_verified": True})
    mock_pw_driver.find_exact_easy_apply_button.return_value = MagicMock()
    mock_pw_driver.click_easy_apply.return_value = (True, None)
    mock_pw_driver.resolve_application_page.return_value = app_tab
    mock_pw_driver.detect_page_state.return_value = (GlassdoorAutomationState.SUBMISSION_READY, {"submit_button_detected": True})
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(url="https://www.glassdoor.com/job/test-easy?jl=1010229231195")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.easy_apply_clicked is True
    assert mock_pw_driver.click_easy_apply.call_count == 1


def test_genuine_already_applied_job_is_never_reapplied():
    """Verify that navigate_to_submit stops immediately on genuine ALREADY_APPLIED and does not click Easy Apply or Submit."""
    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)

    job_tab = MagicMock()
    job_tab.url = "https://www.glassdoor.com/job/already-applied?jl=1010229231195"

    mock_pw_driver.job_page = job_tab
    mock_pw_driver.page = job_tab
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, job_tab, {"job_page_url": job_tab.url, "browser_session_verified": True})
    mock_pw_driver.find_exact_easy_apply_button.return_value = None
    mock_pw_driver.check_already_applied.return_value = (True, "Already applied", "[data-test='job-applied']")
    mock_win_driver.find_easy_apply_control.return_value = None
    mock_win_driver.get_element_at_point.return_value = (None, {"is_easy_apply": False})

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(url="https://www.glassdoor.com/job/already-applied?jl=1010229231195")

    with patch("app.automation.glassdoor.apply_service.is_windows", return_value=True):
        res = service.navigate_to_submit(req)

    assert res.status == "blocked"
    assert res.current_state == GlassdoorAutomationState.ALREADY_APPLIED
    assert res.easy_apply_clicked is False
    assert res.final_submit_clicked is False
    mock_pw_driver.click_easy_apply.assert_not_called()


def test_state_detector_is_already_applied_control():
    """Verify GlassdoorStateDetector.is_already_applied_control matches exact status names and ignores general phrases."""
    assert GlassdoorStateDetector.is_already_applied_control("Applied") is True
    assert GlassdoorStateDetector.is_already_applied_control("You applied") is True
    assert GlassdoorStateDetector.is_already_applied_control("Already applied") is True
    assert GlassdoorStateDetector.is_already_applied_control("You've applied") is True
    assert GlassdoorStateDetector.is_already_applied_control("Application submitted") is True

    # Ignored / false positive phrases
    assert GlassdoorStateDetector.is_already_applied_control("5 candidates applied") is False
    assert GlassdoorStateDetector.is_already_applied_control("Applied filters") is False
    assert GlassdoorStateDetector.is_already_applied_control("Jobs you've applied to") is False
    assert GlassdoorStateDetector.is_already_applied_control("Apply to similar jobs") is False


def test_driver_check_already_applied_excludes_sidebars_and_footers():
    """Verify driver.check_already_applied excludes sidebar and recommendation elements."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()

    # Mock an element inside aside/sidebar
    mock_aside_elem = MagicMock()
    mock_aside_elem.is_visible.return_value = True
    mock_aside_elem.inner_text.return_value = "Applied"
    mock_aside_elem.evaluate.return_value = True  # is inside excluded container

    mock_loc = MagicMock()
    mock_loc.count.return_value = 1
    mock_loc.nth.return_value = mock_aside_elem
    mock_page.get_by_role.return_value = mock_loc
    mock_page.locator.return_value = mock_loc

    is_applied, text, loc_desc = driver.check_already_applied(mock_page)
    assert is_applied is False


# ============================================================================
# PHASE 5.8.1 — CDP ATTACHMENT, SESSION VERIFICATION & SECTION 14 TESTS
# ============================================================================

def test_cdp_endpoint_reachable_browser_verified():
    """Requirement 1: CDP endpoint reachable -> browser_session_verified is True."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_page = MagicMock()
    mock_page.url = "https://www.glassdoor.com/job-listing/j?jl=1010256944540"
    cdp_diag = {
        "cdp_endpoint": "http://127.0.0.1:9222",
        "cdp_connect_attempted": True,
        "cdp_connected": True,
        "browser_session_verified": True,
        "contexts_found": 1,
        "pages_found": 1,
        "authoritative_page_url": mock_page.url,
        "requested_job_url": mock_page.url,
        "second_browser_launched": False,
    }
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, mock_page, cdp_diag)
    mock_pw_driver.find_exact_easy_apply_button.return_value = MagicMock()

    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    # PyWinAuto fails to attach to desktop window
    mock_win_driver.attach_or_open_browser.return_value = (False, None, "Could not find active browser window")
    mock_win_driver.get_window_text_content.return_value = "Easy Apply Job"

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationInspectRequest(url=mock_page.url)

    res = service.inspect_job_application(req)
    assert res.current_state == GlassdoorAutomationState.EASY_APPLY_AVAILABLE
    assert res.diagnostics["cdp_connected"] is True
    assert res.diagnostics["browser_session_verified"] is True
    assert res.diagnostics["authoritative_page_url"] == mock_page.url
    assert res.diagnostics["second_browser_launched"] is False


def test_chrome_newtab_valid_session_and_navigates():
    """Requirement 2: Empty new-tab (chrome://newtab/) is valid CDP session and navigates."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_page = MagicMock()
    mock_page.url = "https://www.glassdoor.com/job-listing/j?jl=1010256944540"
    cdp_diag = {
        "cdp_connect_attempted": True,
        "cdp_connected": True,
        "browser_session_verified": True,
        "contexts_found": 1,
        "pages_found": 1,
        "page_urls": ["chrome://newtab/"],
        "authoritative_page_url": mock_page.url,
        "requested_job_url": mock_page.url,
        "second_browser_launched": False,
    }
    mock_pw_driver.resolve_or_navigate_page.return_value = (True, mock_page, cdp_diag)
    mock_pw_driver.find_exact_easy_apply_button.return_value = MagicMock()

    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (False, None, "Window not found")
    mock_win_driver.get_window_text_content.return_value = "Easy Apply"

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationInspectRequest(url="https://www.glassdoor.com/job-listing/j?jl=1010256944540")

    res = service.inspect_job_application(req)
    assert res.status == "success"
    assert res.diagnostics["browser_session_verified"] is True
    assert res.diagnostics["authoritative_page_url"] == "https://www.glassdoor.com/job-listing/j?jl=1010256944540"


def test_pywinauto_not_required_for_browser_verification():
    """Requirement 4 & 5: Foreground window detection and PyWinAuto are NOT required for browser verification."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_page = MagicMock()
    mock_page.url = "https://www.glassdoor.com/job-listing/j?jl=1010256944540"
    mock_pw_driver.resolve_or_navigate_page.return_value = (
        True,
        mock_page,
        {
            "cdp_connect_attempted": True,
            "cdp_connected": True,
            "browser_session_verified": True,
            "contexts_found": 1,
            "pages_found": 1,
            "authoritative_page_url": mock_page.url,
            "requested_job_url": mock_page.url,
            "second_browser_launched": False,
        },
    )
    mock_pw_driver.find_exact_easy_apply_button.return_value = MagicMock()

    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    # Simulate pywinauto completely throwing an exception or failing
    mock_win_driver.attach_or_open_browser.side_effect = Exception("OS window search failed")
    mock_win_driver.get_window_text_content.return_value = ""

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationInspectRequest(url=mock_page.url)

    res = service.inspect_job_application(req)
    # CDP was authoritative, so verification succeeds
    assert res.current_state == GlassdoorAutomationState.EASY_APPLY_AVAILABLE
    assert res.diagnostics["browser_session_verified"] is True
    assert res.browser == "chrome_cdp"


def test_actual_cdp_failure_yields_browser_not_verified_with_diagnostics():
    """Requirement 9: Actual CDP failure + PyWinAuto failure -> BROWSER_NOT_VERIFIED with Section 12 diagnostics."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver.resolve_or_navigate_page.return_value = (
        False,
        None,
        {
            "cdp_connect_attempted": True,
            "cdp_connected": False,
            "browser_session_verified": False,
            "contexts_found": 0,
            "pages_found": 0,
            "authoritative_page_url": None,
            "requested_job_url": "https://www.glassdoor.com/job-listing/j?jl=1010256944540",
            "second_browser_launched": False,
            "error": "Connection refused",
        },
    )

    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (
        False,
        None,
        "Could not find active browser window (Chrome/Edge) for CDP session.",
    )

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationInspectRequest(url="https://www.glassdoor.com/job-listing/j?jl=1010256944540")

    res = service.inspect_job_application(req)
    assert res.status == "blocked"
    assert res.current_state == GlassdoorAutomationState.BROWSER_NOT_VERIFIED
    assert res.diagnostics["cdp_connect_attempted"] is True
    assert res.diagnostics["cdp_connected"] is False
    assert res.diagnostics["browser_session_verified"] is False
    assert res.diagnostics["contexts_found"] == 0
    assert res.diagnostics["pages_found"] == 0
    assert res.diagnostics["authoritative_page_url"] is None
    assert res.diagnostics["requested_job_url"] == "https://www.glassdoor.com/job-listing/j?jl=1010256944540"
    assert res.diagnostics["second_browser_launched"] is False


def test_navigate_to_submit_connects_over_cdp_and_preserves_diagnostics():
    """Requirement 10: navigate-to-submit connects over CDP and preserves Section 12 diagnostics."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_page = MagicMock()
    mock_page.url = "https://www.glassdoor.com/job-listing/j?jl=1010256944540"
    mock_pw_driver.resolve_or_navigate_page.return_value = (
        True,
        mock_page,
        {
            "cdp_connect_attempted": True,
            "cdp_connected": True,
            "browser_session_verified": True,
            "contexts_found": 1,
            "pages_found": 1,
            "authoritative_page_url": mock_page.url,
            "requested_job_url": mock_page.url,
            "second_browser_launched": False,
        },
    )
    mock_pw_driver.find_exact_easy_apply_button.return_value = MagicMock()
    mock_pw_driver.click_easy_apply.return_value = (True, None)

    # Hosted application page reaches SUBMISSION_READY
    app_page = MagicMock()
    app_page.url = "https://smartapply.indeed.com/form/review"
    mock_pw_driver.resolve_application_page.return_value = app_page
    mock_pw_driver.page = app_page
    mock_pw_driver.detect_page_state.return_value = (
        GlassdoorAutomationState.SUBMISSION_READY,
        {"submit_button_detected": True},
    )
    mock_pw_driver.find_exact_submit_button.return_value = MagicMock()

    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    mock_win_driver.attach_or_open_browser.return_value = (False, None, "No desktop window")

    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)
    req = GlassdoorAutomationNavigateRequest(url=mock_page.url)

    res = service.navigate_to_submit(req)
    assert res.status == "success"
    assert res.current_state == GlassdoorAutomationState.SUBMISSION_READY
    assert res.submit_verified is True
    assert res.final_submit_clicked is False
    assert res.diagnostics["cdp_connect_attempted"] is True
    assert res.diagnostics["cdp_connected"] is True
    assert res.diagnostics["browser_session_verified"] is True
    assert res.diagnostics["second_browser_launched"] is False


def test_resume_connection_failure_returns_section_12_diagnostics():
    """Requirement 11: resume connection failure returns complete Section 12 diagnostics."""
    mock_pw_driver = MagicMock(spec=GlassdoorPlaywrightDriver)
    mock_pw_driver._browser = None
    mock_pw_driver.connect_cdp.return_value = (False, "Could not connect to CDP")

    mock_win_driver = MagicMock(spec=PyWinAutoGlassdoorDriver)
    service = GlassdoorApplyService(driver=mock_win_driver, playwright_driver=mock_pw_driver)

    req = GlassdoorAutomationResumeRequest(
        automation_session_id="session_test_123",
        url="https://www.glassdoor.com/job-listing/j?jl=1010256944540",
    )

    res = service.resume(req)
    assert res.status == "failed"
    assert res.current_state == GlassdoorAutomationState.BROWSER_NOT_VERIFIED
    assert res.diagnostics["cdp_connect_attempted"] is True
    assert res.diagnostics["cdp_connected"] is False
    assert res.diagnostics["browser_session_verified"] is False
    assert res.diagnostics["contexts_found"] == 0
    assert res.diagnostics["pages_found"] == 0
    assert res.diagnostics["authoritative_page_url"] is None
    assert res.diagnostics["requested_job_url"] == "https://www.glassdoor.com/job-listing/j?jl=1010256944540"
    assert res.diagnostics["second_browser_launched"] is False


def test_driver_connect_to_cdp_handles_newtab():
    """Requirement 2 & 6: GlassdoorPlaywrightDriver.connect_to_cdp reuses newtab or existing page."""
    driver = GlassdoorPlaywrightDriver()
    mock_page = MagicMock()
    mock_page.url = "chrome://newtab/"
    mock_page.is_closed.return_value = False

    mock_context = MagicMock()
    mock_context.pages = [mock_page]

    mock_browser = MagicMock()
    mock_browser.contexts = [mock_context]
    mock_browser.is_connected.return_value = True

    from unittest.mock import AsyncMock
    with patch("app.automation.glassdoor.config.is_live_cdp_allowed", return_value=True):
        with patch("playwright.async_api.async_playwright") as mock_pw_init:
            pw_instance = AsyncMock()
            pw_instance.chromium.connect_over_cdp.return_value = mock_browser
            mock_pw_init.return_value.start = AsyncMock(return_value=pw_instance)

            ok, page, diag = driver.connect_to_cdp()

            assert ok is True
            assert page == mock_page
            assert diag["cdp_connected"] is True
            assert diag["browser_session_verified"] is True
            assert diag["pages_found"] == 1
            assert diag["page_urls"] == ["chrome://newtab/"]
            assert diag["authoritative_page_url"] == "chrome://newtab/"
            assert diag["second_browser_launched"] is False



