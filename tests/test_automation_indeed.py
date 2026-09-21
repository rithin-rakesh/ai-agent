"""Comprehensive Unit Tests for Indeed PyWinAuto Automation (Phase 5.4).

Tests dual source validation, state detection, UIA discovery, coordinate fallback verification,
redirect handling, 22-TAB navigation, Submit verification, single-Enter execution, and confirmation.
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routers.automation import get_indeed_apply_service
from app.automation.indeed.apply_service import IndeedApplyService
from app.automation.indeed.models import (
    AutomationState,
    IndeedAutomationApplyRequest,
    IndeedAutomationInspectRequest,
    IndeedAutomationNavigateRequest,
    IndeedAutomationResumeRequest,
    IndeedAutomationResult,
)
from app.automation.indeed.pywinauto_driver import PyWinAutoIndeedDriver
from app.automation.indeed.state_detector import IndeedStateDetector
from app.automation.indeed.url_validator import is_indeed_domain, validate_indeed_request


@pytest.fixture(autouse=True)
def enable_auto_submit_for_indeed_tests(monkeypatch):
    monkeypatch.setenv("AUTO_SUBMIT_ENABLED", "true")


# ---------------------------------------------------------------------------
# 1. Dual Source and URL Validation Tests
# ---------------------------------------------------------------------------


def test_is_indeed_domain():
    """Verify supported and rejected domain patterns."""
    assert is_indeed_domain("indeed.com") is True
    assert is_indeed_domain("www.indeed.com") is True
    assert is_indeed_domain("in.indeed.com") is True
    assert is_indeed_domain("uk.indeed.com") is True
    assert is_indeed_domain("ca.indeed.com") is True

    # Non-Indeed domains
    assert is_indeed_domain("linkedin.com") is False
    assert is_indeed_domain("naukri.com") is False
    assert is_indeed_domain("greenhouse.io") is False
    assert is_indeed_domain("fakeindeed.com.evil.org") is False
    assert is_indeed_domain(None) is False
    assert is_indeed_domain("") is False


def test_validate_indeed_request():
    """Verify dual validation of job source and URL."""
    valid_url = "https://in.indeed.com/viewjob?jk=1234567890abcdef"

    # Valid Indeed source + Indeed URL
    ok, norm, err = validate_indeed_request(valid_url, "indeed")
    assert ok is True
    assert norm == valid_url
    assert err is None

    # Default source="indeed"
    ok_def, _, _ = validate_indeed_request(valid_url)
    assert ok_def is True

    # Unsupported Source Platform
    ok_src, _, err_src = validate_indeed_request(valid_url, "linkedin")
    assert ok_src is False
    assert err_src == "UNSUPPORTED_AUTOMATION_SOURCE"

    ok_naukri, _, err_naukri = validate_indeed_request(valid_url, "naukri")
    assert ok_naukri is False
    assert err_naukri == "UNSUPPORTED_AUTOMATION_SOURCE"

    # Invalid URL domain
    ok_ext, _, err_ext = validate_indeed_request("https://company.greenhouse.io/job/999", "indeed")
    assert ok_ext is False
    assert err_ext == "INVALID_JOB_URL"

    # Invalid URL scheme
    ok_scheme, _, err_scheme = validate_indeed_request("ftp://in.indeed.com/viewjob", "indeed")
    assert ok_scheme is False
    assert err_scheme == "INVALID_JOB_URL"

    # Empty URL
    ok_empty, _, err_empty = validate_indeed_request("", "indeed")
    assert ok_empty is False
    assert err_empty == "INVALID_JOB_URL"


# ---------------------------------------------------------------------------
# 2. State Detector Tests
# ---------------------------------------------------------------------------


def test_state_detector_apply_with_indeed():
    """Verify strict detection of 'Apply with Indeed'."""
    assert IndeedStateDetector.is_apply_with_indeed("Apply with Indeed") is True
    assert IndeedStateDetector.is_apply_with_indeed("apply with indeed") is True
    assert IndeedStateDetector.is_apply_with_indeed("Apply With Indeed") is True

    # Must reject generic Apply, Apply now, and external Apply
    assert IndeedStateDetector.is_apply_with_indeed("Apply now") is False
    assert IndeedStateDetector.is_apply_with_indeed("Apply") is False
    assert IndeedStateDetector.is_apply_with_indeed("Apply on company site") is False
    assert IndeedStateDetector.is_apply_with_indeed("Continue to application") is False


def test_state_detector_external_apply():
    """Verify external application indicators."""
    assert IndeedStateDetector.is_external_apply("Apply on company site") is True
    assert IndeedStateDetector.is_external_apply("Apply on employer site") is True
    assert IndeedStateDetector.is_external_apply("Visit company website") is True
    assert IndeedStateDetector.is_external_apply("Apply with Indeed") is False


def test_state_detector_submit_control():
    """Verify final submission button detection strictly matching exact Submit names."""
    assert IndeedStateDetector.is_submit_control("Submit") is True
    assert IndeedStateDetector.is_submit_control("Submit application") is True
    assert IndeedStateDetector.is_submit_control("Submit your application") is True
    assert IndeedStateDetector.is_submit_control("  Submit  Your  Application  ") is True

    # Must reject non-Submit names
    assert IndeedStateDetector.is_submit_control("Review your application") is False
    assert IndeedStateDetector.is_submit_control("Review application") is False
    assert IndeedStateDetector.is_submit_control("Continue") is False
    assert IndeedStateDetector.is_submit_control("Save application") is False
    assert IndeedStateDetector.is_submit_control("Apply") is False
    assert IndeedStateDetector.is_submit_control("Next") is False
    assert IndeedStateDetector.is_submit_control("Cancel") is False
    assert IndeedStateDetector.is_submit_control("Search") is False


def test_state_detector_confirmation():
    """Verify post-submission confirmation text detection."""
    assert IndeedStateDetector.is_submission_confirmation("Application submitted") is True
    assert IndeedStateDetector.is_submission_confirmation("Your application was submitted to Acme Corp") is True
    assert IndeedStateDetector.is_submission_confirmation("Application successfully submitted") is True
    assert IndeedStateDetector.is_submission_confirmation("You applied on Indeed") is True
    assert IndeedStateDetector.is_submission_confirmation("Welcome to Indeed") is False


def test_state_detector_blocking_states():
    """Verify challenge, login, and closed job detection."""
    captcha_res = IndeedStateDetector.detect_blocking_state("Security check - Verify you are human to continue")
    assert captcha_res is not None
    assert captcha_res[0] == AutomationState.CAPTCHA_OR_CHALLENGE

    login_res = IndeedStateDetector.detect_blocking_state("Sign in to Indeed to apply")
    assert login_res is not None
    assert login_res[0] == AutomationState.LOGIN_REQUIRED

    mfa_res = IndeedStateDetector.detect_blocking_state("Enter the verification code sent to your phone")
    assert mfa_res is not None
    assert mfa_res[0] == AutomationState.MFA_REQUIRED

    applied_res = IndeedStateDetector.detect_blocking_state("You have applied to this job")
    assert applied_res is not None
    assert applied_res[0] == AutomationState.ALREADY_APPLIED

    expired_res = IndeedStateDetector.detect_blocking_state("This job has expired and is no longer accepting applications")
    assert expired_res is not None
    assert expired_res[0] == AutomationState.JOB_UNAVAILABLE


# ---------------------------------------------------------------------------
# 3. Apply Service Inspection & Navigation Tests (Mocked Driver)
# ---------------------------------------------------------------------------


def test_inspect_job_application_success_uia_control():
    """Verify successful inspection when UIA control 'Apply with Indeed' is discovered."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Software Engineer - Acme Corp\nApply with Indeed"

    mock_btn = MagicMock()
    mock_btn.element_info.name = "Apply with Indeed"
    mock_btn.is_visible.return_value = True
    mock_btn.is_enabled.return_value = True
    mock_driver.find_apply_with_indeed_control.return_value = mock_btn

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationInspectRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=111",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.inspect_job_application(req)

    assert result.status == "success"
    assert result.current_state == AutomationState.APPLY_WITH_INDEED_AVAILABLE
    assert result.button_verified is True
    assert result.apply_method == "uia_control"
    assert result.apply_clicked is False  # Zero clicks in inspect


def test_inspect_job_application_success_coordinate_fallback():
    """Verify successful inspection via verified fallback coordinate (420, 563)."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Lead Developer - TechCorp"
    mock_driver.find_apply_with_indeed_control.return_value = None  # UIA direct find misses
    mock_driver.inspect_element_at_coordinates.return_value = (True, "Apply with Indeed", MagicMock())

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationInspectRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=222",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.inspect_job_application(req)

    assert result.status == "success"
    assert result.current_state == AutomationState.APPLY_WITH_INDEED_AVAILABLE
    assert result.button_verified is True
    assert result.apply_method == "coordinate_fallback"
    assert result.apply_coordinates == [420, 563]
    assert result.apply_clicked is False


def test_inspect_job_application_external_apply_blocked():
    """Verify 'Apply on company site' stops automation immediately."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "edge", None)
    mock_driver.get_window_text_content.return_value = "Senior Analyst - BigCorp\nApply on company site"

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationInspectRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=333",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.inspect_job_application(req)

    assert result.status == "blocked"
    assert result.current_state == AutomationState.EXTERNAL_APPLY
    assert result.button_verified is False
    assert result.apply_clicked is False


def test_inspect_job_application_captcha_requires_manual_action():
    """Verify CAPTCHA challenge stops automation and flags manual_action_required=True."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Security check - Verify you are human"

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationInspectRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=444",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.inspect_job_application(req)

    assert result.status == "manual_action_required"
    assert result.current_state == AutomationState.CAPTCHA_OR_CHALLENGE
    assert result.manual_action_required is True


def test_navigate_to_submit_stops_at_submission_ready():
    """Verify navigation with focused Submit control returns SUBMISSION_READY via focused_control strategy."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.side_effect = [
        "AI Engineer\nApply with Indeed",               # Initial inspect
        "Review your application\nContact information", # Post-click redirect
        "Review your application\nSubmit",              # Verification
    ]

    mock_btn = MagicMock()
    mock_btn.element_info.name = "Apply with Indeed"
    mock_driver.find_apply_with_indeed_control.return_value = mock_btn
    mock_driver.click_apply_control.return_value = (True, "uia_control", None, None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.send_tab_sequence.return_value = 22
    mock_driver.get_focused_control_info.return_value = {
        "name": "Submit application",
        "control_type": "Button",
        "is_enabled": True,
        "is_visible": True,
        "is_submit": True,
    }

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationNavigateRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=555",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.navigate_to_submit(req)

    assert result.status == "success"
    assert result.current_state == AutomationState.SUBMISSION_READY
    assert result.button_verified is True
    assert result.apply_clicked is True
    assert result.tabs_sent == 22
    assert result.submit_verified is True
    assert result.final_submit_clicked is False  # Stopped at SUBMISSION_READY
    assert result.diagnostics["submit_verification_strategy"] == "focused_control"
    mock_driver.send_enter_once.assert_not_called()


def test_navigate_to_submit_fallback_uia_scan_single_candidate():
    """Verify navigation when focused control is empty but UIA scan finds exactly 1 Submit button."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.side_effect = [
        "AI Engineer\nApply with Indeed",
        "Review your application",
    ]

    mock_btn = MagicMock()
    mock_btn.element_info.name = "Apply with Indeed"
    mock_driver.find_apply_with_indeed_control.return_value = mock_btn
    mock_driver.click_apply_control.return_value = (True, "uia_control", None, None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.send_tab_sequence.return_value = 22

    # Focused control returns empty
    mock_driver.get_focused_control_info.return_value = {
        "name": "",
        "control_type": "",
        "is_enabled": False,
        "is_visible": False,
        "is_submit": False,
    }

    # UIA scan finds exactly 1 valid Submit control
    mock_submit_elem = MagicMock()
    submit_meta = {
        "name": "Submit your application",
        "control_type": "Button",
        "is_enabled": True,
        "is_visible": True,
        "rectangle": {"left": 400, "top": 600, "right": 600, "bottom": 650, "width": 200, "height": 50},
    }
    mock_driver.find_exact_submit_controls.return_value = [(mock_submit_elem, submit_meta)]

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationNavigateRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=556",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.navigate_to_submit(req)

    assert result.status == "success"
    assert result.current_state == AutomationState.SUBMISSION_READY
    assert result.submit_verified is True
    assert result.final_submit_clicked is False
    assert result.diagnostics["submit_verification_strategy"] == "uia_submit_scan"
    assert result.diagnostics["submit_control"]["name"] == "Submit your application"


def test_navigate_to_submit_unverified_focus_and_zero_uia_candidates():
    """Verify navigation returns SUBMIT_FOCUS_UNVERIFIED when focus is empty and 0 Submit buttons found."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.side_effect = [
        "AI Engineer\nApply with Indeed",
        "Review your application",
    ]

    mock_btn = MagicMock()
    mock_btn.element_info.name = "Apply with Indeed"
    mock_driver.find_apply_with_indeed_control.return_value = mock_btn
    mock_driver.click_apply_control.return_value = (True, "uia_control", None, None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.send_tab_sequence.return_value = 22
    mock_driver.get_focused_control_info.return_value = {
        "name": "",
        "control_type": "",
        "is_enabled": False,
        "is_visible": False,
        "is_submit": False,
    }
    mock_driver.find_exact_submit_controls.return_value = []

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationNavigateRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=666",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.navigate_to_submit(req)

    assert result.status == "blocked"
    assert result.current_state == AutomationState.SUBMIT_FOCUS_UNVERIFIED
    assert result.submit_verified is False
    assert result.final_submit_clicked is False
    assert result.diagnostics["submit_verification_strategy"] == "none"
    mock_driver.send_enter_once.assert_not_called()


def test_navigate_to_submit_unverified_focus_multiple_uia_candidates():
    """Verify navigation returns SUBMIT_FOCUS_UNVERIFIED when multiple Submit buttons found (ambiguous)."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.side_effect = [
        "AI Engineer\nApply with Indeed",
        "Review your application",
    ]

    mock_btn = MagicMock()
    mock_btn.element_info.name = "Apply with Indeed"
    mock_driver.find_apply_with_indeed_control.return_value = mock_btn
    mock_driver.click_apply_control.return_value = (True, "uia_control", None, None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.send_tab_sequence.return_value = 22
    mock_driver.get_focused_control_info.return_value = {
        "name": "",
        "control_type": "",
        "is_enabled": False,
        "is_visible": False,
        "is_submit": False,
    }

    # Multiple candidates
    elem1 = MagicMock()
    elem2 = MagicMock()
    meta1 = {"name": "Submit", "control_type": "Button", "is_enabled": True, "is_visible": True, "rectangle": {"width": 50, "height": 20}}
    meta2 = {"name": "Submit application", "control_type": "Button", "is_enabled": True, "is_visible": True, "rectangle": {"width": 100, "height": 30}}
    mock_driver.find_exact_submit_controls.return_value = [(elem1, meta1), (elem2, meta2)]

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationNavigateRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=667",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.navigate_to_submit(req)

    assert result.status == "blocked"
    assert result.current_state == AutomationState.SUBMIT_FOCUS_UNVERIFIED
    assert result.submit_verified is False
    assert result.diagnostics["submit_verification_strategy"] == "ambiguous"
    assert result.diagnostics["submit_candidates_count"] == 2
    mock_driver.send_enter_once.assert_not_called()


def test_apply_to_job_successful_submission_focused_control():
    """Verify full apply workflow via focused control strategy (single Enter)."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.side_effect = [
        "AI Engineer\nApply with Indeed",
        "Review your application",
        "Review your application\nSubmit",
    ]

    mock_btn = MagicMock()
    mock_btn.element_info.name = "Apply with Indeed"
    mock_driver.find_apply_with_indeed_control.return_value = mock_btn
    mock_driver.click_apply_control.return_value = (True, "uia_control", None, None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.send_tab_sequence.return_value = 22
    mock_driver.get_focused_control_info.return_value = {
        "name": "Submit application",
        "control_type": "Button",
        "is_enabled": True,
        "is_visible": True,
        "is_submit": True,
    }
    mock_driver.send_enter_once.return_value = True
    mock_driver.detect_submission_confirmation.return_value = True  # Post-submit confirmed!

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationApplyRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=777",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.apply_to_job(req)

    assert result.status == "success"
    assert result.current_state == AutomationState.APPLICATION_SUBMITTED
    assert result.submit_verified is True
    assert result.final_submit_clicked is True
    assert result.submission_confirmed is True
    mock_driver.send_enter_once.assert_called_once()
    mock_driver.click_submit_control.assert_not_called()


def test_apply_to_job_successful_submission_fallback_uia_scan():
    """Verify full apply workflow via fallback UIA scan (invokes verified Submit button once, NO blind Enter)."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.side_effect = [
        "AI Engineer\nApply with Indeed",
        "Review your application",
        "Review your application\nSubmit",
    ]

    mock_btn = MagicMock()
    mock_btn.element_info.name = "Apply with Indeed"
    mock_driver.find_apply_with_indeed_control.return_value = mock_btn
    mock_driver.click_apply_control.return_value = (True, "uia_control", None, None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.send_tab_sequence.return_value = 22

    # Focus is empty
    mock_driver.get_focused_control_info.return_value = {
        "name": "",
        "control_type": "",
        "is_enabled": False,
        "is_visible": False,
        "is_submit": False,
    }

    # Exactly 1 UIA Submit control
    mock_submit_elem = MagicMock()
    submit_meta = {
        "name": "Submit your application",
        "control_type": "Button",
        "is_enabled": True,
        "is_visible": True,
        "rectangle": {"left": 400, "top": 600, "right": 600, "bottom": 650, "width": 200, "height": 50},
    }
    mock_driver.find_exact_submit_controls.return_value = [(mock_submit_elem, submit_meta)]
    mock_driver.click_submit_control.return_value = (True, None)
    mock_driver.detect_submission_confirmation.return_value = True

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationApplyRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=778",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.apply_to_job(req)

    assert result.status == "success"
    assert result.current_state == AutomationState.APPLICATION_SUBMITTED
    assert result.submit_verified is True
    assert result.final_submit_clicked is True
    assert result.submission_confirmed is True
    # Crucial check: click_submit_control was called once on mock_submit_elem, send_enter_once was NOT called
    mock_driver.click_submit_control.assert_called_once_with(mock_submit_elem)
    mock_driver.send_enter_once.assert_not_called()


def test_apply_to_job_missing_confirmation_handled_safely():
    """Verify when Submit is activated but confirmation is not detected, returns SUBMISSION_CONFIRMATION_UNVERIFIED without duplicate submission."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.side_effect = [
        "AI Engineer\nApply with Indeed",
        "Review your application",
        "Review your application\nSubmit",
    ]

    mock_btn = MagicMock()
    mock_btn.element_info.name = "Apply with Indeed"
    mock_driver.find_apply_with_indeed_control.return_value = mock_btn
    mock_driver.click_apply_control.return_value = (True, "uia_control", None, None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.send_tab_sequence.return_value = 22
    mock_driver.get_focused_control_info.return_value = {
        "name": "Submit application",
        "control_type": "Button",
        "is_enabled": True,
        "is_visible": True,
        "is_submit": True,
    }
    mock_driver.send_enter_once.return_value = True
    mock_driver.detect_submission_confirmation.return_value = False  # Confirmation timed out

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationApplyRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=888",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.apply_to_job(req)

    assert result.status == "blocked"
    assert result.current_state == AutomationState.SUBMISSION_CONFIRMATION_UNVERIFIED
    assert result.final_submit_clicked is True
    assert result.submission_confirmed is False
    mock_driver.send_enter_once.assert_called_once()  # Exactly once, no duplicates
    mock_driver.click_submit_control.assert_not_called()


def test_non_windows_platform_guard():
    """Verify non-Windows platform returns UNSUPPORTED_PLATFORM gracefully without crashing."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationInspectRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=999",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=False):
        result = service.inspect_job_application(req)

    assert result.status == "blocked"
    assert result.current_state == AutomationState.UNSUPPORTED_PLATFORM


# ---------------------------------------------------------------------------
# 4. FastAPI Endpoints Integration Tests
# ---------------------------------------------------------------------------


def test_api_indeed_inspect_endpoint_p54():
    """Verify POST /automation/indeed/inspect with Phase 5.4 payload."""
    client = TestClient(app)
    mock_service = MagicMock(spec=IndeedApplyService)
    sample_id = uuid4()

    mock_service.inspect_job_application.return_value = IndeedAutomationResult(
        status="success",
        job_id=sample_id,
        url="https://in.indeed.com/viewjob?jk=api101",
        platform="indeed",
        browser="chrome",
        button_text="Apply with Indeed",
        button_verified=True,
        apply_clicked=False,
        current_state=AutomationState.APPLY_WITH_INDEED_AVAILABLE,
    )

    app.dependency_overrides[get_indeed_apply_service] = lambda: mock_service
    try:
        resp = client.post(
            "/automation/indeed/inspect",
            json={"job_id": str(sample_id), "url": "https://in.indeed.com/viewjob?jk=api101", "source": "indeed"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["button_verified"] is True
        assert data["current_state"] == "APPLY_WITH_INDEED_AVAILABLE"
    finally:
        app.dependency_overrides.pop(get_indeed_apply_service, None)


def test_api_indeed_navigate_to_submit_endpoint():
    """Verify POST /automation/indeed/navigate-to-submit endpoint."""
    client = TestClient(app)
    mock_service = MagicMock(spec=IndeedApplyService)
    sample_id = uuid4()

    mock_service.navigate_to_submit.return_value = IndeedAutomationResult(
        status="success",
        job_id=sample_id,
        url="https://in.indeed.com/viewjob?jk=api102",
        platform="indeed",
        button_verified=True,
        apply_clicked=True,
        tabs_sent=22,
        submit_verified=True,
        final_submit_clicked=False,
        current_state=AutomationState.SUBMISSION_READY,
    )

    app.dependency_overrides[get_indeed_apply_service] = lambda: mock_service
    try:
        resp = client.post(
            "/automation/indeed/navigate-to-submit",
            json={"job_id": str(sample_id), "url": "https://in.indeed.com/viewjob?jk=api102"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_state"] == "SUBMISSION_READY"
        assert data["submit_verified"] is True
        assert data["final_submit_clicked"] is False
    finally:
        app.dependency_overrides.pop(get_indeed_apply_service, None)


def test_api_indeed_apply_endpoint():
    """Verify POST /automation/indeed/apply endpoint."""
    client = TestClient(app)
    mock_service = MagicMock(spec=IndeedApplyService)
    sample_id = uuid4()

    mock_service.apply_to_job.return_value = IndeedAutomationResult(
        status="success",
        job_id=sample_id,
        url="https://in.indeed.com/viewjob?jk=api103",
        platform="indeed",
        button_verified=True,
        apply_clicked=True,
        tabs_sent=22,
        submit_verified=True,
        final_submit_clicked=True,
        submission_confirmed=True,
        current_state=AutomationState.APPLICATION_SUBMITTED,
    )

    app.dependency_overrides[get_indeed_apply_service] = lambda: mock_service
    try:
        resp = client.post(
            "/automation/indeed/apply",
            json={"job_id": str(sample_id), "url": "https://in.indeed.com/viewjob?jk=api103"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_state"] == "APPLICATION_SUBMITTED"
        assert data["final_submit_clicked"] is True
        assert data["submission_confirmed"] is True
    finally:
        app.dependency_overrides.pop(get_indeed_apply_service, None)


def test_apply_to_job_captcha_detected_before_submission_blocks():
    """Verify when CAPTCHA appears immediately before Submit, automation halts with manual_action_required."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.side_effect = [
        "AI Engineer\nApply with Indeed",               # 1. inspect
        "Review your application\nContact information", # 2. redirect in navigate_to_submit
        "Security check: I'm not a robot. Please solve the puzzle.", # 3. pre-submission challenge!
    ]

    mock_btn = MagicMock()
    mock_btn.element_info.name = "Apply with Indeed"
    mock_driver.find_apply_with_indeed_control.return_value = mock_btn
    mock_driver.click_apply_control.return_value = (True, "uia_control", None, None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.send_tab_sequence.return_value = 22
    mock_driver.get_focused_control_info.return_value = {
        "name": "Submit application",
        "control_type": "Button",
        "is_enabled": True,
        "is_visible": True,
        "is_submit": True,
    }

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationApplyRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=cap001",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.apply_to_job(req)

    assert result.status == "blocked"
    assert result.current_state == AutomationState.CAPTCHA_OR_CHALLENGE
    assert result.manual_action_required is True
    assert result.final_submit_clicked is False
    assert result.submission_confirmed is False
    # CRUCIAL: Zero clicks or Enters to Submit or CAPTCHA
    mock_driver.send_enter_once.assert_not_called()
    mock_driver.click_submit_control.assert_not_called()


def test_resume_submit_captcha_still_active_blocks():
    """Verify resume-submit halts and remains blocked if CAPTCHA challenge is still present."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.get_window_text_content.return_value = "Security check: Verify you are human\nreCAPTCHA"

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationResumeRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=cap002",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.resume_submit(req)

    assert result.status == "blocked"
    assert result.current_state == AutomationState.CAPTCHA_OR_CHALLENGE
    assert result.manual_action_required is True
    assert result.final_submit_clicked is False
    assert result.submission_confirmed is False
    mock_driver.send_enter_once.assert_not_called()
    mock_driver.click_submit_control.assert_not_called()


def test_resume_submit_after_captcha_solved_success():
    """Verify resume-submit activates Submit once and confirms submission after challenge is resolved."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.get_window_text_content.return_value = "Review your application\nContact information\nResume"
    mock_driver.get_focused_control_info.return_value = {
        "name": "Submit your application",
        "control_type": "Button",
        "is_enabled": True,
        "is_visible": True,
        "is_submit": True,
    }
    mock_driver.send_enter_once.return_value = True
    mock_driver.detect_submission_confirmation.return_value = True

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationResumeRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=cap003",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.resume_submit(req)

    assert result.status == "success"
    assert result.current_state == AutomationState.APPLICATION_SUBMITTED
    assert result.submit_verified is True
    assert result.final_submit_clicked is True
    assert result.submission_confirmed is True
    mock_driver.send_enter_once.assert_called_once()


def test_resume_submit_unverified_submit_blocks():
    """Verify resume-submit blocks with SUBMIT_FOCUS_UNVERIFIED if no valid Submit control exists."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.get_window_text_content.return_value = "Random page content"
    mock_driver.get_focused_control_info.return_value = {
        "name": "",
        "control_type": "",
        "is_enabled": False,
        "is_visible": False,
        "is_submit": False,
    }
    mock_driver.find_exact_submit_controls.return_value = []

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationResumeRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=cap004",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.resume_submit(req)

    assert result.status == "blocked"
    assert result.current_state == AutomationState.SUBMIT_FOCUS_UNVERIFIED
    assert result.submit_verified is False
    assert result.final_submit_clicked is False
    mock_driver.send_enter_once.assert_not_called()
    mock_driver.click_submit_control.assert_not_called()


def test_resume_submit_missing_confirmation_no_duplicate_submit():
    """Verify resume-submit does not retry or double-submit when confirmation is not detected."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.get_window_text_content.return_value = "Review your application"
    mock_driver.get_focused_control_info.return_value = {
        "name": "Submit your application",
        "control_type": "Button",
        "is_enabled": True,
        "is_visible": True,
        "is_submit": True,
    }
    mock_driver.send_enter_once.return_value = True
    mock_driver.detect_submission_confirmation.return_value = False  # Confirmation timeout

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationResumeRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=cap005",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.resume_submit(req)

    assert result.status == "blocked"
    assert result.current_state == AutomationState.SUBMISSION_CONFIRMATION_UNVERIFIED
    assert result.final_submit_clicked is True
    assert result.submission_confirmed is False
    mock_driver.send_enter_once.assert_called_once()  # Strictly once


def test_api_indeed_resume_submit_endpoint():
    """Verify POST /automation/indeed/resume-submit endpoint."""
    client = TestClient(app)
    mock_service = MagicMock(spec=IndeedApplyService)
    sample_id = uuid4()

    mock_service.resume_submit.return_value = IndeedAutomationResult(
        status="success",
        job_id=sample_id,
        url="https://in.indeed.com/viewjob?jk=api104",
        platform="indeed",
        button_verified=True,
        apply_clicked=True,
        tabs_sent=0,
        submit_verified=True,
        final_submit_clicked=True,
        submission_confirmed=True,
        current_state=AutomationState.APPLICATION_SUBMITTED,
    )

    app.dependency_overrides[get_indeed_apply_service] = lambda: mock_service
    try:
        resp = client.post(
            "/automation/indeed/resume-submit",
            json={"job_id": str(sample_id), "url": "https://in.indeed.com/viewjob?jk=api104"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_state"] == "APPLICATION_SUBMITTED"
        assert data["final_submit_clicked"] is True
        assert data["submission_confirmed"] is True
    finally:
        app.dependency_overrides.pop(get_indeed_apply_service, None)


# ---------------------------------------------------------------------------
# 6. Multi-Job Navigation & Stale Page Rejection Tests
# ---------------------------------------------------------------------------


def test_inspect_job_application_navigates_to_target_url():
    """Verify inspect_job_application calls driver with explicit navigation and expected jk."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Software Engineer | Indeed"
    mock_driver.find_apply_with_indeed_control.return_value = None
    mock_driver.inspect_element_at_coordinates.return_value = (False, None, None)

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationInspectRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=targetjk123",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.inspect_job_application(req)

    mock_driver.attach_or_open_browser.assert_called_once_with(
        "https://in.indeed.com/viewjob?jk=targetjk123",
        expected_jk="targetjk123",
        force_navigate=True,
    )
    assert result.current_state == AutomationState.APPLY_WITH_INDEED_NOT_VERIFIED
    assert result.diagnostics.get("expected_jk") == "targetjk123"


def test_inspect_job_application_stale_confirmation_returns_job_navigation_failed():
    """Verify inspect_job_application detects stale submission page from prior task and rejects it."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Your application has been submitted | Indeed"

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationInspectRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=stalecheck456",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.inspect_job_application(req)

    assert result.status == "blocked"
    assert result.current_state == AutomationState.JOB_NAVIGATION_FAILED
    assert "Stale submission confirmation page" in result.message
    assert result.apply_clicked is False
    assert result.button_verified is False


def test_inspect_job_application_missing_apply_with_indeed_returns_not_verified():
    """Verify job page with no Apply with Indeed button ends cleanly with APPLY_WITH_INDEED_NOT_VERIFIED."""
    mock_driver = MagicMock(spec=PyWinAutoIndeedDriver)
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.get_window_text_content.return_value = "Staff Software Engineer - Acme Corp - Bengaluru"
    mock_driver.find_apply_with_indeed_control.return_value = None
    mock_driver.inspect_element_at_coordinates.return_value = (False, None, None)

    service = IndeedApplyService(driver=mock_driver)
    req = IndeedAutomationInspectRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=missingapply789",
        source="indeed",
    )

    with patch("app.automation.indeed.apply_service.is_windows", return_value=True):
        result = service.inspect_job_application(req)

    assert result.current_state == AutomationState.APPLY_WITH_INDEED_NOT_VERIFIED
    assert result.button_verified is False
    assert result.apply_clicked is False
    assert result.diagnostics.get("expected_jk") == "missingapply789"
