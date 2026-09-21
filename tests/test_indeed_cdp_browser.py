"""Unit tests for Indeed Playwright/CDP Browser Attachment (Phase 6.0.1).

Covers all 14 verification requirements:
1. Indeed connects through CDP.
2. Indeed works with chrome://newtab/ and about:blank.
3. Indeed does not require foreground Chrome.
4. Indeed does not require PyWinAuto for browser verification.
5. Existing page reuse.
6. New page creation in context without secondary browser launch.
7. Indeed jk extraction across formats.
8. Invalid Indeed URL rejection.
9. CDP unavailable yields BROWSER_NOT_VERIFIED with structured diagnostics.
10. Indeed CDP safe diagnostics contract (no sensitive tokens/cookies).
11. Question automation unchanged.
12. Resume submission automation unchanged.
13. Human-in-the-loop CAPTCHA detection unchanged.
14. Zero secondary browsers spawned.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import pytest

from app.automation.cdp_browser_manager import PlaywrightCDPBrowserManager
from app.automation.indeed.url_validator import extract_indeed_jk
from app.automation.indeed.apply_service import IndeedApplyService
from app.automation.indeed.field_mapper import IndeedFieldMapper
from app.automation.indeed.models import (
    AutomationState,
    IndeedAutomationApplyRequest,
    IndeedAutomationInspectRequest,
    IndeedAutomationNavigateRequest,
    IndeedAutomationResumeRequest,
)


# ---------------------------------------------------------------------------
# Requirement 7: Indeed jk extraction across formats
# ---------------------------------------------------------------------------

def test_indeed_jk_extraction():
    assert extract_indeed_jk("https://in.indeed.com/viewjob?jk=892187f461c82742") == "892187f461c82742"
    assert extract_indeed_jk("https://www.indeed.com/rc/clk?jk=abcdef1234567890") == "abcdef1234567890"
    assert extract_indeed_jk("https://indeed.com/jobs?jk=xyz123&from=serp") == "xyz123"
    assert extract_indeed_jk("https://indeed.com/viewjob?vjk=vjk12345") == "vjk12345"
    assert extract_indeed_jk("https://in.indeed.com/job/software-engineer-892187f461c82742") == "892187f461c82742"
    assert extract_indeed_jk("https://example.com/other") is None
    assert extract_indeed_jk("") is None
    assert extract_indeed_jk(None) is None


# ---------------------------------------------------------------------------
# Requirement 8: Invalid Indeed URL rejection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_indeed_url_rejection():
    service = IndeedApplyService()
    req = IndeedAutomationInspectRequest(
        job_id=uuid4(),
        url="https://invalid-non-indeed-domain.com/something",
    )
    res = await service.inspect_job_application(req)
    assert res.status == "blocked"
    assert res.current_state in (AutomationState.INVALID_JOB_URL, AutomationState.BROWSER_NOT_VERIFIED)
    assert res.diagnostics.get("cdp_connect_attempted") is False


# ---------------------------------------------------------------------------
# Requirement 1: Indeed connects through CDP
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_indeed_connects_through_cdp():
    manager = PlaywrightCDPBrowserManager(cdp_url="http://127.0.0.1:9222")

    mock_page = AsyncMock()
    mock_page.url = "https://in.indeed.com/viewjob?jk=892187f461c82742"
    mock_page.is_closed = MagicMock(return_value=False)

    mock_context = MagicMock()
    mock_context.pages = [mock_page]

    mock_browser = MagicMock()
    mock_browser.is_connected = MagicMock(return_value=True)
    mock_browser.contexts = [mock_context]

    mock_playwright = MagicMock()
    mock_playwright.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)

    with patch("app.automation.cdp_browser_manager.is_live_cdp_allowed", return_value=True):
        manager._playwright = mock_playwright
        manager._browser = mock_browser

        success, auth_page, diag = await manager.connect_to_cdp()

        assert success is True
        assert auth_page == mock_page
        assert diag["cdp_connected"] is True
        assert diag["cdp_endpoint"] == "http://127.0.0.1:9222"
        assert diag["browser_session_verified"] is True
        assert diag["contexts_found"] == 1
        assert diag["pages_found"] == 1


# ---------------------------------------------------------------------------
# Requirement 2: Indeed works with chrome://newtab/ and about:blank
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_indeed_works_with_chrome_newtab():
    manager = PlaywrightCDPBrowserManager()

    mock_page = AsyncMock()
    mock_page.url = "chrome://newtab/"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.goto = AsyncMock()

    mock_context = MagicMock()
    mock_context.pages = [mock_page]
    mock_context.new_page = AsyncMock()

    mock_browser = MagicMock()
    mock_browser.is_connected = MagicMock(return_value=True)
    mock_browser.contexts = [mock_context]

    with patch.object(manager, "connect_to_cdp", new_callable=AsyncMock) as mock_conn:
        mock_conn.return_value = (True, mock_page, {
            "cdp_endpoint": "http://127.0.0.1:9222",
            "cdp_connect_attempted": True,
            "cdp_connected": True,
            "browser_session_verified": True,
            "contexts_found": 1,
            "pages_found": 1,
            "page_urls": ["chrome://newtab/"],
            "second_browser_launched": False,
        })
        manager._browser = mock_browser

        target_url = "https://in.indeed.com/viewjob?jk=892187f461c82742"
        ok, page, diag = await manager.resolve_or_navigate_page(
            target_url=target_url,
            platform="indeed",
            expected_id="892187f461c82742",
        )

        assert ok is True
        assert page == mock_page
        mock_page.goto.assert_called_once_with(target_url, wait_until="domcontentloaded", timeout=15000)
        mock_context.new_page.assert_not_called()
        assert diag["second_browser_launched"] is False


# ---------------------------------------------------------------------------
# Requirement 3 & 4: No foreground Chrome or PyWinAuto required
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_indeed_does_not_require_foreground_chrome_or_pywinauto():
    service = IndeedApplyService()

    mock_page = AsyncMock()
    mock_page.url = "https://in.indeed.com/viewjob?jk=892187f461c82742"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.inner_text = AsyncMock(return_value="Software Engineer\nApply with Indeed")
    mock_page.locator = MagicMock()
    mock_btn = AsyncMock()
    mock_btn.is_visible = AsyncMock(return_value=True)
    mock_btn.text_content = AsyncMock(return_value="Apply with Indeed")
    mock_page.locator.return_value.first = mock_btn

    mock_browser = MagicMock()
    mock_browser.is_connected = MagicMock(return_value=True)
    mock_context = MagicMock()
    mock_context.pages = [mock_page]

    # Mock PyWinAuto driver failing completely (window in background, minimized, or unsupported)
    mock_driver = MagicMock()
    mock_driver.attach_or_open_browser.return_value = (False, None, "Could not find foreground window")
    mock_driver.verify_browser_window.return_value = (False, "Window minimized")
    mock_driver.find_apply_with_indeed_control.return_value = None
    service.driver = mock_driver

    diag = {
        "cdp_endpoint": "http://127.0.0.1:9222",
        "cdp_connect_attempted": True,
        "cdp_connected": True,
        "browser_session_verified": True,
        "contexts_found": 1,
        "pages_found": 1,
        "page_urls": [mock_page.url],
        "second_browser_launched": False,
    }

    with patch.object(service.cdp_manager, "resolve_or_navigate_page", new_callable=AsyncMock) as mock_res:
        mock_res.return_value = (True, mock_page, diag)

        req = IndeedAutomationInspectRequest(
            job_id=uuid4(),
            url="https://in.indeed.com/viewjob?jk=892187f461c82742",
        )
        res = await service.inspect_job_application(req)

        assert res.status == "success"
        assert res.diagnostics.get("browser_session_verified") is True
        assert res.current_state in (AutomationState.APPLY_WITH_INDEED_AVAILABLE, AutomationState.JOB_PAGE)
        assert res.diagnostics["second_browser_launched"] is False


# ---------------------------------------------------------------------------
# Requirement 5: Existing page reuse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_existing_page_reuse():
    manager = PlaywrightCDPBrowserManager()

    target_url = "https://in.indeed.com/viewjob?jk=892187f461c82742"
    mock_page = AsyncMock()
    mock_page.url = target_url
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.goto = AsyncMock()

    mock_context = MagicMock()
    mock_context.pages = [mock_page]
    mock_context.new_page = AsyncMock()

    mock_browser = MagicMock()
    mock_browser.is_connected = MagicMock(return_value=True)
    mock_browser.contexts = [mock_context]

    with patch.object(manager, "connect_to_cdp", new_callable=AsyncMock) as mock_conn:
        mock_conn.return_value = (True, mock_page, {
            "cdp_endpoint": "http://127.0.0.1:9222",
            "cdp_connect_attempted": True,
            "cdp_connected": True,
            "browser_session_verified": True,
            "contexts_found": 1,
            "pages_found": 1,
            "page_urls": [target_url],
            "second_browser_launched": False,
        })
        manager._browser = mock_browser

        ok, page, diag = await manager.resolve_or_navigate_page(
            target_url=target_url,
            platform="indeed",
            expected_id="892187f461c82742",
        )

        assert ok is True
        assert page == mock_page
        mock_page.goto.assert_not_called()
        mock_context.new_page.assert_not_called()


# ---------------------------------------------------------------------------
# Requirement 6 & 14: New page in existing context & zero secondary browsers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_new_page_creation_in_context_no_secondary_browser():
    manager = PlaywrightCDPBrowserManager()

    existing_page = AsyncMock()
    existing_page.url = "https://www.google.com"
    existing_page.is_closed = MagicMock(return_value=False)

    new_created_page = AsyncMock()
    new_created_page.url = "about:blank"
    new_created_page.is_closed = MagicMock(return_value=False)
    new_created_page.goto = AsyncMock()

    mock_context = MagicMock()
    mock_context.pages = []
    mock_context.new_page = AsyncMock(return_value=new_created_page)

    mock_browser = MagicMock()
    mock_browser.is_connected = MagicMock(return_value=True)
    mock_browser.contexts = [mock_context]

    with patch.object(manager, "connect_to_cdp", new_callable=AsyncMock) as mock_conn:
        mock_conn.return_value = (True, None, {
            "cdp_endpoint": "http://127.0.0.1:9222",
            "cdp_connect_attempted": True,
            "cdp_connected": True,
            "browser_session_verified": True,
            "contexts_found": 1,
            "pages_found": 0,
            "page_urls": [],
            "second_browser_launched": False,
        })
        manager._browser = mock_browser

        target_url = "https://in.indeed.com/viewjob?jk=892187f461c82742"
        ok, page, diag = await manager.resolve_or_navigate_page(
            target_url=target_url,
            platform="indeed",
            expected_id="892187f461c82742",
        )

        assert ok is True
        assert page == new_created_page
        mock_context.new_page.assert_called_once()
        new_created_page.goto.assert_called_once_with(target_url, wait_until="domcontentloaded", timeout=15000)
        assert diag["second_browser_launched"] is False


# ---------------------------------------------------------------------------
# Requirement 9 & 10: CDP unavailable yields BROWSER_NOT_VERIFIED with safe diagnostics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cdp_unavailable_yields_browser_not_verified_with_safe_diagnostics():
    service = IndeedApplyService()
    # Mock CDP failure
    with patch.object(service.cdp_manager, "resolve_or_navigate_page", new_callable=AsyncMock) as mock_res:
        mock_res.return_value = (False, None, {
            "cdp_endpoint": "http://127.0.0.1:9222",
            "cdp_connect_attempted": True,
            "cdp_connected": False,
            "browser_session_verified": False,
            "contexts_found": 0,
            "pages_found": 0,
            "page_urls": [],
            "error": "Failed to connect to CDP endpoint: Connection refused",
            "second_browser_launched": False,
        })

        # Mock driver failure too
        mock_driver = MagicMock()
        mock_driver.attach_or_open_browser.return_value = (False, None, "Could not find browser")
        service.driver = mock_driver

        req = IndeedAutomationInspectRequest(
            job_id=uuid4(),
            url="https://in.indeed.com/viewjob?jk=892187f461c82742",
        )
        res = await service.inspect_job_application(req)

        assert res.status == "blocked"
        assert res.current_state == AutomationState.BROWSER_NOT_VERIFIED
        assert res.diagnostics.get("browser_session_verified") is False
        assert "cdp_endpoint" in res.diagnostics
        assert res.diagnostics["cdp_connected"] is False
        assert res.diagnostics["second_browser_launched"] is False

        # Diagnostic contract: safe keys, no secrets
        diag_str = str(res.diagnostics).lower()
        assert "password" not in diag_str
        assert "cookie" not in diag_str
        assert "bearer" not in diag_str
        assert "token" not in diag_str


# ---------------------------------------------------------------------------
# Requirement 11: Question automation unchanged
# ---------------------------------------------------------------------------

def test_question_automation_unchanged():
    form_elements = [
        {"name": "input_years_exp", "label": "How many years of Python experience do you have?", "type": "text"},
        {"name": "select_degree", "label": "Highest education level achieved", "type": "select"},
    ]
    questions = IndeedFieldMapper.identify_screening_questions(form_elements)
    assert isinstance(questions, list)
    assert len(questions) == 2


# ---------------------------------------------------------------------------
# Requirement 12: Resume submission automation unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resume_submission_automation_unchanged():
    service = IndeedApplyService()
    mock_driver = MagicMock()
    mock_driver.attach_or_open_browser.return_value = (True, "chrome", None)
    mock_driver.verify_browser_window.return_value = (True, None)
    mock_driver.get_window_text_content.return_value = "Review your application\nSubmit application"
    mock_driver.get_focused_control_info.return_value = {
        "name": "Submit application",
        "control_type": "Button",
        "is_enabled": True,
        "is_submit": True,
    }
    mock_driver.send_enter_once.return_value = True
    mock_driver.detect_submission_confirmation.return_value = True
    service.driver = mock_driver

    req = IndeedAutomationResumeRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=892187f461c82742",
    )
    res = await service.resume_submit(req)
    assert res.status == "success"
    assert res.submission_confirmed is True
    assert res.current_state == AutomationState.APPLICATION_SUBMITTED


# ---------------------------------------------------------------------------
# Requirement 13: Human-in-the-loop CAPTCHA detection unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hitl_captcha_detection_unchanged():
    service = IndeedApplyService()
    mock_page = AsyncMock()
    mock_page.url = "https://in.indeed.com/viewjob?jk=892187f461c82742"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.inner_text = AsyncMock(return_value="Please verify you are human. Cloudflare verification required.")

    service.cdp_manager.resolve_or_navigate_page = AsyncMock(return_value=(True, mock_page, {
        "cdp_connected": True,
        "browser_session_verified": True,
        "second_browser_launched": False,
    }))

    req = IndeedAutomationNavigateRequest(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=892187f461c82742",
    )
    res = await service.navigate_to_submit(req)
    assert res.status in ("manual_action_required", "blocked")
    assert res.current_state == AutomationState.CAPTCHA_OR_CHALLENGE
    assert res.manual_action_required is True
    assert res.final_submit_clicked is False
