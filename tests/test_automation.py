"""Unit tests for Playwright browser automation, profile management, and Indeed session adapter."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.automation.browser_profiles import BrowserProfileManager
from app.automation.session_manager import SessionState, SessionStatus
from app.config.settings import Settings, get_settings
from app.platforms.base import PlatformAdapter
from app.platforms.indeed.adapter import IndeedPlatformAdapter
from app.platforms.indeed.session import IndeedSessionManager


# ---------------------------------------------------------------------------
# 1. Configuration & Security Tests
# ---------------------------------------------------------------------------


def test_browser_settings_defaults():
    """Verify default Playwright and Indeed browser settings."""
    settings = Settings()
    assert settings.PLAYWRIGHT_HEADLESS is False
    assert settings.PLAYWRIGHT_BROWSER == "chromium"
    assert settings.PLAYWRIGHT_TIMEOUT_MS == 30000
    assert settings.PLAYWRIGHT_TRACE is False
    assert "browser_sessions/indeed" in settings.INDEED_BROWSER_PROFILE_PATH
    assert "screenshots" in settings.SCREENSHOTS_PATH
    assert "playwright-artifacts" in settings.PLAYWRIGHT_ARTIFACTS_PATH


def test_path_resolution_relative_to_root():
    """Verify relative paths resolve properly to project workspace root."""
    settings = Settings(
        INDEED_BROWSER_PROFILE_PATH="browser_sessions/indeed_test",
        SCREENSHOTS_PATH="screenshots_test",
        PLAYWRIGHT_ARTIFACTS_PATH="traces_test",
    )
    profile_path = settings.get_indeed_profile_path()
    assert profile_path.is_absolute()
    assert str(profile_path).endswith(os.path.join("browser_sessions", "indeed_test"))

    screenshots_path = settings.get_screenshots_path()
    assert screenshots_path.is_absolute()
    assert str(screenshots_path).endswith("screenshots_test")


def test_gitignore_contains_browser_security_exclusions():
    """Verify .gitignore contains exclusions for browser profiles, traces, and screenshots."""
    project_root = Path(__file__).resolve().parent.parent
    gitignore_path = project_root / ".gitignore"
    assert gitignore_path.exists()

    content = gitignore_path.read_text(encoding="utf-8")
    assert "browser_sessions/" in content
    assert "screenshots/" in content
    assert "traces/" in content
    assert "playwright-artifacts/" in content


# ---------------------------------------------------------------------------
# 2. Session Models & Profile Manager Tests
# ---------------------------------------------------------------------------


def test_session_status_model():
    """Verify SessionStatus model serialization and defaults."""
    status = SessionStatus(
        platform="indeed",
        browser="chromium",
        authenticated=True,
        status=SessionState.AUTHENTICATED,
        detected_user="test_user",
        message="Session active",
    )
    assert status.platform == "indeed"
    assert status.authenticated is True
    assert status.status == SessionState.AUTHENTICATED
    assert status.detected_user == "test_user"


def test_browser_profile_manager(tmp_path):
    """Verify BrowserProfileManager creates directory and counts files."""
    manager = BrowserProfileManager()
    test_dir = tmp_path / "indeed_session_profile"
    assert not test_dir.exists()

    ensured = manager.ensure_profile_dir(test_dir)
    assert ensured.exists()
    assert ensured.is_dir()
    assert manager.get_profile_file_count(test_dir) == 0

    # Create dummy file
    (test_dir / "Cookies").write_text("dummy", encoding="utf-8")
    assert manager.profile_exists(test_dir)
    assert manager.get_profile_file_count(test_dir) == 1


# ---------------------------------------------------------------------------
# 3. Indeed Auth State Detection (Mocked Page DOM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_auth_state_authenticated():
    """Verify detection when account menu indicator is present and visible."""
    manager = IndeedSessionManager()
    mock_page = AsyncMock()

    mock_account_element = AsyncMock()
    mock_account_element.is_visible.return_value = True

    async def query_selector_mock(selector):
        if 'button[data-gnav-element-name="AccountMenu"]' in selector:
            return mock_account_element
        return None

    mock_page.query_selector = AsyncMock(side_effect=query_selector_mock)

    is_auth, indicator = await manager.detect_auth_state(mock_page)
    assert is_auth is True
    assert indicator == 'button[data-gnav-element-name="AccountMenu"]'


@pytest.mark.asyncio
async def test_detect_auth_state_unauthenticated():
    """Verify detection when sign-in button is present and visible."""
    manager = IndeedSessionManager()
    mock_page = AsyncMock()

    mock_signin_element = AsyncMock()
    mock_signin_element.is_visible.return_value = True

    async def query_selector_mock(selector):
        if 'a[data-gnav-element-name="SignIn"]' in selector:
            return mock_signin_element
        return None

    mock_page.query_selector = AsyncMock(side_effect=query_selector_mock)
    mock_page.url = "https://www.indeed.com"

    is_auth, indicator = await manager.detect_auth_state(mock_page)
    assert is_auth is False
    assert indicator == 'a[data-gnav-element-name="SignIn"]'


# ---------------------------------------------------------------------------
# 4. Indeed Session & Adapter Tests (Mocked Playwright Lifecycle)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_indeed_session_manager_check_session_status(tmp_path):
    """Verify check_session_status launches persistent context, queries page, and returns SessionStatus."""
    mock_pm = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_context.pages = [mock_page]
    mock_pm.launch_persistent_context.return_value = mock_context

    # Simulate sign-in button present
    mock_signin = AsyncMock()
    mock_signin.is_visible.return_value = True

    async def query_selector_mock(selector):
        if 'a[data-gnav-element-name="SignIn"]' in selector:
            return mock_signin
        return None

    mock_page.query_selector = AsyncMock(side_effect=query_selector_mock)
    mock_page.url = "https://www.indeed.com"

    settings = Settings(INDEED_BROWSER_PROFILE_PATH=str(tmp_path / "indeed"))
    manager = IndeedSessionManager(settings=settings, playwright_manager=mock_pm)

    status = await manager.check_session_status(headless=True)
    assert status.platform == "indeed"
    assert status.authenticated is False
    assert status.status == SessionState.LOGIN_REQUIRED

    mock_pm.launch_persistent_context.assert_called_once()
    mock_pm.close_context.assert_called_once_with(mock_context)


@pytest.mark.asyncio
async def test_indeed_session_manager_open_home_and_screenshot(tmp_path):
    """Verify open_indeed_home navigates and triggers screenshot."""
    mock_pm = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_context.pages = [mock_page]
    mock_pm.launch_persistent_context.return_value = mock_context

    mock_account = AsyncMock()
    mock_account.is_visible.return_value = True

    async def query_selector_mock(selector):
        if 'button[data-gnav-element-name="AccountMenu"]' in selector:
            return mock_account
        return None

    mock_page.query_selector = AsyncMock(side_effect=query_selector_mock)
    mock_page.url = "https://www.indeed.com"

    settings = Settings(
        INDEED_BROWSER_PROFILE_PATH=str(tmp_path / "indeed"),
        SCREENSHOTS_PATH=str(tmp_path / "screenshots"),
    )
    manager = IndeedSessionManager(settings=settings, playwright_manager=mock_pm)

    status, screenshot_path = await manager.open_indeed_home(headless=True, save_screenshot=True)
    assert status.platform == "indeed"
    assert status.authenticated is True
    assert status.status == SessionState.AUTHENTICATED
    assert screenshot_path is not None
    assert str(screenshot_path).endswith("indeed_home.png")

    mock_page.screenshot.assert_called_once()
    mock_pm.close_context.assert_called_once_with(mock_context)


@pytest.mark.asyncio
async def test_indeed_platform_adapter_interface():
    """Verify IndeedPlatformAdapter implements PlatformAdapter contract."""
    mock_manager = AsyncMock(spec=IndeedSessionManager)
    mock_manager.check_session_status.return_value = SessionStatus(
        platform="indeed",
        browser="chromium",
        authenticated=True,
        status=SessionState.AUTHENTICATED,
    )

    adapter = IndeedPlatformAdapter(session_manager=mock_manager)
    assert isinstance(adapter, PlatformAdapter)
    assert adapter.platform_name == "indeed"

    status = await adapter.get_session_status()
    assert status.authenticated is True
    assert status.status == SessionState.AUTHENTICATED


# ---------------------------------------------------------------------------
# 5. FastAPI Automation Router Endpoints
# ---------------------------------------------------------------------------


def test_api_automation_health():
    """Test GET /automation/health endpoint."""
    client = TestClient(app)
    response = client.get("/automation/health")
    assert response.status_code == 200
    data = response.json()

    assert "playwright" in data
    assert data["playwright"]["available"] is True
    assert data["playwright"]["browser"] == "chromium"
    assert "indeed" in data
    assert data["indeed"]["profile_configured"] is True


def test_api_indeed_session_status_endpoint():
    """Test GET /automation/indeed/session endpoint with dependency override."""
    from app.api.routers.automation import get_indeed_session_manager

    client = TestClient(app)
    mock_manager = AsyncMock(spec=IndeedSessionManager)
    mock_manager.check_session_status.return_value = SessionStatus(
        platform="indeed",
        browser="chromium",
        authenticated=False,
        status=SessionState.LOGIN_REQUIRED,
        message="Manual login required",
    )

    app.dependency_overrides[get_indeed_session_manager] = lambda: mock_manager
    try:
        response = client.get("/automation/indeed/session")
        assert response.status_code == 200
        data = response.json()
        assert data["platform"] == "indeed"
        assert data["authenticated"] is False
        assert data["status"] == "login_required"
        assert "login required" in data["message"].lower()
    finally:
        app.dependency_overrides.pop(get_indeed_session_manager, None)


def test_api_root_endpoint_updated_to_phase5():
    """Verify root endpoint reflects Phase 5.1 status."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Phase 5.1" in data["phase"]
