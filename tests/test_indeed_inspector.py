"""Unit tests for Indeed job page navigation, metadata extraction, and application flow inspection."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.automation.session_manager import SessionState, SessionStatus
from app.config.settings import Settings
from app.platforms.indeed.inspector import (
    FormFieldInfo,
    IndeedJobInspectionResult,
    IndeedJobInspector,
)


# ---------------------------------------------------------------------------
# 1. URL Domain Validation Tests
# ---------------------------------------------------------------------------


def test_validate_indeed_url_valid_domains():
    """Verify URL validator accepts valid Indeed domains and subdomains."""
    assert IndeedJobInspector.validate_indeed_url("https://www.indeed.com/viewjob?jk=123456")
    assert IndeedJobInspector.validate_indeed_url("https://in.indeed.com/job/software-engineer-789")
    assert IndeedJobInspector.validate_indeed_url("https://uk.indeed.com/jobs?q=python")
    assert IndeedJobInspector.validate_indeed_url("http://indeed.com/viewjob?jk=abc")


def test_validate_indeed_url_invalid_domains():
    """Verify URL validator rejects invalid or external domains."""
    assert not IndeedJobInspector.validate_indeed_url("https://evil.com/fakejob")
    assert not IndeedJobInspector.validate_indeed_url("https://phishing-indeed.com/login")
    assert not IndeedJobInspector.validate_indeed_url("ftp://indeed.com/job")
    assert not IndeedJobInspector.validate_indeed_url("")
    assert not IndeedJobInspector.validate_indeed_url(None)


# ---------------------------------------------------------------------------
# 2. Session & Policy Guard Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspect_job_unauthenticated_returns_login_required():
    """Verify that unauthenticated session immediately halts with LOGIN_REQUIRED."""
    mock_session_mgr = AsyncMock()
    mock_session_mgr.check_session_status.return_value = SessionStatus(
        platform="indeed",
        browser="chromium",
        authenticated=False,
        status=SessionState.LOGIN_REQUIRED,
    )

    inspector = IndeedJobInspector(session_manager=mock_session_mgr)
    result = await inspector.inspect_job("https://www.indeed.com/viewjob?jk=12345")

    assert result.authenticated is False
    assert result.status == "login_required"
    assert "login required" in result.reason.lower()
    mock_session_mgr.check_session_status.assert_called_once()


@pytest.mark.asyncio
async def test_inspect_job_invalid_url():
    """Verify invalid URL returns error without launching browser."""
    mock_pm = AsyncMock()
    inspector = IndeedJobInspector(playwright_manager=mock_pm)

    result = await inspector.inspect_job("https://attacker.com/malicious")
    assert result.status == "error"
    assert "Invalid Indeed URL" in result.reason
    mock_pm.launch_persistent_context.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Application Flow & Form Discovery Tests (Mocked Page DOM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspect_job_indeed_hosted_apply():
    """Verify detection of Indeed-hosted application (Easily apply / Apply now)."""
    mock_pm = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_context.pages = [mock_page]
    mock_pm.launch_persistent_context.return_value = mock_context

    mock_session_mgr = AsyncMock()
    mock_session_mgr.check_session_status.return_value = SessionStatus(
        platform="indeed",
        browser="chromium",
        authenticated=True,
        status=SessionState.AUTHENTICATED,
    )

    # Mock page properties
    mock_page.url = "https://www.indeed.com/viewjob?jk=abc12345"
    mock_page.title.return_value = "Senior Python Developer - TechCorp - Indeed.com"
    mock_page.content.return_value = "<html><body>Job posting content</body></html>"

    # Mock DOM queries for metadata
    mock_title_el = AsyncMock()
    mock_title_el.is_visible.return_value = True
    mock_title_el.inner_text.return_value = "Senior Python Developer"

    mock_company_el = AsyncMock()
    mock_company_el.is_visible.return_value = True
    mock_company_el.inner_text.return_value = "TechCorp Global"

    mock_location_el = AsyncMock()
    mock_location_el.is_visible.return_value = True
    mock_location_el.inner_text.return_value = "Remote, US"

    mock_apply_btn = AsyncMock()
    mock_apply_btn.is_visible.return_value = True
    mock_apply_btn.inner_text.return_value = "Apply now"

    async def query_selector_mock(selector):
        if 'h1[data-testid="jobsearch-JobInfoHeader-title"]' in selector or '[data-testid="jobsearch-JobInfoHeader-title"]' in selector:
            return mock_title_el
        if '[data-testid="inlineHeader-companyName"]' in selector:
            return mock_company_el
        if '[data-testid="inlineHeader-companyLocation"]' in selector:
            return mock_location_el
        if '#indeedApplyButton' in selector:
            return mock_apply_btn
        return None

    mock_page.query_selector = AsyncMock(side_effect=query_selector_mock)
    mock_page.query_selector_all.return_value = []

    inspector = IndeedJobInspector(playwright_manager=mock_pm, session_manager=mock_session_mgr)
    result = await inspector.inspect_job("https://www.indeed.com/viewjob?jk=abc12345", save_screenshot=False)

    assert result.authenticated is True
    assert result.job_title == "Senior Python Developer"
    assert result.company == "TechCorp Global"
    assert result.location == "Remote, US"
    assert result.application_method == "indeed_hosted"
    assert result.apply_button_found is True
    assert result.apply_button_text == "Apply now"
    assert result.external_url is None
    assert result.status == "review_required"


@pytest.mark.asyncio
async def test_inspect_job_external_apply():
    """Verify detection of External employer site application."""
    mock_pm = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_context.pages = [mock_page]
    mock_pm.launch_persistent_context.return_value = mock_context

    mock_session_mgr = AsyncMock()
    mock_session_mgr.check_session_status.return_value = SessionStatus(
        platform="indeed",
        browser="chromium",
        authenticated=True,
        status=SessionState.AUTHENTICATED,
    )

    mock_page.url = "https://www.indeed.com/viewjob?jk=ext123"
    mock_page.title.return_value = "DevOps Engineer - ScaleUp"
    mock_page.content.return_value = "<html><body>Job posting content</body></html>"

    mock_external_btn = AsyncMock()
    mock_external_btn.is_visible.return_value = True
    mock_external_btn.inner_text.return_value = "Apply on company site"
    mock_external_btn.get_attribute.return_value = "https://scaleup.greenhouse.io/jobs/999"

    async def query_selector_mock(selector):
        if 'a:has-text("Apply on company site")' in selector:
            return mock_external_btn
        return None

    mock_page.query_selector = AsyncMock(side_effect=query_selector_mock)
    mock_page.query_selector_all.return_value = []

    inspector = IndeedJobInspector(playwright_manager=mock_pm, session_manager=mock_session_mgr)
    result = await inspector.inspect_job("https://www.indeed.com/viewjob?jk=ext123", save_screenshot=False)

    assert result.application_method == "external"
    assert result.apply_button_found is True
    assert result.apply_button_text == "Apply on company site"
    assert result.external_url == "https://scaleup.greenhouse.io/jobs/999"


@pytest.mark.asyncio
async def test_inspect_job_no_apply_button():
    """Verify handling when no apply controls exist (e.g. expired job)."""
    mock_pm = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_context.pages = [mock_page]
    mock_pm.launch_persistent_context.return_value = mock_context

    mock_session_mgr = AsyncMock()
    mock_session_mgr.check_session_status.return_value = SessionStatus(
        platform="indeed",
        browser="chromium",
        authenticated=True,
        status=SessionState.AUTHENTICATED,
    )

    mock_page.url = "https://www.indeed.com/viewjob?jk=closed123"
    mock_page.title.return_value = "Expired Job Posting"
    mock_page.content.return_value = "<html><body>Job is closed</body></html>"
    mock_page.query_selector.return_value = None
    mock_page.query_selector_all.return_value = []

    inspector = IndeedJobInspector(playwright_manager=mock_pm, session_manager=mock_session_mgr)
    result = await inspector.inspect_job("https://www.indeed.com/viewjob?jk=closed123", save_screenshot=False)

    assert result.application_method == "none"
    assert result.apply_button_found is False


@pytest.mark.asyncio
async def test_inspect_job_form_field_discovery():
    """Verify read-only inspection of accessible form fields without extracting user values."""
    mock_pm = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_context.pages = [mock_page]
    mock_pm.launch_persistent_context.return_value = mock_context

    mock_session_mgr = AsyncMock()
    mock_session_mgr.check_session_status.return_value = SessionStatus(
        platform="indeed",
        browser="chromium",
        authenticated=True,
        status=SessionState.AUTHENTICATED,
    )

    mock_page.url = "https://www.indeed.com/viewjob?jk=form123"
    mock_page.title.return_value = "Fast Apply Job"
    mock_page.content.return_value = "<html><body>Fast apply form</body></html>"
    mock_page.query_selector.return_value = None

    # Mock application form
    mock_form = AsyncMock()
    mock_form.is_visible.return_value = True
    mock_form.get_attribute.side_effect = lambda attr: "application" if attr == "aria-label" else None
    mock_form.inner_text.return_value = "Submit your application and resume"
    mock_form.query_selector.return_value = AsyncMock()  # file upload exists

    # Mock form input fields
    input_1 = AsyncMock()
    input_1.is_visible.return_value = True
    input_1.get_attribute.side_effect = lambda attr: "text" if attr == "type" else ("full_name" if attr == "name" else "Full Name")
    mock_prop_1 = AsyncMock()
    mock_prop_1.json_value.return_value = "INPUT"
    input_1.get_property.return_value = mock_prop_1

    input_2 = AsyncMock()
    input_2.is_visible.return_value = True
    input_2.get_attribute.side_effect = lambda attr: "file" if attr == "type" else ("resume" if attr == "name" else ("true" if attr == "required" else "Upload Resume"))
    mock_prop_2 = AsyncMock()
    mock_prop_2.json_value.return_value = "INPUT"
    input_2.get_property.return_value = mock_prop_2

    mock_form.query_selector_all.return_value = [input_1, input_2]
    mock_page.query_selector_all.return_value = [mock_form]

    inspector = IndeedJobInspector(playwright_manager=mock_pm, session_manager=mock_session_mgr)
    result = await inspector.inspect_job("https://www.indeed.com/viewjob?jk=form123", save_screenshot=False)

    assert result.form_detected is True
    assert result.field_count == 2
    assert len(result.fields) == 2
    assert result.fields[0].label == "Full Name"
    assert result.fields[0].type == "text"
    assert result.fields[1].type == "file"
    assert result.fields[1].required is True


# ---------------------------------------------------------------------------
# 4. Live-Test Repair Regression Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_job_title_ignores_generic_h1_and_welcome_headings():
    """Verify generic headings like 'Welcome, User' or 'Just a moment...' are not treated as job titles."""
    mock_pm = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_context.pages = [mock_page]
    mock_pm.launch_persistent_context.return_value = mock_context

    mock_session_mgr = AsyncMock()
    mock_session_mgr.check_session_status.return_value = SessionStatus(
        platform="indeed",
        browser="chromium",
        authenticated=True,
        status=SessionState.AUTHENTICATED,
    )

    mock_page.url = "https://www.indeed.com/viewjob?jk=welcome123"
    mock_page.title.return_value = "Indeed Account"
    mock_page.content.return_value = "<html><body>Welcome, Rithin</body></html>"

    # Mock element with Welcome heading
    mock_welcome_el = AsyncMock()
    mock_welcome_el.is_visible.return_value = True
    mock_welcome_el.inner_text.return_value = "Welcome, Rithin"

    async def query_selector_mock(selector):
        if 'data-testid="jobsearch-JobInfoHeader-title"' in selector:
            return mock_welcome_el
        return None

    mock_page.query_selector = AsyncMock(side_effect=query_selector_mock)
    mock_page.query_selector_all.return_value = []

    inspector = IndeedJobInspector(playwright_manager=mock_pm, session_manager=mock_session_mgr)
    result = await inspector.inspect_job("https://www.indeed.com/viewjob?jk=welcome123", save_screenshot=False)

    # Must reject generic "Welcome, Rithin"
    assert result.job_title is None or result.job_title != "Welcome, Rithin"


@pytest.mark.asyncio
async def test_search_form_with_ql_inputs_excluded_from_application_forms():
    """Verify Indeed's search/navigation form (q/l inputs) is strictly excluded from application fields."""
    mock_pm = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_context.pages = [mock_page]
    mock_pm.launch_persistent_context.return_value = mock_context

    mock_session_mgr = AsyncMock()
    mock_session_mgr.check_session_status.return_value = SessionStatus(
        platform="indeed",
        browser="chromium",
        authenticated=True,
        status=SessionState.AUTHENTICATED,
    )

    mock_page.url = "https://www.indeed.com/jobs"
    mock_page.title.return_value = "Job Search"
    mock_page.content.return_value = "<html><body>Search form</body></html>"
    mock_page.query_selector.return_value = None

    # Mock search form
    search_form = AsyncMock()
    search_form.is_visible.return_value = True
    search_form.get_attribute.side_effect = lambda attr: "search" if attr == "role" else ("/jobs" if attr == "action" else None)
    search_form.inner_text.return_value = "Find jobs"
    search_form.query_selector.return_value = None  # no file upload

    # Mock search inputs q and l
    inp_q = AsyncMock()
    inp_q.is_visible.return_value = True
    inp_q.get_attribute.side_effect = lambda attr: "text" if attr == "type" else ("q" if attr == "name" else "what")

    inp_l = AsyncMock()
    inp_l.is_visible.return_value = True
    inp_l.get_attribute.side_effect = lambda attr: "text" if attr == "type" else ("l" if attr == "name" else "where")

    search_form.query_selector_all.return_value = [inp_q, inp_l]
    mock_page.query_selector_all.return_value = [search_form]

    inspector = IndeedJobInspector(playwright_manager=mock_pm, session_manager=mock_session_mgr)
    result = await inspector.inspect_job("https://www.indeed.com/jobs", save_screenshot=False)

    # Search form MUST NOT be detected as an application form
    assert result.form_detected is False
    assert result.field_count == 0
    assert result.fields == []


@pytest.mark.asyncio
async def test_security_challenge_additional_verification_required_classifies_unknown():
    """Verify security challenge with 'Additional Verification Required' returns application_method='unknown'."""
    mock_pm = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_context.pages = [mock_page]
    mock_pm.launch_persistent_context.return_value = mock_context

    mock_session_mgr = AsyncMock()
    mock_session_mgr.check_session_status.return_value = SessionStatus(
        platform="indeed",
        browser="chromium",
        authenticated=True,
        status=SessionState.AUTHENTICATED,
    )

    mock_page.url = "https://www.indeed.com/challenge"
    mock_page.title.return_value = "Just a moment..."
    mock_page.content.return_value = "<html><body><h1>Additional Verification Required</h1><div class='cf-turnstile'></div></body></html>"
    mock_page.query_selector.return_value = None
    mock_page.query_selector_all.return_value = []

    inspector = IndeedJobInspector(playwright_manager=mock_pm, session_manager=mock_session_mgr)
    result = await inspector.inspect_job("https://www.indeed.com/challenge", save_screenshot=False)

    assert result.application_method == "unknown"
    assert result.apply_button_found is False
    assert "could not be reliably determined" in result.reason.lower() or "challenge" in result.reason.lower()


@pytest.mark.asyncio
async def test_inspection_performs_zero_browser_interaction_clicks_or_fills():
    """Verify inspector performs purely read-only inspection without clicking apply or filling inputs."""
    mock_pm = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_context.pages = [mock_page]
    mock_pm.launch_persistent_context.return_value = mock_context

    mock_session_mgr = AsyncMock()
    mock_session_mgr.check_session_status.return_value = SessionStatus(
        platform="indeed",
        browser="chromium",
        authenticated=True,
        status=SessionState.AUTHENTICATED,
    )

    mock_page.url = "https://www.indeed.com/viewjob?jk=readonly123"
    mock_page.title.return_value = "Job Posting"
    mock_page.content.return_value = "<html><body>Job Posting Content</body></html>"

    mock_apply_btn = AsyncMock()
    mock_apply_btn.is_visible.return_value = True
    mock_apply_btn.inner_text.return_value = "Apply now"
    mock_page.query_selector.return_value = mock_apply_btn
    mock_page.query_selector_all.return_value = []

    inspector = IndeedJobInspector(playwright_manager=mock_pm, session_manager=mock_session_mgr)
    result = await inspector.inspect_job("https://www.indeed.com/viewjob?jk=readonly123", save_screenshot=False)

    # Verify zero clicks or fills occurred
    mock_apply_btn.click.assert_not_called()
    mock_page.click.assert_not_called()
    mock_page.fill.assert_not_called()
    assert result.apply_button_found is True


# ---------------------------------------------------------------------------
# 5. FastAPI Endpoint Integration Tests
# ---------------------------------------------------------------------------


def test_api_indeed_inspect_endpoint():
    """Test POST /automation/indeed/inspect endpoint."""
    from app.api.routers.automation import get_indeed_inspector

    client = TestClient(app)
    mock_inspector = AsyncMock(spec=IndeedJobInspector)
    mock_inspector.inspect_job.return_value = IndeedJobInspectionResult(
        platform="indeed",
        authenticated=True,
        job_url="https://www.indeed.com/viewjob?jk=test123",
        job_title="Software Architect",
        company="Enterprise AI",
        location="Remote",
        application_method="indeed_hosted",
        apply_button_found=True,
        apply_button_text="Apply now",
        status="review_required",
    )

    app.dependency_overrides[get_indeed_inspector] = lambda: mock_inspector
    try:
        response = client.post(
            "/automation/indeed/inspect",
            json={"job_url": "https://www.indeed.com/viewjob?jk=test123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_title"] == "Software Architect"
        assert data["application_method"] == "indeed_hosted"
        assert data["apply_button_found"] is True
        assert data["status"] == "review_required"
    finally:
        app.dependency_overrides.pop(get_indeed_inspector, None)


def test_api_indeed_open_with_job_url():
    """Test POST /automation/indeed/open when supplied with a job_url."""
    from app.api.routers.automation import get_indeed_inspector

    client = TestClient(app)
    mock_inspector = AsyncMock(spec=IndeedJobInspector)
    mock_inspector.inspect_job.return_value = IndeedJobInspectionResult(
        platform="indeed",
        authenticated=True,
        job_url="https://www.indeed.com/viewjob?jk=test456",
        job_title="Frontend Lead",
        company="UI Tech",
        application_method="external",
        apply_button_found=True,
        external_url="https://careers.example.com",
        status="review_required",
    )

    app.dependency_overrides[get_indeed_inspector] = lambda: mock_inspector
    try:
        response = client.post(
            "/automation/indeed/open",
            json={"job_url": "https://www.indeed.com/viewjob?jk=test456"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_title"] == "Frontend Lead"
        assert data["application_method"] == "external"
        assert data["external_url"] == "https://careers.example.com"
    finally:
        app.dependency_overrides.pop(get_indeed_inspector, None)
