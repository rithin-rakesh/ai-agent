"""Comprehensive Unit and Integration Tests for Multi-Agent Orchestration (Phase 5.5).

Tests AgentRun lifecycle, queue construction, deduplication, decision tier filtering,
dry run, live apply, CAPTCHA pausing/resuming, and platform routing.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from app.agents.indeed_agent import IndeedApplicationAgent, _extract_indeed_jk
from app.agents.models import (
    AgentRun,
    AgentRunStatus,
    AgentRunSummary,
    ApplicationTask,
    ApplicationTaskResult,
    IndeedAgentRunRequest,
)
from app.agents.platform_router import PlatformRouter
from app.agents.run_manager import AgentRunManager
from app.api.main import app
from app.api.routers.agents import get_agent_run_manager, get_indeed_agent
from app.automation.indeed.apply_service import IndeedApplyService
from app.automation.indeed.models import (
    AutomationState,
    IndeedAutomationApplyRequest,
    IndeedAutomationNavigateRequest,
    IndeedAutomationResult,
    IndeedAutomationResumeRequest,
)
from app.database.repositories.agent_run_repository import AgentRunRepository
from app.database.repositories.application_repository import ApplicationRepository
from app.discovery.profile_discovery_service import ProfileDiscoveryService
from app.models.application import Application
from app.models.job import (
    DiscoveryStats,
    MatchedJobItem,
    MatchingStats,
    ProfileJobSearchResponse,
    SearchPlanSummary,
)


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


def make_mock_match_item(
    title: str = "AI Engineer",
    company: str = "Tech Corp",
    url: Optional[str] = None,
    final_score: float = 85.0,
    decision: str = "strong_match",
    job_id: Optional[UUID] = None,
) -> MatchedJobItem:
    jid = job_id or uuid4()
    u = url or f"https://in.indeed.com/viewjob?jk={uuid4().hex[:16]}"
    return MatchedJobItem(
        job_id=jid,
        title=title,
        company=company,
        location="Bangalore, India",
        source="indeed",
        url=u,
        deterministic_score=final_score,
        semantic_score=final_score,
        final_score=final_score,
        decision=decision,
        reason="Good match",
    )


def make_mock_discovery_response(top_matches: List[MatchedJobItem]) -> ProfileJobSearchResponse:
    return ProfileJobSearchResponse(
        status="success",
        search_plan=SearchPlanSummary(queries_generated=1, queries=[]),
        discovery=DiscoveryStats(raw_jobs=len(top_matches), unique_jobs=len(top_matches)),
        matching=MatchingStats(jobs_scored=len(top_matches)),
        top_matches=top_matches,
    )


# ---------------------------------------------------------------------------
# 1. JK Extractor & Platform Router Tests
# ---------------------------------------------------------------------------


def test_extract_indeed_jk():
    """Verify regex extraction of Indeed jk parameter."""
    assert _extract_indeed_jk("https://in.indeed.com/viewjob?jk=fedcba0987654321") == "fedcba0987654321"
    assert _extract_indeed_jk("https://www.indeed.com/rc/clk?jk=1234567890abcdef&from=vj") == "1234567890abcdef"
    assert _extract_indeed_jk("https://www.indeed.com/viewjob?no_jk_here=true") is None
    assert _extract_indeed_jk("") is None


def test_platform_router_indeed():
    """Verify PlatformRouter correctly identifies and routes Indeed."""
    router = PlatformRouter()
    assert router.is_platform_supported("indeed") is True
    assert router.is_platform_supported("INDEED ") is True
    agent = router.get_agent_for_platform("indeed")
    assert isinstance(agent, IndeedApplicationAgent)


def test_platform_router_rejects_unsupported():
    """Verify PlatformRouter rejects non-Indeed platforms in Phase 5.5."""
    router = PlatformRouter()
    assert router.is_platform_supported("linkedin") is False
    assert router.is_platform_supported("greenhouse") is False
    assert router.is_platform_supported("workday") is False

    with pytest.raises(ValueError, match="Platform 'linkedin' is not supported"):
        router.get_agent_for_platform("linkedin")


# ---------------------------------------------------------------------------
# 2. Agent Run Manager & Persistence Tests
# ---------------------------------------------------------------------------


def test_agent_run_manager_lifecycle():
    """Verify run creation, state updates, metric tracking, and summary generation."""
    repo = AgentRunRepository()
    manager = AgentRunManager(repository=repo)

    run = manager.create_run(platform="indeed", configuration={"max_applications": 5})
    assert run.status == AgentRunStatus.CREATED
    assert run.platform == "indeed"

    manager.update_status(run, AgentRunStatus.DISCOVERING)
    assert run.status == AgentRunStatus.DISCOVERING

    manager.record_discovery_and_matching(run, jobs_discovered=25, jobs_scored=25, jobs_eligible=8)
    assert run.jobs_discovered == 25
    assert run.jobs_eligible == 8

    # Record attempt result
    task_res = ApplicationTaskResult(
        job_id=uuid4(),
        platform="indeed",
        url="https://in.indeed.com/viewjob?jk=111",
        title="AI Engineer",
        company="OpenAI",
        match_score=92.0,
        match_decision="excellent",
        status="success",
        submitted_at=datetime.now(timezone.utc),
    )
    manager.record_attempt_result(run, task_res)
    assert run.jobs_attempted == 1
    assert run.applications_submitted == 1

    summary = manager.get_run_summary(run.run_id)
    assert summary is not None
    assert summary.applications_submitted == 1


# ---------------------------------------------------------------------------
# 3. Queue Construction & Deduplication Tests
# ---------------------------------------------------------------------------


def test_queue_decision_filtering_and_sorting():
    """Verify decisions accepted, low_match/reject filtered, and queue sorted descending."""
    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []

    agent = IndeedApplicationAgent(application_repository=mock_app_repo)

    item1 = make_mock_match_item(title="Job 1", final_score=86.2, decision="strong_match")
    item2 = make_mock_match_item(title="Job 2", final_score=71.5, decision="review")
    item3 = make_mock_match_item(title="Job 3", final_score=94.0, decision="excellent")
    item_low = make_mock_match_item(title="Job Low", final_score=65.0, decision="low_match")
    item_rej = make_mock_match_item(title="Job Rej", final_score=40.0, decision="reject")

    queue = agent._build_application_queue(
        top_matches=[item1, item2, item3, item_low, item_rej],
        allowed_decisions={"excellent", "strong_match", "review"},
        minimum_score=70.0,
    )

    # 3 eligible jobs
    assert len(queue) == 3
    # Sorted descending by score: 94.0 -> 86.2 -> 71.5
    assert queue[0].match_score == 94.0
    assert queue[0].title == "Job 3"
    assert queue[1].match_score == 86.2
    assert queue[2].match_score == 71.5


def test_queue_deduplication_database_and_jk():
    """Verify skipping jobs already in database and deduplicating by jk/url."""
    existing_job_id = uuid4()
    mock_app_repo = MagicMock(spec=ApplicationRepository)
    # Simulate job already submitted in Supabase
    mock_app = MagicMock(spec=Application)
    mock_app.job_id = existing_job_id
    mock_app.status = "submitted"
    mock_app_repo.get_applications_for_jobs.return_value = [mock_app]

    agent = IndeedApplicationAgent(application_repository=mock_app_repo)

    item_submitted = make_mock_match_item(title="Already Submitted", job_id=existing_job_id, final_score=90.0)
    item_dup_jk_1 = make_mock_match_item(title="Job A", url="https://in.indeed.com/viewjob?jk=samejk99", final_score=85.0)
    item_dup_jk_2 = make_mock_match_item(title="Job B", url="https://in.indeed.com/viewjob?jk=samejk99", final_score=82.0)
    item_unique = make_mock_match_item(title="Job Unique", url="https://in.indeed.com/viewjob?jk=uniquejk01", final_score=80.0)

    queue = agent._build_application_queue(
        top_matches=[item_submitted, item_dup_jk_1, item_dup_jk_2, item_unique],
        allowed_decisions={"excellent", "strong_match", "review"},
        minimum_score=70.0,
    )

    # item_submitted skipped, item_dup_jk_2 skipped (duplicate jk) -> only 2 tasks
    assert len(queue) == 2
    assert queue[0].title == "Job A"
    assert queue[1].title == "Job Unique"


# ---------------------------------------------------------------------------
# 4. Agent Execution Tests (End-to-End Orchestration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_run_successful_flow():
    """Verify full successful agent run with discovery, matching, and max_applications limit."""
    mock_disc_service = MagicMock(spec=ProfileDiscoveryService)
    mock_disc_service.discover_jobs_from_profile = AsyncMock()

    item1 = make_mock_match_item(title="Job 1", final_score=90.0, decision="excellent", url="https://in.indeed.com/viewjob?jk=jk001")
    item2 = make_mock_match_item(title="Job 2", final_score=85.0, decision="strong_match", url="https://in.indeed.com/viewjob?jk=jk002")
    item3 = make_mock_match_item(title="Job 3", final_score=80.0, decision="review", url="https://in.indeed.com/viewjob?jk=jk003")
    item4 = make_mock_match_item(title="Job 4", final_score=75.0, decision="review", url="https://in.indeed.com/viewjob?jk=jk004")

    mock_disc_service.discover_jobs_from_profile.return_value = make_mock_discovery_response([item1, item2, item3, item4])

    mock_apply_service = MagicMock(spec=IndeedApplyService)
    mock_apply_service.apply_to_job.return_value = IndeedAutomationResult(
        status="success",
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=mock",
        button_verified=True,
        apply_clicked=True,
        submit_verified=True,
        final_submit_clicked=True,
        submission_confirmed=True,
        current_state=AutomationState.APPLICATION_SUBMITTED,
    )

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []
    mock_app_repo.create_application.return_value = MagicMock(id=uuid4())

    agent = IndeedApplicationAgent(
        discovery_service=mock_disc_service,
        apply_service=mock_apply_service,
        application_repository=mock_app_repo,
    )

    req = IndeedAgentRunRequest(
        max_applications=2,  # Limit to 2 applications
        minimum_score=70.0,
        submit=True,
    )

    summary = await agent.run(req)

    assert summary.status == AgentRunStatus.COMPLETED
    assert summary.jobs_discovered == 4
    assert summary.jobs_eligible == 4
    assert summary.jobs_attempted == 2  # Max 2 applications attempted
    assert summary.applications_submitted == 2
    assert mock_apply_service.apply_to_job.call_count == 2


@pytest.mark.asyncio
async def test_agent_run_dry_run_does_not_submit():
    """Verify submit=False navigates up to SUBMISSION_READY but NEVER submits."""
    mock_disc_service = MagicMock(spec=ProfileDiscoveryService)
    item = make_mock_match_item(title="Dry Run Job", final_score=88.0, decision="strong_match")
    mock_disc_service.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response([item]))

    mock_apply_service = MagicMock(spec=IndeedApplyService)
    mock_apply_service.navigate_to_submit.return_value = IndeedAutomationResult(
        status="success",
        job_id=item.job_id,
        url=item.url,
        button_verified=True,
        apply_clicked=True,
        submit_verified=True,
        final_submit_clicked=False,  # Stopped at SUBMISSION_READY
        current_state=AutomationState.SUBMISSION_READY,
    )

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []

    agent = IndeedApplicationAgent(
        discovery_service=mock_disc_service,
        apply_service=mock_apply_service,
        application_repository=mock_app_repo,
    )

    req = IndeedAgentRunRequest(max_applications=1, submit=False)
    summary = await agent.run(req)

    assert summary.status == AgentRunStatus.COMPLETED
    assert summary.jobs_attempted == 1
    assert summary.applications_submitted == 0  # Dry run has 0 submissions
    mock_apply_service.navigate_to_submit.assert_called_once()
    mock_apply_service.apply_to_job.assert_not_called()


@pytest.mark.asyncio
async def test_agent_run_ordinary_failure_continues_to_next_job():
    """Verify ordinary failures (EXTERNAL_APPLY, ALREADY_APPLIED, SUBMIT_FOCUS_UNVERIFIED) continue queue."""
    mock_disc_service = MagicMock(spec=ProfileDiscoveryService)
    item1 = make_mock_match_item(title="External Job", final_score=90.0, decision="excellent", url="https://in.indeed.com/viewjob?jk=ext1")
    item2 = make_mock_match_item(title="Successful Job", final_score=85.0, decision="strong_match", url="https://in.indeed.com/viewjob?jk=suc1")

    mock_disc_service.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response([item1, item2]))

    mock_apply_service = MagicMock(spec=IndeedApplyService)
    # First job is external apply (skip), second job is successfully submitted
    mock_apply_service.apply_to_job.side_effect = [
        IndeedAutomationResult(
            status="blocked",
            job_id=item1.job_id,
            url=item1.url,
            current_state=AutomationState.EXTERNAL_APPLY,
            message="Apply on company site",
        ),
        IndeedAutomationResult(
            status="success",
            job_id=item2.job_id,
            url=item2.url,
            button_verified=True,
            apply_clicked=True,
            submit_verified=True,
            final_submit_clicked=True,
            submission_confirmed=True,
            current_state=AutomationState.APPLICATION_SUBMITTED,
        ),
    ]

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []
    mock_app_repo.create_application.return_value = MagicMock(id=uuid4())

    agent = IndeedApplicationAgent(
        discovery_service=mock_disc_service,
        apply_service=mock_apply_service,
        application_repository=mock_app_repo,
    )

    req = IndeedAgentRunRequest(max_applications=2, submit=True)
    summary = await agent.run(req)

    assert summary.status == AgentRunStatus.COMPLETED
    assert summary.jobs_attempted == 2
    assert summary.jobs_skipped == 1
    assert summary.applications_submitted == 1


# ---------------------------------------------------------------------------
# 5. CAPTCHA Pause & Resume Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_run_captcha_pauses_and_resumes_cleanly():
    """Verify CAPTCHA pauses entire run, and resume endpoint finishes submission and remaining queue."""
    mock_disc_service = MagicMock(spec=ProfileDiscoveryService)
    item_captcha = make_mock_match_item(title="Captcha Job", final_score=95.0, decision="excellent", url="https://in.indeed.com/viewjob?jk=cap1")
    item_next = make_mock_match_item(title="Next Job", final_score=80.0, decision="review", url="https://in.indeed.com/viewjob?jk=next1")

    mock_disc_service.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response([item_captcha, item_next]))

    mock_apply_service = MagicMock(spec=IndeedApplyService)
    # First attempt hits CAPTCHA
    mock_apply_service.apply_to_job.side_effect = [
        IndeedAutomationResult(
            status="blocked",
            job_id=item_captcha.job_id,
            url=item_captcha.url,
            current_state=AutomationState.CAPTCHA_OR_CHALLENGE,
            manual_action_required=True,
            final_submit_clicked=False,
            submission_confirmed=False,
            message="Security check detected.",
        ),
        # When resumed, next job succeeds
        IndeedAutomationResult(
            status="success",
            job_id=item_next.job_id,
            url=item_next.url,
            button_verified=True,
            apply_clicked=True,
            submit_verified=True,
            final_submit_clicked=True,
            submission_confirmed=True,
            current_state=AutomationState.APPLICATION_SUBMITTED,
        ),
    ]

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []
    mock_app_repo.create_application.return_value = MagicMock(id=uuid4())

    agent = IndeedApplicationAgent(
        discovery_service=mock_disc_service,
        apply_service=mock_apply_service,
        application_repository=mock_app_repo,
    )

    req = IndeedAgentRunRequest(max_applications=2, submit=True)

    # 1. Run encounters CAPTCHA -> enters PAUSED_MANUAL_ACTION
    summary = await agent.run(req)

    assert summary.status == AgentRunStatus.PAUSED_MANUAL_ACTION
    assert summary.manual_action_required is True
    assert summary.pause_reason == "CAPTCHA_OR_CHALLENGE"
    assert summary.current_job_id == item_captcha.job_id
    assert summary.applications_submitted == 0

    # 2. User solves CAPTCHA -> resume_run is called
    mock_apply_service.resume_submit.return_value = IndeedAutomationResult(
        status="success",
        job_id=item_captcha.job_id,
        url=item_captcha.url,
        button_verified=True,
        apply_clicked=True,
        submit_verified=True,
        final_submit_clicked=True,
        submission_confirmed=True,
        current_state=AutomationState.APPLICATION_SUBMITTED,
    )

    resumed_summary = await agent.resume_run(summary.run_id)

    assert resumed_summary.status == AgentRunStatus.COMPLETED
    assert resumed_summary.manual_action_required is False
    assert resumed_summary.applications_submitted == 2
    assert resumed_summary.jobs_attempted == 2


@pytest.mark.asyncio
async def test_agent_run_resume_while_challenge_remains_stays_paused():
    """Verify resume when CAPTCHA is still unsolved remains in PAUSED_MANUAL_ACTION."""
    mock_disc_service = MagicMock(spec=ProfileDiscoveryService)
    item = make_mock_match_item(title="Captcha Job", final_score=90.0, decision="excellent")
    mock_disc_service.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response([item]))

    mock_apply_service = MagicMock(spec=IndeedApplyService)
    mock_apply_service.apply_to_job.return_value = IndeedAutomationResult(
        status="blocked",
        job_id=item.job_id,
        url=item.url,
        current_state=AutomationState.CAPTCHA_OR_CHALLENGE,
        manual_action_required=True,
        final_submit_clicked=False,
    )

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []

    agent = IndeedApplicationAgent(
        discovery_service=mock_disc_service,
        apply_service=mock_apply_service,
        application_repository=mock_app_repo,
    )

    req = IndeedAgentRunRequest(max_applications=1, submit=True)
    summary = await agent.run(req)
    assert summary.status == AgentRunStatus.PAUSED_MANUAL_ACTION

    # Resume called, but challenge is still active
    mock_apply_service.resume_submit.return_value = IndeedAutomationResult(
        status="blocked",
        job_id=item.job_id,
        url=item.url,
        current_state=AutomationState.CAPTCHA_OR_CHALLENGE,
        manual_action_required=True,
        final_submit_clicked=False,
    )

    resumed_summary = await agent.resume_run(summary.run_id)
    assert resumed_summary.status == AgentRunStatus.PAUSED_MANUAL_ACTION
    assert resumed_summary.manual_action_required is True


@pytest.mark.asyncio
async def test_agent_run_enforces_indeed_only_sites():
    """Verify ProfileJobSearchRequest enforces sites=['indeed'] regardless of inputs."""
    mock_disc_service = MagicMock(spec=ProfileDiscoveryService)
    mock_disc_service.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response([]))

    agent = IndeedApplicationAgent(discovery_service=mock_disc_service)
    req = IndeedAgentRunRequest()
    await agent.run(req)

    call_args = mock_disc_service.discover_jobs_from_profile.call_args[1]["request"]
    assert call_args.sites == ["indeed"]


@pytest.mark.asyncio
async def test_agent_run_login_required_pauses_run():
    """Verify LOGIN_REQUIRED pauses the run for manual login."""
    mock_disc_service = MagicMock(spec=ProfileDiscoveryService)
    item = make_mock_match_item(title="Login Job", final_score=90.0, decision="excellent")
    mock_disc_service.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response([item]))

    mock_apply_service = MagicMock(spec=IndeedApplyService)
    mock_apply_service.apply_to_job.return_value = IndeedAutomationResult(
        status="blocked",
        job_id=item.job_id,
        url=item.url,
        current_state=AutomationState.LOGIN_REQUIRED,
        manual_action_required=True,
        final_submit_clicked=False,
    )

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []

    agent = IndeedApplicationAgent(
        discovery_service=mock_disc_service,
        apply_service=mock_apply_service,
        application_repository=mock_app_repo,
    )

    req = IndeedAgentRunRequest(max_applications=1, submit=True)
    summary = await agent.run(req)

    assert summary.status == AgentRunStatus.PAUSED_MANUAL_ACTION
    assert summary.manual_action_required is True
    assert summary.pause_reason == "LOGIN_REQUIRED"


@pytest.mark.asyncio
async def test_agent_run_mfa_required_pauses_run():
    """Verify MFA_REQUIRED pauses the run for manual verification."""
    mock_disc_service = MagicMock(spec=ProfileDiscoveryService)
    item = make_mock_match_item(title="MFA Job", final_score=92.0, decision="excellent")
    mock_disc_service.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response([item]))

    mock_apply_service = MagicMock(spec=IndeedApplyService)
    mock_apply_service.apply_to_job.return_value = IndeedAutomationResult(
        status="blocked",
        job_id=item.job_id,
        url=item.url,
        current_state=AutomationState.MFA_REQUIRED,
        manual_action_required=True,
        final_submit_clicked=False,
    )

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []

    agent = IndeedApplicationAgent(
        discovery_service=mock_disc_service,
        apply_service=mock_apply_service,
        application_repository=mock_app_repo,
    )

    req = IndeedAgentRunRequest(max_applications=1, submit=True)
    summary = await agent.run(req)

    assert summary.status == AgentRunStatus.PAUSED_MANUAL_ACTION
    assert summary.manual_action_required is True
    assert summary.pause_reason == "MFA_REQUIRED"


@pytest.mark.asyncio
async def test_agent_run_browser_not_verified_fails_run_safely():
    """Verify BROWSER_NOT_VERIFIED halts and fails the run safely."""
    mock_disc_service = MagicMock(spec=ProfileDiscoveryService)
    item = make_mock_match_item(title="Browser Fail Job", final_score=85.0, decision="strong_match")
    mock_disc_service.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response([item]))

    mock_apply_service = MagicMock(spec=IndeedApplyService)
    mock_apply_service.apply_to_job.return_value = IndeedAutomationResult(
        status="failed",
        job_id=item.job_id,
        url=item.url,
        current_state=AutomationState.BROWSER_NOT_VERIFIED,
        message="Chrome application window not found",
    )

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []

    agent = IndeedApplicationAgent(
        discovery_service=mock_disc_service,
        apply_service=mock_apply_service,
        application_repository=mock_app_repo,
    )

    req = IndeedAgentRunRequest(max_applications=1, submit=True)
    summary = await agent.run(req)

    assert summary.status == AgentRunStatus.FAILED
    assert summary.jobs_attempted == 1


@pytest.mark.asyncio
async def test_agent_run_already_applied_skipped_and_continues():
    """Verify ALREADY_APPLIED outcome is recorded as skipped and agent moves to next job."""
    mock_disc_service = MagicMock(spec=ProfileDiscoveryService)
    item1 = make_mock_match_item(title="Already Applied Job", final_score=90.0, decision="excellent")
    item2 = make_mock_match_item(title="Fresh Job", final_score=85.0, decision="strong_match")
    mock_disc_service.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response([item1, item2]))

    mock_apply_service = MagicMock(spec=IndeedApplyService)
    mock_apply_service.apply_to_job.side_effect = [
        IndeedAutomationResult(
            status="blocked",
            job_id=item1.job_id,
            url=item1.url,
            current_state=AutomationState.ALREADY_APPLIED,
            message="Already applied on Indeed",
        ),
        IndeedAutomationResult(
            status="success",
            job_id=item2.job_id,
            url=item2.url,
            button_verified=True,
            apply_clicked=True,
            submit_verified=True,
            final_submit_clicked=True,
            submission_confirmed=True,
            current_state=AutomationState.APPLICATION_SUBMITTED,
        ),
    ]

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []
    mock_app_repo.create_application.return_value = MagicMock(id=uuid4())

    agent = IndeedApplicationAgent(
        discovery_service=mock_disc_service,
        apply_service=mock_apply_service,
        application_repository=mock_app_repo,
    )

    req = IndeedAgentRunRequest(max_applications=2, submit=True)
    summary = await agent.run(req)

    assert summary.status == AgentRunStatus.COMPLETED
    assert summary.jobs_attempted == 2
    assert summary.jobs_skipped == 1
    assert summary.applications_submitted == 1


@pytest.mark.asyncio
async def test_uncertain_submission_not_automatically_retried():
    """Verify SUBMISSION_CONFIRMATION_UNVERIFIED is marked failed/unverified and not retried."""
    mock_disc_service = MagicMock(spec=ProfileDiscoveryService)
    item = make_mock_match_item(title="Uncertain Job", final_score=88.0, decision="strong_match")
    mock_disc_service.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response([item]))

    mock_apply_service = MagicMock(spec=IndeedApplyService)
    mock_apply_service.apply_to_job.return_value = IndeedAutomationResult(
        status="failed",
        job_id=item.job_id,
        url=item.url,
        button_verified=True,
        apply_clicked=True,
        submit_verified=True,
        final_submit_clicked=True,
        submission_confirmed=False,
        current_state=AutomationState.SUBMISSION_CONFIRMATION_UNVERIFIED,
        message="Confirmation banner missing",
    )

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []

    agent = IndeedApplicationAgent(
        discovery_service=mock_disc_service,
        apply_service=mock_apply_service,
        application_repository=mock_app_repo,
    )

    req = IndeedAgentRunRequest(max_applications=1, submit=True)
    summary = await agent.run(req)

    assert summary.status == AgentRunStatus.COMPLETED_WITH_ERRORS
    assert summary.applications_submitted == 0
    assert summary.jobs_failed == 1
    assert mock_apply_service.apply_to_job.call_count == 1  # Never retried


# ---------------------------------------------------------------------------
# 6. FastAPI Router Tests
# ---------------------------------------------------------------------------


def test_api_agent_indeed_run_endpoint():
    """Verify POST /agent/indeed/run endpoint."""
    client = TestClient(app)
    mock_agent = MagicMock(spec=IndeedApplicationAgent)
    sample_run_id = uuid4()

    mock_agent.run = AsyncMock(return_value=AgentRunSummary(
        run_id=sample_run_id,
        platform="indeed",
        status=AgentRunStatus.COMPLETED,
        started_at=datetime.now(timezone.utc),
        jobs_discovered=10,
        jobs_scored=10,
        jobs_eligible=5,
        jobs_attempted=3,
        applications_submitted=3,
        jobs_skipped=0,
        jobs_failed=0,
        manual_action_required=False,
    ))

    app.dependency_overrides[get_indeed_agent] = lambda: mock_agent
    try:
        resp = client.post(
            "/agent/indeed/run",
            json={"max_applications": 3, "minimum_score": 70.0, "submit": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == str(sample_run_id)
        assert data["status"] == "COMPLETED"
        assert data["applications_submitted"] == 3
    finally:
        app.dependency_overrides.pop(get_indeed_agent, None)


def test_api_agent_get_run_status_endpoint():
    """Verify GET /agent/runs/{run_id} endpoint."""
    client = TestClient(app)
    mock_manager = MagicMock(spec=AgentRunManager)
    sample_run_id = uuid4()

    mock_manager.get_run_summary.return_value = AgentRunSummary(
        run_id=sample_run_id,
        platform="indeed",
        status=AgentRunStatus.APPLYING,
        started_at=datetime.now(timezone.utc),
        jobs_discovered=15,
        jobs_scored=15,
        jobs_eligible=4,
        jobs_attempted=1,
        applications_submitted=1,
        jobs_skipped=0,
        jobs_failed=0,
        manual_action_required=False,
    )

    app.dependency_overrides[get_agent_run_manager] = lambda: mock_manager
    try:
        resp = client.get(f"/agent/runs/{sample_run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == str(sample_run_id)
        assert data["status"] == "APPLYING"
    finally:
        app.dependency_overrides.pop(get_agent_run_manager, None)


def test_api_agent_resume_run_endpoint():
    """Verify POST /agent/indeed/runs/{run_id}/resume endpoint."""
    client = TestClient(app)
    mock_agent = MagicMock(spec=IndeedApplicationAgent)
    sample_run_id = uuid4()

    mock_agent.resume_run = AsyncMock(return_value=AgentRunSummary(
        run_id=sample_run_id,
        platform="indeed",
        status=AgentRunStatus.COMPLETED,
        started_at=datetime.now(timezone.utc),
        jobs_discovered=5,
        jobs_scored=5,
        jobs_eligible=2,
        jobs_attempted=2,
        applications_submitted=2,
        jobs_skipped=0,
        jobs_failed=0,
        manual_action_required=False,
    ))

    app.dependency_overrides[get_indeed_agent] = lambda: mock_agent
    try:
        resp = client.post(f"/agent/indeed/runs/{sample_run_id}/resume")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == str(sample_run_id)
        assert data["status"] == "COMPLETED"
        assert data["applications_submitted"] == 2
    finally:
        app.dependency_overrides.pop(get_indeed_agent, None)


# ---------------------------------------------------------------------------
# 7. Multi-Job Task Navigation & Isolation Regression Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_job_task_navigation_isolation_and_succession():
    """Verify Job 1 completes with APPLICATION_SUBMITTED and Job 2 explicitly navigates to its own URL."""
    item1 = make_mock_match_item(title="AI Lead 1", final_score=95.0, url="https://in.indeed.com/viewjob?jk=job1url111")
    item2 = make_mock_match_item(title="AI Lead 2", final_score=90.0, url="https://in.indeed.com/viewjob?jk=job2url222")

    mock_disc_service = MagicMock(spec=ProfileDiscoveryService)
    mock_disc_service.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response([item1, item2]))

    mock_apply_service = MagicMock(spec=IndeedApplyService)
    mock_apply_service.apply_to_job.side_effect = [
        IndeedAutomationResult(
            status="success",
            job_id=item1.job_id,
            url=item1.url,
            button_verified=True,
            apply_clicked=True,
            submit_verified=True,
            final_submit_clicked=True,
            submission_confirmed=True,
            current_state=AutomationState.APPLICATION_SUBMITTED,
            message="Application 1 submitted.",
        ),
        IndeedAutomationResult(
            status="success",
            job_id=item2.job_id,
            url=item2.url,
            button_verified=True,
            apply_clicked=True,
            submit_verified=True,
            final_submit_clicked=True,
            submission_confirmed=True,
            current_state=AutomationState.APPLICATION_SUBMITTED,
            message="Application 2 submitted.",
        ),
    ]

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []
    mock_app_repo.create_application.return_value = MagicMock(id=uuid4())

    agent = IndeedApplicationAgent(
        discovery_service=mock_disc_service,
        apply_service=mock_apply_service,
        application_repository=mock_app_repo,
    )

    req = IndeedAgentRunRequest(max_applications=2, submit=True)
    summary = await agent.run(req)

    assert summary.status == AgentRunStatus.COMPLETED
    assert summary.jobs_attempted == 2
    assert summary.applications_submitted == 2
    assert summary.jobs_failed == 0
    assert summary.jobs_skipped == 0

    # Verify both distinct URLs were dispatched to apply_service
    call_args_list = mock_apply_service.apply_to_job.call_args_list
    assert len(call_args_list) == 2
    assert call_args_list[0][0][0].url == item1.url
    assert call_args_list[1][0][0].url == item2.url


@pytest.mark.asyncio
async def test_multi_job_job1_not_verified_agent_continues_to_job2():
    """Verify Job 1 ending with APPLY_WITH_INDEED_NOT_VERIFIED records skip and continues to Job 2."""
    item1 = make_mock_match_item(title="External/Unverified Job", final_score=92.0, url="https://in.indeed.com/viewjob?jk=unverified1")
    item2 = make_mock_match_item(title="Apply With Indeed Job", final_score=88.0, url="https://in.indeed.com/viewjob?jk=verified2")

    mock_disc_service = MagicMock(spec=ProfileDiscoveryService)
    mock_disc_service.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response([item1, item2]))

    mock_apply_service = MagicMock(spec=IndeedApplyService)
    mock_apply_service.apply_to_job.side_effect = [
        IndeedAutomationResult(
            status="blocked",
            job_id=item1.job_id,
            url=item1.url,
            button_verified=False,
            apply_clicked=False,
            current_state=AutomationState.APPLY_WITH_INDEED_NOT_VERIFIED,
            message="Apply with Indeed button not found.",
            diagnostics={"expected_jk": "unverified1", "reason": "Not found"},
        ),
        IndeedAutomationResult(
            status="success",
            job_id=item2.job_id,
            url=item2.url,
            button_verified=True,
            apply_clicked=True,
            submit_verified=True,
            final_submit_clicked=True,
            submission_confirmed=True,
            current_state=AutomationState.APPLICATION_SUBMITTED,
        ),
    ]

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []
    mock_app_repo.create_application.return_value = MagicMock(id=uuid4())

    agent = IndeedApplicationAgent(
        discovery_service=mock_disc_service,
        apply_service=mock_apply_service,
        application_repository=mock_app_repo,
    )

    req = IndeedAgentRunRequest(max_applications=2, submit=True)
    summary = await agent.run(req)

    assert summary.status == AgentRunStatus.COMPLETED
    assert summary.jobs_attempted == 2
    assert summary.jobs_skipped == 1
    assert summary.applications_submitted == 1


@pytest.mark.asyncio
async def test_multi_job_job1_navigation_failed_agent_continues_to_job2():
    """Verify Job 1 ending with JOB_NAVIGATION_FAILED records failure and continues to Job 2."""
    item1 = make_mock_match_item(title="Nav Fail Job", final_score=91.0, url="https://in.indeed.com/viewjob?jk=navfail1")
    item2 = make_mock_match_item(title="Working Job", final_score=87.0, url="https://in.indeed.com/viewjob?jk=work2")

    mock_disc_service = MagicMock(spec=ProfileDiscoveryService)
    mock_disc_service.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response([item1, item2]))

    mock_apply_service = MagicMock(spec=IndeedApplyService)
    mock_apply_service.apply_to_job.side_effect = [
        IndeedAutomationResult(
            status="blocked",
            job_id=item1.job_id,
            url=item1.url,
            button_verified=False,
            apply_clicked=False,
            current_state=AutomationState.JOB_NAVIGATION_FAILED,
            message="Stale submission confirmation page detected from previous task. Target job page failed to load.",
            diagnostics={"expected_jk": "navfail1", "reason": "Stale confirmation page"},
        ),
        IndeedAutomationResult(
            status="success",
            job_id=item2.job_id,
            url=item2.url,
            button_verified=True,
            apply_clicked=True,
            submit_verified=True,
            final_submit_clicked=True,
            submission_confirmed=True,
            current_state=AutomationState.APPLICATION_SUBMITTED,
        ),
    ]

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []
    mock_app_repo.create_application.return_value = MagicMock(id=uuid4())

    agent = IndeedApplicationAgent(
        discovery_service=mock_disc_service,
        apply_service=mock_apply_service,
        application_repository=mock_app_repo,
    )

    req = IndeedAgentRunRequest(max_applications=2, submit=True)
    summary = await agent.run(req)

    assert summary.status == AgentRunStatus.COMPLETED_WITH_ERRORS
    assert summary.jobs_attempted == 2
    assert summary.jobs_failed == 1
    assert summary.applications_submitted == 1
