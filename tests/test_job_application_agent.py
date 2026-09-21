"""Comprehensive Unit and Integration Tests for JobApplicationOrchestrator (Phase 6.0).

Tests the 17-state lifecycle machine, discovery routing, queue ranking,
locking manager, duplicate protection, final submit guards, human intervention pause/resume,
platform routing, and API endpoints.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from app.agents.job_application_agent import JobApplicationOrchestrator
from app.agents.lock_manager import JobLockManager
from app.agents.models import (
    AgentAbortRequest,
    AgentApplyOneRequest,
    AgentResumeRequest,
    AgentRun,
    AgentRunRequest,
    AgentRunResponse,
    AgentRunStatus,
    AgentSkipRequest,
    OrchestratorState,
    QueueItem,
    QueueStatus,
    SingleJobApplyResponse,
)
from app.agents.platform_router import PlatformRouter
from app.agents.run_manager import AgentRunManager
from app.api.main import app
from app.api.routers.agents import get_job_orchestrator
from app.automation.glassdoor.models import (
    GlassdoorAutomationApplyRequest,
    GlassdoorAutomationNavigateRequest,
    GlassdoorAutomationResult,
    GlassdoorAutomationState,
)
from app.automation.indeed.models import (
    AutomationState,
    IndeedAutomationApplyRequest,
    IndeedAutomationNavigateRequest,
    IndeedAutomationResult,
)
from app.database.repositories.application_repository import ApplicationRepository
from app.database.repositories.job_repository import JobRepository
from app.discovery.profile_discovery_service import ProfileDiscoveryService
from app.models.application import Application
from app.models.job import (
    DiscoveryStats,
    Job,
    MatchedJobItem,
    MatchingStats,
    ProfileJobSearchResponse,
    SearchPlanSummary,
)


# ---------------------------------------------------------------------------
# Helpers & Mock Factories
# ---------------------------------------------------------------------------


def make_mock_match(
    title: str = "Software Engineer",
    company: str = "Acme Corp",
    source: str = "indeed",
    url: Optional[str] = None,
    final_score: float = 85.0,
    decision: str = "strong_match",
    job_id: Optional[UUID] = None,
) -> MatchedJobItem:
    jid = job_id or uuid4()
    if not url:
        if source == "glassdoor":
            url = f"https://www.glassdoor.com/job-listing/j?jl={uuid4().int % 1000000000000}"
        else:
            url = f"https://www.indeed.com/viewjob?jk={uuid4().hex[:16]}"
    return MatchedJobItem(
        job_id=jid,
        title=title,
        company=company,
        location="San Francisco, CA",
        source=source,
        url=url,
        deterministic_score=final_score,
        semantic_score=final_score,
        final_score=final_score,
        decision=decision,
        reason="Good fit",
    )


def make_mock_discovery_response(matches: List[MatchedJobItem]) -> ProfileJobSearchResponse:
    return ProfileJobSearchResponse(
        status="success",
        search_plan=SearchPlanSummary(queries_generated=2, queries=[]),
        discovery=DiscoveryStats(raw_jobs=len(matches), unique_jobs=len(matches)),
        matching=MatchingStats(jobs_scored=len(matches)),
        top_matches=matches,
    )


# ---------------------------------------------------------------------------
# 1. Lock Manager Tests
# ---------------------------------------------------------------------------


def test_lock_manager_acquire_and_release(tmp_path):
    lock_file = tmp_path / "test_locks.json"
    mgr = JobLockManager(lock_file=str(lock_file))
    jid = uuid4()
    rid = uuid4()

    assert mgr.acquire_lock(jid, rid) is True
    assert mgr.is_locked(jid) is True

    # Cannot acquire same lock with different run
    other_rid = uuid4()
    assert mgr.acquire_lock(jid, other_rid) is False

    # Same run can re-acquire (idempotent)
    assert mgr.acquire_lock(jid, rid) is True

    # Release
    assert mgr.release_lock(jid, rid) is True
    assert mgr.is_locked(jid) is False


def test_lock_manager_release_all_for_run(tmp_path):
    lock_file = tmp_path / "test_locks.json"
    mgr = JobLockManager(lock_file=str(lock_file))
    jid1 = uuid4()
    jid2 = uuid4()
    rid = uuid4()

    mgr.acquire_lock(jid1, rid)
    mgr.acquire_lock(jid2, rid)
    assert mgr.is_locked(jid1) is True
    assert mgr.is_locked(jid2) is True

    mgr.release_all_for_run(rid)
    assert mgr.is_locked(jid1) is False
    assert mgr.is_locked(jid2) is False


# ---------------------------------------------------------------------------
# 2. Queue Construction & Deduplication Tests
# ---------------------------------------------------------------------------


def test_build_queue_filtering_and_ranking():
    disc_svc = MagicMock(spec=ProfileDiscoveryService)
    app_repo = MagicMock(spec=ApplicationRepository)
    app_repo.get_applications_for_jobs.return_value = []
    orchestrator = JobApplicationOrchestrator(
        discovery_service=disc_svc,
        application_repository=app_repo,
    )

    jid1 = uuid4()
    jid2 = uuid4()
    jid_dup = uuid4()
    jid_low = uuid4()
    jid_reject = uuid4()
    jid_unsupported = uuid4()

    matches = [
        make_mock_match(title="Job Low Score", final_score=50.0, job_id=jid_low),
        make_mock_match(title="Job High Score Indeed", final_score=95.0, job_id=jid1, source="indeed"),
        make_mock_match(title="Job High Score Glassdoor", final_score=88.0, job_id=jid2, source="glassdoor"),
        make_mock_match(title="Job Dup 1", final_score=85.0, job_id=jid_dup),
        make_mock_match(title="Job Dup 2", final_score=82.0, job_id=jid_dup),  # duplicate ID
        make_mock_match(title="Job Reject", final_score=90.0, decision="reject", job_id=jid_reject),
        make_mock_match(title="Job Unsupported", final_score=92.0, source="monster", job_id=jid_unsupported),
    ]

    queue = orchestrator._build_queue(
        top_matches=matches,
        minimum_score=70.0,
        profile_id=uuid4(),
    )

    assert len(queue) == 3
    # Ordered selection:
    next_job = orchestrator.select_next_job(queue)
    assert next_job.job_id == jid1
    assert next_job.match_score == 95.0


def test_build_queue_skips_already_applied_in_db():
    app_repo = MagicMock(spec=ApplicationRepository)
    jid_applied = uuid4()
    jid_unapplied = uuid4()
    prof_id = uuid4()

    mock_app = MagicMock()
    mock_app.job_id = jid_applied
    mock_app.status = "submitted"
    app_repo.get_applications_for_jobs.return_value = [mock_app]

    orchestrator = JobApplicationOrchestrator(application_repository=app_repo)

    matches = [
        make_mock_match(title="Applied Job", job_id=jid_applied, final_score=90.0),
        make_mock_match(title="New Job", job_id=jid_unapplied, final_score=88.0),
    ]

    queue = orchestrator._build_queue(
        top_matches=matches,
        minimum_score=70.0,
        profile_id=prof_id,
    )

    assert len(queue) == 1
    assert queue[0].job_id == jid_unapplied


# ---------------------------------------------------------------------------
# 3. Single Job Execution & Validation Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_single_job_rejects_missing_job_id():
    job_repo = MagicMock(spec=JobRepository)
    job_repo.get_job_by_id.return_value = None
    orchestrator = JobApplicationOrchestrator(job_repository=job_repo)

    fake_id = uuid4()
    req = AgentApplyOneRequest(job_id=fake_id)

    with pytest.raises(ValueError, match="does not exist in database"):
        await orchestrator.run_single_job(req)


@pytest.mark.asyncio
async def test_run_single_job_rejects_unsupported_platform():
    job_repo = MagicMock(spec=JobRepository)
    jid = uuid4()
    job_repo.get_job_by_id.return_value = Job(
        id=jid,
        external_id="ext-unsupported",
        title="Python Dev",
        company="Tech Inc",
        location="Remote",
        source="monster",
        url="https://monster.com/job/123",
        created_at=datetime.now(timezone.utc),
    )
    orchestrator = JobApplicationOrchestrator(job_repository=job_repo)

    req = AgentApplyOneRequest(job_id=jid)
    res = await orchestrator.run_single_job(req)
    assert res.status == "failed"
    assert "not supported" in res.error


@pytest.mark.asyncio
async def test_run_single_job_skips_when_locked():
    job_repo = MagicMock(spec=JobRepository)
    lock_mgr = MagicMock(spec=JobLockManager)
    lock_mgr.acquire_lock.return_value = False  # Locked!

    jid = uuid4()
    job_repo.get_job_by_id.return_value = Job(
        id=jid,
        external_id="ext-indeed-locked",
        title="Python Dev",
        company="Tech Inc",
        location="Remote",
        source="indeed",
        url="https://www.indeed.com/viewjob?jk=1234567890abcdef",
        created_at=datetime.now(timezone.utc),
    )

    orchestrator = JobApplicationOrchestrator(
        job_repository=job_repo,
        lock_manager=lock_mgr,
    )

    req = AgentApplyOneRequest(job_id=jid)
    res = await orchestrator.run_single_job(req)
    assert res.status == "skipped"
    assert "currently locked" in res.error


# ---------------------------------------------------------------------------
# 4. Platform Routing & Submission Guard Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_job_glassdoor_auto_submit_false_stops_at_submission_ready():
    platform_router = MagicMock(spec=PlatformRouter)
    mock_gd_service = MagicMock()
    mock_result = GlassdoorAutomationResult(
        status="blocked",
        url="https://www.glassdoor.com/job-listing/j?jl=1234567890",
        job_id=uuid4(),
        platform="glassdoor",
        current_state=GlassdoorAutomationState.SUBMISSION_READY,
        submit_verified=True,
        final_submit_clicked=False,
        submission_confirmed=False,
    )
    mock_gd_service.navigate_to_submit = AsyncMock(return_value=mock_result)
    platform_router.get_apply_service_for_platform.return_value = mock_gd_service

    app_repo = MagicMock(spec=ApplicationRepository)
    app_repo.get_application_by_job_and_profile.return_value = None

    orchestrator = JobApplicationOrchestrator(
        platform_router=platform_router,
        application_repository=app_repo,
    )

    run = AgentRun(
        run_id=uuid4(),
        platform="glassdoor",
        status=AgentRunStatus.APPLYING,
        started_at=datetime.now(timezone.utc),
    )
    q_item = QueueItem(
        job_id=uuid4(),
        source="glassdoor",
        title="Full Stack Dev",
        company="Apex Corp",
        location="Remote",
        url="https://www.glassdoor.com/job-listing/j?jl=1234567890",
        match_score=90.0,
        status=QueueStatus.QUEUED,
    )

    res = await orchestrator.process_job(
        job=q_item,
        run=run,
        auto_submit=False,
        profile_id=uuid4(),
    )

    # auto_submit=False -> navigates to submit, stops at SUBMISSION_READY
    mock_gd_service.navigate_to_submit.assert_awaited_once()
    assert q_item.status == QueueStatus.SUBMISSION_READY
    assert res.automation_state == "SUBMISSION_READY"
    assert res.status == "success"


@pytest.mark.asyncio
async def test_process_job_indeed_auto_submit_true_submits():
    platform_router = MagicMock(spec=PlatformRouter)
    mock_indeed_service = MagicMock()
    mock_result = IndeedAutomationResult(
        status="success",
        url="https://www.indeed.com/viewjob?jk=1234567890abcdef",
        job_id=uuid4(),
        platform="indeed",
        current_state=AutomationState.APPLICATION_SUBMITTED,
        submit_verified=True,
        final_submit_clicked=True,
        submission_confirmed=True,
    )
    mock_indeed_service.apply_to_job = AsyncMock(return_value=mock_result)
    platform_router.get_apply_service_for_platform.return_value = mock_indeed_service

    app_repo = MagicMock(spec=ApplicationRepository)
    app_repo.get_application_by_job_and_profile.return_value = None

    orchestrator = JobApplicationOrchestrator(
        platform_router=platform_router,
        application_repository=app_repo,
    )

    run = AgentRun(
        run_id=uuid4(),
        platform="indeed",
        status=AgentRunStatus.APPLYING,
        started_at=datetime.now(timezone.utc),
    )
    jid = uuid4()
    prof_id = uuid4()
    q_item = QueueItem(
        job_id=jid,
        source="indeed",
        title="Backend Dev",
        company="Beta LLC",
        location="Remote",
        url="https://www.indeed.com/viewjob?jk=1234567890abcdef",
        match_score=92.0,
        status=QueueStatus.QUEUED,
    )

    res = await orchestrator.process_job(
        job=q_item,
        run=run,
        auto_submit=True,
        profile_id=prof_id,
    )

    mock_indeed_service.apply_to_job.assert_awaited_once()
    assert q_item.status == QueueStatus.SUBMITTED
    assert run.applications_submitted == 1
    assert res.status == "success"
    app_repo.record_application_submission.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Human Intervention / Pause & Resume Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_intervention_pauses_run():
    platform_router = MagicMock(spec=PlatformRouter)
    mock_gd_service = MagicMock()
    mock_result = GlassdoorAutomationResult(
        status="blocked",
        url="https://www.glassdoor.com/job-listing/j?jl=1234567890",
        job_id=uuid4(),
        platform="glassdoor",
        current_state=GlassdoorAutomationState.CAPTCHA_OR_CHALLENGE,
        manual_action_required=True,
        message="Cloudflare challenge detected",
    )
    mock_gd_service.navigate_to_submit = AsyncMock(return_value=mock_result)
    platform_router.get_apply_service_for_platform.return_value = mock_gd_service

    app_repo = MagicMock(spec=ApplicationRepository)
    app_repo.get_application_by_job_and_profile.return_value = None

    disc_svc = MagicMock(spec=ProfileDiscoveryService)
    matches = [make_mock_match(title="Challenge Job", source="glassdoor")]
    disc_svc.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response(matches))

    orchestrator = JobApplicationOrchestrator(
        discovery_service=disc_svc,
        platform_router=platform_router,
        application_repository=app_repo,
    )

    req = AgentRunRequest(
        profile_id=uuid4(),
        sites=["glassdoor"],
        max_jobs=1,
        auto_submit=False,
    )

    resp = await orchestrator.run_application_cycle(req)
    assert resp.status == OrchestratorState.WAITING_FOR_HUMAN_INPUT
    assert resp.manual_actions_required is True
    assert "Cloudflare" in (resp.pause_reason or "")


@pytest.mark.asyncio
async def test_resume_run_success():
    run_mgr = MagicMock(spec=AgentRunManager)
    job_repo = MagicMock(spec=JobRepository)
    platform_router = MagicMock(spec=PlatformRouter)
    app_repo = MagicMock(spec=ApplicationRepository)

    jid = uuid4()
    rid = uuid4()

    mock_run = AgentRun(
        run_id=rid,
        platform="glassdoor",
        status=AgentRunStatus.PAUSED_MANUAL_ACTION,
        started_at=datetime.now(timezone.utc),
        current_job_id=jid,
        manual_action_required=True,
        configuration={"auto_submit": False},
    )
    run_mgr.get_run.return_value = mock_run

    job_repo.get_job_by_id.return_value = Job(
        id=jid,
        external_id="ext-resume",
        title="Engineer",
        company="Gamma Inc",
        location="Remote",
        source="glassdoor",
        url="https://www.glassdoor.com/job-listing/j?jl=1234567890",
        created_at=datetime.now(timezone.utc),
    )

    mock_gd_service = MagicMock()
    mock_result = GlassdoorAutomationResult(
        status="blocked",
        url="https://www.glassdoor.com/job-listing/j?jl=1234567890",
        job_id=jid,
        platform="glassdoor",
        current_state=GlassdoorAutomationState.SUBMISSION_READY,
        submit_verified=True,
    )
    mock_gd_service.navigate_to_submit = AsyncMock(return_value=mock_result)
    platform_router.get_apply_service_for_platform.return_value = mock_gd_service

    orchestrator = JobApplicationOrchestrator(
        run_manager=run_mgr,
        job_repository=job_repo,
        platform_router=platform_router,
        application_repository=app_repo,
    )

    resume_req = AgentResumeRequest(agent_run_id=rid)
    resp = await orchestrator.resume_run(resume_req)

    assert resp.status == OrchestratorState.COMPLETED
    assert resp.manual_actions_required is False


# ---------------------------------------------------------------------------
# 6. Skip and Abort Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skip_job():
    run_mgr = MagicMock(spec=AgentRunManager)
    lock_mgr = MagicMock(spec=JobLockManager)
    rid = uuid4()
    jid = uuid4()

    mock_run = AgentRun(
        run_id=rid,
        platform="indeed",
        status=AgentRunStatus.PAUSED_MANUAL_ACTION,
        started_at=datetime.now(timezone.utc),
        current_job_id=jid,
        manual_action_required=True,
    )
    run_mgr.get_run.return_value = mock_run

    orchestrator = JobApplicationOrchestrator(
        run_manager=run_mgr,
        lock_manager=lock_mgr,
    )

    skip_req = AgentSkipRequest(agent_run_id=rid, job_id=jid)
    resp = await orchestrator.skip_job(skip_req)

    lock_mgr.release_lock.assert_called_once_with(job_id=jid, agent_run_id=rid)
    assert mock_run.jobs_skipped == 1
    assert mock_run.current_job_id is None
    assert resp.status == OrchestratorState.COMPLETED


@pytest.mark.asyncio
async def test_abort_run_releases_all_locks():
    run_mgr = MagicMock(spec=AgentRunManager)
    lock_mgr = MagicMock(spec=JobLockManager)
    rid = uuid4()

    mock_run = AgentRun(
        run_id=rid,
        platform="multi",
        status=AgentRunStatus.APPLYING,
        started_at=datetime.now(timezone.utc),
    )
    run_mgr.get_run.return_value = mock_run

    orchestrator = JobApplicationOrchestrator(
        run_manager=run_mgr,
        lock_manager=lock_mgr,
    )

    abort_req = AgentAbortRequest(agent_run_id=rid)
    resp = await orchestrator.abort_run(abort_req)

    lock_mgr.release_all_for_run.assert_called_once_with(rid)
    assert str(rid) in orchestrator._abort_requested
    assert resp.status == OrchestratorState.COMPLETED


# ---------------------------------------------------------------------------
# 7. FastAPI Router Integration Tests
# ---------------------------------------------------------------------------


def test_api_apply_one_success():
    client = TestClient(app)
    mock_orch = MagicMock(spec=JobApplicationOrchestrator)
    jid = uuid4()

    mock_orch.run_single_job = AsyncMock(
        return_value=SingleJobApplyResponse(
            job_id=jid,
            title="Senior Engineer",
            company="Tech Corp",
            platform="indeed",
            match_score=95.0,
            status="success",
            automation_state="SUBMISSION_READY",
        )
    )

    app.dependency_overrides[get_job_orchestrator] = lambda: mock_orch

    try:
        response = client.post(
            "/agent/apply-one",
            json={"job_id": str(jid), "auto_submit": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == str(jid)
        assert data["status"] == "success"
        assert data["automation_state"] == "SUBMISSION_READY"
    finally:
        app.dependency_overrides.pop(get_job_orchestrator, None)


def test_api_apply_one_not_found():
    client = TestClient(app)
    mock_orch = MagicMock(spec=JobApplicationOrchestrator)
    jid = uuid4()

    mock_orch.run_single_job = AsyncMock(side_effect=ValueError(f"Job with id '{jid}' does not exist in database."))
    app.dependency_overrides[get_job_orchestrator] = lambda: mock_orch

    try:
        response = client.post(
            "/agent/apply-one",
            json={"job_id": str(jid), "auto_submit": False},
        )
        assert response.status_code == 404
        assert "does not exist" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_job_orchestrator, None)


def test_api_run_cycle():
    client = TestClient(app)
    mock_orch = MagicMock(spec=JobApplicationOrchestrator)
    rid = uuid4()

    mock_orch.run_application_cycle = AsyncMock(
        return_value=AgentRunResponse(
            run_id=rid,
            platform="multi",
            status=OrchestratorState.COMPLETED,
            started_at=datetime.now(timezone.utc),
            jobs_discovered=10,
            jobs_matched=10,
            jobs_queued=5,
            jobs_attempted=2,
            jobs_submitted=2,
            jobs_skipped=0,
            jobs_already_applied=0,
            jobs_failed=0,
            manual_actions_required=False,
            results=[],
            remaining_queue=[],
        )
    )

    app.dependency_overrides[get_job_orchestrator] = lambda: mock_orch

    try:
        response = client.post(
            "/agent/run",
            json={
                "sites": ["indeed", "glassdoor"],
                "max_jobs": 2,
                "auto_submit": False,
                "minimum_match_score": 75.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == str(rid)
        assert data["status"] == "COMPLETED"
        assert data["jobs_submitted"] == 2
    finally:
        app.dependency_overrides.pop(get_job_orchestrator, None)


# ---------------------------------------------------------------------------
# 8. Edge Cases & Resilience Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_job_duplicate_guard_db():
    """Verify pre-browser DB check catches submitted jobs before launching browser."""
    platform_router = MagicMock(spec=PlatformRouter)
    app_repo = MagicMock(spec=ApplicationRepository)
    jid = uuid4()
    prof_id = uuid4()

    mock_app = MagicMock()
    mock_app.job_id = jid
    mock_app.status = "submitted"
    app_repo.get_application_by_job_and_profile.return_value = mock_app

    orchestrator = JobApplicationOrchestrator(
        platform_router=platform_router,
        application_repository=app_repo,
    )

    run = AgentRun(
        run_id=uuid4(),
        platform="glassdoor",
        status=AgentRunStatus.APPLYING,
        started_at=datetime.now(timezone.utc),
    )
    q_item = QueueItem(
        job_id=jid,
        source="glassdoor",
        title="Frontend Engineer",
        company="Delta Inc",
        location="Remote",
        url="https://www.glassdoor.com/job-listing/j?jl=1000000000",
        match_score=88.0,
        status=QueueStatus.QUEUED,
    )

    res = await orchestrator.process_job(
        job=q_item,
        run=run,
        auto_submit=False,
        profile_id=prof_id,
    )

    # Platform router shouldn't even be called!
    platform_router.get_apply_service_for_platform.assert_not_called()
    assert res.status == "already_applied"
    assert q_item.status == QueueStatus.ALREADY_APPLIED
    assert run.jobs_already_applied == 1


@pytest.mark.asyncio
async def test_process_job_job_unavailable():
    """Verify listing closed/unavailable state handling."""
    platform_router = MagicMock(spec=PlatformRouter)
    mock_gd_service = MagicMock()
    mock_result = GlassdoorAutomationResult(
        status="blocked",
        url="https://www.glassdoor.com/job-listing/j?jl=1234567890",
        job_id=uuid4(),
        platform="glassdoor",
        current_state=GlassdoorAutomationState.JOB_PAGE,
        message="Job is closed or no longer accepting applications",
    )
    # Simulate UNAVAILABLE state
    mock_result.current_state = "JOB_UNAVAILABLE"
    mock_gd_service.navigate_to_submit = AsyncMock(return_value=mock_result)
    platform_router.get_apply_service_for_platform.return_value = mock_gd_service

    app_repo = MagicMock(spec=ApplicationRepository)
    app_repo.get_application_by_job_and_profile.return_value = None

    orchestrator = JobApplicationOrchestrator(
        platform_router=platform_router,
        application_repository=app_repo,
    )

    run = AgentRun(
        run_id=uuid4(),
        platform="glassdoor",
        status=AgentRunStatus.APPLYING,
        started_at=datetime.now(timezone.utc),
    )
    q_item = QueueItem(
        job_id=uuid4(),
        source="glassdoor",
        title="DevOps",
        company="Epsilon",
        location="Remote",
        url="https://www.glassdoor.com/job-listing/j?jl=1234567890",
        match_score=80.0,
        status=QueueStatus.QUEUED,
    )

    res = await orchestrator.process_job(
        job=q_item,
        run=run,
        auto_submit=False,
        profile_id=uuid4(),
    )

    assert res.status == "blocked"
    assert q_item.status == QueueStatus.FAILED
    assert "closed or unavailable" in (res.error or "")


@pytest.mark.asyncio
async def test_run_cycle_failure_isolation():
    """Verify that a failure in one job does NOT terminate execution of remaining queue."""
    disc_svc = MagicMock(spec=ProfileDiscoveryService)
    matches = [
        make_mock_match(title="Job Fails", source="indeed"),
        make_mock_match(title="Job Succeeds", source="indeed"),
    ]
    disc_svc.discover_jobs_from_profile = AsyncMock(return_value=make_mock_discovery_response(matches))

    app_repo = MagicMock(spec=ApplicationRepository)
    app_repo.get_applications_for_jobs.return_value = []
    app_repo.get_application_by_job_and_profile.return_value = None

    platform_router = MagicMock(spec=PlatformRouter)
    mock_service = MagicMock()

    call_count = 0

    async def mock_navigate(req):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Browser session crashed on Job 1")
        return IndeedAutomationResult(
            status="success",
            url=req.url,
            job_id=req.job_id,
            platform="indeed",
            current_state=AutomationState.SUBMISSION_READY,
            submit_verified=True,
        )

    mock_service.navigate_to_submit = AsyncMock(side_effect=mock_navigate)
    platform_router.get_apply_service_for_platform.return_value = mock_service

    mock_settings = MagicMock()
    mock_settings.APPLICATION_DELAY_SECONDS = 0

    orchestrator = JobApplicationOrchestrator(
        discovery_service=disc_svc,
        platform_router=platform_router,
        application_repository=app_repo,
        settings=mock_settings,
    )

    req = AgentRunRequest(
        profile_id=uuid4(),
        sites=["indeed"],
        max_jobs=2,
        auto_submit=False,
    )

    resp = await orchestrator.run_application_cycle(req)

    assert len(resp.results) == 2
    assert resp.results[0].status == "failed"
    assert "Browser session crashed" in (resp.results[0].error or "")
    assert resp.results[1].status == "success"
    assert resp.status == OrchestratorState.COMPLETED


def test_api_resume_endpoint():
    client = TestClient(app)
    mock_orch = MagicMock(spec=JobApplicationOrchestrator)
    rid = uuid4()

    mock_orch.resume_run = AsyncMock(
        return_value=AgentRunResponse(
            run_id=rid,
            platform="multi",
            status=OrchestratorState.COMPLETED,
            started_at=datetime.now(timezone.utc),
            jobs_discovered=5,
            jobs_matched=5,
            jobs_queued=2,
            jobs_attempted=1,
            jobs_submitted=1,
            jobs_skipped=0,
            jobs_already_applied=0,
            jobs_failed=0,
            manual_actions_required=False,
            results=[],
            remaining_queue=[],
        )
    )

    app.dependency_overrides[get_job_orchestrator] = lambda: mock_orch

    try:
        response = client.post(
            "/agent/resume",
            json={"agent_run_id": str(rid)},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "COMPLETED"
    finally:
        app.dependency_overrides.pop(get_job_orchestrator, None)


def test_api_skip_job_endpoint():
    client = TestClient(app)
    mock_orch = MagicMock(spec=JobApplicationOrchestrator)
    rid = uuid4()
    jid = uuid4()

    mock_orch.skip_job = AsyncMock(
        return_value=AgentRunResponse(
            run_id=rid,
            platform="multi",
            status=OrchestratorState.COMPLETED,
            started_at=datetime.now(timezone.utc),
            jobs_discovered=5,
            jobs_matched=5,
            jobs_queued=2,
            jobs_attempted=1,
            jobs_submitted=0,
            jobs_skipped=1,
            jobs_already_applied=0,
            jobs_failed=0,
            manual_actions_required=False,
            results=[],
            remaining_queue=[],
        )
    )

    app.dependency_overrides[get_job_orchestrator] = lambda: mock_orch

    try:
        response = client.post(
            "/agent/skip-job",
            json={"agent_run_id": str(rid), "job_id": str(jid)},
        )
        assert response.status_code == 200
        assert response.json()["jobs_skipped"] == 1
    finally:
        app.dependency_overrides.pop(get_job_orchestrator, None)


def test_api_abort_endpoint():
    client = TestClient(app)
    mock_orch = MagicMock(spec=JobApplicationOrchestrator)
    rid = uuid4()

    mock_orch.abort_run = AsyncMock(
        return_value=AgentRunResponse(
            run_id=rid,
            platform="multi",
            status=OrchestratorState.COMPLETED,
            started_at=datetime.now(timezone.utc),
            jobs_discovered=5,
            jobs_matched=5,
            jobs_queued=2,
            jobs_attempted=0,
            jobs_submitted=0,
            jobs_skipped=0,
            jobs_already_applied=0,
            jobs_failed=0,
            manual_actions_required=False,
            results=[],
            remaining_queue=[],
        )
    )

    app.dependency_overrides[get_job_orchestrator] = lambda: mock_orch

    try:
        response = client.post(
            "/agent/abort",
            json={"agent_run_id": str(rid)},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "COMPLETED"
    finally:
        app.dependency_overrides.pop(get_job_orchestrator, None)

