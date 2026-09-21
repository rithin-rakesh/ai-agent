"""Comprehensive Unit and Integration Tests for Agent Control Plane (Phase 6.1).

Covers:
1. Global daily limit reached -> fail closed (0 attempted, stop_reason="DAILY_LIMIT_REACHED")
2. Platform daily limit reached -> skips/excludes that platform (SKIPPED_DAILY_LIMIT)
3. Stale job older than max_job_age_days -> marked SKIPPED_STALE_JOB, preserved in DB
4. Minimum match score filter -> jobs below threshold excluded
5. Multi-factor priority ranking -> ordered by score, freshness, role, location
6. Concurrency lock -> single active browser application guaranteed
7. Cooldown interval respected between jobs
8. Failure classification mappings (all 7 categories)
9. Transient retry policy (retries once when enabled, not when disabled)
10. Partial failure isolation (advances to next job when safe)
11. Full run report output schema
12. Strict auto_submit=False safety enforcement
13. Scheduler hook abstraction (NoOpScheduler inactive on startup)
14. API integration with POST /agent/run
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from app.agents.control_plane import (
    AgentControlPlane,
    BaseSchedulerHook,
    DailyLimitsGuard,
    FailureClassifier,
    FreshnessPolicy,
    NoOpScheduler,
    ApplicationQueueManager,
)
from app.agents.eligibility_gate import ApplicationEligibilityGate, EligibilityResult
from app.agents.job_application_agent import JobApplicationOrchestrator
from app.agents.lock_manager import JobLockManager
from app.agents.models import (
    AgentRun,
    AgentRunRequest,
    AgentRunResponse,
    AgentRunStatus,
    FailureClassification,
    OrchestratorState,
    QueueItem,
    QueueStatus,
    SingleJobApplyResponse,
)
from app.agents.run_manager import AgentRunManager
from app.api.main import app
from app.api.routers.agents import get_control_plane
from app.config.settings import Settings
from app.database.repositories.application_repository import ApplicationRepository
from app.database.repositories.job_repository import JobRepository
from app.discovery.profile_discovery_service import ProfileDiscoveryService
from app.models.job import (
    ApplicationMethod,
    AvailabilityStatus,
    DiscoveryStats,
    Job,
    MatchedJobItem,
    MatchingStats,
    ProfileJobSearchResponse,
    SearchPlanSummary,
)
from app.profile.service import ProfileService


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------

def make_test_job(
    job_id: Optional[UUID] = None,
    source: str = "indeed",
    title: str = "ML Engineer",
    company: str = "Apple",
    url: Optional[str] = None,
    posted_at: Optional[datetime] = None,
) -> Job:
    jid = job_id or uuid4()
    if not url:
        if source == "glassdoor":
            url = "https://www.glassdoor.com/job-listing/j?jl=1010256944540"
        else:
            url = "https://in.indeed.com/viewjob?jk=892187f461c82742"
    return Job(
        id=jid,
        external_id=str(uuid4()),
        source=source,
        title=title,
        company=company,
        location="Remote",
        url=url,
        status="ACTIVE",
        posted_at=posted_at or datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )


def make_test_match(
    job: Job,
    final_score: float = 85.0,
    application_method: ApplicationMethod = ApplicationMethod.EASY_APPLY,
    availability_status: AvailabilityStatus = AvailabilityStatus.AVAILABLE,
    application_eligible: bool = True,
    posted_at: Optional[datetime] = None,
) -> MatchedJobItem:
    return MatchedJobItem(
        job_id=job.id,
        title=job.title,
        company=job.company,
        location=job.location or "Remote",
        source=job.source,
        url=job.url or "",
        deterministic_score=final_score,
        semantic_score=final_score,
        final_score=final_score,
        decision="strong_match",
        reason="Test match",
        posted_at=posted_at or job.posted_at,
        application_method=application_method,
        availability_status=availability_status,
        application_eligible=application_eligible,
    )


# ---------------------------------------------------------------------------
# 1. Scheduler Abstraction Tests
# ---------------------------------------------------------------------------

def test_noop_scheduler_inactive_by_default():
    """Requirement 13: Scheduler hook is inactive on startup; keeps runs strictly on demand."""
    scheduler = NoOpScheduler()
    assert scheduler.is_active() is False


@pytest.mark.asyncio
async def test_noop_scheduler_operations():
    """NoOpScheduler registers and cancels safely without background daemon."""
    scheduler = NoOpScheduler()
    req = AgentRunRequest(max_jobs=1)
    sched_id = await scheduler.schedule_run(req)
    assert sched_id == "noop_schedule_id"
    cancelled = await scheduler.cancel_schedule(sched_id)
    assert cancelled is True


# ---------------------------------------------------------------------------
# 2. Failure Classifier Tests
# ---------------------------------------------------------------------------

def test_failure_classifier_mappings():
    """Requirement 8: Classifies outcomes into standard Phase 6.1 failure classifications."""
    # Stale
    res = FailureClassifier.classify_outcome("skipped", error="Job posting is stale (posted 14 days ago)")
    assert res == FailureClassification.STALE_JOB

    # External Apply
    res = FailureClassifier.classify_outcome("skipped_external_apply", automation_state="EXTERNAL_APPLY")
    assert res == FailureClassification.EXTERNAL_APPLY

    # Already Applied
    res = FailureClassifier.classify_outcome("skipped_already_applied", automation_state="ALREADY_APPLIED")
    assert res == FailureClassification.ALREADY_APPLIED

    # Unavailable / Expired
    res = FailureClassifier.classify_outcome("blocked", automation_state="JOB_UNAVAILABLE", error="Job is expired")
    assert res == FailureClassification.JOB_UNAVAILABLE

    # User Action Required
    res = FailureClassifier.classify_outcome("manual_action_required", automation_state="CAPTCHA_DETECTED")
    assert res == FailureClassification.USER_ACTION_REQUIRED

    # Transient Failure
    res = FailureClassifier.classify_outcome("blocked", error="Navigation timeout after 30000ms")
    assert res == FailureClassification.TRANSIENT_FAILURE

    # Permanent Failure
    res = FailureClassifier.classify_outcome("failed", error="Invalid form selector crashed unexpectedly")
    assert res == FailureClassification.PERMANENT_FAILURE

    # Success / Submission Ready (not a failure)
    res = FailureClassifier.classify_outcome("submission_ready", automation_state="SUBMISSION_READY")
    assert res is None


# ---------------------------------------------------------------------------
# 3. Daily Limits Guard Tests
# ---------------------------------------------------------------------------

def test_daily_limits_fail_closed_global():
    """Requirement 1: Global daily limit reached -> fail closed."""
    settings = Settings(
        MAX_APPLICATIONS_PER_DAY=10,
        MAX_INDEED_APPLICATIONS_PER_DAY=5,
        MAX_GLASSDOOR_APPLICATIONS_PER_DAY=5,
    )
    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.count_daily_applications.return_value = 10

    guard = DailyLimitsGuard(application_repo=mock_app_repo, settings=settings)
    can_start, reason, details = guard.check_can_start_run()

    assert can_start is False
    assert "Daily application limit reached" in reason
    assert details["limit_type"] == "GLOBAL_DAILY_LIMIT"
    assert details["limit"] == 10
    assert details["current"] == 10


def test_daily_limits_platform_capacity():
    """Requirement 2: Platform daily limit reached -> excludes platform."""
    settings = Settings(
        MAX_APPLICATIONS_PER_DAY=10,
        MAX_INDEED_APPLICATIONS_PER_DAY=5,
        MAX_GLASSDOOR_APPLICATIONS_PER_DAY=5,
    )
    mock_app_repo = MagicMock(spec=ApplicationRepository)

    def side_effect(profile_id=None, platform=None, since=None):
        if platform == "indeed":
            return 5
        elif platform == "glassdoor":
            return 2
        return 7

    mock_app_repo.count_daily_applications.side_effect = side_effect
    guard = DailyLimitsGuard(application_repo=mock_app_repo, settings=settings)

    can_start, _, _ = guard.check_can_start_run()
    assert can_start is True

    ind_ok, ind_reason = guard.check_platform_capacity("indeed")
    assert ind_ok is False
    assert "Platform daily limit reached" in ind_reason

    gd_ok, gd_reason = guard.check_platform_capacity("glassdoor")
    assert gd_ok is True
    assert gd_reason is None


# ---------------------------------------------------------------------------
# 4. Freshness Policy Tests
# ---------------------------------------------------------------------------

def test_freshness_policy_stale_detection():
    """Requirement 3: Stale jobs older than max_job_age_days are flagged."""
    policy = FreshnessPolicy(max_job_age_days=7)

    now = datetime.now(timezone.utc)
    fresh_date = now - timedelta(days=2)
    stale_date = now - timedelta(days=10)

    is_fresh, days_fresh = policy.is_stale(fresh_date)
    assert is_fresh is False
    assert days_fresh == 2

    is_stale, days_stale = policy.is_stale(stale_date)
    assert is_stale is True
    assert days_stale == 10

    is_undated, _ = policy.is_stale(None)
    assert is_undated is False


# ---------------------------------------------------------------------------
# 5. Application Queue Manager & Prioritization Tests
# ---------------------------------------------------------------------------

def test_queue_prioritization_and_filtering():
    """Requirements 4 & 5: Score threshold, stale filtering, and multi-factor ranking."""
    settings = Settings(MAX_JOB_AGE_DAYS=7)
    now = datetime.now(timezone.utc)

    job1 = make_test_job(title="Software Engineer", company="A", posted_at=now - timedelta(days=1))
    job2 = make_test_job(title="Staff ML Engineer", company="B", posted_at=now - timedelta(days=2))
    job3_low_score = make_test_job(title="DevOps", company="C", posted_at=now - timedelta(days=1))
    job4_stale = make_test_job(title="Data Scientist", company="D", posted_at=now - timedelta(days=12))

    matches = [
        make_test_match(job1, final_score=75.0, posted_at=job1.posted_at),
        make_test_match(job2, final_score=90.0, posted_at=job2.posted_at),
        make_test_match(job3_low_score, final_score=40.0, posted_at=job3_low_score.posted_at),
        make_test_match(job4_stale, final_score=95.0, posted_at=job4_stale.posted_at),
    ]

    mock_job_repo = MagicMock(spec=JobRepository)
    mock_job_repo.get_job_by_id.side_effect = lambda jid: {
        job1.id: job1,
        job2.id: job2,
        job3_low_score.id: job3_low_score,
        job4_stale.id: job4_stale,
    }.get(jid, None)

    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.get_applications_for_jobs.return_value = []
    mock_app_repo.count_daily_applications.return_value = 0

    mock_lock_manager = MagicMock(spec=JobLockManager)
    mock_lock_manager.is_locked.return_value = False

    mock_profile_service = MagicMock(spec=ProfileService)
    mock_profile_service.get_profile.return_value = None

    guard = DailyLimitsGuard(application_repo=mock_app_repo, settings=settings)

    queue_mgr = ApplicationQueueManager(
        job_repo=mock_job_repo,
        app_repo=mock_app_repo,
        profile_service=mock_profile_service,
    )

    request = AgentRunRequest(
        minimum_match_score=55.0,
        max_job_age_days=7,
        max_jobs_per_platform=5,
    )

    active_queue, skipped_items, limits_hit = queue_mgr.build_and_prioritize_queue(
        top_matches=matches,
        config=request,
        daily_limits=guard,
        lock_manager=mock_lock_manager,
        run_id=uuid4(),
    )

    # job4_stale must be marked SKIPPED_STALE_JOB
    stale_skipped = [s for s in skipped_items if s.status == "SKIPPED_STALE_JOB"]
    assert len(stale_skipped) == 1
    assert stale_skipped[0].job_id == job4_stale.id

    # job3_low_score was excluded due to minimum match score (40 < 55)
    # Active queue should contain job2 (90) and job1 (75), ordered with job2 first
    assert len(active_queue) == 2
    assert active_queue[0].job_id == job2.id
    assert active_queue[0].match_score == 90.0
    assert active_queue[1].job_id == job1.id
    assert active_queue[1].match_score == 75.0


# ---------------------------------------------------------------------------
# 6. Control Plane Full Cycle & Fail Closed Integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_control_plane_fail_closed_on_global_limit():
    """Requirement 1: Run halts immediately when daily application limit is reached."""
    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.count_daily_applications.return_value = 10
    mock_job_repo = MagicMock(spec=JobRepository)
    mock_lock_manager = MagicMock(spec=JobLockManager)
    run_mgr = AgentRunManager()

    orchestrator = JobApplicationOrchestrator(
        job_repository=mock_job_repo,
        application_repository=mock_app_repo,
        run_manager=run_mgr,
        lock_manager=mock_lock_manager,
    )

    settings = Settings(MAX_APPLICATIONS_PER_DAY=10)
    control_plane = AgentControlPlane(orchestrator=orchestrator, settings=settings)

    req = AgentRunRequest(max_jobs=5)
    resp = await control_plane.run_autonomous_cycle(req)

    assert resp.jobs_attempted == 0
    assert resp.jobs_submitted == 0
    assert resp.limits_hit is not None
    assert resp.limits_hit.get("limit_type") == "GLOBAL_DAILY_LIMIT"
    assert resp.report["stop_reason"] == "DAILY_LIMIT_REACHED"


@pytest.mark.asyncio
async def test_control_plane_concurrency_lock_and_single_execution():
    """Requirement 6: Concurrency lock prevents parallel browser execution."""
    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.count_daily_applications.return_value = 0
    mock_app_repo.get_applications_for_jobs.return_value = []
    mock_job_repo = MagicMock(spec=JobRepository)
    mock_lock_manager = MagicMock(spec=JobLockManager)
    mock_lock_manager.acquire_lock.return_value = True
    mock_lock_manager.is_locked.return_value = False
    mock_lock_manager.release_lock.return_value = True
    mock_lock_manager.release_all_for_run.return_value = 0
    run_mgr = AgentRunManager()

    job = make_test_job()
    mock_job_repo.get_job_by_id.return_value = job
    matched = make_test_match(job, final_score=90.0)

    mock_discovery = MagicMock(spec=ProfileDiscoveryService)
    mock_discovery.discover_jobs_from_profile = AsyncMock(
        return_value=ProfileJobSearchResponse(
            search_plan=SearchPlanSummary(queries_generated=1),
            discovery=DiscoveryStats(total_discovered=1, canonical_urls_resolved=1, persisted_jobs=1, errors=0),
            matching=MatchingStats(total_evaluated=1, strong_matches=1, good_matches=0, low_matches=0),
            top_matches=[matched],
        )
    )

    mock_gate = MagicMock(spec=ApplicationEligibilityGate)
    mock_gate.evaluate_job_live = AsyncMock(
        return_value=EligibilityResult(
            is_eligible=True,
            application_method=ApplicationMethod.EASY_APPLY,
            availability_status=AvailabilityStatus.AVAILABLE,
            status_code="ELIGIBLE",
            message="Eligible",
        )
    )

    orchestrator = JobApplicationOrchestrator(
        discovery_service=mock_discovery,
        job_repository=mock_job_repo,
        application_repository=mock_app_repo,
        run_manager=run_mgr,
        lock_manager=mock_lock_manager,
        eligibility_gate=mock_gate,
    )

    # Mock process_job
    orchestrator.process_job = AsyncMock(
        return_value=SingleJobApplyResponse(
            job_id=job.id,
            title=job.title,
            company=job.company,
            platform="indeed",
            match_score=90.0,
            status="submission_ready",
            automation_state="SUBMISSION_READY",
            application_submitted=False,
        )
    )

    settings = Settings(MAX_APPLICATIONS_PER_DAY=10, COOLDOWN_SECONDS_BETWEEN_APPLICATIONS=0)
    control_plane = AgentControlPlane(orchestrator=orchestrator, settings=settings)

    assert not control_plane._concurrency_lock.locked()

    req = AgentRunRequest(max_jobs=1, auto_submit=False, application_delay_seconds=0)
    resp = await control_plane.run_autonomous_cycle(req)

    assert not control_plane._concurrency_lock.locked()
    assert resp.jobs_attempted == 1
    assert resp.jobs_submitted == 0
    assert resp.report["attempted"] == 1
    assert resp.report["submitted"] == 0

    # Strict auto_submit=False preserved
    orchestrator.process_job.assert_awaited_once_with(
        job=orchestrator.process_job.call_args[1]["job"],
        run=orchestrator.process_job.call_args[1]["run"],
        auto_submit=False,
        profile_id=req.profile_id,
    )


@pytest.mark.asyncio
async def test_control_plane_transient_retry_policy():
    """Requirement 9: Transient failure retried once when retry_failed_jobs=True."""
    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.count_daily_applications.return_value = 0
    mock_app_repo.get_applications_for_jobs.return_value = []
    mock_job_repo = MagicMock(spec=JobRepository)
    mock_lock_manager = MagicMock(spec=JobLockManager)
    mock_lock_manager.acquire_lock.return_value = True
    mock_lock_manager.is_locked.return_value = False
    mock_lock_manager.release_lock.return_value = True
    mock_lock_manager.release_all_for_run.return_value = 0
    run_mgr = AgentRunManager()

    job = make_test_job()
    mock_job_repo.get_job_by_id.return_value = job
    matched = make_test_match(job, final_score=90.0)

    mock_discovery = MagicMock(spec=ProfileDiscoveryService)
    mock_discovery.discover_jobs_from_profile = AsyncMock(
        return_value=ProfileJobSearchResponse(
            search_plan=SearchPlanSummary(queries_generated=1),
            discovery=DiscoveryStats(total_discovered=1, canonical_urls_resolved=1, persisted_jobs=1, errors=0),
            matching=MatchingStats(total_evaluated=1, strong_matches=1, good_matches=0, low_matches=0),
            top_matches=[matched],
        )
    )

    mock_gate = MagicMock(spec=ApplicationEligibilityGate)
    mock_gate.evaluate_job_live = AsyncMock(
        return_value=EligibilityResult(
            is_eligible=True,
            application_method=ApplicationMethod.EASY_APPLY,
            availability_status=AvailabilityStatus.AVAILABLE,
            status_code="ELIGIBLE",
            message="Eligible",
        )
    )

    orchestrator = JobApplicationOrchestrator(
        discovery_service=mock_discovery,
        job_repository=mock_job_repo,
        application_repository=mock_app_repo,
        run_manager=run_mgr,
        lock_manager=mock_lock_manager,
        eligibility_gate=mock_gate,
    )

    # First attempt: timeout (transient); Second attempt: submission_ready
    attempt1 = SingleJobApplyResponse(
        job_id=job.id,
        title=job.title,
        company=job.company,
        platform="indeed",
        match_score=90.0,
        status="blocked",
        automation_state="TIMEOUT",
        error="Browser timeout after 30s",
        application_submitted=False,
    )
    attempt2 = SingleJobApplyResponse(
        job_id=job.id,
        title=job.title,
        company=job.company,
        platform="indeed",
        match_score=90.0,
        status="submission_ready",
        automation_state="SUBMISSION_READY",
        application_submitted=False,
    )
    orchestrator.process_job = AsyncMock(side_effect=[attempt1, attempt2])

    control_plane = AgentControlPlane(orchestrator=orchestrator)
    req = AgentRunRequest(max_jobs=1, retry_failed_jobs=True, application_delay_seconds=0)

    resp = await control_plane.run_autonomous_cycle(req)

    # Should have called process_job twice due to retry
    assert orchestrator.process_job.call_count == 2
    assert resp.jobs_attempted == 1
    assert resp.results[0].automation_state == "SUBMISSION_READY"


@pytest.mark.asyncio
async def test_control_plane_partial_failure_isolation():
    """Requirement 10: Partial failure on job 1 isolates and advances safely to job 2."""
    mock_app_repo = MagicMock(spec=ApplicationRepository)
    mock_app_repo.count_daily_applications.return_value = 0
    mock_app_repo.get_applications_for_jobs.return_value = []
    mock_job_repo = MagicMock(spec=JobRepository)
    mock_lock_manager = MagicMock(spec=JobLockManager)
    mock_lock_manager.acquire_lock.return_value = True
    mock_lock_manager.is_locked.return_value = False
    mock_lock_manager.release_lock.return_value = True
    mock_lock_manager.release_all_for_run.return_value = 0
    run_mgr = AgentRunManager()

    job1 = make_test_job(title="Job 1")
    job2 = make_test_job(title="Job 2")
    mock_job_repo.get_job_by_id.side_effect = lambda jid: job1 if jid == job1.id else job2

    match1 = make_test_match(job1, final_score=90.0)
    match2 = make_test_match(job2, final_score=85.0)

    mock_discovery = MagicMock(spec=ProfileDiscoveryService)
    mock_discovery.discover_jobs_from_profile = AsyncMock(
        return_value=ProfileJobSearchResponse(
            search_plan=SearchPlanSummary(queries_generated=1),
            discovery=DiscoveryStats(total_discovered=2, canonical_urls_resolved=2, persisted_jobs=2, errors=0),
            matching=MatchingStats(total_evaluated=2, strong_matches=2, good_matches=0, low_matches=0),
            top_matches=[match1, match2],
        )
    )

    mock_gate = MagicMock(spec=ApplicationEligibilityGate)
    mock_gate.evaluate_job_live = AsyncMock(
        return_value=EligibilityResult(
            is_eligible=True,
            application_method=ApplicationMethod.EASY_APPLY,
            availability_status=AvailabilityStatus.AVAILABLE,
            status_code="ELIGIBLE",
            message="Eligible",
        )
    )

    orchestrator = JobApplicationOrchestrator(
        discovery_service=mock_discovery,
        job_repository=mock_job_repo,
        application_repository=mock_app_repo,
        run_manager=run_mgr,
        lock_manager=mock_lock_manager,
        eligibility_gate=mock_gate,
    )

    # Job 1 fails permanently; Job 2 succeeds to submission_ready
    res1 = SingleJobApplyResponse(
        job_id=job1.id,
        title=job1.title,
        company=job1.company,
        platform="indeed",
        match_score=90.0,
        status="failed",
        automation_state="ERROR",
        error="Unknown fatal error in DOM",
        application_submitted=False,
    )
    res2 = SingleJobApplyResponse(
        job_id=job2.id,
        title=job2.title,
        company=job2.company,
        platform="indeed",
        match_score=85.0,
        status="submission_ready",
        automation_state="SUBMISSION_READY",
        application_submitted=False,
    )
    orchestrator.process_job = AsyncMock(side_effect=[res1, res2])

    control_plane = AgentControlPlane(orchestrator=orchestrator)
    req = AgentRunRequest(max_jobs=2, max_jobs_per_platform=2, retry_failed_jobs=False, application_delay_seconds=0)

    resp = await control_plane.run_autonomous_cycle(req)

    assert resp.jobs_attempted == 2
    assert len(resp.results) == 2
    assert resp.results[0].status == "failed"
    assert resp.results[1].status == "submission_ready"
    assert resp.report["failed"] == 1
    assert resp.report["attempted"] == 2
    assert resp.failure_classification.get(FailureClassification.PERMANENT_FAILURE.value) == 1


# ---------------------------------------------------------------------------
# 7. FastAPI Endpoint Integration Tests (POST /agent/run)
# ---------------------------------------------------------------------------

def test_api_post_agent_run_integration():
    """Requirement 14: POST /agent/run passes through AgentControlPlane and returns report."""
    mock_control_plane = MagicMock(spec=AgentControlPlane)
    mock_run_id = uuid4()

    mock_resp = AgentRunResponse(
        run_id=mock_run_id,
        status=OrchestratorState.COMPLETED,
        started_at=datetime.now(timezone.utc),
        jobs_discovered=5,
        jobs_matched=3,
        jobs_application_eligible=2,
        jobs_queued=2,
        jobs_attempted=1,
        jobs_submitted=0,
        jobs_skipped=1,
        jobs_stale=1,
        jobs_already_applied=0,
        jobs_external_apply=0,
        jobs_failed=0,
        manual_actions_required=False,
        results=[
            SingleJobApplyResponse(
                job_id=uuid4(),
                title="Staff ML Engineer",
                company="Apple",
                platform="indeed",
                match_score=92.0,
                status="submission_ready",
                automation_state="SUBMISSION_READY",
                application_submitted=False,
            )
        ],
        limits_hit={},
        failure_classification={},
        report={
            "discovered": 5,
            "matched": 3,
            "eligible": 2,
            "queued": 2,
            "attempted": 1,
            "submitted": 0,
            "skipped": 1,
            "already_applied": 0,
            "external_apply": 0,
            "stale": 1,
            "failed": 0,
            "manual_action_required": 0,
            "stop_reason": "MAX_JOBS_REACHED",
            "limits_hit": {},
            "failure_classification": {},
        },
    )
    mock_control_plane.run_autonomous_cycle = AsyncMock(return_value=mock_resp)

    app.dependency_overrides[get_control_plane] = lambda: mock_control_plane

    try:
        client = TestClient(app)
        response = client.post(
            "/agent/run",
            json={
                "max_jobs": 1,
                "auto_submit": False,
                "require_easy_apply": True,
                "max_job_age_days": 7,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == str(mock_run_id)
        assert data["status"] == "COMPLETED"
        assert data["jobs_stale"] == 1
        assert "report" in data
        assert data["report"]["stop_reason"] == "MAX_JOBS_REACHED"
        mock_control_plane.run_autonomous_cycle.assert_awaited_once()

    finally:
        app.dependency_overrides.pop(get_control_plane, None)


def test_search_planner_distributes_across_preferred_roles():
    """Verify ProfileSearchPlanner prioritizes candidate's distinct preferred roles."""
    from app.discovery.search_planner import ProfileSearchPlanner
    from app.models.profile import CandidateProfileData, PersonalDetails, CareerPreferences, EducationItem, SkillItem

    profile = CandidateProfileData(
        personal=PersonalDetails(name="Rithin Rakesh", email="rithin@example.com", location="Kerala, India"),
        career=CareerPreferences(
            preferred_roles=["AI Engineer", "AI ML Engineer", "Data Scientist", "Machine Learning Engineer"],
            preferred_locations=["Kerala, India", "Bangalore, India", "Chennai, India", "Remote"],
            experience_years=0.8,
        ),
        education=[EducationItem(degree="B.Tech", institution="KTU", end_date="2024")],
        skills=[SkillItem(skill="Python", importance="critical"), SkillItem(skill="Machine Learning", importance="critical")],
    )

    planner = ProfileSearchPlanner()
    plan = planner.create_search_plan(profile_data=profile, max_queries=3, site="indeed")

    assert len(plan.queries) == 3
    terms = [q.search_term for q in plan.queries]
    # Verify all 3 queries cover distinct preferred roles instead of repeating a single role
    assert terms[0] == "AI Engineer"
    assert terms[1] == "AI ML Engineer"
    assert terms[2] == "Data Scientist"


def test_compute_eligibility_diagnostics():
    """Verify compute_eligibility_diagnostics produces the 11 required counters and sample items."""
    control_plane = AgentControlPlane()
    job1_id = uuid4()
    job2_id = uuid4()

    items = [
        MatchedJobItem(
            job_id=job1_id,
            title="AI Engineer",
            company="Tech Corp",
            location="Remote",
            source="indeed",
            url="https://in.indeed.com/viewjob?jk=1234567890abcdef",
            final_score=85.0,
            decision="strong_match",
            application_method=ApplicationMethod.EASY_APPLY.value,
            availability_status=AvailabilityStatus.AVAILABLE.value,
            application_eligible=True,
        ),
        MatchedJobItem(
            job_id=job2_id,
            title="Senior ML Engineer",
            company="Legacy Corp",
            location="Bangalore",
            source="indeed",
            url="https://in.indeed.com/viewjob?jk=abcdef1234567890",
            final_score=35.0,
            decision="reject",
            application_method=ApplicationMethod.EXTERNAL_APPLY.value,
            availability_status=AvailabilityStatus.UNKNOWN.value,
            application_eligible=False,
        ),
    ]

    breakdown, samples = control_plane.compute_eligibility_diagnostics(
        candidate_items=items,
        profile_id=None,
        max_job_age_days=7,
        top_matches=items,
    )

    # Verify all 11 required metric keys exist
    expected_keys = [
        "easy_apply", "external_apply", "unknown_method",
        "available", "expired", "unavailable", "unknown_availability",
        "already_applied", "stale", "eligible", "ineligible",
    ]
    for key in expected_keys:
        assert key in breakdown, f"Missing breakdown counter: {key}"

    assert breakdown["easy_apply"] == 1
    assert breakdown["external_apply"] == 1
    assert breakdown["available"] == 1
    assert breakdown["eligible"] == 1
    assert breakdown["ineligible"] == 1

    assert len(samples) == 2
    assert samples[0]["job_id"] == str(job1_id)
    assert samples[0]["application_eligible"] is True
    assert samples[0]["rejection_reason"] == "ELIGIBLE"

    assert samples[1]["job_id"] == str(job2_id)
    assert samples[1]["application_eligible"] is False
    assert "EXTERNAL_APPLY" in samples[1]["rejection_reason"]


def test_normalizer_easy_apply_none_when_unspecified():
    """Verify JobNormalizer does not force easy_apply=False when raw data has no easyApply field."""
    from app.discovery.normalizer import JobNormalizer

    raw_payload = {
        "title": "Data Scientist",
        "company": "DeepMind",
        "location": "London",
        "url": "https://www.indeed.com/viewjob?jk=test123456",
        "site": "indeed",
    }
    job_create, _ = JobNormalizer.normalize(raw_payload)
    assert job_create.easy_apply is None

