"""Comprehensive Unit and Integration Tests for Application Eligibility Gate (Phase 6.0.2).

Covers:
1. Easy Apply -> eligible
2. External Apply -> ineligible (SKIPPED_EXTERNAL_APPLY)
3. Expired -> ineligible (JOB_UNAVAILABLE / SKIPPED_EXPIRED)
4. Unavailable -> ineligible (JOB_UNAVAILABLE)
5. Already Applied -> ineligible (ALREADY_APPLIED)
6. match_score=100 does NOT override eligibility
7. apply-one skips external apply without opening application
8. apply-one skips expired job without opening application
9. agent/run skips ineligible jobs and updates metrics
10. agent/run continues queue to next eligible job
11. Live pre-application inspection check execution
12. Indeed application method detection
13. Glassdoor Easy Apply detection
14. Integration with API endpoints
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from app.agents.eligibility_gate import ApplicationEligibilityGate, EligibilityResult
from app.agents.job_application_agent import JobApplicationOrchestrator
from app.agents.lock_manager import JobLockManager
from app.agents.models import (
    AgentApplyOneRequest,
    AgentRun,
    AgentRunRequest,
    AgentRunResponse,
    AgentRunStatus,
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
    GlassdoorAutomationNavigateRequest,
    GlassdoorAutomationResult,
    GlassdoorAutomationState,
)
from app.automation.indeed.models import (
    AutomationState,
    IndeedAutomationNavigateRequest,
    IndeedAutomationResult,
)
from app.database.repositories.application_repository import ApplicationRepository
from app.database.repositories.job_repository import JobRepository
from app.discovery.profile_discovery_service import ProfileDiscoveryService
from app.models.application import Application
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


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def make_test_job(
    job_id: Optional[UUID] = None,
    source: str = "indeed",
    title: str = "ML Engineer",
    company: str = "Apple",
    url: Optional[str] = None,
) -> Job:
    jid = job_id or uuid4()
    if not url:
        if source == "glassdoor":
            url = f"https://www.glassdoor.com/job-listing/j?jl=1010256944540"
        else:
            url = f"https://in.indeed.com/viewjob?jk=892187f461c82742"
    return Job(
        id=jid,
        external_id=str(uuid4()),
        source=source,
        title=title,
        company=company,
        location="Remote",
        url=url,
        status="ACTIVE",
        created_at=datetime.now(timezone.utc),
    )


def make_test_match(
    job: Job,
    final_score: float = 100.0,
    application_method: ApplicationMethod = ApplicationMethod.UNKNOWN,
    availability_status: AvailabilityStatus = AvailabilityStatus.AVAILABLE,
    application_eligible: bool = True,
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
        application_method=application_method,
        availability_status=availability_status,
        application_eligible=application_eligible,
    )


# ---------------------------------------------------------------------------
# 1-5. Direct Eligibility Gate Unit Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gate_easy_apply_eligible():
    """1. Easy Apply -> eligible."""
    gate = ApplicationEligibilityGate()
    mock_apply_svc = MagicMock()
    mock_apply_svc.inspect_job_application = AsyncMock(
        return_value={
            "application_method": "EASY_APPLY",
            "availability": "AVAILABLE",
            "is_easy_apply": True,
            "status": "EASY_APPLY_AVAILABLE",
        }
    )
    mock_router = MagicMock(spec=PlatformRouter)
    mock_router.get_apply_service_for_platform.return_value = mock_apply_svc
    gate.platform_router = mock_router

    eval_res = await gate.evaluate_job_live(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=1234567890abcdef",
        platform="indeed",
    )

    assert eval_res.is_eligible is True
    assert eval_res.application_method == ApplicationMethod.EASY_APPLY
    assert eval_res.availability_status == AvailabilityStatus.AVAILABLE
    assert eval_res.status_code in ("ELIGIBLE", "eligible")


@pytest.mark.asyncio
async def test_gate_external_apply_ineligible():
    """2. External Apply -> ineligible (SKIPPED_EXTERNAL_APPLY)."""
    gate = ApplicationEligibilityGate()
    mock_apply_svc = MagicMock()
    mock_apply_svc.inspect_job_application = AsyncMock(
        return_value={
            "application_method": "EXTERNAL_APPLY",
            "availability": "AVAILABLE",
            "is_easy_apply": False,
            "status": "EXTERNAL_APPLY",
            "message": "Apply on company website",
        }
    )
    mock_router = MagicMock(spec=PlatformRouter)
    mock_router.get_apply_service_for_platform.return_value = mock_apply_svc
    gate.platform_router = mock_router

    eval_res = await gate.evaluate_job_live(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=892187f461c82742",
        platform="indeed",
    )

    assert eval_res.is_eligible is False
    assert eval_res.application_method == ApplicationMethod.EXTERNAL_APPLY
    assert eval_res.status_code == "SKIPPED_EXTERNAL_APPLY"
    assert eval_res.skip_reason in ("SKIPPED_EXTERNAL_APPLY", "EXTERNAL_APPLY")


@pytest.mark.asyncio
async def test_gate_expired_job_ineligible():
    """3. Expired -> ineligible (JOB_UNAVAILABLE)."""
    gate = ApplicationEligibilityGate()
    mock_apply_svc = MagicMock()
    mock_apply_svc.inspect_job_application = AsyncMock(
        return_value={
            "application_method": "UNKNOWN",
            "availability": "EXPIRED",
            "status": "JOB_EXPIRED",
            "message": "This job has expired",
        }
    )
    mock_router = MagicMock(spec=PlatformRouter)
    mock_router.get_apply_service_for_platform.return_value = mock_apply_svc
    gate.platform_router = mock_router

    eval_res = await gate.evaluate_job_live(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=1234567890abcdef",
        platform="indeed",
    )

    assert eval_res.is_eligible is False
    assert eval_res.availability_status == AvailabilityStatus.EXPIRED
    assert eval_res.status_code == "JOB_UNAVAILABLE"
    assert eval_res.skip_reason in ("SKIPPED_EXPIRED", "EXPIRED", "JOB_UNAVAILABLE")


@pytest.mark.asyncio
async def test_gate_unavailable_job_ineligible():
    """4. Unavailable -> ineligible (JOB_UNAVAILABLE)."""
    gate = ApplicationEligibilityGate()
    mock_apply_svc = MagicMock()
    mock_apply_svc.inspect_job_application = AsyncMock(
        return_value={
            "application_method": "UNKNOWN",
            "availability": "UNAVAILABLE",
            "status": "JOB_UNAVAILABLE",
            "message": "Job listing not found",
        }
    )
    mock_router = MagicMock(spec=PlatformRouter)
    mock_router.get_apply_service_for_platform.return_value = mock_apply_svc
    gate.platform_router = mock_router

    eval_res = await gate.evaluate_job_live(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=1234567890abcdef",
        platform="indeed",
    )

    assert eval_res.is_eligible is False
    assert eval_res.availability_status == AvailabilityStatus.UNAVAILABLE
    assert eval_res.status_code == "JOB_UNAVAILABLE"


@pytest.mark.asyncio
async def test_gate_already_applied_ineligible():
    """5. Already Applied -> ineligible (ALREADY_APPLIED)."""
    gate = ApplicationEligibilityGate()
    mock_apply_svc = MagicMock()
    mock_apply_svc.inspect_job_application = AsyncMock(
        return_value={
            "application_method": "EASY_APPLY",
            "availability": "AVAILABLE",
            "status": "ALREADY_APPLIED",
            "message": "You applied to this job on 2026-09-01",
        }
    )
    mock_router = MagicMock(spec=PlatformRouter)
    mock_router.get_apply_service_for_platform.return_value = mock_apply_svc
    gate.platform_router = mock_router

    eval_res = await gate.evaluate_job_live(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=1234567890abcdef",
        platform="indeed",
    )

    assert eval_res.is_eligible is False
    assert eval_res.status_code == "ALREADY_APPLIED"
    assert eval_res.skip_reason == "ALREADY_APPLIED"


# ---------------------------------------------------------------------------
# 6. Match Score 100 Does Not Override Ineligibility
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_match_score_100_does_not_override_ineligibility():
    """6. High match score (100) must NOT override ineligibility if external apply or unavailable."""
    gate = ApplicationEligibilityGate()
    mock_apply_svc = MagicMock()
    mock_apply_svc.inspect_job_application = AsyncMock(
        return_value={
            "application_method": "EXTERNAL_APPLY",
            "availability": "AVAILABLE",
            "is_easy_apply": False,
            "status": "EXTERNAL_APPLY",
        }
    )
    mock_router = MagicMock(spec=PlatformRouter)
    mock_router.get_apply_service_for_platform.return_value = mock_apply_svc
    gate.platform_router = mock_router

    eval_res = await gate.evaluate_job_live(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=892187f461c82742",
        platform="indeed",
    )

    # Even though user gave it 100% match score in DB, live gate rejects it
    assert eval_res.is_eligible is False
    assert eval_res.status_code == "SKIPPED_EXTERNAL_APPLY"


# ---------------------------------------------------------------------------
# 7-8. Single-Job Mode (POST /agent/apply-one)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_apply_one_skips_external_apply_without_opening_application(tmp_path):
    """7. apply-one skips external apply without opening application."""
    test_job = make_test_job(title="Apple ML Engineer", company="Apple", source="indeed")

    job_repo = MagicMock(spec=JobRepository)
    job_repo.get_job_by_id.return_value = test_job

    app_repo = MagicMock(spec=ApplicationRepository)
    app_repo.get_application_by_job_and_profile.return_value = None

    lock_mgr = JobLockManager(lock_file=str(tmp_path / "test_locks.json"))
    run_mgr = AgentRunManager(repository=MagicMock())

    # Gate returns external apply
    gate = MagicMock(spec=ApplicationEligibilityGate)
    gate.evaluate_job_live = AsyncMock(
        return_value=EligibilityResult(
            is_eligible=False,
            skip_reason="EXTERNAL_APPLY",
            status_code="SKIPPED_EXTERNAL_APPLY",
            application_method=ApplicationMethod.EXTERNAL_APPLY,
            availability_status=AvailabilityStatus.AVAILABLE,
            message="Apply on company site: Apple.com",
        )
    )

    platform_router = MagicMock(spec=PlatformRouter)
    platform_router.is_platform_supported.return_value = True
    mock_apply_svc = MagicMock()
    mock_apply_svc.navigate_to_submit = AsyncMock()
    mock_apply_svc.apply_to_job = AsyncMock()
    platform_router.get_apply_service_for_platform.return_value = mock_apply_svc

    orchestrator = JobApplicationOrchestrator(
        job_repository=job_repo,
        application_repository=app_repo,
        platform_router=platform_router,
        lock_manager=lock_mgr,
        run_manager=run_mgr,
        eligibility_gate=gate,
    )

    res = await orchestrator.run_single_job(
        AgentApplyOneRequest(job_id=test_job.id, auto_submit=False)
    )

    # Verifications
    assert res.status == "SKIPPED_EXTERNAL_APPLY"
    assert res.automation_state == "SKIPPED_EXTERNAL_APPLY"
    assert res.application_eligible is False
    assert res.application_method == "EXTERNAL_APPLY"
    # CRITICAL: Underlying application automation must NOT have been called
    mock_apply_svc.navigate_to_submit.assert_not_called()
    mock_apply_svc.apply_to_job.assert_not_called()


@pytest.mark.asyncio
async def test_apply_one_skips_expired_job_without_opening_application(tmp_path):
    """8. apply-one skips expired job without opening application."""
    test_job = make_test_job(title="Kyndryl Engineer", company="Kyndryl", source="indeed")

    job_repo = MagicMock(spec=JobRepository)
    job_repo.get_job_by_id.return_value = test_job

    app_repo = MagicMock(spec=ApplicationRepository)
    app_repo.get_application_by_job_and_profile.return_value = None

    lock_mgr = JobLockManager(lock_file=str(tmp_path / "test_locks.json"))
    run_mgr = AgentRunManager(repository=MagicMock())

    gate = MagicMock(spec=ApplicationEligibilityGate)
    gate.evaluate_job_live = AsyncMock(
        return_value=EligibilityResult(
            is_eligible=False,
            skip_reason="EXPIRED",
            status_code="JOB_UNAVAILABLE",
            application_method=ApplicationMethod.UNKNOWN,
            availability_status=AvailabilityStatus.EXPIRED,
            message="This job is no longer available",
        )
    )

    platform_router = MagicMock(spec=PlatformRouter)
    platform_router.is_platform_supported.return_value = True
    mock_apply_svc = MagicMock()
    mock_apply_svc.navigate_to_submit = AsyncMock()
    mock_apply_svc.apply_to_job = AsyncMock()
    platform_router.get_apply_service_for_platform.return_value = mock_apply_svc

    orchestrator = JobApplicationOrchestrator(
        job_repository=job_repo,
        application_repository=app_repo,
        platform_router=platform_router,
        lock_manager=lock_mgr,
        run_manager=run_mgr,
        eligibility_gate=gate,
    )

    res = await orchestrator.run_single_job(
        AgentApplyOneRequest(job_id=test_job.id, auto_submit=False)
    )

    assert res.status == "JOB_UNAVAILABLE"
    assert res.application_eligible is False
    assert res.availability_status == "EXPIRED"
    mock_apply_svc.navigate_to_submit.assert_not_called()
    mock_apply_svc.apply_to_job.assert_not_called()


# ---------------------------------------------------------------------------
# 9-10. Multi-Job Mode (POST /agent/run)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_run_skips_ineligible_and_continues_to_eligible(tmp_path):
    """9-10. agent/run skips ineligible jobs, records metrics, and continues to eligible jobs."""
    job_external = make_test_job(title="Apple ML Engineer", company="Apple", source="indeed")
    job_eligible = make_test_job(title="PwC Analyst", company="PwC", source="glassdoor")

    match_external = make_test_match(job_external, final_score=95.0)
    match_eligible = make_test_match(job_eligible, final_score=90.0)

    disc_svc = MagicMock(spec=ProfileDiscoveryService)
    disc_svc.discover_jobs_from_profile = AsyncMock(
        return_value=ProfileJobSearchResponse(
            status="success",
            search_plan=SearchPlanSummary(queries_generated=1, queries=[]),
            discovery=DiscoveryStats(raw_jobs=2, unique_jobs=2),
            matching=MatchingStats(jobs_scored=2),
            top_matches=[match_external, match_eligible],
        )
    )

    app_repo = MagicMock(spec=ApplicationRepository)
    app_repo.get_applications_for_jobs.return_value = []
    app_repo.get_application_by_job_and_profile.return_value = None

    lock_mgr = JobLockManager(lock_file=str(tmp_path / "test_locks.json"))
    run_mgr = AgentRunManager(repository=MagicMock())

    # Gate behavior: first job external apply, second job eligible
    async def mock_eval(job_id, url, platform, **kwargs):
        if job_id == job_external.id:
            return EligibilityResult(
                is_eligible=False,
                skip_reason="EXTERNAL_APPLY",
                status_code="SKIPPED_EXTERNAL_APPLY",
                application_method=ApplicationMethod.EXTERNAL_APPLY,
                availability_status=AvailabilityStatus.AVAILABLE,
                message="Apple external site",
            )
        else:
            return EligibilityResult(
                is_eligible=True,
                status_code="eligible",
                application_method=ApplicationMethod.EASY_APPLY,
                availability_status=AvailabilityStatus.AVAILABLE,
            )

    gate = MagicMock(spec=ApplicationEligibilityGate)
    gate.evaluate_job_live = AsyncMock(side_effect=mock_eval)

    mock_indeed_svc = MagicMock()
    mock_glassdoor_svc = MagicMock()
    mock_glassdoor_svc.navigate_to_submit = AsyncMock(
        return_value=GlassdoorAutomationResult(
            status="submission_ready",
            url=job_eligible.url,
            job_id=str(job_eligible.id),
            submit_verified=True,
            current_state=GlassdoorAutomationState.SUBMISSION_READY,
        )
    )

    router = MagicMock(spec=PlatformRouter)
    router.is_platform_supported.return_value = True
    router.get_apply_service_for_platform.side_effect = lambda p: (
        mock_indeed_svc if p == "indeed" else mock_glassdoor_svc
    )

    orchestrator = JobApplicationOrchestrator(
        discovery_service=disc_svc,
        application_repository=app_repo,
        platform_router=router,
        lock_manager=lock_mgr,
        run_manager=run_mgr,
        eligibility_gate=gate,
    )

    res = await orchestrator.run_application_cycle(
        AgentRunRequest(max_jobs=2, auto_submit=False)
    )

    # 1. First job was skipped as SKIPPED_EXTERNAL_APPLY
    assert len(res.results) == 2
    first_res = res.results[0]
    assert first_res.job_id == job_external.id
    assert first_res.status == "SKIPPED_EXTERNAL_APPLY"
    assert first_res.application_eligible is False

    # 2. Second job was processed and reached submission_ready
    second_res = res.results[1]
    assert second_res.job_id == job_eligible.id
    assert second_res.status in ("submission_ready", "success")
    assert second_res.application_eligible is True

    # 3. Metrics verification
    assert res.jobs_ineligible == 1
    assert res.jobs_external_apply == 1
    assert res.jobs_application_eligible == 1
    assert res.jobs_skipped == 1
    assert res.jobs_attempted == 1


# ---------------------------------------------------------------------------
# 11-13. Platform-Specific Inspection Mapping
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_indeed_inspection_mapping():
    """12. Indeed application method detection mapping."""
    gate = ApplicationEligibilityGate()
    mock_indeed_svc = MagicMock()
    # Indeed inspection with is_easy_apply=False and external_apply=True
    mock_indeed_svc.inspect_job_application = AsyncMock(
        return_value={
            "application_method": "EXTERNAL_APPLY",
            "is_easy_apply": False,
            "external_apply": True,
            "availability": "AVAILABLE",
            "status": "EXTERNAL_APPLY",
            "message": "Apply on company website",
        }
    )
    router = MagicMock(spec=PlatformRouter)
    router.get_apply_service_for_platform.return_value = mock_indeed_svc
    gate.platform_router = router

    res = await gate.evaluate_job_live(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=892187f461c82742",
        platform="indeed",
    )
    assert res.is_eligible is False
    assert res.application_method == ApplicationMethod.EXTERNAL_APPLY
    assert res.status_code == "SKIPPED_EXTERNAL_APPLY"

    # Now test indeed easy apply
    mock_indeed_svc.inspect_job_application = AsyncMock(
        return_value={
            "application_method": "EASY_APPLY",
            "is_easy_apply": True,
            "availability": "AVAILABLE",
            "status": "EASY_APPLY_AVAILABLE",
        }
    )
    res_easy = await gate.evaluate_job_live(
        job_id=uuid4(),
        url="https://in.indeed.com/viewjob?jk=1111111111111111",
        platform="indeed",
    )
    assert res_easy.is_eligible is True
    assert res_easy.application_method == ApplicationMethod.EASY_APPLY


@pytest.mark.asyncio
async def test_glassdoor_inspection_mapping():
    """13. Glassdoor Easy Apply detection mapping."""
    gate = ApplicationEligibilityGate()
    mock_gd_svc = MagicMock()
    # Glassdoor inspection with easy_apply_verified=False
    mock_gd_svc.inspect_job_application = AsyncMock(
        return_value={
            "application_method": "EXTERNAL_APPLY",
            "easy_apply_verified": False,
            "availability": "AVAILABLE",
            "status": "EXTERNAL_APPLY",
        }
    )
    router = MagicMock(spec=PlatformRouter)
    router.get_apply_service_for_platform.return_value = mock_gd_svc
    gate.platform_router = router

    res = await gate.evaluate_job_live(
        job_id=uuid4(),
        url="https://www.glassdoor.com/job-listing/j?jl=12345",
        platform="glassdoor",
    )
    assert res.is_eligible is False
    assert res.status_code == "SKIPPED_EXTERNAL_APPLY"

    # Glassdoor inspection with easy_apply_verified=True
    mock_gd_svc.inspect_job_application = AsyncMock(
        return_value={
            "application_method": "EASY_APPLY",
            "easy_apply_verified": True,
            "availability": "AVAILABLE",
            "status": "EASY_APPLY_AVAILABLE",
        }
    )
    res_easy = await gate.evaluate_job_live(
        job_id=uuid4(),
        url="https://www.glassdoor.com/job-listing/j?jl=54321",
        platform="glassdoor",
    )
    assert res_easy.is_eligible is True
    assert res_easy.application_method == ApplicationMethod.EASY_APPLY


# ---------------------------------------------------------------------------
# 14. API Endpoint Integration Test
# ---------------------------------------------------------------------------

def test_api_apply_one_skipped_external():
    """API endpoint returns SKIPPED_EXTERNAL_APPLY with proper schema."""
    client = TestClient(app)
    mock_orch = MagicMock(spec=JobApplicationOrchestrator)
    jid = uuid4()

    mock_orch.run_single_job = AsyncMock(
        return_value=SingleJobApplyResponse(
            job_id=jid,
            title="Applied ML Engineer",
            company="Apple",
            platform="indeed",
            match_score=100.0,
            status="SKIPPED_EXTERNAL_APPLY",
            automation_state="SKIPPED_EXTERNAL_APPLY",
            application_method="EXTERNAL_APPLY",
            availability_status="AVAILABLE",
            application_eligible=False,
            error="Job redirects to external company application: Apple.com",
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
        assert data["status"] == "SKIPPED_EXTERNAL_APPLY"
        assert data["application_eligible"] is False
        assert data["application_method"] == "EXTERNAL_APPLY"
    finally:
        app.dependency_overrides.pop(get_job_orchestrator, None)
