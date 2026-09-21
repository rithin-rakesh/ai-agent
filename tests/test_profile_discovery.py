import asyncio
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.config.settings import Settings
from app.discovery.jobspy_client import SourceResult
from app.discovery.profile_discovery_service import ProfileDiscoveryService
from app.discovery.search_planner import ProfileSearchPlanner
from app.matching.service import MatchService
from app.models.job import (
    Job,
    JobCreate,
    MatchedJobItem,
    ProfileJobSearchRequest,
    ProfileJobSearchResponse,
    SearchPlan,
    SearchQuery,
)
from app.models.match import MatchResult
from app.models.profile import (
    CandidateProfileData,
    CareerPreferences,
    EducationItem,
    ExperienceItem,
    PersonalDetails,
    SkillItem,
)


@pytest.fixture
def sample_profile_data() -> CandidateProfileData:
    """Fixture providing rich candidate profile data."""
    return CandidateProfileData(
        personal=PersonalDetails(
            name="Alice Cybersecurity",
            email="alice@example.com",
            phone="+91 9999999999",
            location="Kochi, Kerala",
        ),
        career=CareerPreferences(
            experience_years=3.5,
            preferred_roles=[
                "Cybersecurity Analyst",
                "Security Engineer",
                "SOC Analyst",
            ],
            preferred_locations=[
                "Kochi, Kerala",
                "Bangalore, Karnataka",
                "Remote",
            ],
            remote_preference="any",
            job_type="full-time",
            salary_min=800000,
            salary_max=1800000,
        ),
        skills=[
            SkillItem(skill="Cybersecurity", category="Security", proficiency="expert", years_experience=3.5, importance="critical", weight=1.5),
            SkillItem(skill="SIEM", category="Security", proficiency="advanced", years_experience=3.0, importance="high", weight=1.3),
            SkillItem(skill="Vulnerability Assessment", category="Security", proficiency="advanced", years_experience=2.5, importance="high", weight=1.2),
            SkillItem(skill="Python", category="Programming", proficiency="intermediate", years_experience=2.0, importance="medium", weight=1.0),
            SkillItem(skill="Linux", category="Tools", proficiency="advanced", years_experience=3.0, importance="medium", weight=1.0),
        ],
        experience=[
            ExperienceItem(
                company="Cyber Defense Corp",
                title="SOC Analyst",
                start_date="2022-01-01",
                end_date=None,
                responsibilities="Threat monitoring, vulnerability scanning, and incident response.",
            )
        ],
        education=[
            EducationItem(
                degree="B.Tech",
                field="Computer Science and Engineering",
                institution="Tech University",
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1. Search Planner Unit Tests
# ---------------------------------------------------------------------------


def test_planner_generates_queries_from_preferred_roles(sample_profile_data):
    """Verify preferred roles in candidate profile produce prioritized search queries."""
    planner = ProfileSearchPlanner()
    plan = planner.create_search_plan(sample_profile_data, max_queries=10, site="indeed")

    assert plan.query_count > 0
    assert len(plan.queries) <= 10

    terms = [q.search_term for q in plan.queries]
    assert "Cybersecurity Analyst" in terms
    assert "Security Engineer" in terms
    assert "SOC Analyst" in terms


def test_planner_role_and_domain_breadth(sample_profile_data):
    """Verify planner derives broader domain terms (e.g. Vulnerability Assessment, Cloud Security)."""
    planner = ProfileSearchPlanner()
    plan = planner.create_search_plan(sample_profile_data, max_queries=15, site="indeed")

    terms = [q.search_term.lower() for q in plan.queries]
    # Check domain expansions
    assert any("vulnerability" in t or "information security" in t or "cloud security" in t for t in terms)


def test_planner_no_duplicate_queries(sample_profile_data):
    """Verify near-duplicate queries are normalized and removed."""
    # Add duplicate-like preferred roles
    sample_profile_data.career.preferred_roles.extend(["Cyber Security Analyst", "cybersecurity analyst "])
    planner = ProfileSearchPlanner()
    plan = planner.create_search_plan(sample_profile_data, max_queries=20, site="indeed")

    normalized_keys = set()
    for q in plan.queries:
        key = (q.search_term.lower().replace(" ", "").replace("-", ""), q.location.lower())
        assert key not in normalized_keys, f"Duplicate query found: {q}"
        normalized_keys.add(key)


def test_planner_max_queries_enforced(sample_profile_data):
    """Verify max_queries upper bound is strictly respected."""
    planner = ProfileSearchPlanner()
    plan = planner.create_search_plan(sample_profile_data, max_queries=4, site="indeed")

    assert plan.query_count == 4
    assert len(plan.queries) == 4


def test_planner_no_personal_credentials_or_pii_in_queries(sample_profile_data):
    """Verify generated search terms contain zero candidate PII (name, email, phone)."""
    planner = ProfileSearchPlanner()
    plan = planner.create_search_plan(sample_profile_data, max_queries=15, site="indeed")

    for q in plan.queries:
        assert sample_profile_data.personal.name.lower() not in q.search_term.lower()
        assert sample_profile_data.personal.email.lower() not in q.search_term.lower()
        assert "alice" not in q.search_term.lower()
        assert "9999999999" not in q.search_term


def test_planner_standalone_stopwords_excluded(sample_profile_data):
    """Verify standalone tools/stopwords like 'Linux' or 'Git' alone are not searched without role context."""
    planner = ProfileSearchPlanner()
    plan = planner.create_search_plan(sample_profile_data, max_queries=15, site="indeed")

    terms = [q.search_term.lower() for q in plan.queries]
    assert "linux" not in terms
    assert "git" not in terms
    assert "docker" not in terms


def test_planner_nvidia_expansion_fallback_on_failure(sample_profile_data):
    """Verify NVIDIA query expansion gracefully falls back to deterministic planning on error."""
    mock_nvidia_client = MagicMock()
    mock_nvidia_client.post.side_effect = Exception("NVIDIA network error")

    planner = ProfileSearchPlanner(nvidia_client=mock_nvidia_client)
    plan = planner.create_search_plan(sample_profile_data, max_queries=10, site="indeed", use_nvidia=True)

    # Must still produce valid deterministic plan without crashing
    assert plan.query_count > 0
    assert len(plan.queries) > 0


# ---------------------------------------------------------------------------
# 2. Profile Discovery Service Unit & Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discover_jobs_from_profile_multi_query_execution(sample_profile_data):
    """Verify discovery executes multiple JobSpy queries, aggregates results, and deduplicates."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    # Mock JobSpy client returning jobs across queries
    mock_jobspy = AsyncMock()

    raw_job_1 = {
        "id": "ind-101",
        "site": "indeed",
        "title": "SOC Analyst L1",
        "company": "SecureNet India",
        "location": "Kochi, Kerala",
        "job_url": "https://www.indeed.com/viewjob?jk=101",
        "description": "Looking for SOC Analyst with SIEM and Vulnerability Assessment experience.",
    }
    raw_job_2 = {
        "id": "ind-102",
        "site": "indeed",
        "title": "Cybersecurity Engineer",
        "company": "Infosec Solutions",
        "location": "Bangalore, Karnataka",
        "job_url": "https://www.indeed.com/viewjob?jk=102",
        "description": "Information Security Engineer role.",
    }
    # Duplicate of job 1 returned under a second query
    raw_job_1_dup = dict(raw_job_1)

    async def search_single_source_mock(source, search_term, location, **kwargs):
        if "soc" in search_term.lower():
            return SourceResult(
                source=source,
                status="success",
                jobs_returned=2,
                raw_jobs=[raw_job_1, raw_job_2],
            )
        elif "cybersecurity" in search_term.lower():
            return SourceResult(
                source=source,
                status="success",
                jobs_returned=1,
                raw_jobs=[raw_job_1_dup],  # duplicate
            )
        return SourceResult(source=source, status="success", jobs_returned=0, raw_jobs=[])

    mock_jobspy.search_single_source = AsyncMock(side_effect=search_single_source_mock)

    # Mock Job repository
    mock_job_repo = MagicMock()
    job_uuid_1 = uuid4()
    job_uuid_2 = uuid4()

    stored_job_1 = Job(
        id=job_uuid_1,
        source="indeed",
        external_id="ind-101",
        title="SOC Analyst L1",
        company="SecureNet India",
        location="Kochi, Kerala",
        url="https://www.indeed.com/viewjob?jk=101",
        created_at=datetime.now(timezone.utc),
    )
    stored_job_2 = Job(
        id=job_uuid_2,
        source="indeed",
        external_id="ind-102",
        title="Cybersecurity Engineer",
        company="Infosec Solutions",
        location="Bangalore, Karnataka",
        url="https://www.indeed.com/viewjob?jk=102",
        created_at=datetime.now(timezone.utc),
    )

    mock_job_repo.upsert_jobs.return_value = ([stored_job_1, stored_job_2], 2, 0)

    from app.models.match import MatchBreakdown
    mock_match_svc = MagicMock()
    mock_match_results = [
        MatchResult(
            job_id=job_uuid_1,
            profile_id=uuid4(),
            breakdown=MatchBreakdown(
                skills=85.0,
                title=90.0,
                experience=80.0,
                location=90.0,
                salary=80.0,
            ),
            final_score=86.5,
            decision="excellent",
            reasons=["Strong skill alignment: Cybersecurity, SIEM"],
        ),
        MatchResult(
            job_id=job_uuid_2,
            profile_id=uuid4(),
            breakdown=MatchBreakdown(
                skills=75.0,
                title=80.0,
                experience=75.0,
                location=70.0,
                salary=80.0,
            ),
            final_score=76.0,
            decision="strong_match",
            reasons=["Good role alignment: Security Engineer"],
        ),
    ]
    mock_match_svc.match_discovered_jobs.return_value = mock_match_results
    mock_match_svc.match_job.side_effect = mock_match_results

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        jobspy_client=mock_jobspy,
        job_repository=mock_job_repo,
        match_service=mock_match_svc,
    )

    request = ProfileJobSearchRequest(
        sites=["indeed"],
        results_per_query=20,
        hours_old=72,
        country_indeed="India",
        max_queries=5,
        run_matching=True,
        use_semantic=False,
        top_k=10,
        minimum_score=55.0,
    )

    response = await service.discover_jobs_from_profile(request)

    # Verify Response Contract
    assert response.status == "success"
    assert response.search_plan.queries_generated > 0
    assert response.discovery.raw_jobs == 4
    assert response.discovery.unique_jobs == 2
    assert response.discovery.duplicates_removed == 2
    assert response.discovery.queries_succeeded > 0
    assert response.matching.jobs_scored == 2
    assert response.matching.excellent == 1
    assert response.matching.strong_match == 1

    # Verify Ranked Top Matches with original URLs
    assert len(response.top_matches) == 2
    assert response.top_matches[0].url == "https://www.indeed.com/viewjob?jk=101"
    assert response.top_matches[0].final_score >= response.top_matches[1].final_score
    assert response.top_matches[0].title == "SOC Analyst L1"
    assert response.top_matches[1].url == "https://www.indeed.com/viewjob?jk=102"


@pytest.mark.asyncio
async def test_single_query_failure_does_not_abort_entire_discovery_run(sample_profile_data):
    """Verify that if one JobSpy query fails or times out, the rest succeed and complete."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    mock_jobspy = AsyncMock()
    call_count = 0

    async def search_single_source_flaky(source, search_term, location, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First query fails
            return SourceResult(source=source, status="failed", error="JobSpy timeout", jobs_returned=0)
        # Subsequent queries succeed
        return SourceResult(
            source=source,
            status="success",
            jobs_returned=1,
            raw_jobs=[
                {
                    "id": f"ind-{call_count}",
                    "site": "indeed",
                    "title": f"Security Specialist {call_count}",
                    "company": "SecCorp",
                    "location": "Remote",
                    "job_url": f"https://www.indeed.com/viewjob?jk={call_count}",
                }
            ],
        )

    mock_jobspy.search_single_source = AsyncMock(side_effect=search_single_source_flaky)

    mock_job_repo = MagicMock()
    mock_job_repo.upsert_jobs.return_value = ([], 0, 0)

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        jobspy_client=mock_jobspy,
        job_repository=mock_job_repo,
    )

    request = ProfileJobSearchRequest(
        sites=["indeed"],
        max_queries=3,
        run_matching=False,
    )

    response = await service.discover_jobs_from_profile(request)

    assert response.status == "partial_success"
    assert response.discovery.queries_failed == 1
    assert response.discovery.queries_succeeded == 2


# ---------------------------------------------------------------------------
# 3. FastAPI Endpoint Integration Tests
# ---------------------------------------------------------------------------


def test_api_jobs_search_from_profile_endpoint():
    """Verify POST /jobs/search/from-profile returns valid schema and preserved URLs."""
    from app.api.routers.jobs import get_profile_discovery_service

    client = TestClient(app)
    mock_service = AsyncMock(spec=ProfileDiscoveryService)

    sample_uuid = uuid4()
    mock_service.discover_jobs_from_profile.return_value = ProfileJobSearchResponse(
        status="success",
        search_plan={
            "queries_generated": 3,
            "queries": [
                {
                    "search_term": "Machine Learning Engineer",
                    "location": "Kochi, Kerala",
                    "source": "indeed",
                    "reason": "Preferred role",
                    "priority": 1,
                }
            ],
        },
        discovery={
            "raw_jobs": 50,
            "unique_jobs": 35,
            "duplicates_removed": 15,
            "queries_succeeded": 3,
            "queries_failed": 0,
            "execution_time_ms": 1200,
        },
        matching={
            "jobs_scored": 35,
            "excellent": 5,
            "strong_match": 10,
            "review": 12,
            "weak_match": 5,
            "unqualified": 3,
        },
        top_matches=[
            MatchedJobItem(
                job_id=sample_uuid,
                title="Senior AI Engineer",
                company="DeepMind Solutions",
                location="Remote",
                source="indeed",
                url="https://www.indeed.com/viewjob?jk=ai999",
                deterministic_score=84.0,
                semantic_score=88.0,
                final_score=85.2,
                decision="excellent",
                reason="Excellent fit with PyTorch and Machine Learning",
            )
        ],
    )

    app.dependency_overrides[get_profile_discovery_service] = lambda: mock_service
    try:
        payload = {
            "sites": ["indeed"],
            "results_per_query": 20,
            "hours_old": 168,
            "country_indeed": "India",
            "max_queries": 5,
            "run_matching": True,
            "use_semantic": True,
            "top_k": 10,
            "minimum_score": 55.0,
        }
        response = client.post("/jobs/search/from-profile", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert data["search_plan"]["queries_generated"] == 3
        assert data["discovery"]["unique_jobs"] == 35
        assert len(data["top_matches"]) == 1
        assert data["top_matches"][0]["title"] == "Senior AI Engineer"
        assert data["top_matches"][0]["url"] == "https://www.indeed.com/viewjob?jk=ai999"
        assert data["top_matches"][0]["final_score"] == 85.2
    finally:
        app.dependency_overrides.pop(get_profile_discovery_service, None)


def test_existing_manual_jobs_search_endpoint_still_works():
    """Verify POST /jobs/search remains fully functional and backward compatible."""
    from app.api.routers.jobs import get_discovery_service
    from app.discovery.service import JobDiscoveryService
    from app.models.job import JobSearchResponse, SourceStatus

    client = TestClient(app)
    mock_discovery = AsyncMock(spec=JobDiscoveryService)
    mock_discovery.discover_jobs.return_value = JobSearchResponse(
        jobs_found=5,
        new_jobs=5,
        duplicates=0,
        sources={"indeed": SourceStatus(status="success", jobs_returned=5)},
        execution_time_ms=500,
        jobs=[],
    )

    app.dependency_overrides[get_discovery_service] = lambda: mock_discovery
    try:
        response = client.post(
            "/jobs/search",
            json={"sites": ["indeed"], "search_term": "Python", "location": "Remote"},
        )
        assert response.status_code == 200
        assert response.json()["jobs_found"] == 5
    finally:
        app.dependency_overrides.pop(get_discovery_service, None)


@pytest.mark.asyncio
async def test_top_matches_strictly_excludes_rejected_and_low_match_tiers(sample_profile_data):
    """Verify that jobs with decision 'reject' or 'low_match' are excluded from top_matches even if score >= minimum_score."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    mock_jobspy = AsyncMock()
    mock_jobspy.search_single_source.return_value = SourceResult(
        source="indeed",
        status="success",
        jobs_returned=3,
        raw_jobs=[
            {"id": "j1", "site": "indeed", "title": "Good AI Role", "company": "Co1", "location": "Kochi", "job_url": "https://indeed.com/1"},
            {"id": "j2", "site": "indeed", "title": "Low Match Role", "company": "Co2", "location": "Kochi", "job_url": "https://indeed.com/2"},
            {"id": "j3", "site": "indeed", "title": "Rejected Role", "company": "Co3", "location": "Kochi", "job_url": "https://indeed.com/3"},
        ],
    )

    job1 = Job(id=uuid4(), source="indeed", external_id="j1", title="Good AI Role", company="Co1", location="Kochi", url="https://indeed.com/1", created_at=datetime.now(timezone.utc))
    job2 = Job(id=uuid4(), source="indeed", external_id="j2", title="Low Match Role", company="Co2", location="Kochi", url="https://indeed.com/2", created_at=datetime.now(timezone.utc))
    job3 = Job(id=uuid4(), source="indeed", external_id="j3", title="Rejected Role", company="Co3", location="Kochi", url="https://indeed.com/3", created_at=datetime.now(timezone.utc))

    mock_job_repo = MagicMock()
    mock_job_repo.upsert_jobs.return_value = ([job1, job2, job3], 3, 0)

    from app.models.match import MatchBreakdown
    mock_match_svc = MagicMock()
    mock_match_svc.match_discovered_jobs.return_value = [
        MatchResult(job_id=job1.id, profile_id=uuid4(), breakdown=MatchBreakdown(skills=90, title=90), final_score=88.0, decision="excellent"),
        MatchResult(job_id=job2.id, profile_id=uuid4(), breakdown=MatchBreakdown(skills=60, title=60), final_score=62.0, decision="low_match"),
        MatchResult(job_id=job3.id, profile_id=uuid4(), breakdown=MatchBreakdown(skills=70, title=0), final_score=63.6, decision="reject"),
    ]

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        jobspy_client=mock_jobspy,
        job_repository=mock_job_repo,
        match_service=mock_match_svc,
    )

    request = ProfileJobSearchRequest(
        sites=["indeed"],
        max_queries=1,
        run_matching=True,
        minimum_score=55.0,
    )

    response = await service.discover_jobs_from_profile(request)

    # Top matches should ONLY include the 'excellent' tier job, NOT the low_match or rejected job
    assert len(response.top_matches) == 1
    assert response.top_matches[0].title == "Good AI Role"
    assert response.top_matches[0].decision == "excellent"
    assert response.matching.jobs_scored == 3
    assert response.matching.excellent == 1
    assert response.matching.low_match == 1
    assert response.matching.reject == 1


# ---------------------------------------------------------------------------
# 4. Phase 5.8 Provider-Level Execution & Diagnostics Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_indeed_single_query_success_returns_structured_job_list(sample_profile_data):
    """Area 1: Indeed single query success returns structured job list with valid canonical URLs."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    mock_jobspy = AsyncMock()
    mock_jobspy.search_single_source.return_value = SourceResult(
        source="indeed",
        status="success",
        jobs_returned=1,
        raw_jobs=[
            {
                "id": "ind-101",
                "site": "indeed",
                "title": "SOC Analyst",
                "company": "SecureNet",
                "location": "Kochi, Kerala",
                "job_url": "https://www.indeed.com/viewjob?jk=soc101",
            }
        ],
    )

    persisted_job = Job(
        id=uuid4(),
        source="indeed",
        external_id="ind-101",
        title="SOC Analyst",
        company="SecureNet",
        location="Kochi, Kerala",
        url="https://www.indeed.com/viewjob?jk=soc101",
        created_at=datetime.now(timezone.utc),
    )
    mock_job_repo = MagicMock()
    mock_job_repo.upsert_jobs.return_value = ([persisted_job], 1, 0)

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        jobspy_client=mock_jobspy,
        job_repository=mock_job_repo,
    )

    req = ProfileJobSearchRequest(sites=["indeed"], max_queries=1, run_matching=False)
    resp = await service.discover_jobs_from_profile(req)

    assert resp.status == "success"
    assert resp.discovery.queries_succeeded == 1
    assert resp.discovery.queries_failed == 0
    assert len(resp.jobs) == 1
    assert resp.jobs[0].title == "SOC Analyst"
    assert resp.jobs[0].url == "https://www.indeed.com/viewjob?jk=soc101"
    assert resp.jobs[0].id == persisted_job.id


@pytest.mark.asyncio
async def test_indeed_query_failure_logs_diagnostic_and_marks_query_failed(sample_profile_data):
    """Area 2: Indeed query failure logs diagnostic and marks query_failed."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    mock_jobspy = AsyncMock()
    mock_jobspy.search_single_source.return_value = SourceResult(
        source="indeed",
        status="failed",
        provider_status="BRIDGE_UNAVAILABLE",
        error="JobSpy bridge port 9423 unreachable",
        jobs_returned=0,
    )

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        jobspy_client=mock_jobspy,
        job_repository=MagicMock(),
    )

    req = ProfileJobSearchRequest(sites=["indeed"], max_queries=1, run_matching=False)
    resp = await service.discover_jobs_from_profile(req)

    assert resp.status == "failed"
    assert resp.discovery.queries_failed == 1
    assert resp.discovery.queries_succeeded == 0
    assert len(resp.discovery.query_results) == 1
    qr = resp.discovery.query_results[0]
    assert qr.provider == "jobspy"
    assert qr.status == "BRIDGE_UNAVAILABLE"
    assert "unreachable" in (qr.error or "")


@pytest.mark.asyncio
async def test_glassdoor_single_query_via_apify_success_returns_jobs(sample_profile_data):
    """Area 3: Glassdoor single query via Apify success returns jobs."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    mock_apify = AsyncMock()
    mock_apify.search.return_value = SourceResult(
        source="glassdoor",
        status="success",
        provider_status="SUCCESS",
        jobs_returned=1,
        raw_jobs=[
            {
                "id": "gd-202",
                "site": "glassdoor",
                "title": "Security Engineer",
                "company": "CloudGuard",
                "location": "Bengaluru, Karnataka",
                "job_url": "https://www.glassdoor.co.in/job-listing/sec-eng-jl.htm?jl=1008",
            }
        ],
    )

    persisted_job = Job(
        id=uuid4(),
        source="glassdoor",
        external_id="gd-202",
        title="Security Engineer",
        company="CloudGuard",
        location="Bangalore, Karnataka",
        url="https://www.glassdoor.co.in/job-listing/sec-eng-jl.htm?jl=1008",
        created_at=datetime.now(timezone.utc),
    )
    mock_job_repo = MagicMock()
    mock_job_repo.upsert_jobs.return_value = ([persisted_job], 1, 0)

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        apify_provider=mock_apify,
        job_repository=mock_job_repo,
    )

    req = ProfileJobSearchRequest(sites=["glassdoor"], max_queries=1, run_matching=False)
    resp = await service.discover_jobs_from_profile(req)

    assert resp.status == "success"
    assert resp.discovery.queries_succeeded == 1
    assert len(resp.jobs) == 1
    assert resp.jobs[0].source == "glassdoor"
    assert resp.jobs[0].url == "https://www.glassdoor.co.in/job-listing/sec-eng-jl.htm?jl=1008"


@pytest.mark.asyncio
async def test_glassdoor_query_via_apify_failure_logs_diagnostic(sample_profile_data):
    """Area 4: Glassdoor query via Apify failure logs diagnostic and marks query_failed."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    mock_apify = AsyncMock()
    mock_apify.search.return_value = SourceResult(
        source="glassdoor",
        status="failed",
        provider_status="APIFY_DAILY_LIMIT_REACHED",
        error="Daily run limit (5) reached",
        jobs_returned=0,
    )

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        apify_provider=mock_apify,
        job_repository=MagicMock(),
    )

    req = ProfileJobSearchRequest(sites=["glassdoor"], max_queries=1, run_matching=False)
    resp = await service.discover_jobs_from_profile(req)

    assert resp.status == "failed"
    assert resp.discovery.queries_failed == 1
    qr = resp.discovery.query_results[0]
    assert qr.provider == "apify"
    assert qr.status == "APIFY_DAILY_LIMIT_REACHED"
    assert "limit" in qr.error


@pytest.mark.asyncio
async def test_discovery_query_result_populated_with_all_fields(sample_profile_data):
    """Area 5: DiscoveryQueryResult populated with all required diagnostic fields."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    mock_jobspy = AsyncMock()
    mock_jobspy.search_single_source.return_value = SourceResult(
        source="indeed",
        status="success",
        provider_status="SUCCESS",
        jobs_returned=2,
        raw_jobs=[],
    )

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        jobspy_client=mock_jobspy,
        job_repository=MagicMock(),
    )

    req = ProfileJobSearchRequest(sites=["indeed"], max_queries=1, run_matching=False)
    resp = await service.discover_jobs_from_profile(req)

    assert len(resp.discovery.query_results) == 1
    qr = resp.discovery.query_results[0]
    assert qr.query_id.startswith("query_")
    assert qr.search_term
    assert qr.location
    assert qr.source == "indeed"
    assert qr.provider == "jobspy"
    assert qr.status == "SUCCESS"
    assert qr.jobs_returned == 2
    assert qr.duration_ms >= 0


@pytest.mark.asyncio
async def test_partial_failure_returns_partial_success_status(sample_profile_data):
    """Area 6 & 9: Partial failure returns status='partial_success' and runs matching."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    call_count = 0
    job_uuid = uuid4()

    async def search_mock(source, search_term, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return SourceResult(source=source, status="failed", provider_status="TIMEOUT", error="Timed out", jobs_returned=0)
        return SourceResult(
            source=source,
            status="success",
            provider_status="SUCCESS",
            jobs_returned=1,
            raw_jobs=[
                {
                    "id": "ind-part-1",
                    "site": "indeed",
                    "title": "SOC Analyst",
                    "company": "SecureNet",
                    "location": "Kochi, Kerala",
                    "job_url": "https://www.indeed.com/viewjob?jk=part1",
                }
            ],
        )

    mock_jobspy = AsyncMock()
    mock_jobspy.search_single_source = AsyncMock(side_effect=search_mock)

    persisted_job = Job(
        id=job_uuid,
        source="indeed",
        external_id="ind-part-1",
        title="SOC Analyst",
        company="SecureNet",
        location="Kochi, Kerala",
        url="https://www.indeed.com/viewjob?jk=part1",
        created_at=datetime.now(timezone.utc),
    )
    mock_job_repo = MagicMock()
    mock_job_repo.upsert_jobs.return_value = ([persisted_job], 1, 0)

    from app.models.match import MatchBreakdown
    mock_match = MagicMock()
    mock_match.match_discovered_jobs.return_value = [
        MatchResult(job_id=job_uuid, profile_id=uuid4(), breakdown=MatchBreakdown(skills=85, title=85), final_score=85.0, decision="strong_match")
    ]

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        jobspy_client=mock_jobspy,
        job_repository=mock_job_repo,
        match_service=mock_match,
    )

    req = ProfileJobSearchRequest(sites=["indeed"], max_queries=2, run_matching=True, minimum_score=50.0)
    resp = await service.discover_jobs_from_profile(req)

    assert resp.status == "partial_success"
    assert resp.discovery.queries_succeeded == 1
    assert resp.discovery.queries_failed == 1
    assert resp.matching is not None
    assert resp.matching.jobs_scored == 1
    assert len(resp.top_matches) == 1


@pytest.mark.asyncio
async def test_all_queries_fail_returns_failed_status(sample_profile_data):
    """Area 7 & 10: All queries fail returns status='failed' and matching does not run."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    mock_jobspy = AsyncMock()
    mock_jobspy.search_single_source.return_value = SourceResult(
        source="indeed",
        status="failed",
        provider_status="FAILED",
        error="Network error",
        jobs_returned=0,
    )

    mock_match = MagicMock()

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        jobspy_client=mock_jobspy,
        job_repository=MagicMock(),
        match_service=mock_match,
    )

    req = ProfileJobSearchRequest(sites=["indeed"], max_queries=2, run_matching=True)
    resp = await service.discover_jobs_from_profile(req)

    assert resp.status == "failed"
    assert resp.discovery.queries_succeeded == 0
    assert resp.discovery.queries_failed == 2
    assert resp.discovery.raw_jobs == 0
    assert resp.matching is None
    assert resp.top_matches == []
    mock_match.match_discovered_jobs.assert_not_called()


@pytest.mark.asyncio
async def test_all_queries_succeed_returns_success_status(sample_profile_data):
    """Area 8: All queries succeed returns status='success'."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    mock_jobspy = AsyncMock()
    mock_jobspy.search_single_source.return_value = SourceResult(
        source="indeed",
        status="success",
        provider_status="SUCCESS",
        jobs_returned=1,
        raw_jobs=[
            {
                "id": "ind-all-1",
                "site": "indeed",
                "title": "Engineer",
                "company": "Co",
                "location": "Remote",
                "job_url": "https://www.indeed.com/viewjob?jk=all1",
            }
        ],
    )

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        jobspy_client=mock_jobspy,
        job_repository=MagicMock(upsert_jobs=MagicMock(return_value=([], 0, 0))),
    )

    req = ProfileJobSearchRequest(sites=["indeed"], max_queries=2, run_matching=False)
    resp = await service.discover_jobs_from_profile(req)

    assert resp.status == "success"
    assert resp.discovery.queries_succeeded == 2
    assert resp.discovery.queries_failed == 0


def test_location_normalization_remote_and_cities():
    """Area 11, 12, 13, 14: Remote mapping and city normalization for Bangalore, Chennai, Kerala."""
    from app.discovery.india_validator import normalize_query_location

    loc, is_rem = normalize_query_location("Remote", provider="indeed")
    assert is_rem is True
    assert loc == ""

    loc_gd, is_rem_gd = normalize_query_location("Remote", provider="glassdoor")
    assert is_rem_gd is True
    assert loc_gd == "India"

    loc, is_rem = normalize_query_location("bangalore", provider="indeed")
    assert loc == "Bangalore, India"
    assert is_rem is False

    loc, is_rem = normalize_query_location("chennai", provider="indeed")
    assert loc == "Chennai, India"
    assert is_rem is False

    loc, is_rem = normalize_query_location("kochi", provider="indeed")
    assert loc == "Kochi, Kerala, India"

    loc, is_rem = normalize_query_location("trivandrum", provider="indeed")
    assert loc == "Trivandrum, Kerala, India"

    loc, is_rem = normalize_query_location("kozhikode", provider="indeed")
    assert loc == "Kozhikode, Kerala, India"


@pytest.mark.asyncio
async def test_glassdoor_query_restricted_to_india(sample_profile_data):
    """Area 15: Glassdoor query enforces India search and passes country='India'."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    mock_apify = AsyncMock()
    mock_apify.search.return_value = SourceResult(
        source="glassdoor",
        status="success",
        provider_status="SUCCESS",
        jobs_returned=1,
        raw_jobs=[
            {
                "id": "gd-in-1",
                "site": "glassdoor",
                "title": "Engineer",
                "company": "Co",
                "location": "Mumbai, Maharashtra",
                "job_url": "https://www.glassdoor.co.in/job-listing/test.htm?jl=999",
            }
        ],
    )

    mock_job_repo = MagicMock(upsert_jobs=MagicMock(return_value=([], 0, 0)))

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        apify_provider=mock_apify,
        job_repository=mock_job_repo,
    )

    req = ProfileJobSearchRequest(sites=["glassdoor"], max_queries=1, run_matching=False)
    await service.discover_jobs_from_profile(req)

    assert mock_apify.search.call_count == 1
    call_kwargs = mock_apify.search.call_args.kwargs
    assert call_kwargs.get("country") == "India"


@pytest.mark.asyncio
async def test_glassdoor_concurrency_semaphore_limits_runs(sample_profile_data):
    """Area 16: Glassdoor concurrency semaphore restricts concurrent Apify executions to 1."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    concurrent_runs = 0
    max_concurrent_observed = 0

    async def mock_apify_search(**kwargs):
        nonlocal concurrent_runs, max_concurrent_observed
        concurrent_runs += 1
        if concurrent_runs > max_concurrent_observed:
            max_concurrent_observed = concurrent_runs
        await asyncio.sleep(0.05)
        concurrent_runs -= 1
        return SourceResult(source="glassdoor", status="success", jobs_returned=0, raw_jobs=[])

    mock_apify = AsyncMock()
    mock_apify.search = AsyncMock(side_effect=mock_apify_search)

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        apify_provider=mock_apify,
        job_repository=MagicMock(upsert_jobs=MagicMock(return_value=([], 0, 0))),
    )

    req = ProfileJobSearchRequest(sites=["glassdoor"], max_queries=3, run_matching=False)
    await service.discover_jobs_from_profile(req)

    assert max_concurrent_observed == 1


@pytest.mark.asyncio
async def test_apify_budget_guard_blocks_without_crashing_discovery(sample_profile_data):
    """Area 17: Apify budget guard blocks run without unhandled exception."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    mock_apify = AsyncMock()
    mock_apify.search.return_value = SourceResult(
        source="glassdoor",
        status="failed",
        provider_status="APIFY_DAILY_LIMIT_REACHED",
        error="Daily runs limit reached",
        jobs_returned=0,
    )

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        apify_provider=mock_apify,
        job_repository=MagicMock(),
    )

    req = ProfileJobSearchRequest(sites=["glassdoor"], max_queries=1, run_matching=False)
    resp = await service.discover_jobs_from_profile(req)

    assert resp.status == "failed"
    assert resp.discovery.queries_failed == 1
    assert resp.discovery.query_results[0].status == "APIFY_DAILY_LIMIT_REACHED"


@pytest.mark.asyncio
async def test_strict_provider_routing_isolation(sample_profile_data):
    """Area 18 & 19: Glassdoor NEVER routes to JobSpy; Indeed NEVER routes to Apify."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    mock_jobspy = AsyncMock()
    mock_jobspy.search_single_source.return_value = SourceResult(source="indeed", status="success", jobs_returned=0, raw_jobs=[])

    mock_apify = AsyncMock()
    mock_apify.search.return_value = SourceResult(source="glassdoor", status="success", jobs_returned=0, raw_jobs=[])

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        jobspy_client=mock_jobspy,
        apify_provider=mock_apify,
        job_repository=MagicMock(upsert_jobs=MagicMock(return_value=([], 0, 0))),
    )

    # 1. Glassdoor search only
    req_gd = ProfileJobSearchRequest(sites=["glassdoor"], max_queries=1, run_matching=False)
    await service.discover_jobs_from_profile(req_gd)

    mock_apify.search.assert_called_once()
    mock_jobspy.search_single_source.assert_not_called()

    mock_apify.reset_mock()
    mock_jobspy.reset_mock()

    # 2. Indeed search only
    req_ind = ProfileJobSearchRequest(sites=["indeed"], max_queries=1, run_matching=False)
    await service.discover_jobs_from_profile(req_ind)

    mock_jobspy.search_single_source.assert_called_once()
    mock_apify.search.assert_not_called()


@pytest.mark.asyncio
async def test_multiple_queries_execute_asynchronously(sample_profile_data):
    """Area 20: Multiple queries execute asynchronously via asyncio.gather."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    sleep_time = 0.1
    async def slow_search(**kwargs):
        await asyncio.sleep(sleep_time)
        return SourceResult(source="indeed", status="success", jobs_returned=0, raw_jobs=[])

    mock_jobspy = AsyncMock()
    mock_jobspy.search_single_source = AsyncMock(side_effect=slow_search)

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        jobspy_client=mock_jobspy,
        job_repository=MagicMock(upsert_jobs=MagicMock(return_value=([], 0, 0))),
    )

    req = ProfileJobSearchRequest(sites=["indeed"], max_queries=3, run_matching=False)
    t0 = time.perf_counter()
    await service.discover_jobs_from_profile(req)
    total_time = time.perf_counter() - t0

    # 3 concurrent queries taking 0.1s each should take much less than 3 * 0.1 = 0.3s
    assert total_time < 0.28


@pytest.mark.asyncio
async def test_cross_query_deduplication_merges_identical_jobs(sample_profile_data):
    """Area 21: Cross-query deduplication merges identical jobs from multiple queries."""
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    mock_jobspy = AsyncMock()
    mock_jobspy.search_single_source.return_value = SourceResult(
        source="indeed",
        status="success",
        jobs_returned=1,
        raw_jobs=[
            {
                "id": "ind-dup-1",
                "site": "indeed",
                "title": "Duplicate Role",
                "company": "Company A",
                "location": "Bangalore, Karnataka",
                "job_url": "https://www.indeed.com/viewjob?jk=dup1",
            }
        ],
    )

    persisted_job = Job(
        id=uuid4(),
        source="indeed",
        external_id="ind-dup-1",
        title="Duplicate Role",
        company="Company A",
        location="Bangalore, Karnataka",
        url="https://www.indeed.com/viewjob?jk=dup1",
        created_at=datetime.now(timezone.utc),
    )

    mock_job_repo = MagicMock()
    mock_job_repo.upsert_jobs.return_value = ([persisted_job], 1, 0)

    service = ProfileDiscoveryService(
        profile_service=mock_profile_svc,
        jobspy_client=mock_jobspy,
        job_repository=mock_job_repo,
    )

    req = ProfileJobSearchRequest(sites=["indeed"], max_queries=2, run_matching=False)
    resp = await service.discover_jobs_from_profile(req)

    assert resp.discovery.raw_jobs == 2
    assert resp.discovery.unique_jobs == 1
    assert resp.discovery.duplicates_removed == 1
    mock_job_repo.upsert_jobs.assert_called_once()
    assert len(mock_job_repo.upsert_jobs.call_args[0][0]) == 1


def test_canonical_job_url_validation_and_rejection():
    """Area 22: Validate canonical URL helper accepts valid job URLs and rejects bad URLs."""
    from app.discovery.profile_discovery_service import is_valid_canonical_job_url

    # Valid Indeed URLs
    assert is_valid_canonical_job_url("indeed", "https://www.indeed.com/viewjob?jk=1234567890abcdef")
    assert is_valid_canonical_job_url("indeed", "https://in.indeed.com/viewjob?jk=abcdef1234567890")
    assert is_valid_canonical_job_url("indeed", "https://www.indeed.com/job/software-engineer-1234")

    # Invalid Indeed URLs
    assert not is_valid_canonical_job_url("indeed", None)
    assert not is_valid_canonical_job_url("indeed", "")
    assert not is_valid_canonical_job_url("indeed", "https://indeed.com")
    assert not is_valid_canonical_job_url("indeed", "https://in.indeed.com/")
    assert not is_valid_canonical_job_url("indeed", "https://www.indeed.com/jobs?q=python")
    assert not is_valid_canonical_job_url("indeed", "https://www.indeed.com/q-python-jobs.html")
    assert not is_valid_canonical_job_url("indeed", "https://apify.com/dataset/123")

    # Valid Glassdoor URLs
    assert is_valid_canonical_job_url("glassdoor", "https://www.glassdoor.co.in/job-listing/engineer-jl.htm?jl=100800")
    assert is_valid_canonical_job_url("glassdoor", "https://www.glassdoor.com/Job/india-python-developer-jobs-SRCH_IL.0,5_IN115_KO6,22.htm")

    # Invalid Glassdoor URLs
    assert not is_valid_canonical_job_url("glassdoor", "https://glassdoor.com")
    assert not is_valid_canonical_job_url("glassdoor", "https://www.glassdoor.co.in")
    assert not is_valid_canonical_job_url("glassdoor", "https://api.apify.com/v2/datasets/xyz")
    assert not is_valid_canonical_job_url("glassdoor", "https://www.indeed.com/viewjob?jk=123")


@pytest.mark.asyncio
async def test_apify_token_redacted_in_query_result_diagnostics(sample_profile_data):
    """Test that Apify API tokens are redacted from query_results error strings."""
    mock_settings = Settings(APIFY_API_TOKEN="apify_api_SECRETTOKEN12345")

    mock_profile_svc = MagicMock()
    mock_profile_svc.get_active_profile_data.return_value = sample_profile_data

    mock_apify = AsyncMock()
    mock_apify.search.return_value = SourceResult(
        source="glassdoor",
        status="failed",
        provider_status="FAILED",
        error="Actor error with token apify_api_SECRETTOKEN12345 failed",
        jobs_returned=0,
    )

    service = ProfileDiscoveryService(
        settings=mock_settings,
        profile_service=mock_profile_svc,
        apify_provider=mock_apify,
        job_repository=MagicMock(),
    )

    req = ProfileJobSearchRequest(sites=["glassdoor"], max_queries=1, run_matching=False)
    resp = await service.discover_jobs_from_profile(req)

    qr = resp.discovery.query_results[0]
    assert "SECRETTOKEN12345" not in qr.error
    assert "[REDACTED]" in qr.error
