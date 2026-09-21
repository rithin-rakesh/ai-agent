"""Unit tests for Phase 2.2 Job Discovery module, normalizer, and resilience logic."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from fastapi.testclient import TestClient

from app.api.main import app
from app.config.settings import Settings
from app.database.repositories.job_repository import JobRepository
from app.discovery.jobspy_client import JobSpyClient, SourceResult, SUPPORTED_JOB_SOURCES
from app.discovery.normalizer import JobNormalizer, normalize_text_key, normalize_url
from app.discovery.service import JobDiscoveryService
from app.models.job import Job, JobCreate, JobSearchRequest, JobSearchResponse

client = TestClient(app)


# Sample Mock Job Payloads
MOCK_LINKEDIN_JOB = {
    "id": "li-12345",
    "site": "linkedin",
    "jobUrl": "https://www.linkedin.com/jobs/view/12345?refId=abc&trackingId=xyz&utm_source=feed",
    "title": "AI ML Engineer",
    "company": "Tech Corp",
    "location": "Kochi, Kerala, India",
    "datePosted": "2026-08-15T00:00:00.000Z",
    "jobType": "fulltime",
    "isRemote": False,
    "minAmount": 1200000.0,
    "maxAmount": 1800000.0,
    "description": "Develop cutting edge AI models with PyTorch and Transformers.",
}

MOCK_INDEED_JOB = {
    "id": "in-67890",
    "site": "indeed",
    "jobUrl": "https://in.indeed.com/viewjob?jk=67890&utm_medium=email",
    "title": "Machine Learning Specialist",
    "company": "Data Solutions Ltd",
    "location": "Kochi, Kerala",
    "datePosted": "2026-08-16T10:00:00.000Z",
    "jobType": "fulltime",
    "isRemote": True,
    "minAmount": None,
    "maxAmount": None,
    "description": "Looking for ML Specialist for computer vision workflows.",
}


class MockJobRepository:
    """Mock repository for unit testing without live Supabase connection."""

    def __init__(self) -> None:
        self.jobs_db: Dict[str, Job] = {}

    def upsert_job(self, job_in: JobCreate) -> tuple[Job, bool]:
        key = f"{job_in.source}:{job_in.external_id}"
        is_new = key not in self.jobs_db
        job = Job(
            id=uuid.uuid4(),
            source=job_in.source,
            external_id=job_in.external_id,
            title=job_in.title,
            company=job_in.company,
            location=job_in.location,
            description=job_in.description,
            url=job_in.url,
            salary_min=job_in.salary_min,
            salary_max=job_in.salary_max,
            experience_text=job_in.experience_text,
            job_type=job_in.job_type,
            remote=job_in.remote,
            posted_at=job_in.posted_at,
            easy_apply=job_in.easy_apply,
            raw_data=job_in.raw_data,
            created_at=datetime.now(timezone.utc),
        )
        self.jobs_db[key] = job
        return job, is_new

    def upsert_jobs(self, jobs: List[JobCreate]) -> tuple[List[Job], int, int]:
        saved_jobs = []
        new_cnt = 0
        dup_cnt = 0
        for j in jobs:
            saved, is_new = self.upsert_job(j)
            saved_jobs.append(saved)
            if is_new:
                new_cnt += 1
            else:
                dup_cnt += 1
        return saved_jobs, new_cnt, dup_cnt

    def list_jobs(self, limit: int = 50, offset: int = 0, source: str = None, search: str = None):
        res = list(self.jobs_db.values())
        if source:
            res = [j for j in res if j.source == source]
        return res[offset : offset + limit], len(res)

    def get_job_by_id(self, job_id: uuid.UUID):
        for j in self.jobs_db.values():
            if j.id == job_id:
                return j
        return None

    def get_job_stats(self):
        from app.models.job import JobStats
        return JobStats(
            total_jobs=len(self.jobs_db),
            sources={"linkedin": sum(1 for j in self.jobs_db.values() if j.source == "linkedin")},
            remote_jobs=sum(1 for j in self.jobs_db.values() if j.remote),
            easy_apply_jobs=0,
        )


# ============================================================================
# 1. Normalization & Canonicalization Unit Tests
# ============================================================================

def test_url_normalization():
    """Verify that tracking query parameters and fragments are stripped cleanly."""
    dirty_url = "https://www.linkedin.com/jobs/view/12345/?utm_source=google&utm_medium=cpc&refId=999#details"
    clean = normalize_url(dirty_url)
    assert clean == "https://www.linkedin.com/jobs/view/12345"

    indeed_dirty = "https://in.indeed.com/viewjob?jk=abc&trk=job_feed&utm_campaign=daily"
    clean_indeed = normalize_url(indeed_dirty)
    assert clean_indeed == "https://in.indeed.com/viewjob?jk=abc"


def test_job_normalizer_complete_mapping():
    """Verify that JobNormalizer maps all fields correctly and creates dedup keys."""
    job_create, dedup_keys = JobNormalizer.normalize(MOCK_LINKEDIN_JOB)

    assert job_create.source == "linkedin"
    assert job_create.external_id == "li-12345"
    assert job_create.title == "AI ML Engineer"
    assert job_create.company == "Tech Corp"
    assert job_create.location == "Kochi, Kerala, India"
    assert job_create.salary_min == 1200000.0
    assert job_create.salary_max == 1800000.0
    assert job_create.job_type == "fulltime"
    assert job_create.remote is False
    assert job_create.url == "https://www.linkedin.com/jobs/view/12345"
    assert job_create.posted_at is not None
    assert dedup_keys["source_external_id"] == "linkedin:li-12345"
    assert "tech corp" in dedup_keys["signature"]


def test_job_normalizer_handles_missing_fields_gracefully():
    """Verify that minimal or null-heavy payloads do not crash normalizer."""
    minimal_raw = {"site": "glassdoor"}
    job_create, dedup_keys = JobNormalizer.normalize(minimal_raw)

    assert job_create.source == "glassdoor"
    assert job_create.external_id.startswith("gl-")  # Deterministic hash fallback
    assert job_create.title == "Untitled Position"
    assert job_create.company == "Unknown Company"
    assert job_create.location is None
    assert job_create.salary_min is None
    assert job_create.salary_max is None
    assert job_create.remote is False
    assert job_create.raw_data == minimal_raw


# ============================================================================
# 2. Resilient Job Discovery & Multi-Source Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_all_four_sources_succeed():
    """Verify behavior when all 4 sources return successful jobs."""
    mock_client = AsyncMock(spec=JobSpyClient)
    mock_client.search_jobs.return_value = [
        SourceResult(source="linkedin", status="success", jobs_returned=1, raw_jobs=[MOCK_LINKEDIN_JOB]),
        SourceResult(source="indeed", status="success", jobs_returned=1, raw_jobs=[MOCK_INDEED_JOB]),
        SourceResult(source="naukri", status="success", jobs_returned=0, raw_jobs=[]),
        SourceResult(source="glassdoor", status="success", jobs_returned=0, raw_jobs=[]),
    ]

    mock_repo = MockJobRepository()
    service = JobDiscoveryService(jobspy_client=mock_client, job_repository=mock_repo)

    request = JobSearchRequest(
        sites=["linkedin", "indeed", "naukri", "glassdoor"],
        search_term="AI ML Engineer",
        location="Kochi, Kerala",
    )

    response = await service.discover_jobs(request)

    assert response.jobs_found == 2
    assert response.new_jobs == 2
    assert response.duplicates == 0
    assert response.sources["linkedin"].status == "success"
    assert response.sources["indeed"].status == "success"
    assert response.sources["naukri"].status == "success"
    assert response.sources["glassdoor"].status == "success"


@pytest.mark.asyncio
async def test_linkedin_succeeds_naukri_fails():
    """Verify that a failure in Naukri (e.g. 406 reCAPTCHA) does NOT fail the search or block LinkedIn results."""
    mock_client = AsyncMock(spec=JobSpyClient)
    mock_client.search_jobs.return_value = [
        SourceResult(source="linkedin", status="success", jobs_returned=1, raw_jobs=[MOCK_LINKEDIN_JOB]),
        SourceResult(
            source="naukri",
            status="failed",
            jobs_returned=0,
            error="HTTP 406: recaptcha required",
            duration_ms=450,
        ),
    ]

    mock_repo = MockJobRepository()
    service = JobDiscoveryService(jobspy_client=mock_client, job_repository=mock_repo)

    request = JobSearchRequest(
        sites=["linkedin", "naukri"],
        search_term="AI ML Engineer",
        location="Kochi, Kerala",
    )

    response = await service.discover_jobs(request)

    # LinkedIn succeeded and its job was processed
    assert response.jobs_found == 1
    assert response.new_jobs == 1
    assert response.sources["linkedin"].status == "success"
    assert response.sources["linkedin"].jobs_returned == 1

    # Naukri failure was recorded with its diagnostic message
    assert response.sources["naukri"].status == "failed"
    assert "recaptcha required" in response.sources["naukri"].error
    assert response.sources["naukri"].jobs_returned == 0


@pytest.mark.asyncio
async def test_linkedin_indeed_succeed_naukri_glassdoor_fail():
    """Verify the exact verified environment condition: LinkedIn+Indeed succeed while Naukri+Glassdoor fail."""
    mock_client = AsyncMock(spec=JobSpyClient)
    mock_client.search_jobs.return_value = [
        SourceResult(source="linkedin", status="success", jobs_returned=1, raw_jobs=[MOCK_LINKEDIN_JOB]),
        SourceResult(source="indeed", status="success", jobs_returned=1, raw_jobs=[MOCK_INDEED_JOB]),
        SourceResult(source="naukri", status="failed", jobs_returned=0, error="HTTP 406: recaptcha required"),
        SourceResult(source="glassdoor", status="failed", jobs_returned=0, error="HTTP 400: location not parsed"),
    ]

    mock_repo = MockJobRepository()
    service = JobDiscoveryService(jobspy_client=mock_client, job_repository=mock_repo)

    request = JobSearchRequest(
        sites=["linkedin", "indeed", "naukri", "glassdoor"],
        search_term="AI ML Engineer",
        location="Kochi, Kerala",
    )

    response = await service.discover_jobs(request)

    assert response.jobs_found == 2
    assert response.new_jobs == 2
    assert response.sources["linkedin"].status == "success"
    assert response.sources["indeed"].status == "success"
    assert response.sources["naukri"].status == "failed"
    assert response.sources["glassdoor"].status == "failed"
    assert "recaptcha" in response.sources["naukri"].error
    assert "location not parsed" in response.sources["glassdoor"].error


@pytest.mark.asyncio
async def test_deduplication_by_id_url_and_signature():
    """Verify that duplicate jobs (by external ID, canonical URL, and title/company signature) are filtered."""
    duplicate_by_url = dict(MOCK_LINKEDIN_JOB, id="li-99999", jobUrl="https://www.linkedin.com/jobs/view/12345?utm_campaign=share")
    duplicate_by_signature = {
        "id": "other-1",
        "site": "other",
        "title": "ai ml engineer",
        "company": "tech corp",
        "location": "kochi, kerala, india",
        "jobUrl": "https://example.com/unique-url",
    }

    mock_client = AsyncMock(spec=JobSpyClient)
    mock_client.search_jobs.return_value = [
        SourceResult(
            source="linkedin",
            status="success",
            jobs_returned=4,
            raw_jobs=[
                MOCK_LINKEDIN_JOB,
                MOCK_LINKEDIN_JOB,  # Exact duplicate ID
                duplicate_by_url,   # Duplicate canonical URL
                duplicate_by_signature,  # Duplicate title/company/location
            ],
        ),
    ]

    mock_repo = MockJobRepository()
    service = JobDiscoveryService(jobspy_client=mock_client, job_repository=mock_repo)

    request = JobSearchRequest(sites=["linkedin"], search_term="AI ML Engineer", location="Kochi, Kerala")
    response = await service.discover_jobs(request)

    # Only 1 unique job should survive deduplication
    assert response.jobs_found == 1
    assert response.new_jobs == 1
    assert response.duplicates == 3


# ============================================================================
# 3. FastAPI Endpoints Integration Unit Tests
# ============================================================================

def test_api_search_jobs_endpoint():
    """Verify POST /jobs/search with mock discovery service dependency override."""
    from app.api.routers.jobs import get_discovery_service

    mock_discovery = AsyncMock()
    mock_discovery.discover_jobs.return_value = JobSearchResponse(
        jobs_found=1,
        new_jobs=1,
        duplicates=0,
        sources={
            "linkedin": {"status": "success", "jobs_returned": 1, "error": None, "duration_ms": 100},
            "naukri": {"status": "failed", "jobs_returned": 0, "error": "HTTP 406: recaptcha required", "duration_ms": 50},
        },
        execution_time_ms=150,
        jobs=[],
    )

    app.dependency_overrides[get_discovery_service] = lambda: mock_discovery

    try:
        payload = {
            "sites": ["linkedin", "naukri"],
            "search_term": "AI ML Engineer",
            "location": "Kochi, Kerala",
            "results_wanted": 10,
            "hours_old": 72,
        }
        res = client.post("/jobs/search", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["jobs_found"] == 1
        assert data["sources"]["linkedin"]["status"] == "success"
        assert data["sources"]["naukri"]["status"] == "failed"
    finally:
        app.dependency_overrides.clear()


def test_api_list_jobs_and_stats():
    """Verify GET /jobs and GET /jobs/stats endpoints."""
    from app.api.routers.jobs import get_job_repository
    from app.models.job import JobStats

    mock_repo = MagicMock(spec=JobRepository)
    mock_repo.list_jobs.return_value = ([], 0)
    mock_repo.get_job_stats.return_value = JobStats(
        total_jobs=10,
        sources={"linkedin": 8, "indeed": 2},
        remote_jobs=3,
        easy_apply_jobs=1,
    )

    app.dependency_overrides[get_job_repository] = lambda: mock_repo

    try:
        res_list = client.get("/jobs?limit=10&offset=0")
        assert res_list.status_code == 200
        assert res_list.json()["total"] == 0

        res_stats = client.get("/jobs/stats")
        assert res_stats.status_code == 200
        assert res_stats.json()["total_jobs"] == 10
        assert res_stats.json()["sources"]["linkedin"] == 8
    finally:
        app.dependency_overrides.clear()
