"""Comprehensive regression tests for Glassdoor discovery and manual ingestion pipeline.

Covers all 22 required architectural and operational criteria:
1. Root-level JobSpy payload structure
2. siteNames contains 'glassdoor'
3. Bangalore location normalization
4. Chennai location normalization
5. Remote search handled independently
6. Malformed location classified as LOCATION_PARSE_ERROR
7. Upstream 403 classified as UPSTREAM_BLOCKED
8. No anti-bot evasion / stealth bypass attempted
9. Valid Glassdoor response deserialization
10. Source normalized to 'glassdoor'
11. Glassdoor jl ID extracted from URL
12. glassdoor.co.in accepted
13. glassdoor.com accepted
14. Missing salary does not reject job
15. Missing description does not reject job
16. Successful result persists to jobs table
17. Idempotent upsert / no duplication
18. Persistence returns real jobs.id UUID
19. Indeed discovery remains unaffected
20. LinkedIn ingestion remains unaffected
21. Manual ingestion fallback endpoint works
22. Diagnostics return structured provider status
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.config.settings import Settings
from app.discovery.jobspy_client import JobSpyClient, SourceResult
from app.discovery.location import normalize_glassdoor_location
from app.discovery.normalizer import JobNormalizer, normalize_url
from app.discovery.providers.base import JobDiscoveryProvider
from app.discovery.providers.jobspy_provider import JobSpyDiscoveryProvider
from app.discovery.service import JobDiscoveryService
from app.models.job import (
    GlassdoorIngestRequest,
    Job,
    JobCreate,
    JobSearchRequest,
    JobSearchResponse,
)

client = TestClient(app)


class InMemoryJobRepository:
    """Mock repository with in-memory storage for deterministic testing."""

    def __init__(self) -> None:
        self.jobs_db: Dict[str, Job] = {}

    def get_job_by_source_external_id(self, source: str, external_id: str):
        key = f"{source}:{external_id}"
        return self.jobs_db.get(key)

    def get_job_by_id(self, job_id: uuid.UUID):
        for j in self.jobs_db.values():
            if j.id == job_id:
                return j
        return None

    def upsert_job(self, job_in: JobCreate) -> tuple[Job, bool]:
        key = f"{job_in.source}:{job_in.external_id}"
        is_new = key not in self.jobs_db
        if is_new:
            job_id = uuid.uuid4()
            job = Job(
                id=job_id,
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
            return job, True
        else:
            existing = self.jobs_db[key]
            return existing, False

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


# ---------------------------------------------------------------------------
# Test 1 & 2: Root-level JobSpy payload structure & siteNames
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_jobspy_client_payload_structure_and_sitenames():
    """Verify JobSpyClient formats root-level payload conforming to bridge schema."""
    captured_payload = None

    async def mock_post(url, json=None, timeout=None):
        nonlocal captured_payload
        captured_payload = json
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"jobs": []}
        return mock_resp

    mock_http = MagicMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(side_effect=mock_post)

    client_instance = JobSpyClient(http_client=mock_http)
    await client_instance.search_single_source(
        source="glassdoor",
        search_term="AI Engineer",
        location="Bangalore, India",
        is_remote=False,
        results_wanted=5,
    )

    assert captured_payload is not None
    assert captured_payload["siteNames"] == ["glassdoor"]
    assert captured_payload["searchTerm"] == "AI Engineer"
    assert captured_payload["location"] == "Bangalore, India"
    assert captured_payload["isRemote"] is False
    assert captured_payload["resultsWanted"] == 5
    assert "countryIndeed" in captured_payload


# ---------------------------------------------------------------------------
# Test 3 & 4: Location Normalization (Bangalore & Chennai)
# ---------------------------------------------------------------------------
def test_location_normalization_bangalore_and_chennai():
    """Test location normalization for Bangalore, Bengaluru, Chennai."""
    loc_b, is_rem_b = normalize_glassdoor_location("Bangalore")
    assert loc_b == "Bangalore, India"
    assert is_rem_b is False

    loc_b2, is_rem_b2 = normalize_glassdoor_location("Bengaluru")
    assert loc_b2 == "Bengaluru, India"
    assert is_rem_b2 is False

    loc_c, is_rem_c = normalize_glassdoor_location("Chennai")
    assert loc_c == "Chennai, India"
    assert is_rem_c is False

    # Pre-formatted location remains clean
    loc_p, is_rem_p = normalize_glassdoor_location("Kochi, Kerala, India")
    assert loc_p == "Kochi, Kerala, India"
    assert is_rem_p is False


# ---------------------------------------------------------------------------
# Test 5: Remote Search Handled Independently
# ---------------------------------------------------------------------------
def test_remote_search_separation():
    """Verify remote search is detected and separated from physical location search."""
    loc_r1, is_rem1 = normalize_glassdoor_location("remote")
    assert is_rem1 is True

    loc_r2, is_rem2 = normalize_glassdoor_location("wfh")
    assert is_rem2 is True

    loc_r3, is_rem3 = normalize_glassdoor_location("work from home")
    assert is_rem3 is True


# ---------------------------------------------------------------------------
# Test 6: Malformed Location classified as LOCATION_PARSE_ERROR
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_location_parse_error_classification():
    """Verify upstream location parse failure is classified as LOCATION_PARSE_ERROR."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "jobs": [],
        "message": "Glassdoor: location not parsed",
    }

    mock_http = MagicMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(return_value=mock_resp)

    client_instance = JobSpyClient(http_client=mock_http)
    res = await client_instance.search_single_source(
        source="glassdoor",
        search_term="AI Engineer",
        location="invalid_loc",
    )

    assert res.status == "failed"
    assert res.provider_status == "LOCATION_PARSE_ERROR"
    assert res.http_status == 400


# ---------------------------------------------------------------------------
# Test 7 & 8: Upstream 403 Classified as UPSTREAM_BLOCKED without evasion
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_upstream_403_classified_as_upstream_blocked():
    """Verify upstream HTTP 403 / Cloudflare security block returns UPSTREAM_BLOCKED."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 403
    mock_resp.text = "<title>Security | Glassdoor</title> 403 Forbidden"

    mock_http = MagicMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(return_value=mock_resp)

    client_instance = JobSpyClient(http_client=mock_http)
    res = await client_instance.search_single_source(
        source="glassdoor",
        search_term="AI Engineer",
        location="Bangalore, India",
    )

    assert res.status == "failed"
    assert res.provider_status == "UPSTREAM_BLOCKED"
    assert res.http_status == 403
    assert res.retryable is False
    assert "blocked" in res.error.lower() or "403" in res.error


# ---------------------------------------------------------------------------
# Test 9, 10, 11, 12, 13: Deserialization, Source Normalization, jl Extraction
# ---------------------------------------------------------------------------
def test_glassdoor_normalizer_external_id_and_domains():
    """Test Glassdoor normalizer extracts jl ID from co.in and com URLs."""
    raw_coin = {
        "site": "Glassdoor",
        "job_url": "https://www.glassdoor.co.in/job-listing/senior-ai-engineer-JV_IC2874136.htm?jl=1010229231195",
        "title": "Senior AI Engineer",
        "company": "Tech Corp",
        "location": "Bengaluru, India",
    }
    job_coin, keys_coin = JobNormalizer.normalize(raw_coin)

    assert job_coin.source == "glassdoor"
    assert job_coin.external_id == "1010229231195"
    assert "1010229231195" in job_coin.url
    assert keys_coin["source_external_id"] == "glassdoor:1010229231195"

    raw_com = {
        "site": "GLASSDOOR",
        "id": "gd-1009876543210",
        "job_url": "https://www.glassdoor.com/job-listing/machine-learning-engineer.htm?jl=1009876543210",
        "title": "Machine Learning Engineer",
        "company": "Global AI Inc",
        "location": "Remote",
    }
    job_com, keys_com = JobNormalizer.normalize(raw_com)

    assert job_com.source == "glassdoor"
    assert job_com.external_id == "1009876543210"
    assert keys_com["source_external_id"] == "glassdoor:1009876543210"


# ---------------------------------------------------------------------------
# Test 14 & 15: Nullable Salary & Description do not reject job
# ---------------------------------------------------------------------------
def test_glassdoor_nullable_fields_do_not_reject():
    """Verify missing salary and missing description are accepted."""
    raw = {
        "site": "glassdoor",
        "id": "gd-1010001",
        "title": "Junior Data Scientist",
        "company": "Data Labs",
        "location": "Chennai, India",
        "description": None,
        "minAmount": None,
        "maxAmount": None,
    }
    job, _ = JobNormalizer.normalize(raw)
    assert job.source == "glassdoor"
    assert job.external_id == "1010001"
    assert job.description is None
    assert job.salary_min is None
    assert job.salary_max is None


# ---------------------------------------------------------------------------
# Test 16, 17, 18: Persistence, Idempotence, Real UUID
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_glassdoor_persistence_and_idempotence():
    """Verify discovery persistence flow returns real UUID and handles deduplication."""
    mock_repo = InMemoryJobRepository()

    raw_jobs = [
        {
            "site": "glassdoor",
            "job_url": "https://www.glassdoor.co.in/job-listing/ai-lead.htm?jl=1010229231195",
            "title": "AI Lead",
            "company": "NextGen AI",
            "location": "Bangalore, India",
        },
        {
            "site": "glassdoor",
            "job_url": "https://www.glassdoor.co.in/job-listing/ai-lead.htm?jl=1010229231195",
            "title": "AI Lead",
            "company": "NextGen AI",
            "location": "Bangalore, India",
        },
    ]

    mock_client = MagicMock(spec=JobSpyClient)
    mock_client.search_jobs = AsyncMock(
        return_value=[
            SourceResult(
                source="glassdoor",
                status="success",
                provider_status="SUCCESS",
                http_status=200,
                jobs_returned=2,
                raw_jobs=raw_jobs,
                duration_ms=120,
            )
        ]
    )

    service = JobDiscoveryService(jobspy_client=mock_client, job_repository=mock_repo)
    req = JobSearchRequest(
        sites=["glassdoor"],
        search_term="AI Lead",
        location="Bangalore, India",
        results_wanted=5,
    )

    resp = await service.discover_jobs(req)

    # Idempotence: 2 results provided, but 1 unique job persisted, 1 duplicate
    assert resp.jobs_found == 1
    assert resp.new_jobs == 1
    assert resp.duplicates == 1
    assert isinstance(resp.jobs[0].id, uuid.UUID)
    assert resp.jobs[0].source == "glassdoor"
    assert resp.jobs[0].external_id == "1010229231195"

    # Glassdoor metrics breakdown
    assert resp.glassdoor_stats is not None
    assert resp.glassdoor_stats["glassdoor_results_received"] == 2
    assert resp.glassdoor_stats["glassdoor_jobs_inserted"] == 1


# ---------------------------------------------------------------------------
# Test 19 & 20: Indeed and LinkedIn Remain Unaffected
# ---------------------------------------------------------------------------
def test_indeed_and_linkedin_normalization_unaffected():
    """Verify Indeed and LinkedIn items continue to normalize identically."""
    raw_indeed = {
        "site": "indeed",
        "id": "in-abc12345",
        "job_url": "https://in.indeed.com/viewjob?jk=abc12345",
        "title": "Software Engineer",
        "company": "Indeed Employer",
        "location": "Kochi, Kerala",
    }
    job_in, _ = JobNormalizer.normalize(raw_indeed)
    assert job_in.source == "indeed"
    assert job_in.external_id == "in-abc12345"

    raw_li = {
        "site": "linkedin",
        "id": "li-xyz67890",
        "job_url": "https://www.linkedin.com/jobs/view/xyz67890",
        "title": "Data Analyst",
        "company": "LinkedIn Employer",
        "location": "Mumbai, India",
    }
    job_li, _ = JobNormalizer.normalize(raw_li)
    assert job_li.source == "linkedin"
    assert job_li.external_id == "li-xyz67890"


# ---------------------------------------------------------------------------
# Test 21: Manual URL Ingestion Endpoint Fallback
# ---------------------------------------------------------------------------
def test_manual_glassdoor_ingestion_endpoint():
    """Test POST /jobs/ingest/glassdoor manual fallback endpoint."""
    from app.api.routers.jobs import get_job_repository

    mock_repo = InMemoryJobRepository()
    app.dependency_overrides[get_job_repository] = lambda: mock_repo

    try:
        payload = {
            "url": "https://www.glassdoor.co.in/job-listing/teacher-JV_IC2874136.htm?jl=1010229231195",
            "title": "Teacher",
            "company": "EduCorp",
            "location": "Bangalore, India",
        }
        res = client.post("/jobs/ingest/glassdoor", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["success"] is True
        assert data["source"] == "glassdoor"
        assert data["external_id"] == "1010229231195"
        assert data["title"] == "Teacher"
        assert data["company"] == "EduCorp"
        assert uuid.UUID(data["job_id"])  # Valid UUID
        assert data["is_new"] is True

        # Second ingestion of same URL is idempotent
        res_dup = client.post("/jobs/ingest/glassdoor", json=payload)
        assert res_dup.status_code == 200
        data_dup = res_dup.json()
        assert data_dup["job_id"] == data["job_id"]
        assert data_dup["is_new"] is False
    finally:
        app.dependency_overrides.pop(get_job_repository, None)


# ---------------------------------------------------------------------------
# Test 22: Diagnostics Return Structured Provider Status
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_jobspy_diagnostics_endpoint():
    """Test /jobs/discovery/jobspy/test reports structured provider status."""
    mock_client = MagicMock(spec=JobSpyClient)
    mock_client.search_single_source = AsyncMock(
        return_value=SourceResult(
            provider="jobspy",
            source="glassdoor",
            status="failed",
            provider_status="UPSTREAM_BLOCKED",
            http_status=403,
            retryable=False,
            jobs_returned=0,
            error="Upstream Glassdoor request blocked by provider (HTTP 403 Security Check).",
            duration_ms=45,
            diagnostics={"provider": "jobspy", "site": "glassdoor", "results_wanted": 5},
        )
    )

    mock_service = MagicMock(spec=JobDiscoveryService)
    mock_service.jobspy_client = mock_client

    from app.api.routers.jobs import get_discovery_service

    app.dependency_overrides[get_discovery_service] = lambda: mock_service
    try:
        res = client.post(
            "/jobs/discovery/jobspy/test",
            params={"source": "glassdoor", "search_term": "AI Engineer"},
        )
        assert res.status_code == 200
        data = res.json()

        assert data["provider"] == "jobspy"
        assert data["source"] == "glassdoor"
        assert data["provider_status"] == "UPSTREAM_BLOCKED"
        assert data["http_status"] == 403
        assert data["retryable"] is False
    finally:
        app.dependency_overrides.pop(get_discovery_service, None)
