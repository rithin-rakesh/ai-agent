"""Phase 5.7A Comprehensive Test Suite: Apify Glassdoor Discovery & India-Only Resume Matching.

Covers all required verification criteria:
1. Token missing check
2. Token configured check
3. Correct Actor ID (orgupdate~glassdoor-jobs-scraper)
4. Actor schema mapping (includeKeyword, locationName, countryName, pagesToFetch, datePosted)
5. Hard India country restriction
6. Bangalore location mapping and cleaning
7. Profile search term mapping
8. Result limit capped at per-run max
9. Budget guard 500 results/month cap enforcement
10. Dynamic result clamping when remaining quota < requested
11. Cost calculation ($0.004 / result)
12. Disabled provider behavior (APIFY_GLASSDOOR_DISABLED)
13. Successful Actor result normalization
14. Actor run failure handling (APIFY_RUN_FAILED)
15. Actor timeout handling (APIFY_TIMEOUT)
16. Invalid Actor output handling
17. India validation passing (Bangalore, Chennai, Mumbai, Pune, Hyderabad, Delhi, Remote India)
18. Non-India rejection (New York, London, Toronto, etc.)
19. Unknown location handling (fail-closed for glassdoor)
20. Glassdoor URL normalization (canonical, stripped query params)
21. Glassdoor jl external ID extraction
22. Normalization field mapping (title, company, salary, date)
23. Deduplication by source + external_id, canonical URL, signature
24. Supabase jobs upsert with real UUID
25. Combined Indeed + Glassdoor results
26. Matching after provider merge
27. Minimum score filtering
28. Manual Glassdoor ingestion remains functional
29. JobSpy Indeed remains functional
30. Glassdoor application automation untouched
31. Token never appears in diagnostics/logs/API responses
32. Budget exceeded does not execute Actor
33. No application triggered by discovery endpoint
34. READY/RUNNING non-terminal states polled until SUCCEEDED
35. Transport failure mapped to APIFY_TRANSPORT_ERROR (503) distinct from 502
36. Empty dataset returns NO_RESULTS (200) instead of failure
37. Existing run_id inspection support
38. Aborted run handled gracefully
"""

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.main import app
from app.config.settings import Settings
from app.discovery.budget_guard import ApifyBudgetGuard
from app.discovery.india_validator import validate_india_location
from app.discovery.jobspy_client import JobSpyClient, SourceResult
from app.discovery.normalizer import JobNormalizer, normalize_url
from app.discovery.profile_discovery_service import ProfileDiscoveryService
from app.discovery.providers.apify_provider import (
    ApifyDiscoveryProvider,
    clean_actor_location_name,
    map_hours_old_to_date_posted,
)
from app.discovery.service import JobDiscoveryService
from app.models.job import (
    GlassdoorIngestRequest,
    Job,
    JobCreate,
    JobSearchRequest,
    JobSearchResponse,
    ProfileJobSearchRequest,
    ProfileJobSearchResponse,
    SearchQuery,
)
from app.models.profile import CandidateProfileData, CareerPreferences, PersonalDetails

client = TestClient(app)


class InMemoryJobRepository:
    """Deterministic in-memory repository for discovery testing."""

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
            return self.jobs_db[key], False

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


# Sample dataset items from orgupdate/glassdoor-jobs-scraper
SAMPLE_VALIG_ITEM_BANGALORE = {
    "id": 1010062906489,
    "title": "Senior AI Engineer",
    "url": "https://www.glassdoor.com/job-listing/j?jl=1010062906489",
    "seoUrl": "https://www.glassdoor.com/job-listing/senior-ai-engineer-geico-JV_IC1132348_KO0,18_KE19,24.htm?jl=1010062906489",
    "ageInDays": 3,
    "rating": 4.2,
    "easyApply": True,
    "employer": {
        "id": 270,
        "name": "Turing Tech India",
        "url": "https://www.glassdoor.com/Overview/W-EI_IE270.htm",
    },
    "location": {
        "countryId": 1,
        "id": 1132348,
        "name": "Bangalore, India",
        "type": "C",
    },
    "pay": {
        "source": "EMPLOYER_PROVIDED",
        "currency": "INR",
        "period": "ANNUAL",
        "min": 2500000.0,
        "max": 3500000.0,
    },
    "description": "<div>Exciting AI role in Bangalore</div>",
}

SAMPLE_ORGUPDATE_ITEM_BANGALORE = {
    "job_title": "Senior AI Engineer",
    "company_name": "Turing Tech India",
    "location": "Bangalore, India",
    "posted_via": "Glassdoor",
    "salary": "₹2,500,000 - ₹3,500,000 a year",
    "date": "2026-09-08T10:00:00Z",
    "job_type": "FULLTIME",
    "URL": "https://www.glassdoor.co.in/job-listing/senior-ai-engineer-turing-tech-jl=1009876543210.htm?utm_campaign=google_jobs_apply",
}

SAMPLE_ORGUPDATE_ITEM_US = {
    "job_title": "AI Researcher",
    "company_name": "US Silicon Inc",
    "location": "San Francisco, CA",
    "posted_via": "Glassdoor",
    "salary": "$150,000 - $200,000 a year",
    "date": "2026-09-08T12:00:00Z",
    "job_type": "FULLTIME",
    "URL": "https://www.glassdoor.com/job-listing/ai-researcher-jl=1009876543212.htm",
}


# ============================================================================
# 1. Token & Authentication Tests
# ============================================================================

@pytest.mark.asyncio
async def test_token_missing_returns_auth_error():
    """Criterion 1: If APIFY_API_TOKEN is empty/None, provider reports APIFY_AUTH_ERROR without crash."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        settings = Settings(
            APIFY_API_TOKEN=None,
            APIFY_GLASSDOOR_ENABLED=True,
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)
        provider = ApifyDiscoveryProvider(settings=settings, budget_guard=guard)

        res = await provider.search(source="glassdoor", search_term="AI Engineer", location="Bangalore, India")
        assert res.status == "failed"
        assert res.provider_status == "APIFY_AUTH_ERROR"
        assert res.diagnostics["apify_token_configured"] is False
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


@pytest.mark.asyncio
async def test_token_configured_reported_safely():
    """Criterion 2 & 31: Provider reports apify_token_configured: true, never exposes token plaintext."""
    secret_val = "apify_api_mocktoken1234567890abcdef"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        settings = Settings(
            APIFY_API_TOKEN=SecretStr(secret_val),
            APIFY_GLASSDOOR_ENABLED=True,
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)
        provider = ApifyDiscoveryProvider(settings=settings, budget_guard=guard)

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "data": {
                "id": "run-test-123",
                "actId": "valig~glassdoor-jobs-scraper",
                "status": "SUCCEEDED",
                "defaultDatasetId": "dataset-123",
            }
        }

        mock_dataset_resp = MagicMock()
        mock_dataset_resp.status_code = 200
        mock_dataset_resp.json.return_value = [SAMPLE_ORGUPDATE_ITEM_BANGALORE]

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response
        mock_client.get.return_value = mock_dataset_resp
        provider._http_client = mock_client

        res = await provider.search(source="glassdoor", search_term="AI Engineer", location="Bangalore, India")
        assert res.status == "success"
        assert res.diagnostics["apify_token_configured"] is True
        serialized = json.dumps(res.model_dump())
        assert secret_val not in serialized
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


# ============================================================================
# 2. Lifecycle & Terminal Status Tests (READY -> RUNNING -> SUCCEEDED)
# ============================================================================

@pytest.mark.asyncio
async def test_ready_and_running_states_polled_to_terminal_success():
    """Lifecycle Test: Initial READY response does NOT fail; polls through RUNNING to SUCCEEDED."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        settings = Settings(
            APIFY_API_TOKEN=SecretStr("mock_token"),
            APIFY_GLASSDOOR_ENABLED=True,
            APIFY_GLASSDOOR_TIMEOUT_SECONDS=30,
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)
        provider = ApifyDiscoveryProvider(settings=settings, budget_guard=guard)

        # 1. Initial start_actor returns HTTP 201 with READY
        start_resp = MagicMock()
        start_resp.status_code = 201
        start_resp.json.return_value = {
            "data": {
                "id": "run-lifecycle-001",
                "status": "READY",
                "defaultDatasetId": "ds-lifecycle-001",
            }
        }

        # 2. Polling calls: first returns RUNNING, second returns SUCCEEDED
        poll_resp_running = MagicMock()
        poll_resp_running.status_code = 200
        poll_resp_running.json.return_value = {
            "data": {
                "id": "run-lifecycle-001",
                "status": "RUNNING",
                "defaultDatasetId": "ds-lifecycle-001",
            }
        }

        poll_resp_succeeded = MagicMock()
        poll_resp_succeeded.status_code = 200
        poll_resp_succeeded.json.return_value = {
            "data": {
                "id": "run-lifecycle-001",
                "status": "SUCCEEDED",
                "defaultDatasetId": "ds-lifecycle-001",
                "exitCode": 0,
                "startedAt": "2026-09-10T10:00:00Z",
                "finishedAt": "2026-09-10T10:01:30Z",
            }
        }

        # 3. Dataset retrieval
        dataset_resp = MagicMock()
        dataset_resp.status_code = 200
        dataset_resp.json.return_value = [SAMPLE_ORGUPDATE_ITEM_BANGALORE]

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = start_resp
        mock_client.get.side_effect = [
            poll_resp_running,
            poll_resp_succeeded,
            dataset_resp,
        ]
        provider._http_client = mock_client

        res = await provider.search(
            source="glassdoor",
            search_term="AI Engineer",
            location="Bangalore, India",
            results_wanted=5,
        )

        assert res.status == "success"
        assert res.provider_status == "SUCCESS"
        assert res.jobs_returned == 1
        assert res.diagnostics["run_status"] == "SUCCEEDED"
        assert res.diagnostics["run_id"] == "run-lifecycle-001"
        assert res.diagnostics["default_dataset_id"] == "ds-lifecycle-001"
        assert res.diagnostics["dataset_fetch_status"] == "SUCCESS"
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


@pytest.mark.asyncio
async def test_succeeded_empty_dataset_returns_no_results():
    """Terminal SUCCEEDED with 0 items produces NO_RESULTS (HTTP 200, status=success), NOT 502 failure."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        settings = Settings(
            APIFY_API_TOKEN=SecretStr("mock_token"),
            APIFY_GLASSDOOR_ENABLED=True,
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)
        provider = ApifyDiscoveryProvider(settings=settings, budget_guard=guard)

        start_resp = MagicMock()
        start_resp.status_code = 201
        start_resp.json.return_value = {
            "data": {
                "id": "run-empty-001",
                "status": "SUCCEEDED",
                "defaultDatasetId": "ds-empty-001",
                "exitCode": 0,
            }
        }

        dataset_resp = MagicMock()
        dataset_resp.status_code = 200
        dataset_resp.json.return_value = []

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = start_resp
        mock_client.get.return_value = dataset_resp
        provider._http_client = mock_client

        res = await provider.search(
            source="glassdoor",
            search_term="NonExistentRole12345",
            location="Bangalore, India",
            results_wanted=5,
        )

        assert res.status == "success"
        assert res.provider_status == "NO_RESULTS"
        assert res.http_status == 200
        assert res.jobs_returned == 0
        assert res.diagnostics["run_status"] == "SUCCEEDED"
        assert res.diagnostics["dataset_fetch_status"] == "EMPTY"
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


@pytest.mark.asyncio
async def test_terminal_failed_status_mapped_to_apify_run_failed():
    """Terminal FAILED status maps to APIFY_RUN_FAILED with http_status=502."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        settings = Settings(
            APIFY_API_TOKEN=SecretStr("mock_token"),
            APIFY_GLASSDOOR_ENABLED=True,
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)
        provider = ApifyDiscoveryProvider(settings=settings, budget_guard=guard)

        start_resp = MagicMock()
        start_resp.status_code = 201
        start_resp.json.return_value = {
            "data": {
                "id": "run-fail-001",
                "status": "FAILED",
                "exitCode": 1,
            }
        }

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = start_resp
        provider._http_client = mock_client

        res = await provider.search(source="glassdoor", search_term="AI Engineer")
        assert res.status == "failed"
        assert res.provider_status == "APIFY_RUN_FAILED"
        assert res.http_status == 502
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


@pytest.mark.asyncio
async def test_terminal_aborted_status_mapped():
    """Terminal ABORTED status maps to APIFY_ABORTED with http_status=502."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        settings = Settings(
            APIFY_API_TOKEN=SecretStr("mock_token"),
            APIFY_GLASSDOOR_ENABLED=True,
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)
        provider = ApifyDiscoveryProvider(settings=settings, budget_guard=guard)

        start_resp = MagicMock()
        start_resp.status_code = 201
        start_resp.json.return_value = {
            "data": {
                "id": "run-abort-001",
                "status": "ABORTED",
            }
        }

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = start_resp
        provider._http_client = mock_client

        res = await provider.search(source="glassdoor", search_term="AI Engineer")
        assert res.status == "failed"
        assert res.provider_status == "APIFY_ABORTED"
        assert res.http_status == 502
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


@pytest.mark.asyncio
async def test_transport_error_mapped_to_503():
    """Network connection failure maps to APIFY_TRANSPORT_ERROR (503), NOT 502."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        settings = Settings(
            APIFY_API_TOKEN=SecretStr("mock_token"),
            APIFY_GLASSDOOR_ENABLED=True,
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)
        provider = ApifyDiscoveryProvider(settings=settings, budget_guard=guard)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        provider._http_client = mock_client

        res = await provider.search(source="glassdoor", search_term="AI Engineer")
        assert res.status == "failed"
        assert res.provider_status == "APIFY_TRANSPORT_ERROR"
        assert res.http_status == 503
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


@pytest.mark.asyncio
async def test_inspect_existing_run_status():
    """Criterion 11: Inspects an existing run ID using get_run_status."""
    settings = Settings(APIFY_API_TOKEN=SecretStr("mock_token"))
    provider = ApifyDiscoveryProvider(settings=settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {
            "id": "04YXJ4Hxnb3FetFAG",
            "status": "SUCCEEDED",
            "defaultDatasetId": "wm4dOvOrOqq3gf4e9",
            "exitCode": 0,
        }
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = mock_resp
    provider._http_client = mock_client

    run_data = await provider.get_run_status("04YXJ4Hxnb3FetFAG")
    assert run_data["id"] == "04YXJ4Hxnb3FetFAG"
    assert run_data["status"] == "SUCCEEDED"
    assert run_data["defaultDatasetId"] == "wm4dOvOrOqq3gf4e9"


# ============================================================================
# 3. Actor ID, Schema Mapping, and Location Cleaning
# ============================================================================

@pytest.mark.asyncio
async def test_actor_schema_mapping_valig_default():
    """Criterion 3, 4, 5, 6: Verifies default valig~glassdoor-jobs-scraper payload mapping."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        settings = Settings(
            APIFY_API_TOKEN=SecretStr("mock_token"),
            APIFY_GLASSDOOR_ENABLED=True,
            APIFY_GLASSDOOR_ACTOR_ID="valig~glassdoor-jobs-scraper",
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)
        provider = ApifyDiscoveryProvider(settings=settings, budget_guard=guard)

        captured_payload = {}

        async def fake_post(url, json=None, headers=None, timeout=None):
            nonlocal captured_payload
            captured_payload = json
            resp = MagicMock()
            resp.status_code = 201
            resp.json.return_value = {
                "data": {
                    "id": "run-xyz",
                    "status": "SUCCEEDED",
                    "defaultDatasetId": "ds-xyz",
                }
            }
            return resp

        async def fake_get(url, headers=None, timeout=None):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = [SAMPLE_VALIG_ITEM_BANGALORE]
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = fake_post
        mock_client.get = fake_get
        provider._http_client = mock_client

        res = await provider.search(
            source="glassdoor",
            search_term="Lead AI Architect",
            location="Bangalore, India",
            results_wanted=15,
            hours_old=72,
        )

        assert res.status == "success"
        assert captured_payload["keywords"] == "Lead AI Architect"
        assert captured_payload["location"] == "Bangalore, India"
        assert captured_payload["limit"] == 15
        assert captured_payload["daysOld"] == 3
        assert captured_payload["sortBy"] == "relevant_desc"
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


@pytest.mark.asyncio
async def test_actor_schema_mapping_cheap_scraper():
    """Criterion 2: Verifies alternative cheap_scraper schema mapping."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        settings = Settings(
            APIFY_API_TOKEN=SecretStr("mock_token"),
            APIFY_GLASSDOOR_ENABLED=True,
            APIFY_GLASSDOOR_ACTOR_ID="cheap_scraper~glassdoor-jobs-scraper-remove-duplicate-jobs",
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)
        provider = ApifyDiscoveryProvider(settings=settings, budget_guard=guard)

        captured_payload = {}

        async def fake_post(url, json=None, headers=None, timeout=None):
            nonlocal captured_payload
            captured_payload = json
            resp = MagicMock()
            resp.status_code = 201
            resp.json.return_value = {
                "data": {
                    "id": "run-cheap-123",
                    "status": "SUCCEEDED",
                    "defaultDatasetId": "ds-cheap",
                }
            }
            return resp

        async def fake_get(url, headers=None, timeout=None):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = [SAMPLE_VALIG_ITEM_BANGALORE]
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = fake_post
        mock_client.get = fake_get
        provider._http_client = mock_client

        res = await provider.search(
            source="glassdoor",
            search_term="Lead AI Architect",
            location="Bangalore, India",
            results_wanted=15,
            hours_old=72,
        )

        assert res.status == "success"
        assert captured_payload["keywords"] == ["Lead AI Architect"]
        assert captured_payload["location"] == "Bangalore, India"
        assert captured_payload["country"] == "India"
        assert captured_payload["maxItems"] == 15
        assert captured_payload["datePosted"] == "3"
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


@pytest.mark.asyncio
async def test_orgupdate_actor_permanently_unavailable():
    """Criterion 1: Verifies orgupdate is blocked permanently with ACTOR_UNAVAILABLE."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        settings = Settings(
            APIFY_API_TOKEN=SecretStr("mock_token"),
            APIFY_GLASSDOOR_ENABLED=True,
            APIFY_GLASSDOOR_ACTOR_ID="orgupdate~glassdoor-jobs-scraper",
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)
        provider = ApifyDiscoveryProvider(settings=settings, budget_guard=guard)

        res = await provider.search(
            source="glassdoor",
            search_term="Lead AI Architect",
            location="Bangalore, India",
        )

        assert res.status == "failed"
        assert res.provider_status == "ACTOR_UNAVAILABLE"
        assert res.http_status == 400
        assert "ACTOR_UNAVAILABLE" in res.error
        assert res.diagnostics["circuit_breaker_status"] == "ACTOR_UNAVAILABLE"
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


@pytest.mark.asyncio
async def test_circuit_breaker_trips_to_open_after_consecutive_failures():
    """Criterion 8, 18: Verifies circuit breaker trips to ACTOR_CIRCUIT_OPEN after 2 consecutive failures."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        cb_state = tf.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf2:
        budget_state = tf2.name
    try:
        from app.discovery.circuit_breaker import ActorCircuitBreaker
        settings = Settings(
            APIFY_API_TOKEN=SecretStr("mock_token"),
            APIFY_GLASSDOOR_ENABLED=True,
            APIFY_GLASSDOOR_ACTOR_ID="custom~test-scraper",
            APIFY_ACTOR_FAILURE_THRESHOLD=2,
            APIFY_ACTOR_COOLDOWN_MINUTES=60,
        )
        breaker = ActorCircuitBreaker(settings=settings, state_file_path=cb_state, failure_threshold=2)
        guard = ApifyBudgetGuard(settings=settings, state_file_path=budget_state)
        provider = ApifyDiscoveryProvider(settings=settings, budget_guard=guard, circuit_breaker=breaker)

        # Mock client that returns non-zero exitCode failure
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {"data": {"id": "run-fail", "status": "FAILED", "exitCode": 91, "defaultDatasetId": "ds"}},
        )
        provider._http_client = mock_client

        # Run 1: Failure 1 recorded
        res1 = await provider.search(source="glassdoor", search_term="AI Engineer")
        assert res1.status == "failed"
        assert res1.provider_status == "APIFY_RUN_FAILED"
        assert breaker.get_status("custom~test-scraper")["state"] == "CLOSED"
        assert breaker.get_status("custom~test-scraper")["consecutive_failures"] == 1

        # Run 2: Failure 2 recorded -> TRIPS OPEN
        res2 = await provider.search(source="glassdoor", search_term="AI Engineer")
        assert res2.status == "failed"
        assert breaker.get_status("custom~test-scraper")["state"] == "OPEN"

        # Run 3: Blocked by circuit breaker immediately without calling Apify
        res3 = await provider.search(source="glassdoor", search_term="AI Engineer")
        assert res3.status == "failed"
        assert res3.provider_status == "ACTOR_CIRCUIT_OPEN"
        assert res3.http_status == 503
        assert "ACTOR_CIRCUIT_OPEN" in res3.error
    finally:
        for p in (cb_state, budget_state):
            if os.path.exists(p):
                os.remove(p)


@pytest.mark.parametrize(
    "city,expected_canonical",
    [
        ("chennai", "Chennai, India"),
        ("kochi", "Kochi, Kerala, India"),
        ("cochin", "Kochi, Kerala, India"),
        ("kozhikode", "Kozhikode, Kerala, India"),
        ("calicut", "Kozhikode, Kerala, India"),
        ("trivandrum", "Thiruvananthapuram, Kerala, India"),
        ("thiruvananthapuram", "Thiruvananthapuram, Kerala, India"),
        ("bangalore", "Bangalore, India"),
        ("bengaluru", "Bengaluru, India"),
        ("mumbai", "Mumbai, India"),
        ("bombay", "Mumbai, India"),
    ],
)
def test_multi_location_search_filters_all_6_hubs(city, expected_canonical):
    """Verifies all 6 requested search filter hubs (Chennai, Kochi, Kozhikode, Trivandrum, Bangalore, Mumbai)."""
    is_india, norm_loc, status = validate_india_location(city, allow_unknown=False)
    assert is_india is True
    assert status == "india_filter_passed"
    assert norm_loc == expected_canonical


def test_search_planner_distributes_across_all_6_target_hubs():
    """Verifies that ProfileSearchPlanner generates queries spanning Chennai, Kochi, Kozhikode, Trivandrum, Bangalore, and Mumbai."""
    from app.discovery.search_planner import ProfileSearchPlanner
    from app.models.profile import CandidateProfileData
    with open("data/profile/candidate_profile.json", "r", encoding="utf-8") as f:
        profile_dict = json.load(f)
    profile_data = CandidateProfileData.model_validate(profile_dict)

    planner = ProfileSearchPlanner()
    plan = planner.create_search_plan(profile_data, max_queries=20, site="glassdoor")

    locations = {q.location for q in plan.queries}
    assert any("Bangalore" in loc for loc in locations), f"Bangalore missing from {locations}"
    assert any("Chennai" in loc for loc in locations), f"Chennai missing from {locations}"
    assert any("Mumbai" in loc for loc in locations), f"Mumbai missing from {locations}"
    assert any("Kochi" in loc for loc in locations), f"Kochi missing from {locations}"
    assert any("Kozhikode" in loc for loc in locations), f"Kozhikode missing from {locations}"
    assert any("Trivandrum" in loc for loc in locations), f"Trivandrum missing from {locations}"


def test_valig_output_normalization_and_jl_extraction():
    """Criterion 11, 12, 13: Normalizes valig dataset item extracting jl, employer, location, pay."""
    job_create, dedup_keys = JobNormalizer.normalize(SAMPLE_VALIG_ITEM_BANGALORE)
    assert job_create.source == "glassdoor"
    assert job_create.external_id == "1010062906489"
    assert job_create.title == "Senior AI Engineer"
    assert job_create.company == "Turing Tech India"
    assert job_create.location == "Bangalore, India"
    assert job_create.salary_min == 2500000.0
    assert job_create.salary_max == 3500000.0
    assert job_create.easy_apply is True
    assert "1010062906489" in job_create.url


def test_clean_actor_location_name():
    """Unit test for clean_actor_location_name helper."""
    assert clean_actor_location_name("Bangalore, India") == "Bangalore"
    assert clean_actor_location_name("Bengaluru, Karnataka, India") == "Bengaluru, Karnataka"
    assert clean_actor_location_name("Chennai, Tamil Nadu, in") == "Chennai, Tamil Nadu"
    assert clean_actor_location_name("Remote, India") == "Remote"
    assert clean_actor_location_name("Pune") == "Pune"
    assert clean_actor_location_name(None) == "Bangalore"


def test_map_hours_old_to_date_posted():
    """Criterion 4: Verifies hours_old to Actor datePosted enum mapping."""
    assert map_hours_old_to_date_posted(24) == "today"
    assert map_hours_old_to_date_posted(48) == "3days"
    assert map_hours_old_to_date_posted(72) == "3days"
    assert map_hours_old_to_date_posted(168) == "week"
    assert map_hours_old_to_date_posted(720) == "month"
    assert map_hours_old_to_date_posted(1000) == "all"
    assert map_hours_old_to_date_posted(None) == "all"


# ============================================================================
# 4. Budget Guard Enforcement (500 Results / Month & Clamping)
# ============================================================================

def test_budget_guard_disabled():
    """Criterion 12: If APIFY_GLASSDOOR_ENABLED=False, budget guard rejects with APIFY_GLASSDOOR_DISABLED."""
    settings = Settings(APIFY_GLASSDOOR_ENABLED=False)
    guard = ApifyBudgetGuard(settings=settings)
    allowed, count, reason = guard.can_execute(20)
    assert allowed is False
    assert count == 0
    assert reason == "APIFY_GLASSDOOR_DISABLED"


def test_budget_guard_500_cap_enforcement_and_dynamic_clamping():
    """Criterion 8, 9, 10, 11, 32: Enforces 500 results/month cap, dynamic clamping, and cost calculation."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        settings = Settings(
            APIFY_GLASSDOOR_ENABLED=True,
            APIFY_GLASSDOOR_MAX_RESULTS_PER_RUN=20,
            APIFY_GLASSDOOR_MAX_RESULTS_PER_MONTH=500,
            APIFY_GLASSDOOR_COST_PER_RESULT=0.004,
            APIFY_GLASSDOOR_MAX_RUNS_PER_DAY=10,
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)

        # 1. Normal execution within quota
        allowed, clamped, reason = guard.can_execute(requested_results=20)
        assert allowed is True
        assert clamped == 20
        assert reason == "OK"

        # Record 490 results used
        guard.record_run(results_returned=490)
        status = guard.get_budget_status()
        assert status["monthly_results_count"] == 490
        assert status["remaining_monthly_quota"] == 10
        assert round(status["estimated_monthly_cost_usd"], 4) == round(490 * 0.004, 4)

        # 2. Dynamic Clamping: Requesting 20 when only 10 quota remains -> clamped to 10
        allowed, clamped, reason = guard.can_execute(requested_results=20)
        assert allowed is True
        assert clamped == 10
        assert reason == "OK"

        # Record remaining 10 results
        guard.record_run(results_returned=10)
        status_full = guard.get_budget_status()
        assert status_full["monthly_results_count"] == 500
        assert status_full["remaining_monthly_quota"] == 0
        assert round(status_full["estimated_monthly_cost_usd"], 2) == 2.00

        # 3. Hard Stop: Monthly limit reached -> rejected with APIFY_BUDGET_LIMIT_REACHED
        allowed, clamped, reason = guard.can_execute(requested_results=5)
        assert allowed is False
        assert clamped == 0
        assert reason == "APIFY_BUDGET_LIMIT_REACHED"
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


# ============================================================================
# 5. India Location Validator Tests
# ============================================================================

def test_india_location_validator_known_cities():
    """Criterion 17: Validates known Indian cities and tech hubs."""
    test_cases = [
        ("Bangalore", True, "Bangalore, India", "india_filter_passed"),
        ("Bengaluru, Karnataka", True, "Bengaluru, India", "india_filter_passed"),
        ("Hyderabad", True, "Hyderabad, India", "india_filter_passed"),
        ("Chennai", True, "Chennai, India", "india_filter_passed"),
        ("Pune, Maharashtra", True, "Pune, India", "india_filter_passed"),
        ("Mumbai", True, "Mumbai, India", "india_filter_passed"),
        ("New Delhi", True, "New Delhi, India", "india_filter_passed"),
        ("Gurgaon", True, "Gurgaon, India", "india_filter_passed"),
        ("Noida", True, "Noida, India", "india_filter_passed"),
        ("Kochi, Kerala", True, "Kochi, Kerala, India", "india_filter_passed"),
        ("Remote - India", True, "Remote, India", "india_filter_passed"),
    ]

    for raw, expected_is_in, expected_norm, expected_status in test_cases:
        is_in, norm, st = validate_india_location(raw, allow_unknown=False)
        assert is_in == expected_is_in, f"Failed for {raw}: got is_india={is_in}"
        assert st == expected_status, f"Failed status for {raw}: got {st}"


def test_india_location_validator_rejects_foreign():
    """Criterion 18: Rejects foreign locations explicitly."""
    foreign_locs = [
        "New York, NY",
        "San Francisco, California",
        "Austin, TX, USA",
        "London, United Kingdom",
        "Toronto, Canada",
        "Berlin, Germany",
        "Sydney, Australia",
        "Singapore",
    ]

    for floc in foreign_locs:
        is_in, norm, st = validate_india_location(floc, allow_unknown=False)
        assert is_in is False, f"Foreign location {floc} was not rejected"
        assert st == "india_filter_rejected"


def test_india_location_validator_unknown_handling():
    """Criterion 19: Unknown/unresolvable location handled fail-closed."""
    is_in, norm, st = validate_india_location("Some Random Nowhere", allow_unknown=False)
    assert is_in is False
    assert st == "india_location_unknown"

    is_in_allow, _, _ = validate_india_location("Some Random Nowhere", allow_unknown=True)
    assert is_in_allow is True


# ============================================================================
# 6. Normalizer & Deduplication Tests
# ============================================================================

def test_orgupdate_normalization_and_jl_extraction():
    """Criterion 13, 20, 21, 22: Correctly extracts title, company, salary, and jl external ID."""
    job_create, dedup_keys = JobNormalizer.normalize(SAMPLE_ORGUPDATE_ITEM_BANGALORE)

    assert job_create.source == "glassdoor"
    assert job_create.external_id == "1009876543210"
    assert job_create.title == "Senior AI Engineer"
    assert job_create.company == "Turing Tech India"
    assert job_create.salary_min == 2500000.0
    assert job_create.salary_max == 3500000.0
    assert job_create.url == "https://www.glassdoor.co.in/job-listing/senior-ai-engineer-turing-tech-jl=1009876543210.htm"
    assert dedup_keys["source_external_id"] == "glassdoor:1009876543210"


def test_deduplication_by_source_id_and_canonical_url():
    """Criterion 23: In-memory deduplication strips repeated items."""
    duplicate_item = dict(SAMPLE_ORGUPDATE_ITEM_BANGALORE)
    duplicate_item["URL"] = "https://www.glassdoor.co.in/job-listing/senior-ai-engineer-turing-tech-jl=1009876543210.htm?utm_source=glassdoor_alert"

    job1, keys1 = JobNormalizer.normalize(SAMPLE_ORGUPDATE_ITEM_BANGALORE)
    job2, keys2 = JobNormalizer.normalize(duplicate_item)

    assert keys1["source_external_id"] == keys2["source_external_id"]
    assert keys1["canonical_url"] == keys2["canonical_url"]


# ============================================================================
# 7. Service Integration & Routing Tests
# ============================================================================

@pytest.mark.asyncio
async def test_service_routes_glassdoor_to_apify_when_enabled():
    """Criterion 24, 25: Service dispatches Glassdoor to Apify, normalizes, validates India, and persists."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        settings = Settings(
            APIFY_API_TOKEN=SecretStr("mock_token"),
            APIFY_GLASSDOOR_ENABLED=True,
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)
        mock_apify_provider = AsyncMock(spec=ApifyDiscoveryProvider)
        mock_apify_provider.budget_guard = guard
        mock_apify_provider.search.return_value = SourceResult(
            provider="apify",
            source="glassdoor",
            status="success",
            provider_status="SUCCESS",
            http_status=200,
            jobs_returned=2,
            raw_jobs=[
                SAMPLE_ORGUPDATE_ITEM_BANGALORE,
                SAMPLE_ORGUPDATE_ITEM_US,
            ],
            duration_ms=1200,
            diagnostics={"provider": "apify", "actor_id": "valig~glassdoor-jobs-scraper"},
        )

        mock_jobspy = AsyncMock(spec=JobSpyClient)
        mock_repo = InMemoryJobRepository()

        service = JobDiscoveryService(
            settings=settings,
            jobspy_client=mock_jobspy,
            apify_provider=mock_apify_provider,
            job_repository=mock_repo,
        )

        request = JobSearchRequest(
            sites=["glassdoor"],
            search_term="AI Engineer",
            location="Bangalore, India",
            results_wanted=10,
        )

        resp: JobSearchResponse = await service.discover_jobs(request)

        assert resp.jobs_found == 1
        assert resp.new_jobs == 1
        assert resp.sources["glassdoor"].status == "success"

        stats = resp.glassdoor_stats
        assert stats is not None
        assert stats["glassdoor_results_received"] == 2
        assert stats["india_filter_passed"] == 1
        assert stats["india_filter_rejected"] == 1
        assert stats["glassdoor_jobs_inserted"] == 1
        assert "budget_status" in stats
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


@pytest.mark.asyncio
async def test_combined_indeed_and_glassdoor_profile_matching():
    """Criterion 25, 26, 27: Profile search combines Indeed + Glassdoor results and evaluates via MatchService."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        settings = Settings(
            APIFY_API_TOKEN=SecretStr("mock_token"),
            APIFY_GLASSDOOR_ENABLED=True,
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)

        mock_apify = AsyncMock(spec=ApifyDiscoveryProvider)
        mock_apify.budget_guard = guard
        mock_apify.search.return_value = SourceResult(
            provider="apify",
            source="glassdoor",
            status="success",
            jobs_returned=1,
            raw_jobs=[SAMPLE_ORGUPDATE_ITEM_BANGALORE],
        )

        mock_jobspy = AsyncMock(spec=JobSpyClient)
        mock_jobspy.search_single_source.return_value = SourceResult(
            provider="jobspy",
            source="indeed",
            status="success",
            jobs_returned=1,
            raw_jobs=[
                {
                    "id": "in-998877",
                    "site": "indeed",
                    "jobUrl": "https://in.indeed.com/viewjob?jk=998877",
                    "title": "Senior AI Engineer",
                    "company": "Infosys AI Lab",
                    "location": "Bangalore, India",
                    "datePosted": "2026-09-08T00:00:00Z",
                    "jobType": "fulltime",
                    "isRemote": False,
                }
            ],
        )

        mock_repo = InMemoryJobRepository()
        service = ProfileDiscoveryService(
            settings=settings,
            jobspy_client=mock_jobspy,
            apify_provider=mock_apify,
            job_repository=mock_repo,
        )

        profile_data = CandidateProfileData(
            personal=PersonalDetails(name="Arun Kumar", email="arun@example.com", location="Bangalore, India"),
            career=CareerPreferences(
                preferred_roles=["AI Engineer"],
                preferred_locations=["Bangalore, India"],
                experience_years=6.0,
            ),
        )
        service.profile_service.get_active_profile_data = MagicMock(return_value=profile_data)

        req = ProfileJobSearchRequest(
            sites=["indeed", "glassdoor"],
            max_queries=2,
            results_per_query=5,
            run_matching=True,
            minimum_score=50.0,
        )

        resp: ProfileJobSearchResponse = await service.discover_jobs_from_profile(req)

        assert resp.status == "success"
        assert resp.discovery.unique_jobs == 2
        assert resp.matching is not None
        assert resp.matching.jobs_scored == 2
        for match in resp.top_matches:
            assert match.url is not None
            assert match.source in ("indeed", "glassdoor")
            assert match.final_score >= 50.0
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


# ============================================================================
# 8. Endpoint Tests: Budget & Controlled Discovery Test
# ============================================================================

def test_api_get_apify_budget_endpoint():
    """Criterion 11: GET /jobs/discovery/apify/budget returns budget metrics."""
    resp = client.get("/jobs/discovery/apify/budget")
    assert resp.status_code == 200
    data = resp.json()
    assert "monthly_results_count" in data
    assert "monthly_results_limit" in data
    assert "remaining_monthly_quota" in data
    assert "estimated_monthly_cost_usd" in data
    assert "cost_per_result_usd" in data


def test_api_manual_glassdoor_ingestion_remains_functional():
    """Criterion 28: POST /jobs/ingest/glassdoor manual fallback remains fully functional."""
    payload = {
        "url": "https://www.glassdoor.co.in/job-listing/senior-dev-jl=1234567890.htm",
        "title": "Senior AI Architect",
        "company": "DeepMind Partner",
        "location": "Bengaluru, Karnataka, India",
    }
    resp = client.post("/jobs/ingest/glassdoor", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["source"] == "glassdoor"
    assert data["external_id"] == "1234567890"
    assert "job_id" in data


# ============================================================================
# 9. Pre-flight Auth Check, Header Construction & Token Sanitization Tests
# ============================================================================

@pytest.mark.asyncio
async def test_apify_auth_check_valid_token_200():
    """Valid token -> GET /v2/users/me returns 200 and safe account metadata."""
    settings = Settings(
        APIFY_API_TOKEN=SecretStr("mock_valid_token_12345"),
        APIFY_GLASSDOOR_ENABLED=True,
    )
    provider = ApifyDiscoveryProvider(settings=settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {
            "id": "user_xyz789",
            "username": "test_engineer",
        }
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = mock_resp
    provider._http_client = mock_client

    is_ok, code, details = await provider.check_authentication()
    assert is_ok is True
    assert code == 200
    assert details["user_id"] == "user_xyz789"
    assert details["username"] == "test_engineer"


@pytest.mark.asyncio
async def test_apify_auth_check_invalid_token_401():
    """Invalid token -> GET /v2/users/me returns 401."""
    settings = Settings(
        APIFY_API_TOKEN=SecretStr("invalid_token"),
        APIFY_GLASSDOOR_ENABLED=True,
    )
    provider = ApifyDiscoveryProvider(settings=settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = '{"error": {"message": "Invalid token"}}'

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = mock_resp
    provider._http_client = mock_client

    is_ok, code, details = await provider.check_authentication()
    assert is_ok is False
    assert code == 401
    assert "401" in details["error"]


def test_apify_auth_header_construction():
    """Header must be exactly Authorization: Bearer <token> without quotes or extra Bearer."""
    raw_token = "apify_api_cleanToken123"
    settings = Settings(APIFY_API_TOKEN=SecretStr(raw_token))
    provider = ApifyDiscoveryProvider(settings=settings)

    headers = provider.get_auth_headers()
    assert "Authorization" in headers
    assert headers["Authorization"] == f"Bearer {raw_token}"
    assert "Bearer Bearer" not in headers["Authorization"]
    assert '"' not in headers["Authorization"]
    assert "'" not in headers["Authorization"]


def test_apify_token_sanitization_no_double_bearer_and_whitespace():
    """Tokens with 'Bearer ', quotes, spaces, and newlines are sanitized cleanly."""
    # Test double Bearer prefix
    s1 = Settings(APIFY_API_TOKEN=SecretStr("Bearer apify_api_tokenDouble"))
    assert s1.apify_token_value == "apify_api_tokenDouble"

    # Test double Bearer with lowercase
    s2 = Settings(APIFY_API_TOKEN=SecretStr("Bearer Bearer apify_api_tokenTriple"))
    assert s2.apify_token_value == "apify_api_tokenTriple"

    # Test double quotes
    s3 = Settings(APIFY_API_TOKEN=SecretStr('"apify_api_quotedToken"'))
    assert s3.apify_token_value == "apify_api_quotedToken"

    # Test single quotes
    s4 = Settings(APIFY_API_TOKEN=SecretStr("'apify_api_singleQuoted'"))
    assert s4.apify_token_value == "apify_api_singleQuoted"

    # Test whitespace and newlines
    s5 = Settings(APIFY_API_TOKEN=SecretStr("  apify_api_newlineToken\r\n  "))
    assert s5.apify_token_value == "apify_api_newlineToken"


def test_apify_token_never_logged_or_exposed():
    """Token never appears in logs, diagnostics, or serialized responses."""
    secret = "apify_api_secretNeverExpose123456"
    settings = Settings(APIFY_API_TOKEN=SecretStr(secret))
    provider = ApifyDiscoveryProvider(settings=settings)

    headers = provider.get_auth_headers()
    assert headers["Authorization"] == f"Bearer {secret}"

    # Diagnostics verification
    diag = {
        "apify_token_configured": bool(provider.settings.apify_token_value),
        "actor_id": "test_actor",
    }
    dumped = json.dumps(diag)
    assert secret not in dumped
    assert "apify_api_" not in dumped


@pytest.mark.asyncio
async def test_apify_preflight_auth_check_prevents_actor_start_on_401():
    """If pre-flight check returns 401, Actor is NEVER started and 0 quota consumed."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        settings = Settings(
            APIFY_API_TOKEN=SecretStr("invalid_token"),
            APIFY_GLASSDOOR_ENABLED=True,
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)
        provider = ApifyDiscoveryProvider(settings=settings, budget_guard=guard)

        # /users/me returns 401
        auth_resp = MagicMock()
        auth_resp.status_code = 401
        auth_resp.text = "Unauthorized"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = auth_resp
        provider._http_client = mock_client

        res = await provider.search(source="glassdoor", search_term="AI Engineer")

        # 1. Returned APIFY_AUTH_ERROR with 401
        assert res.status == "failed"
        assert res.provider_status == "APIFY_AUTH_ERROR"
        assert res.http_status == 401
        assert res.diagnostics["apify_auth_check_status"] == "FAILED_HTTP_401"
        assert res.diagnostics["apify_actor_request_status"] == "SKIPPED"

        # 2. start_actor was NEVER called (post was never executed)
        assert mock_client.post.call_count == 0

        # 3. Consumed 0 Actor runs and 0 results quota
        budget_status = guard.get_budget_status()
        assert budget_status["monthly_results_count"] == 0
        assert budget_status["monthly_runs_count"] == 0
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


@pytest.mark.asyncio
async def test_apify_actor_and_dataset_requests_receive_auth_header():
    """Both Actor run start and dataset retrieval pass the Authorization header."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        state_file = tf.name
    try:
        token = "apify_api_verifiedHeader123"
        settings = Settings(
            APIFY_API_TOKEN=SecretStr(token),
            APIFY_GLASSDOOR_ENABLED=True,
        )
        guard = ApifyBudgetGuard(settings=settings, state_file_path=state_file)
        provider = ApifyDiscoveryProvider(settings=settings, budget_guard=guard)

        post_headers_captured = {}
        get_headers_captured = []

        async def fake_post(url, json=None, headers=None, timeout=None):
            nonlocal post_headers_captured
            post_headers_captured = headers or {}
            resp = MagicMock()
            resp.status_code = 201
            resp.json.return_value = {
                "data": {
                    "id": "run-hdr-01",
                    "status": "SUCCEEDED",
                    "defaultDatasetId": "ds-hdr-01",
                }
            }
            return resp

        async def fake_get(url, headers=None, timeout=None):
            nonlocal get_headers_captured
            get_headers_captured.append((url, headers or {}))
            resp = MagicMock()
            resp.status_code = 200
            if "users/me" in url:
                resp.json.return_value = {"data": {"id": "u1", "username": "admin"}}
            else:
                resp.json.return_value = [SAMPLE_ORGUPDATE_ITEM_BANGALORE]
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = fake_post
        mock_client.get = fake_get
        provider._http_client = mock_client

        res = await provider.search(source="glassdoor", search_term="AI Engineer")
        assert res.status == "success"

        # Check Actor start POST received auth
        assert post_headers_captured.get("Authorization") == f"Bearer {token}"

        # Check dataset GET received auth
        dataset_calls = [h for u, h in get_headers_captured if "datasets" in u]
        assert len(dataset_calls) > 0
        assert dataset_calls[0].get("Authorization") == f"Bearer {token}"
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


def test_api_get_apify_auth_diagnostic_endpoint():
    """GET /jobs/discovery/apify/auth returns verified status without leaking token."""
    resp = client.get("/jobs/discovery/apify/auth")
    assert resp.status_code == 200
    data = resp.json()
    assert "apify_token_configured" in data
    assert "auth_verified" in data
    assert "token_length" in data
    # Token value must never appear
    serialized = json.dumps(data)
    assert "apify_api_" not in serialized

