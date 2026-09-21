"""API router for job discovery, profile-driven broad search, and retrieval endpoints."""

import logging
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field


from app.database.repositories.job_repository import JobRepository
from app.discovery.profile_discovery_service import ProfileDiscoveryService
from app.discovery.service import JobDiscoveryService
from app.automation.glassdoor.url_validator import (
    extract_glassdoor_job_id,
    validate_glassdoor_request,
)
from app.discovery.india_validator import validate_india_location
from app.discovery.jobspy_client import JobSpyClient, SourceResult
from app.discovery.normalizer import JobNormalizer
from app.models.job import (
    GlassdoorIngestRequest,
    GlassdoorIngestResponse,
    Job,
    JobCreate,
    JobListResponse,
    JobSearchRequest,
    JobSearchResponse,
    JobStats,
    ProfileJobSearchRequest,
    ProfileJobSearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/jobs",
    tags=["Jobs"],
)


def get_job_repository() -> JobRepository:
    """Dependency provider for JobRepository."""
    return JobRepository()


def get_discovery_service(
    repo: JobRepository = Depends(get_job_repository),
) -> JobDiscoveryService:
    """Dependency provider for JobDiscoveryService."""
    return JobDiscoveryService(job_repository=repo)


def get_profile_discovery_service(
    repo: JobRepository = Depends(get_job_repository),
) -> ProfileDiscoveryService:
    """Dependency provider for ProfileDiscoveryService."""
    return ProfileDiscoveryService(job_repository=repo)


@router.post(
    "/search",
    response_model=JobSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Discover and persist jobs across multiple job boards",
    description=(
        "Queries JobSpy MCP server independently for each specified source, handles individual failures gracefully, "
        "normalizes and deduplicates results, persists them to Supabase, and returns a detailed summary."
    ),
)
async def search_jobs(
    request: JobSearchRequest,
    discovery_service: JobDiscoveryService = Depends(get_discovery_service),
) -> JobSearchResponse:
    """Trigger manual multi-source job search and persistence."""
    try:
        response = await discovery_service.discover_jobs(request)
        return response
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        ) from val_err
    except Exception as exc:
        logger.error("Job discovery search endpoint error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal discovery error: {str(exc)}",
        ) from exc


@router.post(
    "/search/from-profile",
    response_model=ProfileJobSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Automated broad discovery from candidate profile",
    description=(
        "Generates a broad, bounded multi-query search plan directly from the candidate career profile, "
        "executes JobSpy searches, deduplicates across all queries, persists jobs to Supabase, "
        "evaluates matches using deterministic and optional NVIDIA semantic scoring, and returns ranked jobs with direct URLs."
    ),
)
async def search_jobs_from_profile(
    request: ProfileJobSearchRequest,
    profile_id: Optional[UUID] = Query(default=None, description="Optional profile ID to target"),
    profile_discovery_service: ProfileDiscoveryService = Depends(get_profile_discovery_service),
) -> ProfileJobSearchResponse:
    """Trigger profile-driven automated broad discovery, deduplication, matching, and ranked URL delivery."""
    try:
        response = await profile_discovery_service.discover_jobs_from_profile(
            request=request,
            profile_id=profile_id,
        )
        return response
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        ) from val_err
    except Exception as exc:
        logger.error("Profile-driven discovery endpoint error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal profile discovery error: {str(exc)}",
        ) from exc


@router.get(
    "/stats",
    response_model=JobStats,
    status_code=status.HTTP_200_OK,
    summary="Get job database statistics",
    description="Returns aggregate metrics on stored jobs, breakdown by source platform, and remote counts.",
)
async def get_job_stats(
    repository: JobRepository = Depends(get_job_repository),
) -> JobStats:
    """Fetch stored job statistics."""
    try:
        return repository.get_job_stats()
    except Exception as exc:
        logger.error("Error fetching job stats: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job statistics",
        ) from exc


@router.get(
    "",
    response_model=JobListResponse,
    status_code=status.HTTP_200_OK,
    summary="List stored jobs",
    description="Retrieves a paginated list of stored jobs with optional source and keyword filtering.",
)
async def list_jobs(
    limit: int = Query(default=50, ge=1, le=200, description="Max jobs to return"),
    offset: int = Query(default=0, ge=0, description="Number of jobs to skip"),
    source: Optional[str] = Query(default=None, description="Filter by source board (e.g. linkedin, indeed)"),
    search: Optional[str] = Query(default=None, description="Search keyword in title, company, or location"),
    repository: JobRepository = Depends(get_job_repository),
) -> JobListResponse:
    """List jobs with pagination and filtering."""
    try:
        jobs, total = repository.list_jobs(
            limit=limit,
            offset=offset,
            source=source,
            search=search,
        )
        return JobListResponse(
            total=total,
            limit=limit,
            offset=offset,
            jobs=jobs,
        )
    except Exception as exc:
        logger.error("Error listing jobs: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list jobs",
        ) from exc


@router.get(
    "/{job_id}",
    response_model=Job,
    status_code=status.HTTP_200_OK,
    summary="Get single job details",
    description="Retrieves full job details and original raw payload by primary UUID.",
)
async def get_job(
    job_id: UUID,
    repository: JobRepository = Depends(get_job_repository),
) -> Job:
    """Fetch single job by UUID."""
    job = repository.get_job_by_id(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found",
        )
    return job


@router.post(
    "/ingest/glassdoor",
    response_model=GlassdoorIngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Manually ingest a Glassdoor job URL into database",
    description="Validates a Glassdoor job URL, extracts its external listing ID, normalizes the record, and upserts it into the jobs table returning a real UUID.",
)
async def ingest_glassdoor_job(
    request: GlassdoorIngestRequest,
    repository: JobRepository = Depends(get_job_repository),
) -> GlassdoorIngestResponse:
    """Ingest a Glassdoor job URL fallback without browser scraping."""
    is_valid, normalized_url, err_msg = validate_glassdoor_request(request.url)
    if not is_valid or not normalized_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid Glassdoor URL: {err_msg}",
        )

    external_id = extract_glassdoor_job_id(normalized_url)
    if not external_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract stable Glassdoor listing ID (jl) from URL.",
        )

    title = (request.title or "").strip() or "Glassdoor Opportunity"
    company = (request.company or "").strip() or "Glassdoor Employer"
    location = (request.location or "").strip() or None

    job_in = JobCreate(
        source="glassdoor",
        external_id=external_id,
        title=title,
        company=company,
        location=location,
        url=normalized_url,
        easy_apply=True,
        raw_data={"source": "manual_ingest", "url": normalized_url},
    )

    saved_job, is_new = repository.upsert_job(job_in)
    if not saved_job:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist Glassdoor job into database.",
        )

    return GlassdoorIngestResponse(
        success=True,
        job_id=saved_job.id,
        source="glassdoor",
        external_id=external_id,
        url=normalized_url,
        title=saved_job.title,
        company=saved_job.company,
        location=saved_job.location,
        is_new=is_new,
        message="Glassdoor job ingested successfully.",
    )


@router.get(
    "/discovery/jobspy/health",
    status_code=status.HTTP_200_OK,
    summary="Health check for JobSpy bridge connection",
    description="Verifies whether the JobSpy bridge server is reachable.",
)
async def jobspy_health(
    discovery_service: JobDiscoveryService = Depends(get_discovery_service),
):
    """Check connectivity to JobSpy bridge."""
    return await discovery_service.jobspy_client.health_check()


@router.post(
    "/discovery/jobspy/test",
    status_code=status.HTTP_200_OK,
    summary="Diagnostic test for JobSpy source search",
    description="Executes a test query against JobSpy and reports granular provider status (SUCCESS, UPSTREAM_BLOCKED, LOCATION_PARSE_ERROR, etc.).",
)
async def jobspy_test(
    source: str = Query(default="glassdoor", description="Job board to test"),
    search_term: str = Query(default="AI Engineer", description="Job query"),
    location: str = Query(default="Bangalore, India", description="Target location"),
    is_remote: bool = Query(default=False, description="Search remote jobs"),
    results_wanted: int = Query(default=5, ge=1, le=20, description="Results count"),
    discovery_service: JobDiscoveryService = Depends(get_discovery_service),
):
    """Run a single test query and return raw diagnostic SourceResult."""
    res = await discovery_service.jobspy_client.search_single_source(
        source=source,
        search_term=search_term,
        location=location,
        is_remote=is_remote,
        results_wanted=results_wanted,
    )
    return res.model_dump()


@router.get(
    "/discovery/apify/auth",
    status_code=status.HTTP_200_OK,
    summary="Diagnostic verification for Apify API token and authentication",
    description="Tests GET /v2/users/me against Apify using provider token loading and returns safe metadata without exposing the secret.",
)
async def apify_auth_diagnostic(
    discovery_service: JobDiscoveryService = Depends(get_discovery_service),
):
    """Safely verify Apify API token authentication against /v2/users/me."""
    is_ok, code, details = await discovery_service.apify_provider.check_authentication()
    token = discovery_service.apify_provider.settings.apify_token_value
    return {
        "apify_token_configured": bool(token),
        "token_length": len(token) if token else 0,
        "token_starts_with_apify_api": token.startswith("apify_api_") if token else False,
        "auth_verified": is_ok,
        "http_status": code,
        "details": details,
    }


@router.get(
    "/discovery/apify/budget",
    status_code=status.HTTP_200_OK,
    summary="Get Apify Glassdoor discovery budget and usage status",
    description="Returns current monthly result count, monthly limit (500), remaining quota, estimated cost ($0.004/item), and daily run count.",
)
async def get_apify_budget(
    discovery_service: JobDiscoveryService = Depends(get_discovery_service),
):
    """Get current Apify budget guard status."""
    return discovery_service.apify_provider.budget_guard.get_budget_status()



class ApifyDiscoveryTestRequest(BaseModel):
    search_term: str = Field(default="AI Engineer", description="Job query keyword")
    location: str = Field(default="Bangalore, India", description="Target location")
    results_wanted: int = Field(default=5, ge=1, le=20, description="Results count (capped at 20)")
    hours_old: Optional[int] = Field(default=168, description="Maximum job posting age in hours")
    persist: bool = Field(default=True, description="Whether to normalize, validate India, and upsert to Supabase jobs table")


@router.post(
    "/discovery/apify/test",
    status_code=status.HTTP_200_OK,
    summary="Diagnostic controlled test for Apify Glassdoor discovery",
    description="Executes a single test query against Apify Glassdoor actor with budget enforcement, India location validation, and returns diagnostics without calling automation.",
)
async def apify_glassdoor_test(
    body: Optional[ApifyDiscoveryTestRequest] = None,
    search_term: Optional[str] = Query(default=None, description="Job query keyword"),
    location: Optional[str] = Query(default=None, description="Target location"),
    results_wanted: Optional[int] = Query(default=None, ge=1, le=20, description="Results count (capped at 20)"),
    hours_old: Optional[int] = Query(default=None, description="Maximum job posting age in hours"),
    persist: Optional[bool] = Query(default=None, description="Persist normalized results to database"),
    discovery_service: JobDiscoveryService = Depends(get_discovery_service),
):
    """Run a controlled Apify Glassdoor discovery test with budget guard, India validation, and optional Supabase persistence."""
    effective_req = body or ApifyDiscoveryTestRequest()
    term = search_term if search_term is not None else effective_req.search_term
    loc = location if location is not None else effective_req.location
    rw = results_wanted if results_wanted is not None else effective_req.results_wanted
    ho = hours_old if hours_old is not None else effective_req.hours_old
    should_persist = persist if persist is not None else effective_req.persist

    res = await discovery_service.apify_provider.search(
        source="glassdoor",
        search_term=term,
        location=loc,
        results_wanted=rw,
        hours_old=ho or 168,
        country="India",
    )
    res_dict = res.model_dump()
    persisted_results = []

    if should_persist and res.status == "success" and res.raw_jobs:
        for raw in res.raw_jobs:
            try:
                job_create, dedup_keys = JobNormalizer.normalize(raw, default_source="glassdoor")
                is_india, norm_loc, filter_status = validate_india_location(job_create.location, allow_unknown=False)
                if not is_india:
                    logger.info("Test run: Glassdoor job '%s' rejected by India filter: %s", job_create.title, job_create.location)
                    continue
                job_create.location = norm_loc
                saved_job, is_new = discovery_service.job_repository.upsert_job(job_create)
                if saved_job:
                    persisted_results.append({
                        "id": str(saved_job.id),
                        "source": saved_job.source,
                        "external_id": saved_job.external_id,
                        "title": saved_job.title,
                        "company": saved_job.company,
                        "location": saved_job.location,
                        "url": saved_job.url,
                        "is_new": is_new,
                    })
            except Exception as e:
                logger.warning("Test run job normalization/upsert error: %s", e)

    res_dict["persisted_jobs"] = persisted_results
    res_dict["persisted_count"] = len(persisted_results)
    return res_dict


