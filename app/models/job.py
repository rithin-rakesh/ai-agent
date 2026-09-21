"""Pydantic models for Job entity and Profile-Driven Discovery."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field


class ApplicationMethod(str, Enum):
    """Application method type for a job posting."""

    EASY_APPLY = "EASY_APPLY"
    EXTERNAL_APPLY = "EXTERNAL_APPLY"
    UNKNOWN = "UNKNOWN"


class AvailabilityStatus(str, Enum):
    """Normalized posting availability status."""

    UNKNOWN = "UNKNOWN"
    AVAILABLE = "AVAILABLE"
    EXPIRED = "EXPIRED"
    UNAVAILABLE = "UNAVAILABLE"


class JobBase(BaseModel):
    """Base schema for a job posting."""

    source: str = Field(..., description="Job source/platform (e.g. linkedin, indeed, glassdoor)")
    external_id: str = Field(..., description="Unique job ID on the external platform")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Hiring company name")
    location: Optional[str] = Field(default=None, description="Job location or Remote")
    description: Optional[str] = Field(default=None, description="Full job description text")
    url: Optional[str] = Field(default=None, description="Direct URL to the job listing")
    salary_min: Optional[float] = Field(default=None, ge=0.0, description="Minimum estimated or posted salary")
    salary_max: Optional[float] = Field(default=None, ge=0.0, description="Maximum estimated or posted salary")
    experience_text: Optional[str] = Field(default=None, description="Experience level requirement text")
    job_type: Optional[str] = Field(default=None, description="Employment type (e.g. full-time, contract)")
    remote: bool = Field(default=False, description="Whether the position is remote")
    posted_at: Optional[datetime] = Field(default=None, description="When the job was originally posted")
    easy_apply: Optional[bool] = Field(default=None, description="Whether platform supports easy / quick apply")
    application_method: str = Field(default=ApplicationMethod.UNKNOWN.value, description="Application method: EASY_APPLY, EXTERNAL_APPLY, UNKNOWN")
    availability_status: str = Field(default=AvailabilityStatus.UNKNOWN.value, description="Availability status: UNKNOWN, AVAILABLE, EXPIRED, UNAVAILABLE")
    application_eligible: bool = Field(default=True, description="Whether job is currently eligible for automated application")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw job payload from discovery/scraper")


class JobCreate(JobBase):
    """Schema for creating or upserting a new job."""

    pass


class Job(JobBase):
    """Complete Job model with database identifiers and timestamps."""

    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SourceStatus(BaseModel):
    """Status report for a single job search source."""

    status: str = Field(..., description="'success' or 'failed'")
    provider_status: Optional[str] = Field(default=None, description="Granular provider status e.g. UPSTREAM_BLOCKED, LOCATION_PARSE_ERROR, SUCCESS")
    http_status: Optional[int] = Field(default=None, description="HTTP status code from upstream or bridge")
    jobs_returned: int = Field(default=0, description="Number of jobs returned by this source")
    error: Optional[str] = Field(default=None, description="Error message if this source failed")
    duration_ms: Optional[int] = Field(default=None, description="Execution duration in milliseconds")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic metadata for the search")


class JobSearchRequest(BaseModel):
    """Request schema for initiating a job search across multiple sources."""

    sites: list[str] = Field(
        default=["linkedin", "indeed", "naukri", "glassdoor"],
        description="List of job sites to search (subset of linkedin, indeed, naukri, glassdoor)",
    )
    search_term: str = Field(..., min_length=1, description="Search term or job title")
    location: str = Field(default="remote", description="Job location or Remote")
    is_remote: bool = Field(default=False, description="Whether to filter for remote jobs explicitly")
    results_wanted: int = Field(default=20, ge=1, le=100, description="Number of results desired per site")
    hours_old: int = Field(default=72, ge=1, description="Maximum age of postings in hours")
    country_indeed: str = Field(default="India", description="Country code/name for Indeed queries")


class JobSearchResponse(BaseModel):
    """Response schema returned by the discovery API."""

    jobs_found: int = Field(..., description="Total unique jobs discovered across all sources")
    new_jobs: int = Field(..., description="Count of new jobs inserted into the database")
    duplicates: int = Field(..., description="Count of duplicate jobs skipped or updated")
    sources: Dict[str, SourceStatus] = Field(..., description="Breakdown of status per source")
    execution_time_ms: int = Field(..., description="Total search duration in milliseconds")
    jobs: list[Job] = Field(default_factory=list, description="List of persisted/discovered jobs")
    glassdoor_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Breakdown of Glassdoor discovery and persistence metrics",
    )


class GlassdoorIngestRequest(BaseModel):
    """Request schema for manually ingesting a Glassdoor job URL."""

    url: str = Field(..., description="Direct Glassdoor job listing URL")
    title: Optional[str] = Field(default=None, description="Job title override")
    company: Optional[str] = Field(default=None, description="Company name override")
    location: Optional[str] = Field(default=None, description="Job location override")


class GlassdoorIngestResponse(BaseModel):
    """Response schema for manual Glassdoor job URL ingestion."""

    success: bool
    job_id: UUID
    source: str = "glassdoor"
    external_id: str
    url: str
    title: str
    company: str
    location: Optional[str] = None
    is_new: bool
    message: str


class JobStats(BaseModel):
    """Aggregated statistics for jobs stored in the database."""

    total_jobs: int = Field(..., description="Total count of jobs stored")
    sources: Dict[str, int] = Field(default_factory=dict, description="Job counts broken down by source")
    remote_jobs: int = Field(default=0, description="Count of remote positions")
    easy_apply_jobs: int = Field(default=0, description="Count of easy-apply positions")
    latest_posted_at: Optional[datetime] = Field(default=None, description="Timestamp of most recent job posting")


class JobListResponse(BaseModel):
    """Paginated list of jobs."""

    total: int = Field(..., description="Total matching jobs in database")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")
    jobs: list[Job] = Field(default_factory=list, description="List of jobs")


# ---------------------------------------------------------------------------
# Phase 5.3: Profile-Driven Broad Discovery Models
# ---------------------------------------------------------------------------


class SearchQuery(BaseModel):
    """A single bounded search query generated by the search planner."""

    query_id: str = Field(default_factory=lambda: f"query_{uuid4().hex[:8]}", description="Unique identifier for the query")
    search_term: str = Field(..., description="Query string for job board search")
    location: str = Field(default="remote", description="Target location for the query")
    source: str = Field(default="indeed", description="Job board platform (e.g. indeed)")
    reason: str = Field(..., description="Rationale for generating this query")
    priority: int = Field(default=1, ge=1, le=5, description="Search priority (1=highest)")


class SearchPlan(BaseModel):
    """Structured plan of queries derived from the candidate profile."""

    queries: List[SearchQuery] = Field(default_factory=list, description="List of planned queries")
    generated_from: str = Field(default="candidate_profile", description="Source basis for the plan")
    query_count: int = Field(default=0, description="Total count of planned queries")


class SearchPlanSummary(BaseModel):
    """Summary of the executed search plan for API responses."""

    queries_generated: int = Field(..., description="Count of queries generated in plan")
    queries: List[SearchQuery] = Field(default_factory=list, description="Generated query list")


class DiscoveryQueryResult(BaseModel):
    """Detailed result and diagnostic record for an individual query execution."""

    query_id: str = Field(..., description="Unique query identifier")
    source: str = Field(..., description="Target job board platform (e.g. indeed, glassdoor)")
    provider: str = Field(..., description="Underlying provider (e.g. jobspy, apify)")
    search_term: str = Field(..., description="Search term queried")
    location: str = Field(..., description="Normalized location queried")
    remote: bool = Field(default=False, description="Whether query was executed as remote search")
    status: str = Field(..., description="Execution status e.g. SUCCESS, FAILED, APIFY_BUDGET_LIMIT_REACHED")
    jobs_returned: int = Field(default=0, description="Count of raw jobs returned by this query")
    error: Optional[str] = Field(default=None, description="Diagnostic error message if query failed (zero secrets)")
    duration_ms: int = Field(default=0, description="Query execution duration in milliseconds")


class DiscoveryStats(BaseModel):
    """Aggregated statistics from the multi-query discovery execution."""

    raw_jobs: int = Field(default=0, description="Total raw jobs collected across all queries")
    unique_jobs: int = Field(default=0, description="Count of unique jobs after cross-query deduplication")
    duplicates_removed: int = Field(default=0, description="Count of redundant/duplicate jobs removed")
    queries_succeeded: int = Field(default=0, description="Count of search queries that succeeded")
    queries_failed: int = Field(default=0, description="Count of search queries that encountered errors")
    query_results: List[DiscoveryQueryResult] = Field(default_factory=list, description="Per-query diagnostic execution results")
    execution_time_ms: Optional[int] = Field(default=None, description="Total discovery duration in milliseconds")


class MatchingStats(BaseModel):
    """Breakdown of decision tiers for scored jobs."""

    jobs_scored: int = Field(default=0, description="Total jobs evaluated by matching engine")
    excellent: int = Field(default=0, description="Count of excellent matches (score >= 90)")
    strong_match: int = Field(default=0, description="Count of strong matches (80 <= score < 90)")
    review: int = Field(default=0, description="Count of review-tier matches (70 <= score < 80)")
    low_match: int = Field(default=0, description="Count of low matches (60 <= score < 70)")
    weak_match: int = Field(default=0, description="Alias for low_match for backward compatibility")
    reject: int = Field(default=0, description="Count of rejected/disqualified jobs (score < 60 or gated)")
    unqualified: int = Field(default=0, description="Alias for reject for backward compatibility")
    matching_time_ms: Optional[int] = Field(default=0, description="Time spent in matching engine (ms)")


class MatchedJobItem(BaseModel):
    """Ranked job item returned with match scores and original job URL."""

    job_id: UUID = Field(..., description="Job primary key UUID")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Hiring company")
    location: Optional[str] = Field(default=None, description="Job location")
    source: str = Field(default="indeed", description="Job board platform")
    url: Optional[str] = Field(default=None, description="Original direct job listing URL")
    deterministic_score: Optional[float] = Field(default=None, description="Deterministic match score (0-100)")
    semantic_score: Optional[float] = Field(default=None, description="Semantic match score (0-100) if evaluated")
    final_score: float = Field(..., description="Overall combined match score (0-100)")
    decision: str = Field(..., description="Match tier decision (e.g. excellent, strong_match, review)")
    reason: Optional[str] = Field(default=None, description="Summary explanation of candidate-job fit")
    application_method: str = Field(default=ApplicationMethod.UNKNOWN.value, description="Application method: EASY_APPLY, EXTERNAL_APPLY, UNKNOWN")
    availability_status: str = Field(default=AvailabilityStatus.AVAILABLE.value, description="Availability status: UNKNOWN, AVAILABLE, EXPIRED, UNAVAILABLE")
    application_eligible: bool = Field(default=True, description="Whether job is currently eligible for automated application")

    @property
    def id(self) -> UUID:
        """Convenience alias for job_id matching discovery response format."""
        return self.job_id

    @property
    def match_score(self) -> float:
        """Convenience alias for final_score matching discovery response format."""
        return self.final_score


class ProfileJobSearchRequest(BaseModel):
    """Request schema for profile-driven automated broad discovery."""

    sites: List[str] = Field(
        default=["indeed"],
        description="List of job boards to query (default ['indeed'])",
    )
    results_per_query: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Number of jobs requested per search query",
    )
    hours_old: int = Field(
        default=168,
        ge=1,
        description="Maximum job posting age in hours (168 = 7 days)",
    )
    country_indeed: str = Field(
        default="India",
        description="Country code or name for Indeed queries",
    )
    max_queries: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum number of search queries to generate",
    )
    run_matching: bool = Field(
        default=True,
        description="Whether to run matching engine and rank discovered jobs",
    )
    use_semantic: bool = Field(
        default=True,
        description="Whether to include NVIDIA semantic matching for qualifying jobs",
    )
    use_nvidia_expansion: bool = Field(
        default=False,
        description="Whether to optionally query NVIDIA for additional search phrase expansion",
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of ranked top matches to return in response",
    )
    minimum_score: float = Field(
        default=55.0,
        ge=0.0,
        le=100.0,
        description="Minimum match score required to appear in top matches",
    )


class DiscoveredJobItem(BaseModel):
    """Normalized, validated, and persisted job item returned in profile discovery response."""

    id: UUID = Field(..., description="Persisted job UUID in database")
    source: str = Field(..., description="Job platform (indeed, glassdoor)")
    external_id: str = Field(..., description="Platform external identifier")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Hiring company")
    location: str = Field(..., description="Normalized job location")
    url: str = Field(..., description="Direct, canonical job listing URL")


class ProfileJobSearchResponse(BaseModel):
    """Comprehensive response for profile-driven job discovery and ranking."""

    status: str = Field(default="success", description="Execution status: success, partial_success, failed")
    search_plan: SearchPlanSummary = Field(..., description="Generated search plan summary")
    discovery: DiscoveryStats = Field(..., description="Discovery and deduplication metrics with query results")
    jobs: List[DiscoveredJobItem] = Field(default_factory=list, description="List of all unique discovered jobs with canonical URLs")
    matching: Optional[MatchingStats] = Field(default=None, description="Match score tier breakdown")
    top_matches: List[MatchedJobItem] = Field(default_factory=list, description="Ranked top matched jobs with URLs")
    total_execution_time_ms: Optional[int] = Field(default=None, description="Total pipeline execution duration in milliseconds")
