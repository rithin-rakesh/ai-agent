"""Pydantic Models for Multi-Agent Job Application Platform (Phase 5.5).

Defines platform-neutral orchestration models, run states, application tasks, and API schemas.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field


class AgentRunStatus(str, Enum):
    """Lifecycle status states for an Agent Run."""

    CREATED = "CREATED"
    DISCOVERING = "DISCOVERING"
    MATCHING = "MATCHING"
    BUILDING_QUEUE = "BUILDING_QUEUE"
    APPLYING = "APPLYING"
    PAUSED_MANUAL_ACTION = "PAUSED_MANUAL_ACTION"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_ERRORS = "COMPLETED_WITH_ERRORS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class OrchestratorState(str, Enum):
    """Explicit 17-state lifecycle states for the Autonomous Job Application Orchestrator (Phase 6.0)."""

    IDLE = "IDLE"
    DISCOVERING = "DISCOVERING"
    DISCOVERY_COMPLETE = "DISCOVERY_COMPLETE"
    MATCHING = "MATCHING"
    QUEUE_READY = "QUEUE_READY"
    SELECTING_JOB = "SELECTING_JOB"
    JOB_SELECTED = "JOB_SELECTED"
    APPLICATION_STARTING = "APPLICATION_STARTING"
    APPLICATION_IN_PROGRESS = "APPLICATION_IN_PROGRESS"
    WAITING_FOR_HUMAN_INPUT = "WAITING_FOR_HUMAN_INPUT"
    SUBMISSION_READY = "SUBMISSION_READY"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    ALREADY_APPLIED = "ALREADY_APPLIED"
    JOB_UNAVAILABLE = "JOB_UNAVAILABLE"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class FailureClassification(str, Enum):
    """Failure categorization for agent control plane policy routing."""

    TRANSIENT_FAILURE = "TRANSIENT_FAILURE"
    PERMANENT_FAILURE = "PERMANENT_FAILURE"
    USER_ACTION_REQUIRED = "USER_ACTION_REQUIRED"
    ALREADY_APPLIED = "ALREADY_APPLIED"
    JOB_UNAVAILABLE = "JOB_UNAVAILABLE"
    EXTERNAL_APPLY = "EXTERNAL_APPLY"
    STALE_JOB = "STALE_JOB"


class QueueStatus(str, Enum):
    """Status states for items in the autonomous application queue."""

    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    SUBMISSION_READY = "SUBMISSION_READY"
    SUBMITTED = "SUBMITTED"
    SKIPPED = "SKIPPED"
    ALREADY_APPLIED = "ALREADY_APPLIED"
    FAILED = "FAILED"
    NEEDS_USER_INPUT = "NEEDS_USER_INPUT"
    SKIPPED_STALE_JOB = "SKIPPED_STALE_JOB"
    SKIPPED_DAILY_LIMIT = "SKIPPED_DAILY_LIMIT"


class QueueItem(BaseModel):
    """Platform-neutral representation of a job application task in the autonomous queue."""

    job_id: UUID = Field(..., description="Internal Job primary key UUID")
    source: str = Field(..., description="Target platform (indeed, glassdoor)")
    external_id: Optional[str] = Field(default=None, description="External platform listing ID")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Employer company name")
    location: Optional[str] = Field(default=None, description="Job location")
    url: str = Field(..., description="Direct job listing URL")
    match_score: float = Field(..., description="Calculated match score (0-100)")
    status: QueueStatus = Field(default=QueueStatus.QUEUED, description="Queue item status")
    priority: int = Field(default=0, description="Priority ranking")
    decision: Optional[str] = Field(default=None, description="Match tier decision (e.g. strong_match, review)")
    attempt_count: int = Field(default=0, description="Number of application attempts")
    last_attempt_at: Optional[datetime] = Field(default=None, description="Timestamp of most recent attempt")
    error: Optional[str] = Field(default=None, description="Error message if failed or blocked")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Platform diagnostics")
    posted_at: Optional[datetime] = Field(default=None, description="Original posting timestamp if known")
    failure_class: Optional[str] = Field(default=None, description="Classification of failure if applicable")

    model_config = ConfigDict(from_attributes=True)


class ApplicationTask(BaseModel):
    """Platform-neutral representation of a job application task in the queue."""

    job_id: UUID = Field(..., description="Internal Job unique identifier")
    external_job_id: Optional[str] = Field(default=None, description="External job ID (e.g. Indeed jk)")
    platform: str = Field(default="indeed", description="Target job platform")
    url: str = Field(..., description="Direct job application posting URL")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Employer company name")
    location: Optional[str] = Field(default=None, description="Job location")
    match_score: float = Field(..., description="Calculated matching score (0-100)")
    match_decision: str = Field(..., description="Match decision tier (e.g. excellent, strong_match, review)")
    profile_id: Optional[UUID] = Field(default=None, description="Candidate profile ID")
    status: str = Field(default="PENDING", description="Task status (PENDING, APPLIED, SKIPPED, BLOCKED, FAILED)")
    attempt_count: int = Field(default=0, description="Number of application attempts")
    last_attempt_at: Optional[datetime] = Field(default=None, description="Timestamp of most recent attempt")

    model_config = ConfigDict(from_attributes=True)


class ApplicationTaskResult(BaseModel):
    """Result record of an attempted application task."""

    job_id: UUID = Field(..., description="Job identifier")
    external_job_id: Optional[str] = Field(default=None, description="External job key (e.g. Indeed jk)")
    platform: str = Field(default="indeed", description="Target platform")
    url: str = Field(..., description="Job URL")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    match_score: float = Field(..., description="Final match score")
    match_decision: str = Field(..., description="Decision tier")
    status: str = Field(..., description="Outcome: success, blocked, skipped, failed, manual_action_required")
    automation_state: Optional[str] = Field(default=None, description="Detailed automation state (e.g. APPLICATION_SUBMITTED)")
    attempted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Attempt timestamp")
    submitted_at: Optional[datetime] = Field(default=None, description="Timestamp when confirmed submitted")
    failure_reason: Optional[str] = Field(default=None, description="Reason if blocked or failed")
    diagnostics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Diagnostic data or coordinates")

    model_config = ConfigDict(from_attributes=True)


class IndeedAgentRunRequest(BaseModel):
    """Request payload to initiate an Indeed Job Application Agent run."""

    max_applications: int = Field(default=3, ge=1, le=50, description="Maximum number of application attempts")
    results_per_query: int = Field(default=15, ge=1, le=100, description="Jobs fetched per search query")
    hours_old: int = Field(default=336, ge=1, le=720, description="Filter for recently posted jobs (in hours)")
    max_queries: int = Field(default=8, ge=1, le=20, description="Maximum search queries generated from profile")
    minimum_score: float = Field(default=70.0, ge=0.0, le=100.0, description="Minimum match score required for eligibility")
    allowed_decisions: List[str] = Field(
        default=["excellent", "strong_match", "review"],
        description="Allowed match decision tiers eligible for queue entry",
    )
    use_semantic: bool = False
    use_nvidia_expansion: bool = False
    submit: bool = Field(default=True, description="If False, dry-run mode (does not perform final submit)")
    profile_id: Optional[UUID] = Field(default=None, description="Optional profile ID. If omitted, uses active profile.")


class AgentRun(BaseModel):
    """Full domain model for an autonomous agent execution run."""

    run_id: UUID = Field(default_factory=uuid4, description="Unique agent run UUID")
    platform: str = Field(default="indeed", description="Platform identifier")
    profile_id: Optional[UUID] = Field(default=None, description="Candidate profile ID")
    status: AgentRunStatus = Field(default=AgentRunStatus.CREATED, description="Current run lifecycle state")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Run start time")
    completed_at: Optional[datetime] = Field(default=None, description="Run completion time")
    current_job_id: Optional[UUID] = Field(default=None, description="Currently active or paused job ID")
    jobs_discovered: int = Field(default=0, description="Total jobs discovered from queries")
    jobs_scored: int = Field(default=0, description="Total jobs scored by matching engine")
    jobs_eligible: int = Field(default=0, description="Jobs passing score & decision thresholds")
    jobs_attempted: int = Field(default=0, description="Total jobs where application was attempted")
    applications_submitted: int = Field(default=0, description="Total applications successfully confirmed submitted")
    jobs_skipped: int = Field(default=0, description="Jobs skipped (duplicate, already applied, external)")
    jobs_already_applied: int = Field(default=0, description="Total jobs skipped because already applied")
    jobs_ineligible: int = Field(default=0, description="Total jobs found not application-eligible")
    jobs_external_apply: int = Field(default=0, description="Total jobs skipped because external apply")
    jobs_expired: int = Field(default=0, description="Total jobs skipped because expired")
    jobs_unavailable: int = Field(default=0, description="Total jobs skipped because unavailable")
    jobs_stale: int = Field(default=0, description="Total jobs skipped because posted_at exceeded max_job_age_days")
    jobs_application_eligible: int = Field(default=0, description="Total jobs confirmed application-eligible")
    jobs_failed: int = Field(default=0, description="Jobs failed due to automation or focus issues")
    manual_action_required: bool = Field(default=False, description="True if run paused waiting for user action")
    pause_reason: Optional[str] = Field(default=None, description="Reason why run is paused")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Configuration parameters used for run")
    limits_hit: Dict[str, Any] = Field(default_factory=dict, description="Details of any daily or platform limits reached")
    failure_classification: Dict[str, int] = Field(default_factory=dict, description="Counts per failure category")
    results: List[ApplicationTaskResult] = Field(default_factory=list, description="Per-job attempt results")
    remaining_queue: List[ApplicationTask] = Field(default_factory=list, description="Remaining queued tasks if paused")

    model_config = ConfigDict(from_attributes=True)


class AgentRunSummary(BaseModel):
    """User-facing summary of an Agent Run for API responses."""

    run_id: UUID
    platform: str
    status: AgentRunStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    jobs_discovered: int
    jobs_scored: int
    jobs_eligible: int
    jobs_attempted: int
    applications_submitted: int
    jobs_skipped: int
    jobs_already_applied: int = 0
    jobs_ineligible: int = 0
    jobs_external_apply: int = 0
    jobs_expired: int = 0
    jobs_unavailable: int = 0
    jobs_stale: int = 0
    jobs_application_eligible: int = 0
    jobs_failed: int
    manual_action_required: bool = False
    pause_reason: Optional[str] = None
    current_job_id: Optional[UUID] = None
    limits_hit: Dict[str, Any] = Field(default_factory=dict)
    failure_classification: Dict[str, int] = Field(default_factory=dict)
    results: List[ApplicationTaskResult] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


# ==============================================================================
# PHASE 6.0 — AUTONOMOUS ORCHESTRATOR API SCHEMAS
# ==============================================================================

class AgentRunRequest(BaseModel):
    """Request payload to initiate a multi-job autonomous application cycle."""

    profile_id: Optional[UUID] = Field(default=None, description="Candidate profile ID (defaults to active profile)")
    minimum_match_score: float = Field(default=55.0, ge=0.0, le=100.0, description="Minimum match score required for eligibility")
    max_jobs: int = Field(default=1, ge=1, le=50, description="Maximum number of applications to process")
    max_jobs_per_platform: int = Field(default=1, ge=1, le=50, description="Maximum applications to process per platform")
    auto_submit: bool = Field(default=False, description="If True, allows final submit; if False, stops at SUBMISSION_READY")
    require_easy_apply: bool = Field(default=True, description="When True, only jobs supporting Easy Apply are application-eligible")
    max_job_age_days: int = Field(default=7, ge=1, le=365, description="Maximum age of job postings in days before considered stale")
    application_delay_seconds: int = Field(default=5, ge=0, le=300, description="Delay between consecutive job applications in seconds")
    retry_failed_jobs: bool = Field(default=False, description="Whether to retry jobs that fail with transient errors")
    allow_external_apply: bool = Field(default=False, description="Whether to allow applying to external company site jobs")
    sites: List[str] = Field(default=["indeed", "glassdoor"], description="Platforms to include in discovery")
    use_semantic: bool = Field(default=False, description="Whether to use semantic embedding matching")
    max_queries: int = Field(default=3, ge=1, le=20, description="Maximum search queries in discovery")


class AgentApplyOneRequest(BaseModel):
    """Request payload to execute a single job application through the state machine."""

    job_id: UUID = Field(..., description="Real Supabase jobs.id UUID")
    profile_id: Optional[UUID] = Field(default=None, description="Candidate profile ID")
    auto_submit: bool = Field(default=False, description="If True, allows final submit; if False, stops at SUBMISSION_READY")


class AgentResumeRequest(BaseModel):
    """Request payload to resume a paused agent run after manual action."""

    agent_run_id: UUID = Field(..., description="Active or paused agent run UUID")


class AgentSkipRequest(BaseModel):
    """Request payload to skip the current or specified job and proceed with remaining queue."""

    agent_run_id: UUID = Field(..., description="Target agent run UUID")
    job_id: Optional[UUID] = Field(default=None, description="Optional job ID to skip (defaults to current active job)")


class AgentAbortRequest(BaseModel):
    """Request payload to abort an active agent run and release all locks."""

    agent_run_id: UUID = Field(..., description="Target agent run UUID to cancel")


class SingleJobApplyResponse(BaseModel):
    """Result schema for a single job application attempt."""

    job_id: UUID
    title: str
    company: str
    platform: str
    match_score: float
    status: str = Field(..., description="Outcome: success, blocked, skipped, already_applied, failed, manual_action_required, SKIPPED_EXTERNAL_APPLY, etc.")
    automation_state: Optional[str] = None
    application_method: Optional[str] = Field(default=None, description="Application method: EASY_APPLY, EXTERNAL_APPLY, UNKNOWN")
    availability_status: Optional[str] = Field(default=None, description="Availability: AVAILABLE, EXPIRED, UNAVAILABLE, UNKNOWN")
    application_eligible: bool = Field(default=True, description="Whether job is currently eligible for automated application")
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class AgentRunResponse(BaseModel):
    """Complete response schema for an autonomous agent run cycle."""

    run_id: UUID
    platform: str = "multi"
    status: OrchestratorState
    started_at: datetime
    completed_at: Optional[datetime] = None
    jobs_discovered: int = 0
    jobs_matched: int = 0
    jobs_queued: int = 0
    jobs_attempted: int = 0
    jobs_submitted: int = 0
    jobs_skipped: int = 0
    jobs_already_applied: int = 0
    jobs_ineligible: int = 0
    jobs_external_apply: int = 0
    jobs_expired: int = 0
    jobs_unavailable: int = 0
    jobs_stale: int = 0
    jobs_application_eligible: int = 0
    jobs_failed: int = 0
    manual_actions_required: bool = False
    pause_reason: Optional[str] = None
    current_job_id: Optional[UUID] = None
    limits_hit: Dict[str, Any] = Field(default_factory=dict)
    failure_classification: Dict[str, int] = Field(default_factory=dict)
    pipeline: Dict[str, Any] = Field(default_factory=dict, description="Detailed discovery and execution pipeline diagnostics")
    queue_empty_reason: Optional[str] = Field(default=None, description="Granular reason when queue is empty")
    stop_reason: Optional[str] = Field(default=None, description="Run stopping condition (None if queue has jobs)")
    eligibility_breakdown: Dict[str, int] = Field(default_factory=dict, description="Structured diagnostics breakdown of candidate job eligibility")
    eligibility_samples: List[Dict[str, Any]] = Field(default_factory=list, description="Per-job eligibility diagnostics for up to 10 sample jobs")
    report: Optional[Dict[str, Any]] = None
    results: List[SingleJobApplyResponse] = Field(default_factory=list)
    remaining_queue: List[QueueItem] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)

