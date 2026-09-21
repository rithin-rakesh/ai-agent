"""Pydantic models for Job Match entity and Deterministic Match Engine results."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Deterministic Match Engine Models
# ---------------------------------------------------------------------------


class MatchBreakdown(BaseModel):
    """Detailed category-by-category score breakdown (0-100 scale per category, or None if unavailable)."""

    skills: float = Field(default=0.0, ge=0.0, le=100.0, description="Skill overlap score")
    title: float = Field(default=0.0, ge=0.0, le=100.0, description="Job title similarity score")
    experience: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Years of experience compatibility score")
    location: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Location & remote compatibility score")
    education: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Education requirement score")
    salary: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Salary range overlap score")
    job_type: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Job type alignment score")


class MatchResult(BaseModel):
    """Structured result of evaluating a single job against a candidate profile."""

    job_id: UUID = Field(..., description="Referenced job ID")
    profile_id: UUID = Field(..., description="Referenced user profile ID")
    final_score: float = Field(..., ge=0.0, le=100.0, description="Weighted composite match score (0-100)")
    decision: str = Field(..., description="Decision: excellent, strong_match, review, low_match, reject")
    breakdown: MatchBreakdown = Field(..., description="Category sub-scores")
    deterministic_score: Optional[float] = Field(default=None, description="Pure deterministic score before semantic weighting")
    embedding_score: Optional[float] = Field(default=None, description="Semantic embedding cosine similarity (0-100)")
    llm_score: Optional[float] = Field(default=None, description="NVIDIA LLM reasoning score (0-100)")
    matched_skills: List[str] = Field(default_factory=list, description="Skills present in both job and candidate")
    missing_skills: List[str] = Field(default_factory=list, description="Job skills missing from candidate")
    reasons: List[str] = Field(default_factory=list, description="Natural language explanations supporting the score")
    warnings: List[str] = Field(default_factory=list, description="Caveats or potential mismatches detected")
    semantic_evaluation: Optional[Dict[str, Any]] = Field(default=None, description="Detailed NVIDIA semantic evaluation")

    # Helpers mapping to database schema fields
    @property
    def skill_score(self) -> float:
        return self.breakdown.skills

    @property
    def title_score(self) -> float:
        return self.breakdown.title

    @property
    def experience_score(self) -> Optional[float]:
        return self.breakdown.experience

    @property
    def location_score(self) -> Optional[float]:
        return self.breakdown.location

    @property
    def salary_score(self) -> Optional[float]:
        return self.breakdown.salary


class MatchRunRequest(BaseModel):
    """Request payload to trigger matching against stored jobs."""

    profile_id: Optional[UUID] = Field(default=None, description="Optional profile ID. If omitted, active profile is used.")
    limit: Optional[int] = Field(default=100, ge=1, le=1000, description="Maximum number of jobs to evaluate")
    force_recalculate: bool = Field(default=False, description="Whether to recompute matches for already evaluated jobs")
    use_semantic: bool = Field(default=True, description="Whether to apply NVIDIA semantic embedding similarity & reasoning")


class MatchBatchResult(BaseModel):
    """Summary of batch matching execution."""

    profile_id: UUID
    jobs_processed: int = Field(default=0, description="Total jobs evaluated")
    matches_created: int = Field(default=0, description="New match records inserted")
    matches_updated: int = Field(default=0, description="Existing match records updated")
    top_matches: List[Dict[str, Any]] = Field(default_factory=list, description="Top ranked job matches with job details")


# ---------------------------------------------------------------------------
# Database Models (Supabase 'job_matches' table)
# ---------------------------------------------------------------------------


class JobMatchBase(BaseModel):
    """Base schema for a job match record in Supabase."""

    job_id: UUID = Field(..., description="Referenced job ID")
    profile_id: UUID = Field(..., description="Referenced user profile ID")
    match_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Overall match score (0-100)")
    skill_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Skill alignment score")
    title_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Title alignment score")
    experience_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Experience alignment score")
    location_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Location match score")
    salary_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Salary match score")
    llm_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="LLM qualitative evaluation score (populated in Phase 4)")
    reason: Optional[str] = Field(default=None, description="Explanation or reasoning behind the match evaluation")
    decision: Optional[str] = Field(default=None, description="Match decision: excellent, strong_match, review, low_match, reject")


class JobMatchCreate(JobMatchBase):
    """Schema for creating or upserting a job match entry."""

    pass


class JobMatch(JobMatchBase):
    """Complete JobMatch model with database identifiers and timestamps."""

    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
