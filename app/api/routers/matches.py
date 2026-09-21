"""FastAPI Router for Job Matching endpoints."""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.matching.service import MatchService
from app.models.match import (
    JobMatch,
    MatchBatchResult,
    MatchResult,
    MatchRunRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/matches", tags=["Matches"])


def get_match_service() -> MatchService:
    """Dependency injection for MatchService."""
    return MatchService()


@router.post(
    "/run",
    response_model=MatchBatchResult,
    summary="Run matching across stored jobs",
    description="Evaluates jobs stored in the database against candidate profile and persists scores in job_matches with optional NVIDIA semantic evaluation.",
)
def run_matching(
    request: MatchRunRequest,
    service: MatchService = Depends(get_match_service),
) -> MatchBatchResult:
    try:
        return service.match_jobs(
            profile_id=request.profile_id,
            limit=request.limit,
            force_recalculate=request.force_recalculate,
            use_semantic=request.use_semantic,
        )
    except Exception as exc:
        logger.error("Failed to run matching process: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Matching calculation failed: {str(exc)}",
        )


@router.get(
    "/top",
    response_model=List[Dict[str, Any]],
    summary="Get top ranked job matches",
    description="Retrieve the highest-scoring job matches joined with full job details.",
)
def get_top_matches(
    profile_id: Optional[UUID] = Query(default=None, description="Optional profile UUID"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results to return"),
    service: MatchService = Depends(get_match_service),
) -> List[Dict[str, Any]]:
    return service.get_top_matches(profile_id=profile_id, limit=limit)


@router.get(
    "",
    response_model=List[JobMatch],
    summary="List job matches",
    description="Retrieve a paginated list of job matches with optional score and decision filtering.",
)
def list_matches(
    profile_id: Optional[UUID] = Query(default=None, description="Optional profile UUID"),
    limit: int = Query(default=50, ge=1, le=200, description="Items per page"),
    offset: int = Query(default=0, ge=0, description="Items offset"),
    min_score: Optional[float] = Query(default=None, ge=0.0, le=100.0, description="Filter matches by min score"),
    decision: Optional[str] = Query(default=None, description="Filter matches by decision: excellent, strong_match, review, low_match, reject"),
    service: MatchService = Depends(get_match_service),
) -> List[JobMatch]:
    matches, _ = service.list_matches(
        profile_id=profile_id,
        limit=limit,
        offset=offset,
        min_score=min_score,
        decision=decision,
    )
    return matches


@router.get(
    "/{job_id}",
    response_model=MatchResult,
    summary="Evaluate or get match for a specific job",
    description="Computes match breakdown and natural language explanation for a specific job.",
)
def get_or_evaluate_job_match(
    job_id: UUID,
    profile_id: Optional[UUID] = Query(default=None, description="Optional profile UUID"),
    use_semantic: bool = Query(default=True, description="Whether to apply semantic matching"),
    service: MatchService = Depends(get_match_service),
) -> MatchResult:
    result = service.match_job(job_id=job_id, profile_id=profile_id, use_semantic=use_semantic)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unable to evaluate match for job '{job_id}'. Verify that the job exists in the database.",
        )
    return result
