"""Matching Service Orchestrator.

Orchestrates candidate profile loading, batch job evaluation, deterministic scoring,
NVIDIA semantic embedding similarity & reasoning, persistence to Supabase 'job_matches',
and ranking of evaluated job opportunities.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from app.config.settings import Settings, get_settings
from app.database.repositories.job_repository import JobRepository
from app.database.repositories.match_repository import MatchRepository
from app.matching.matcher import DeterministicMatchEngine
from app.matching.semantic_matcher import NVIDIASemanticMatcher, SemanticMatcher
from app.models.job import Job
from app.models.match import (
    JobMatch,
    JobMatchCreate,
    MatchBatchResult,
    MatchResult,
)
from app.models.profile import CandidateProfileData
from app.profile.service import ProfileService

logger = logging.getLogger(__name__)


def _format_combined_reason(
    result: MatchResult,
    relevance: Optional[str] = None,
    strengths: Optional[List[str]] = None,
    concerns: Optional[List[str]] = None,
) -> str:
    """Format a comprehensive explanation combining deterministic and semantic reasoning."""
    reason_parts: List[str] = []

    # Deterministic highlights
    if result.reasons:
        reason_parts.extend(result.reasons[:2])

    # Semantic evaluation highlights
    if relevance:
        reason_parts.append(f"Semantic Alignment: {relevance.capitalize()}")

    if strengths:
        reason_parts.append(f"Strengths: {'; '.join(strengths[:2])}")

    if concerns:
        reason_parts.append(f"Concerns: {'; '.join(concerns[:2])}")

    combined = "; ".join(reason_parts)
    if len(combined) > 450:
        combined = combined[:447] + "..."
    return combined if combined else "Evaluation completed."


class MatchService:
    """Service layer orchestrating multi-modal job matching workflows."""

    def __init__(
        self,
        engine: Optional[DeterministicMatchEngine] = None,
        semantic_matcher: Optional[SemanticMatcher] = None,
        match_repository: Optional[MatchRepository] = None,
        job_repository: Optional[JobRepository] = None,
        profile_service: Optional[ProfileService] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or DeterministicMatchEngine()
        self.semantic_matcher = semantic_matcher or NVIDIASemanticMatcher(settings=self.settings)
        self.match_repo = match_repository or MatchRepository()
        self.job_repo = job_repository or JobRepository()
        self.profile_service = profile_service or ProfileService()

    def match_job(
        self,
        job_id: UUID,
        profile_id: Optional[UUID] = None,
        use_semantic: bool = True,
    ) -> Optional[MatchResult]:
        """Evaluate a single job against the candidate profile and persist the match."""
        profile = self.profile_service.get_profile(profile_id)
        if not profile:
            logger.error("Cannot perform matching: Candidate profile not found.")
            return None

        job = self.job_repo.get_job_by_id(job_id)
        if not job:
            logger.error("Job with ID %s not found.", job_id)
            return None

        profile_data = self.profile_service.get_active_profile_data(profile.id)
        result = self.engine.evaluate_match(job, profile_data, profile.id)

        # 1. Store pure deterministic score
        result.deterministic_score = result.final_score

        # 2. Semantic Evaluation if enabled, deterministic score qualifies, and not rejected
        can_run_semantic = (
            use_semantic
            and self.settings.SEMANTIC_MATCHING_ENABLED
            and result.decision not in ("reject", "unqualified")
            and result.final_score >= self.settings.DETERMINISTIC_SEMANTIC_THRESHOLD
        )

        semantic_strengths = []
        semantic_concerns = []
        semantic_relevance = None

        if can_run_semantic:
            try:
                sem_eval = self.semantic_matcher.evaluate_semantic_match(job, profile_data, result)
                result.embedding_score = sem_eval.embedding_score
                result.llm_score = sem_eval.reasoning_score
                result.semantic_evaluation = sem_eval.model_dump()
                semantic_strengths = sem_eval.strengths
                semantic_concerns = sem_eval.concerns
                semantic_relevance = sem_eval.relevance

                # Compute Phase 4 Combined Score (70% det, 20% emb, 10% rsn)
                final_score, decision = self.engine.scoring_config.calculate_combined_score(
                    deterministic_score=result.deterministic_score,
                    embedding_score=sem_eval.embedding_score,
                    reasoning_score=sem_eval.reasoning_score,
                )
                result.final_score = final_score
                result.decision = decision
            except Exception as exc:
                logger.warning("Semantic evaluation pipeline error for job %s: %s", job.id, exc)

        reason_str = _format_combined_reason(
            result=result,
            relevance=semantic_relevance,
            strengths=semantic_strengths,
            concerns=semantic_concerns,
        )

        # Build database payload
        match_create = JobMatchCreate(
            job_id=job.id,
            profile_id=profile.id,
            match_score=result.final_score,
            skill_score=result.skill_score,
            title_score=result.title_score,
            experience_score=result.experience_score,
            location_score=result.location_score,
            salary_score=result.salary_score,
            llm_score=result.llm_score,
            reason=reason_str,
            decision=result.decision,
        )

        self.match_repo.upsert_match(match_create)
        return result

    def match_discovered_jobs(
        self,
        jobs: List[Job],
        profile_id: Optional[UUID] = None,
        use_semantic: bool = True,
    ) -> List[MatchResult]:
        """Evaluate pre-loaded Job models against candidate profile in-memory.

        Loads candidate profile and ProfileData ONCE, then iterates over jobs,
        evaluating deterministic matches, applying qualification gates, running
        semantic matching only for qualifying non-rejected candidates, and persisting to DB.
        """
        if not jobs:
            return []

        profile = self.profile_service.get_profile(profile_id)
        if not profile:
            logger.error("Cannot perform matching: Candidate profile not found.")
            return []

        active_profile_id = profile.id
        profile_data = self.profile_service.get_active_profile_data(active_profile_id)

        results: List[MatchResult] = []
        for job in jobs:
            try:
                result = self.engine.evaluate_match(job, profile_data, active_profile_id)
                result.deterministic_score = result.final_score

                can_run_semantic = (
                    use_semantic
                    and self.settings.SEMANTIC_MATCHING_ENABLED
                    and result.decision not in ("reject", "unqualified")
                    and result.final_score >= self.settings.DETERMINISTIC_SEMANTIC_THRESHOLD
                )

                semantic_strengths = []
                semantic_concerns = []
                semantic_relevance = None

                if can_run_semantic:
                    try:
                        sem_eval = self.semantic_matcher.evaluate_semantic_match(job, profile_data, result)
                        result.embedding_score = sem_eval.embedding_score
                        result.llm_score = sem_eval.reasoning_score
                        result.semantic_evaluation = sem_eval.model_dump()
                        semantic_strengths = sem_eval.strengths
                        semantic_concerns = sem_eval.concerns
                        semantic_relevance = sem_eval.relevance

                        final_score, decision = self.engine.scoring_config.calculate_combined_score(
                            deterministic_score=result.deterministic_score,
                            embedding_score=sem_eval.embedding_score,
                            reasoning_score=sem_eval.reasoning_score,
                        )
                        result.final_score = final_score
                        result.decision = decision
                    except Exception as exc:
                        logger.warning("Semantic evaluation error for job %s: %s", job.id, exc)

                reason_str = _format_combined_reason(
                    result=result,
                    relevance=semantic_relevance,
                    strengths=semantic_strengths,
                    concerns=semantic_concerns,
                )

                match_create = JobMatchCreate(
                    job_id=job.id,
                    profile_id=active_profile_id,
                    match_score=result.final_score,
                    skill_score=result.skill_score,
                    title_score=result.title_score,
                    experience_score=result.experience_score,
                    location_score=result.location_score,
                    salary_score=result.salary_score,
                    llm_score=result.llm_score,
                    reason=reason_str,
                    decision=result.decision,
                )

                self.match_repo.upsert_match(match_create)
                results.append(result)
            except Exception as exc:
                logger.warning("Error evaluating job %s in match_discovered_jobs: %s", job.id, exc)

        return results

    def match_jobs(
        self,
        profile_id: Optional[UUID] = None,
        limit: Optional[int] = 100,
        force_recalculate: bool = False,
        use_semantic: bool = True,
    ) -> MatchBatchResult:
        """Evaluate all available jobs against candidate profile, persist results, and rank top matches."""
        profile = self.profile_service.get_profile(profile_id)
        if not profile:
            raise ValueError("Candidate profile could not be found or initialized.")

        active_profile_id = profile.id
        profile_data = self.profile_service.get_active_profile_data(active_profile_id)

        # Retrieve stored jobs from database
        max_jobs = limit or 100
        jobs, _ = self.job_repo.list_jobs(limit=max_jobs, offset=0)
        logger.info(
            "Starting batch matching for profile %s across %d jobs (limit=%d, use_semantic=%s)...",
            active_profile_id,
            len(jobs),
            max_jobs,
            use_semantic,
        )

        processed = 0
        created = 0
        updated = 0

        for job in jobs:
            existing_match = self.match_repo.get_match(job.id, active_profile_id)
            if existing_match and not force_recalculate:
                processed += 1
                continue

            result = self.engine.evaluate_match(job, profile_data, active_profile_id)
            result.deterministic_score = result.final_score

            can_run_semantic = (
                use_semantic
                and self.settings.SEMANTIC_MATCHING_ENABLED
                and result.decision not in ("reject", "unqualified")
                and result.final_score >= self.settings.DETERMINISTIC_SEMANTIC_THRESHOLD
            )

            semantic_strengths = []
            semantic_concerns = []
            semantic_relevance = None

            if can_run_semantic:
                try:
                    sem_eval = self.semantic_matcher.evaluate_semantic_match(job, profile_data, result)
                    result.embedding_score = sem_eval.embedding_score
                    result.llm_score = sem_eval.reasoning_score
                    result.semantic_evaluation = sem_eval.model_dump()
                    semantic_strengths = sem_eval.strengths
                    semantic_concerns = sem_eval.concerns
                    semantic_relevance = sem_eval.relevance

                    final_score, decision = self.engine.scoring_config.calculate_combined_score(
                        deterministic_score=result.deterministic_score,
                        embedding_score=sem_eval.embedding_score,
                        reasoning_score=sem_eval.reasoning_score,
                    )
                    result.final_score = final_score
                    result.decision = decision
                except Exception as exc:
                    logger.warning("Semantic evaluation pipeline error in batch for job %s: %s", job.id, exc)

            reason_str = _format_combined_reason(
                result=result,
                relevance=semantic_relevance,
                strengths=semantic_strengths,
                concerns=semantic_concerns,
            )

            match_create = JobMatchCreate(
                job_id=job.id,
                profile_id=active_profile_id,
                match_score=result.final_score,
                skill_score=result.skill_score,
                title_score=result.title_score,
                experience_score=result.experience_score,
                location_score=result.location_score,
                salary_score=result.salary_score,
                llm_score=result.llm_score,
                reason=reason_str,
                decision=result.decision,
            )

            saved = self.match_repo.upsert_match(match_create)
            if saved:
                if existing_match:
                    updated += 1
                else:
                    created += 1
            processed += 1

        top_matches = self.match_repo.get_top_matches(active_profile_id, limit=20)

        logger.info(
            "Batch match complete: %d jobs processed, %d created, %d updated. Top score: %s",
            processed,
            created,
            updated,
            top_matches[0].get("match_score") if top_matches else "N/A",
        )

        return MatchBatchResult(
            profile_id=active_profile_id,
            jobs_processed=processed,
            matches_created=created,
            matches_updated=updated,
            top_matches=top_matches,
        )

    def get_match(self, job_id: UUID, profile_id: Optional[UUID] = None) -> Optional[JobMatch]:
        """Fetch a specific match result from database."""
        profile = self.profile_service.get_profile(profile_id)
        if not profile:
            return None
        return self.match_repo.get_match(job_id, profile.id)

    def get_top_matches(self, profile_id: Optional[UUID] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve top ranked matches joined with job details."""
        profile = self.profile_service.get_profile(profile_id)
        if not profile:
            return []
        return self.match_repo.get_top_matches(profile.id, limit=limit)

    def list_matches(
        self,
        profile_id: Optional[UUID] = None,
        limit: int = 50,
        offset: int = 0,
        min_score: Optional[float] = None,
        decision: Optional[str] = None,
    ) -> Tuple[List[JobMatch], int]:
        """List matches for active profile with pagination and score filtering."""
        profile = self.profile_service.get_profile(profile_id)
        if not profile:
            return [], 0
        return self.match_repo.list_matches(
            profile_id=profile.id,
            limit=limit,
            offset=offset,
            min_score=min_score,
            decision=decision,
        )
