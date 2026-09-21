"""Deterministic Match Engine.

Orchestrates category-by-category evaluations (Skills, Title, Experience, Location,
Salary, Education, Job Type), computes weighted composite scores, derives decisions,
and generates structured MatchResult models.
"""

import logging
from typing import Optional
from uuid import UUID

from app.matching.education_matcher import EducationMatcher
from app.matching.experience_matcher import ExperienceMatcher
from app.matching.explanation import ExplanationGenerator
from app.matching.job_type_matcher import JobTypeMatcher
from app.matching.location_matcher import LocationMatcher
from app.matching.salary_matcher import SalaryMatcher
from app.matching.scorer import ScoringConfig
from app.matching.skill_matcher import SkillMatcher
from app.matching.title_matcher import TitleMatcher
from app.models.job import Job
from app.models.match import MatchBreakdown, MatchResult
from app.models.profile import CandidateProfileData

logger = logging.getLogger(__name__)


class DeterministicMatchEngine:
    """Core deterministic rule-based matching engine."""

    def __init__(
        self,
        config: Optional[ScoringConfig] = None,
        skill_matcher: Optional[SkillMatcher] = None,
        title_matcher: Optional[TitleMatcher] = None,
        experience_matcher: Optional[ExperienceMatcher] = None,
        location_matcher: Optional[LocationMatcher] = None,
        salary_matcher: Optional[SalaryMatcher] = None,
        education_matcher: Optional[EducationMatcher] = None,
        job_type_matcher: Optional[JobTypeMatcher] = None,
        explanation_generator: Optional[ExplanationGenerator] = None,
    ) -> None:
        self.config = config or ScoringConfig()
        self.skill_matcher = skill_matcher or SkillMatcher()
        self.title_matcher = title_matcher or TitleMatcher()
        self.experience_matcher = experience_matcher or ExperienceMatcher()
        self.location_matcher = location_matcher or LocationMatcher()
        self.salary_matcher = salary_matcher or SalaryMatcher()
        self.education_matcher = education_matcher or EducationMatcher()
        self.job_type_matcher = job_type_matcher or JobTypeMatcher()
        self.explanation_generator = explanation_generator or ExplanationGenerator()

    @property
    def scoring_config(self) -> ScoringConfig:
        """Alias property for engine scoring configuration."""
        return self.config

    def evaluate_match(
        self,
        job: Job,
        profile_data: CandidateProfileData,
        profile_id: UUID,
    ) -> MatchResult:
        """Run all deterministic sub-matchers and return a complete MatchResult.

        Guaranteed to produce the exact same score for the same (job, profile_data) input.
        """
        # 1. Skills Matching
        skills_score, matched_skills, missing_skills, skill_diag = self.skill_matcher.match(
            job, profile_data
        )

        # 2. Title Matching
        title_score, matched_role = self.title_matcher.match(
            job.title, profile_data.career.preferred_roles
        )

        # 3. Experience Matching
        exp_score, min_years, max_years, exp_status = self.experience_matcher.match(
            candidate_years=profile_data.career.experience_years,
            job_experience_text=job.experience_text,
            job_description=job.description,
        )

        # 4. Location Matching
        loc_score, loc_status = self.location_matcher.match(
            job_location=job.location,
            job_is_remote=job.remote,
            candidate_location=profile_data.personal.location,
            preferred_locations=profile_data.career.preferred_locations,
            remote_preference=profile_data.career.remote_preference,
        )

        # 5. Education Matching
        combined_text = f"{job.title}\n{job.description or ''}"
        edu_score, edu_status = self.education_matcher.match(
            job_text=combined_text,
            candidate_degree=profile_data.career.education_degree,
            candidate_field=profile_data.career.education_field,
        )

        # 6. Salary Matching
        salary_score, is_salary_compat, salary_status = self.salary_matcher.match(
            job_salary_min=float(job.salary_min) if job.salary_min is not None else None,
            job_salary_max=float(job.salary_max) if job.salary_max is not None else None,
            candidate_salary_min=profile_data.career.salary_min,
            candidate_salary_max=profile_data.career.salary_max,
        )

        # 7. Job Type Matching
        job_type_score, job_type_status = self.job_type_matcher.match(
            job_type=job.job_type,
            candidate_job_type=profile_data.career.job_type,
        )

        # Gate 1: Extreme Experience Mismatch Gate
        exp_mismatch_fatal = False
        if min_years is not None:
            deficit = min_years - profile_data.career.experience_years
            if deficit > 4.0 or (profile_data.career.experience_years <= 3.0 and min_years >= 8.0) or (min_years >= 10.0 and profile_data.career.experience_years < 6.0):
                exp_mismatch_fatal = True

        # Gate 2: Severe Title / Domain Mismatch Gate
        title_mismatch_fatal = False
        if profile_data.career.preferred_roles:
            domain_relevant = self.title_matcher.is_domain_relevant(
                job.title, profile_data.career.preferred_roles
            )
            if not domain_relevant and title_score < 20.0:
                title_mismatch_fatal = True

        # Gate 3: Skill Mismatch Gate (Zero skill match)
        skill_mismatch_fatal = False
        if skills_score < 15.0 and len(matched_skills) == 0:
            skill_mismatch_fatal = True

        # Build Breakdown (unavailable fields are None and will have weight redistributed)
        breakdown = MatchBreakdown(
            skills=skills_score,
            title=title_score,
            experience=exp_score,
            location=loc_score,
            education=edu_score,
            salary=salary_score,
            job_type=job_type_score,
        )

        # Calculate Final Composite Score and Decision Tier with weight redistribution
        final_score, decision = self.config.calculate_final_score(breakdown)

        # Enforce Hard Qualification Gates
        if exp_mismatch_fatal:
            decision = "reject"
            final_score = min(final_score, 45.0)
        elif title_mismatch_fatal:
            decision = "reject"
            final_score = min(final_score, 40.0)
        elif skill_mismatch_fatal:
            decision = "reject"
            final_score = min(final_score, 35.0)

        # Generate Human-Readable Explanations
        summary_reason, reasons, warnings = self.explanation_generator.generate(
            final_score=final_score,
            decision=decision,
            breakdown=breakdown,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            job_title=job.title,
            matched_role=matched_role or (profile_data.career.preferred_roles[0] if profile_data.career.preferred_roles else ""),
            exp_status=exp_status,
            min_years=min_years,
            candidate_years=profile_data.career.experience_years,
            loc_status=loc_status,
            salary_status=salary_status,
        )

        if exp_mismatch_fatal:
            warnings.insert(0, f"Extreme experience mismatch: job requires {min_years}+ years (candidate has {profile_data.career.experience_years} years)")
        if title_mismatch_fatal:
            warnings.insert(0, f"Severe title/domain mismatch: job title '{job.title}' does not align with preferred roles")

        result = MatchResult(
            job_id=job.id,
            profile_id=profile_id,
            final_score=final_score,
            decision=decision,
            breakdown=breakdown,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            reasons=reasons,
            warnings=warnings,
        )

        logger.debug(
            "Evaluated job '%s' (ID: %s) -> Final Score: %.2f (%s)",
            job.title,
            job.id,
            final_score,
            decision,
        )
        return result
