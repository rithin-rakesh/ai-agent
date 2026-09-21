"""Deterministic Job Matching Engine.

Provides multi-category rule-based matching, scoring, explanation generation,
and match persistence.
"""

from app.matching.education_matcher import EducationMatcher
from app.matching.experience_matcher import ExperienceMatcher
from app.matching.explanation import ExplanationGenerator
from app.matching.job_type_matcher import JobTypeMatcher
from app.matching.location_matcher import LocationMatcher
from app.matching.matcher import DeterministicMatchEngine
from app.matching.salary_matcher import SalaryMatcher
from app.matching.scorer import (
    CombinedScoringWeights,
    DecisionThresholds,
    ScoringConfig,
    ScoringWeights,
)
from app.matching.semantic_matcher import (
    NVIDIASemanticMatcher,
    PlaceholderSemanticMatcher,
    SemanticMatcher,
)
from app.matching.service import MatchService
from app.matching.skill_matcher import SkillMatcher
from app.matching.title_matcher import TitleMatcher

__all__ = [
    "DeterministicMatchEngine",
    "MatchService",
    "ScoringConfig",
    "ScoringWeights",
    "CombinedScoringWeights",
    "DecisionThresholds",
    "SkillMatcher",
    "TitleMatcher",
    "ExperienceMatcher",
    "LocationMatcher",
    "SalaryMatcher",
    "EducationMatcher",
    "JobTypeMatcher",
    "ExplanationGenerator",
    "SemanticMatcher",
    "PlaceholderSemanticMatcher",
    "NVIDIASemanticMatcher",
]
