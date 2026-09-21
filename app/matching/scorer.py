"""Scoring Configuration, Weights, and Thresholds for Match Engine.

Defines category weights for deterministic matching and composite weights
for Phase 4 multi-modal scoring (70% deterministic, 20% embedding, 10% reasoning).
"""

from typing import Optional, Tuple
from pydantic import BaseModel, Field

from app.models.match import MatchBreakdown


class ScoringWeights(BaseModel):
    """Configurable weights assigned to each deterministic matching category. Must sum to 100."""

    skills: float = Field(default=40.0, ge=0.0, le=100.0, description="Weight for skill match")
    title: float = Field(default=20.0, ge=0.0, le=100.0, description="Weight for title similarity")
    experience: float = Field(default=15.0, ge=0.0, le=100.0, description="Weight for experience alignment")
    location: float = Field(default=10.0, ge=0.0, le=100.0, description="Weight for location / remote match")
    education: float = Field(default=5.0, ge=0.0, le=100.0, description="Weight for education match")
    salary: float = Field(default=5.0, ge=0.0, le=100.0, description="Weight for salary compatibility")
    job_type: float = Field(default=5.0, ge=0.0, le=100.0, description="Weight for job type match")

    @property
    def total(self) -> float:
        return (
            self.skills
            + self.title
            + self.experience
            + self.location
            + self.education
            + self.salary
            + self.job_type
        )


class CombinedScoringWeights(BaseModel):
    """Weights for Phase 4 multi-modal composite score combination."""

    deterministic: float = Field(default=0.70, ge=0.0, le=1.0, description="Weight for deterministic match (70%)")
    embedding: float = Field(default=0.20, ge=0.0, le=1.0, description="Weight for embedding similarity (20%)")
    reasoning: float = Field(default=0.10, ge=0.0, le=1.0, description="Weight for LLM reasoning score (10%)")

    @property
    def total(self) -> float:
        return self.deterministic + self.embedding + self.reasoning


class DecisionThresholds(BaseModel):
    """Score boundaries mapping final score to qualitative decision categories."""

    excellent: float = Field(default=90.0, description="Score >= 90 is excellent match")
    strong_match: float = Field(default=80.0, description="Score >= 80 is strong match")
    review: float = Field(default=70.0, description="Score >= 70 warrants manual review")
    low_match: float = Field(default=60.0, description="Score >= 60 is low match")


class ScoringConfig(BaseModel):
    """Complete scoring engine configuration."""

    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    combined_weights: CombinedScoringWeights = Field(default_factory=CombinedScoringWeights)
    thresholds: DecisionThresholds = Field(default_factory=DecisionThresholds)

    def derive_decision(self, score: float) -> str:
        """Derive qualitative decision category from score."""
        if score >= self.thresholds.excellent:
            return "excellent"
        elif score >= self.thresholds.strong_match:
            return "strong_match"
        elif score >= self.thresholds.review:
            return "review"
        elif score >= self.thresholds.low_match:
            return "low_match"
        else:
            return "reject"

    def calculate_final_score(self, breakdown: MatchBreakdown) -> Tuple[float, str]:
        """Compute the weighted composite match score (0-100) and derive decision label.

        Weights for missing/unknown components (where score is None) are automatically
        excluded and redistributed across available components.
        """
        components = [
            ("skills", breakdown.skills, self.weights.skills),
            ("title", breakdown.title, self.weights.title),
            ("experience", breakdown.experience, self.weights.experience),
            ("location", breakdown.location, self.weights.location),
            ("education", breakdown.education, self.weights.education),
            ("salary", breakdown.salary, self.weights.salary),
            ("job_type", breakdown.job_type, self.weights.job_type),
        ]

        total_active_weight = 0.0
        weighted_sum = 0.0
        for name, score, weight in components:
            if score is not None and weight > 0:
                weighted_sum += score * weight
                total_active_weight += weight

        if total_active_weight <= 0:
            total_active_weight = 100.0

        final_score = round(min(100.0, max(0.0, weighted_sum / total_active_weight)), 2)
        decision = self.derive_decision(final_score)
        return final_score, decision

    def calculate_combined_score(
        self,
        deterministic_score: float,
        embedding_score: Optional[float] = None,
        reasoning_score: Optional[float] = None,
    ) -> Tuple[float, str]:
        """Compute Phase 4 multi-modal combined score with graceful fallback.

        Formula:
        - When all 3 available: det * 0.70 + emb * 0.20 + rsn * 0.10
        - When embedding only: (det * 0.70 + emb * 0.20) / 0.90
        - When reasoning only: (det * 0.70 + rsn * 0.10) / 0.80
        - When none: deterministic_score
        """
        w_det = self.combined_weights.deterministic
        w_emb = self.combined_weights.embedding
        w_rsn = self.combined_weights.reasoning

        active_weights = w_det
        weighted_sum = deterministic_score * w_det

        if embedding_score is not None:
            weighted_sum += embedding_score * w_emb
            active_weights += w_emb

        if reasoning_score is not None:
            weighted_sum += reasoning_score * w_rsn
            active_weights += w_rsn

        if active_weights <= 0:
            final_score = round(deterministic_score, 2)
        else:
            final_score = round(min(100.0, max(0.0, weighted_sum / active_weights)), 2)

        decision = self.derive_decision(final_score)
        return final_score, decision
