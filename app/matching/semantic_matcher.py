"""Semantic Matcher Interface and NVIDIA Provider Implementation.

Combines embedding vector similarity (nvidia/nemotron-3-embed-1b) with LLM qualitative
reasoning to produce an in-depth evaluation of candidate-to-job fit.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional

from app.config.settings import Settings, get_settings
from app.llm.embeddings import (
    NVIDIAEmbeddingProvider,
    build_candidate_embedding_text,
    build_job_embedding_text,
)
from app.llm.reasoning import NVIDIAReasoningProvider
from app.llm.similarity import calculate_semantic_similarity
from app.models.job import Job
from app.models.llm import ReasoningOutput, SemanticMatchEvaluation
from app.models.match import MatchResult
from app.models.profile import CandidateProfileData

logger = logging.getLogger(__name__)


class SemanticMatcher(ABC):
    """Abstract interface for LLM-assisted qualitative job matching and embedding similarity."""

    async def evaluate_job_fit(
        self,
        job_title: str,
        job_description: str,
        candidate_summary: str,
        candidate_skills: List[str],
    ) -> Optional[float]:
        """Legacy helper for qualitative fit evaluation."""
        return None

    @abstractmethod
    def evaluate_semantic_match(
        self,
        job: Job,
        profile_data: CandidateProfileData,
        deterministic_result: MatchResult,
    ) -> SemanticMatchEvaluation:
        """Evaluate full semantic fit returning embeddings score, reasoning output, and diagnostics."""
        pass

    @abstractmethod
    def generate_embedding(self, text: str, input_type: str = "query") -> List[float]:
        """Generate semantic embedding vector."""
        pass


class PlaceholderSemanticMatcher(SemanticMatcher):
    """Placeholder implementation that performs zero API calls and returns empty evaluations."""

    async def evaluate_job_fit(
        self,
        job_title: str,
        job_description: str,
        candidate_summary: str,
        candidate_skills: List[str],
    ) -> Optional[float]:
        return None

    def evaluate_semantic_match(
        self,
        job: Job,
        profile_data: CandidateProfileData,
        deterministic_result: MatchResult,
    ) -> SemanticMatchEvaluation:
        return SemanticMatchEvaluation()

    async def generate_embedding(self, text: str, input_type: str = "query") -> List[float]:
        return []


class NVIDIASemanticMatcher(SemanticMatcher):
    """Production NVIDIA-powered semantic matcher utilizing Nemotron embeddings & reasoning."""

    def __init__(
        self,
        embedding_provider: Optional[NVIDIAEmbeddingProvider] = None,
        reasoning_provider: Optional[NVIDIAReasoningProvider] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_provider = embedding_provider or NVIDIAEmbeddingProvider(settings=self.settings)
        self.reasoning_provider = reasoning_provider or NVIDIAReasoningProvider(settings=self.settings)

    def generate_embedding(self, text: str, input_type: str = "query") -> List[float]:
        """Generate or retrieve a cached text embedding vector."""
        return self.embedding_provider.get_embedding(text, input_type=input_type)

    def evaluate_semantic_match(
        self,
        job: Job,
        profile_data: CandidateProfileData,
        deterministic_result: MatchResult,
    ) -> SemanticMatchEvaluation:
        """Execute full semantic evaluation pipeline: embedding similarity + reasoning model.

        Resilience guarantee:
        - If embedding generation fails, embedding_score is None, and error is recorded.
        - If reasoning model fails, reasoning_score is None, and error is recorded.
        """
        eval_result = SemanticMatchEvaluation()
        errors = []

        # 1. Embedding Similarity Step
        try:
            cand_text = build_candidate_embedding_text(profile_data)
            job_text = build_job_embedding_text(job)

            cand_vec = self.embedding_provider.get_embedding(cand_text, input_type="query")
            job_vec = self.embedding_provider.get_embedding(job_text, input_type="passage")

            eval_result.embedding_score = calculate_semantic_similarity(cand_vec, job_vec)
        except Exception as exc:
            logger.warning("NVIDIA Embedding generation failed for job %s: %s", job.id, exc)
            errors.append(f"Embedding error: {exc}")

        # 2. Reasoning Model Step
        try:
            reasoning_out: ReasoningOutput = self.reasoning_provider.evaluate_match(
                job=job,
                profile_data=profile_data,
                deterministic_result=deterministic_result,
            )
            eval_result.reasoning_score = reasoning_out.semantic_score
            eval_result.relevance = reasoning_out.relevance
            eval_result.strengths = reasoning_out.strengths
            eval_result.concerns = reasoning_out.concerns
            eval_result.recommendation = reasoning_out.recommendation
        except Exception as exc:
            logger.warning("NVIDIA Reasoning evaluation failed for job %s: %s", job.id, exc)
            errors.append(f"Reasoning error: {exc}")

        if errors:
            eval_result.error = "; ".join(errors)

        return eval_result
