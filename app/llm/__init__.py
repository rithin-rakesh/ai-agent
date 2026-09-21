"""NVIDIA LLM and Semantic Matching Package."""

from app.llm.cache import SemanticCache
from app.llm.embeddings import (
    NVIDIAEmbeddingProvider,
    build_candidate_embedding_text,
    build_job_embedding_text,
)
from app.llm.nvidia_client import (
    NVIDIAAuthError,
    NVIDIAClient,
    NVIDIAClientError,
    NVIDIAModelNotFoundError,
)
from app.llm.reasoning import NVIDIAReasoningProvider
from app.llm.similarity import (
    calculate_semantic_similarity,
    cosine_similarity,
)

__all__ = [
    "NVIDIAClient",
    "NVIDIAClientError",
    "NVIDIAAuthError",
    "NVIDIAModelNotFoundError",
    "NVIDIAEmbeddingProvider",
    "build_candidate_embedding_text",
    "build_job_embedding_text",
    "NVIDIAReasoningProvider",
    "SemanticCache",
    "cosine_similarity",
    "calculate_semantic_similarity",
]
