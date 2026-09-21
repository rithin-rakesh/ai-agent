"""Pydantic models for NVIDIA LLM and Semantic Matching responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


class LLMHealthResponse(BaseModel):
    """Health check response for NVIDIA NIM LLM provider."""

    provider: str = Field(default="nvidia", description="LLM provider name")
    configured: bool = Field(..., description="Whether NVIDIA API key is configured")
    reasoning_model: str = Field(..., description="Configured reasoning model identifier")
    embedding_model: str = Field(..., description="Configured embedding model identifier")
    base_url: str = Field(..., description="NVIDIA NIM API base URL")
    status: str = Field(..., description="Operational status: available, unconfigured, or error")


class ReasoningOutput(BaseModel):
    """Structured output expected from NVIDIA Reasoning LLM."""

    semantic_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Qualitative semantic relevance score (0.0 - 100.0)",
    )
    relevance: str = Field(
        default="medium",
        description="Relevance category: high, medium, low, none",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Key alignment strengths between candidate profile and job posting",
    )
    concerns: List[str] = Field(
        default_factory=list,
        description="Potential gaps, missing requirements, or experience mismatches",
    )
    recommendation: str = Field(
        default="review",
        description="Match recommendation: excellent, strong_match, review, low_match, reject",
    )


class SemanticMatchEvaluation(BaseModel):
    """Complete semantic evaluation outcome combining embeddings and reasoning."""

    embedding_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Cosine similarity normalized to 0-100 scale",
    )
    reasoning_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="LLM qualitative reasoning score (0-100)",
    )
    relevance: Optional[str] = Field(default=None, description="Semantic relevance level")
    strengths: List[str] = Field(default_factory=list, description="Extracted strengths")
    concerns: List[str] = Field(default_factory=list, description="Extracted concerns")
    recommendation: Optional[str] = Field(default=None, description="Recommended tier")
    raw_reasoning_text: Optional[str] = Field(default=None, description="Raw model response")
    error: Optional[str] = Field(default=None, description="Error message if semantic evaluation failed")
