"""FastAPI Router for NVIDIA LLM and Semantic Matching health/status endpoints."""

import logging
from fastapi import APIRouter, Depends
from app.config.settings import Settings, get_settings
from app.models.llm import LLMHealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM & Semantic Matching"])


@router.get(
    "/health",
    response_model=LLMHealthResponse,
    summary="Check NVIDIA NIM provider status",
    description="Returns the operational status and model identifiers for NVIDIA LLM and Embeddings without exposing any API keys.",
)
def get_llm_health(
    settings: Settings = Depends(get_settings),
) -> LLMHealthResponse:
    """Safely check NVIDIA NIM configuration status without exposing sensitive credentials."""
    api_key_configured = bool(settings.get_nvidia_api_key())
    status_str = "available" if api_key_configured else "unconfigured"

    return LLMHealthResponse(
        provider="nvidia",
        configured=api_key_configured,
        reasoning_model=settings.NVIDIA_REASONING_MODEL,
        embedding_model=settings.NVIDIA_EMBEDDING_MODEL,
        base_url=settings.NVIDIA_BASE_URL,
        status=status_str,
    )
