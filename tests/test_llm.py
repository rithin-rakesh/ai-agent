"""Unit tests for NVIDIA LLM Client, Embeddings, Similarity, Reasoning, and Cache."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.main import app
from app.config.settings import Settings
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
from app.llm.reasoning import NVIDIAReasoningProvider, _clean_json_response
from app.llm.similarity import calculate_semantic_similarity, cosine_similarity
from app.models.job import Job
from app.models.llm import LLMHealthResponse, ReasoningOutput
from app.models.match import MatchBreakdown, MatchResult
from app.models.profile import (
    CandidateProfileData,
    CareerPreferences,
    EducationItem,
    ExperienceItem,
    PersonalDetails,
    SkillItem,
)


@pytest.fixture
def dummy_settings():
    """Settings fixture with mock NVIDIA API key."""
    return Settings(
        NVIDIA_API_KEY=SecretStr("nvapi-test-dummy-key-12345"),
        NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1/",
        NVIDIA_REASONING_MODEL="nvidia/nemotron-3.5-lightning-30b-a3b",
        NVIDIA_EMBEDDING_MODEL="nvidia/nemotron-3-embed-1b",
        NVIDIA_REQUEST_TIMEOUT_SECONDS=10,
    )


@pytest.fixture
def sample_profile_data():
    return CandidateProfileData(
        personal=PersonalDetails(
            name="Alice Smith",
            email="alice@example.com",
            phone="+1-555-123-4567",  # Must be excluded from embedding text
            location="Kochi, Kerala",
        ),
        career=CareerPreferences(
            experience_years=3.0,
            preferred_roles=["AI Engineer", "ML Engineer"],
            preferred_locations=["Kochi", "Remote"],
            remote_preference="any",
            job_type="full-time",
            education_degree="Bachelor of Technology",
            education_field="Computer Science",
        ),
        skills=[
            SkillItem(skill="Python", importance="critical", years_experience=3.0),
            SkillItem(skill="PyTorch", importance="critical", years_experience=2.5),
            SkillItem(skill="FastAPI", importance="high", years_experience=2.0),
        ],
        experience=[
            ExperienceItem(
                company="Tech Corp",
                title="Machine Learning Engineer",
                responsibilities="Developed PyTorch models and FastAPI services.",
                skills_used=["Python", "PyTorch", "FastAPI"],
            )
        ],
        education=[
            EducationItem(
                degree="Bachelor of Technology",
                field="Computer Science",
                institution="State University",
            )
        ],
    )


@pytest.fixture
def sample_job_model():
    return Job(
        id=uuid4(),
        source="linkedin",
        external_id="job-12345",
        created_at=datetime.utcnow(),
        title="AI Engineer",
        company="NextGen AI Labs",
        location="Kochi, Kerala",
        remote=True,
        job_type="full-time",
        experience_text="2-5 years",
        description="Looking for an AI Engineer proficient in Python, PyTorch, and microservices architectures.",
    )


# ---------------------------------------------------------------------------
# 1. Configuration & Secret Masking Tests
# ---------------------------------------------------------------------------


def test_nvidia_settings_configuration(dummy_settings):
    """Test URL normalization and secret retrieval."""
    assert dummy_settings.NVIDIA_BASE_URL == "https://integrate.api.nvidia.com/v1"
    assert dummy_settings.get_nvidia_api_key() == "nvapi-test-dummy-key-12345"

    # Ensure representation masks the key
    repr_str = str(dummy_settings.NVIDIA_API_KEY)
    assert "nvapi-test-dummy-key-12345" not in repr_str
    assert "**********" in repr_str or "SecretStr" in repr_str


def test_nvidia_client_missing_key():
    """Client should raise NVIDIAAuthError if key is missing."""
    empty_settings = Settings(NVIDIA_API_KEY=None)
    client = NVIDIAClient(empty_settings)
    assert client.is_configured is False

    with pytest.raises(NVIDIAAuthError, match="NVIDIA_API_KEY is not configured"):
        client.post("/embeddings", {})


# ---------------------------------------------------------------------------
# 2. Similarity & Normalization Tests
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors():
    vec_a = [0.5, 0.5, 0.5, 0.5]
    vec_b = [0.5, 0.5, 0.5, 0.5]
    assert cosine_similarity(vec_a, vec_b) == pytest.approx(1.0, abs=1e-5)
    assert calculate_semantic_similarity(vec_a, vec_b) == 100.0


def test_cosine_similarity_orthogonal_vectors():
    vec_a = [1.0, 0.0]
    vec_b = [0.0, 1.0]
    assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0, abs=1e-5)
    assert calculate_semantic_similarity(vec_a, vec_b) == 0.0


def test_cosine_similarity_opposite_vectors():
    vec_a = [1.0, 0.0]
    vec_b = [-1.0, 0.0]
    assert cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0, abs=1e-5)
    assert calculate_semantic_similarity(vec_a, vec_b) == 0.0


def test_cosine_similarity_zero_vector_guard():
    vec_a = [0.0, 0.0, 0.0]
    vec_b = [1.0, 2.0, 3.0]
    assert cosine_similarity(vec_a, vec_b) == 0.0
    assert calculate_semantic_similarity(vec_a, vec_b) == 0.0


def test_cosine_similarity_mismatched_dimensions():
    vec_a = [1.0, 2.0]
    vec_b = [1.0, 2.0, 3.0]
    assert cosine_similarity(vec_a, vec_b) == 0.0


# ---------------------------------------------------------------------------
# 3. Privacy-Safe Text Templates
# ---------------------------------------------------------------------------


def test_build_candidate_embedding_text_excludes_pii(sample_profile_data):
    """Ensure candidate text contains technical information and excludes contact details."""
    text = build_candidate_embedding_text(sample_profile_data)
    assert "AI Engineer" in text
    assert "Python" in text
    assert "PyTorch" in text
    assert "3.0 years" in text

    # STRICT PII EXCLUSION:
    assert "alice@example.com" not in text
    assert "+1-555-123-4567" not in text


def test_build_job_embedding_text(sample_job_model):
    """Ensure job text formats title, company, and description."""
    text = build_job_embedding_text(sample_job_model)
    assert "AI Engineer" in text
    assert "NextGen AI Labs" in text
    assert "Remote" in text
    assert "Python, PyTorch" in text


# ---------------------------------------------------------------------------
# 4. Semantic Cache Tests
# ---------------------------------------------------------------------------


def test_semantic_cache_hit_and_miss():
    cache = SemanticCache()
    assert cache.stats["hits"] == 0
    assert cache.stats["misses"] == 0

    text = "AI Engineer with Python skills"
    model = "nvidia/nemotron-3-embed-1b"
    dummy_vec = [0.1, 0.2, 0.3]

    # Miss
    assert cache.get_embedding(text, model, "query") is None
    assert cache.stats["misses"] == 1

    # Store
    cache.set_embedding(text, model, "query", dummy_vec)

    # Hit
    cached = cache.get_embedding(text, model, "query")
    assert cached == dummy_vec
    assert cache.stats["hits"] == 1

    # Different input_type misses
    assert cache.get_embedding(text, model, "passage") is None
    assert cache.stats["misses"] == 2

    # Clear
    cache.clear()
    assert cache.stats["embedding_entries"] == 0


# ---------------------------------------------------------------------------
# 5. Embedding Provider Tests
# ---------------------------------------------------------------------------


def test_embedding_provider_query_and_passage(dummy_settings):
    mock_client = MagicMock(spec=NVIDIAClient)
    mock_client.post.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]
    }

    provider = NVIDIAEmbeddingProvider(client=mock_client, settings=dummy_settings)

    # 1. Query embedding
    vec = provider.get_embedding("Find AI jobs", input_type="query")
    assert vec == [0.1, 0.2, 0.3, 0.4]
    mock_client.post.assert_called_with(
        "/embeddings",
        {
            "model": "nvidia/nemotron-3-embed-1b",
            "input": ["Find AI jobs"],
            "input_type": "query",
        },
    )

    # 2. Passage embedding
    vec_job = provider.get_embedding("Job description text", input_type="passage")
    assert vec_job == [0.1, 0.2, 0.3, 0.4]
    mock_client.post.assert_called_with(
        "/embeddings",
        {
            "model": "nvidia/nemotron-3-embed-1b",
            "input": ["Job description text"],
            "input_type": "passage",
        },
    )


def test_embedding_provider_invalid_input_type(dummy_settings):
    mock_client = MagicMock(spec=NVIDIAClient)
    provider = NVIDIAEmbeddingProvider(client=mock_client, settings=dummy_settings)

    with pytest.raises(ValueError, match="Invalid input_type"):
        provider.get_embedding("text", input_type="invalid_type")


def test_embedding_provider_malformed_response(dummy_settings):
    mock_client = MagicMock(spec=NVIDIAClient)
    mock_client.post.return_value = {"data": []}
    provider = NVIDIAEmbeddingProvider(client=mock_client, settings=dummy_settings)

    with pytest.raises(ValueError, match="missing data array"):
        provider.get_embedding("text", input_type="query")


# ---------------------------------------------------------------------------
# 6. Reasoning Provider Tests
# ---------------------------------------------------------------------------


def test_clean_json_response_strip_markdown():
    raw_markdown = "```json\n{\n  \"semantic_score\": 88.5,\n  \"relevance\": \"high\"\n}\n```"
    cleaned = _clean_json_response(raw_markdown)
    assert cleaned == '{\n  "semantic_score": 88.5,\n  "relevance": "high"\n}'


def test_reasoning_provider_evaluation(dummy_settings, sample_profile_data, sample_job_model):
    mock_client = MagicMock(spec=NVIDIAClient)
    mock_client.post.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "semantic_score": 85.0,
                            "relevance": "high",
                            "strengths": ["Strong PyTorch alignment", "Experience meets requirement"],
                            "concerns": ["Job mentions microservices architectures"],
                            "recommendation": "strong_match",
                        }
                    )
                }
            }
        ]
    }

    dummy_det_result = MatchResult(
        job_id=sample_job_model.id,
        profile_id=sample_profile_data.personal.id if hasattr(sample_profile_data.personal, "id") else sample_job_model.id,
        final_score=78.0,
        decision="review",
        breakdown=MatchBreakdown(skills=80.0, title=90.0, experience=75.0, location=100.0),
        matched_skills=["Python", "PyTorch"],
        missing_skills=["Microservices"],
    )

    provider = NVIDIAReasoningProvider(client=mock_client, settings=dummy_settings)
    eval_out = provider.evaluate_match(sample_job_model, sample_profile_data, dummy_det_result)

    assert isinstance(eval_out, ReasoningOutput)
    assert eval_out.semantic_score == 85.0
    assert eval_out.relevance == "high"
    assert "Strong PyTorch alignment" in eval_out.strengths
    assert eval_out.recommendation == "strong_match"


# ---------------------------------------------------------------------------
# 7. FastAPI LLM Health Endpoint
# ---------------------------------------------------------------------------


def test_api_get_llm_health(dummy_settings):
    from app.config.settings import get_settings

    client = TestClient(app)
    app.dependency_overrides[get_settings] = lambda: dummy_settings
    try:
        response = client.get("/llm/health")
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "nvidia"
        assert data["configured"] is True
        assert data["status"] == "available"
        assert data["embedding_model"] == "nvidia/nemotron-3-embed-1b"
        assert data["reasoning_model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"

        # Verify no secret leaked in response
        assert "nvapi-test-dummy-key" not in json.dumps(data)
    finally:
        app.dependency_overrides.pop(get_settings, None)
