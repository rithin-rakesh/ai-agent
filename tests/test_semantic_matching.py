"""Unit tests for Phase 4 Semantic Matcher, Combined Scoring, and MatchService Integration."""

from unittest.mock import MagicMock, patch
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.config.settings import Settings
from app.llm.embeddings import NVIDIAEmbeddingProvider
from app.llm.reasoning import NVIDIAReasoningProvider
from app.matching.matcher import DeterministicMatchEngine
from app.matching.scorer import CombinedScoringWeights, ScoringConfig
from app.matching.semantic_matcher import NVIDIASemanticMatcher, PlaceholderSemanticMatcher
from app.matching.service import MatchService
from app.models.job import Job
from app.models.llm import ReasoningOutput, SemanticMatchEvaluation
from app.models.match import MatchBreakdown, MatchResult, MatchRunRequest
from app.models.profile import (
    CandidateProfileData,
    CareerPreferences,
    EducationItem,
    ExperienceItem,
    PersonalDetails,
    Profile,
    SkillItem,
)


@pytest.fixture
def test_profile_data():
    return CandidateProfileData(
        personal=PersonalDetails(
            name="John Doe",
            email="john@example.com",
            location="Kochi, Kerala",
        ),
        career=CareerPreferences(
            experience_years=3.0,
            preferred_roles=["AI Engineer", "Machine Learning Engineer"],
            preferred_locations=["Kochi", "Kerala"],
            remote_preference="any",
            job_type="full-time",
            education_degree="Bachelor of Technology",
            education_field="Computer Science",
        ),
        skills=[
            SkillItem(skill="Python", importance="critical", years_experience=3.0),
            SkillItem(skill="PyTorch", importance="critical", years_experience=2.5),
            SkillItem(skill="TensorFlow", importance="high", years_experience=2.0),
            SkillItem(skill="FastAPI", importance="high", years_experience=2.0),
        ],
        experience=[
            ExperienceItem(
                company="AI Labs",
                title="ML Engineer",
                responsibilities="Engineered PyTorch neural networks.",
                skills_used=["Python", "PyTorch"],
            )
        ],
        education=[
            EducationItem(
                degree="Bachelor of Technology",
                field="Computer Science",
            )
        ],
    )


from datetime import datetime


@pytest.fixture
def sample_job():
    return Job(
        id=uuid4(),
        source="linkedin",
        external_id="ext-sample-123",
        created_at=datetime.utcnow(),
        title="AI Engineer",
        company="Cognitive Tech",
        location="Kochi, Kerala, India",
        remote=False,
        job_type="full-time",
        experience_text="2-4 years",
        description="We need an AI Engineer experienced in Python, PyTorch, and deep learning architectures.",
    )


# ---------------------------------------------------------------------------
# 1. Combined Scoring Formula Tests
# ---------------------------------------------------------------------------


def test_combined_scoring_exact_math():
    """Verify 70% deterministic + 20% embedding + 10% reasoning combination."""
    config = ScoringConfig()

    # Case 1: 80 det, 90 emb, 70 rsn -> (80*0.70) + (90*0.20) + (70*0.10) = 56 + 18 + 7 = 81.0
    score, decision = config.calculate_combined_score(
        deterministic_score=80.0,
        embedding_score=90.0,
        reasoning_score=70.0,
    )
    assert score == 81.0
    assert decision == "strong_match"

    # Case 2: 95 det, 95 emb, 90 rsn -> (95*0.70) + (95*0.20) + (90*0.10) = 66.5 + 19.0 + 9.0 = 94.5
    score_high, decision_high = config.calculate_combined_score(
        deterministic_score=95.0,
        embedding_score=95.0,
        reasoning_score=90.0,
    )
    assert score_high == 94.5
    assert decision_high == "excellent"


def test_combined_scoring_fallbacks():
    config = ScoringConfig()

    # When reasoning is missing (fallback to 70/20 proportion)
    score_no_rsn, _ = config.calculate_combined_score(
        deterministic_score=90.0,
        embedding_score=90.0,
        reasoning_score=None,
    )
    assert score_no_rsn == 90.0

    # When both are missing (pure deterministic)
    score_det_only, _ = config.calculate_combined_score(
        deterministic_score=75.0,
        embedding_score=None,
        reasoning_score=None,
    )
    assert score_det_only == 75.0


# ---------------------------------------------------------------------------
# 2. Semantic Matcher Provider Tests
# ---------------------------------------------------------------------------


def test_nvidia_semantic_matcher_success(test_profile_data, sample_job):
    mock_emb = MagicMock(spec=NVIDIAEmbeddingProvider)
    # Return two identical vectors for 100% cosine similarity
    mock_emb.get_embedding.side_effect = lambda text, input_type: [0.5, 0.5, 0.5, 0.5]

    mock_rsn = MagicMock(spec=NVIDIAReasoningProvider)
    mock_rsn.evaluate_match.return_value = ReasoningOutput(
        semantic_score=88.0,
        relevance="high",
        strengths=["PyTorch alignment", "Location in Kochi"],
        concerns=[],
        recommendation="strong_match",
    )

    matcher = NVIDIASemanticMatcher(
        embedding_provider=mock_emb,
        reasoning_provider=mock_rsn,
    )

    dummy_det = MatchResult(
        job_id=sample_job.id,
        profile_id=sample_job.id,
        final_score=82.0,
        decision="strong_match",
        breakdown=MatchBreakdown(skills=90.0, title=85.0, experience=80.0, location=100.0),
    )

    eval_res = matcher.evaluate_semantic_match(sample_job, test_profile_data, dummy_det)

    assert isinstance(eval_res, SemanticMatchEvaluation)
    assert eval_res.embedding_score == 100.0
    assert eval_res.reasoning_score == 88.0
    assert eval_res.relevance == "high"
    assert "PyTorch alignment" in eval_res.strengths


def test_nvidia_semantic_matcher_embedding_failure_resilience(test_profile_data, sample_job):
    mock_emb = MagicMock(spec=NVIDIAEmbeddingProvider)
    mock_emb.get_embedding.side_effect = RuntimeError("Embedding service unavailable")

    mock_rsn = MagicMock(spec=NVIDIAReasoningProvider)
    mock_rsn.evaluate_match.return_value = ReasoningOutput(
        semantic_score=85.0,
        relevance="high",
        strengths=["Strong skills"],
        concerns=[],
        recommendation="strong_match",
    )

    matcher = NVIDIASemanticMatcher(
        embedding_provider=mock_emb,
        reasoning_provider=mock_rsn,
    )

    dummy_det = MatchResult(
        job_id=sample_job.id,
        profile_id=sample_job.id,
        final_score=75.0,
        decision="review",
        breakdown=MatchBreakdown(),
    )

    eval_res = matcher.evaluate_semantic_match(sample_job, test_profile_data, dummy_det)

    # Embedding failed, but reasoning succeeded
    assert eval_res.embedding_score is None
    assert eval_res.reasoning_score == 85.0
    assert "Embedding error" in eval_res.error


# ---------------------------------------------------------------------------
# 3. MatchService Integration Tests
# ---------------------------------------------------------------------------


def test_match_service_semantic_threshold_gating(test_profile_data, sample_job):
    """Jobs below DETERMINISTIC_SEMANTIC_THRESHOLD (60) must skip reasoning to save cost."""
    mock_engine = MagicMock(spec=DeterministicMatchEngine)
    mock_engine.scoring_config = ScoringConfig()
    # Return a low deterministic score (45.0)
    mock_engine.evaluate_match.return_value = MatchResult(
        job_id=sample_job.id,
        profile_id=sample_job.id,
        final_score=45.0,
        decision="reject",
        breakdown=MatchBreakdown(),
        reasons=["Low title match"],
    )

    mock_semantic = MagicMock(spec=NVIDIASemanticMatcher)
    mock_match_repo = MagicMock()
    mock_job_repo = MagicMock()
    mock_job_repo.get_job_by_id.return_value = sample_job
    mock_profile_service = MagicMock()
    mock_profile_service.get_profile.return_value = Profile(
        id=uuid4(),
        name="Test Candidate",
        email="candidate@example.com",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    mock_profile_service.get_active_profile_data.return_value = test_profile_data

    service = MatchService(
        engine=mock_engine,
        semantic_matcher=mock_semantic,
        match_repository=mock_match_repo,
        job_repository=mock_job_repo,
        profile_service=mock_profile_service,
    )

    result = service.match_job(job_id=sample_job.id, use_semantic=True)

    assert result.final_score == 45.0
    assert result.decision == "reject"
    assert result.llm_score is None
    # Semantic matcher was skipped because 45.0 < 60.0 threshold
    mock_semantic.evaluate_semantic_match.assert_not_called()


def test_match_service_qualifying_semantic_execution(test_profile_data, sample_job):
    """Qualifying job (deterministic score >= 60) triggers semantic scoring and combines scores."""
    mock_engine = MagicMock(spec=DeterministicMatchEngine)
    mock_engine.scoring_config = ScoringConfig()
    mock_engine.evaluate_match.return_value = MatchResult(
        job_id=sample_job.id,
        profile_id=sample_job.id,
        final_score=80.0,
        decision="strong_match",
        breakdown=MatchBreakdown(skills=85.0, title=80.0),
        reasons=["Good skill overlap"],
    )

    mock_semantic = MagicMock(spec=NVIDIASemanticMatcher)
    mock_semantic.evaluate_semantic_match.return_value = SemanticMatchEvaluation(
        embedding_score=90.0,
        reasoning_score=85.0,
        relevance="high",
        strengths=["Deep ML knowledge"],
        concerns=[],
        recommendation="strong_match",
    )

    mock_match_repo = MagicMock()
    mock_job_repo = MagicMock()
    mock_job_repo.get_job_by_id.return_value = sample_job
    mock_profile_service = MagicMock()
    mock_profile_service.get_profile.return_value = Profile(
        id=uuid4(),
        name="Test Candidate",
        email="candidate@example.com",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    mock_profile_service.get_active_profile_data.return_value = test_profile_data

    service = MatchService(
        engine=mock_engine,
        semantic_matcher=mock_semantic,
        match_repository=mock_match_repo,
        job_repository=mock_job_repo,
        profile_service=mock_profile_service,
    )

    result = service.match_job(job_id=sample_job.id, use_semantic=True)

    # Combined = 80*0.70 + 90*0.20 + 85*0.10 = 56 + 18 + 8.5 = 82.5
    assert result.final_score == 82.5
    assert result.deterministic_score == 80.0
    assert result.embedding_score == 90.0
    assert result.llm_score == 85.0
    assert result.decision == "strong_match"

    # Verify repository upsert was called with populated llm_score
    saved_payload = mock_match_repo.upsert_match.call_args[0][0]
    assert saved_payload.match_score == 82.5
    assert saved_payload.llm_score == 85.0
    assert "Semantic Alignment: High" in saved_payload.reason


def test_api_run_matching_with_semantic_flag():
    from app.api.routers.matches import get_match_service

    client = TestClient(app)
    mock_service = MagicMock(spec=MatchService)
    mock_service.match_jobs.return_value = {
        "profile_id": uuid4(),
        "jobs_processed": 5,
        "matches_created": 5,
        "matches_updated": 0,
        "top_matches": [],
    }

    app.dependency_overrides[get_match_service] = lambda: mock_service
    try:
        response = client.post(
            "/matches/run",
            json={"limit": 50, "force_recalculate": True, "use_semantic": True},
        )
        assert response.status_code == 200
        mock_service.match_jobs.assert_called_with(
            profile_id=None,
            limit=50,
            force_recalculate=True,
            use_semantic=True,
        )
    finally:
        app.dependency_overrides.pop(get_match_service, None)
