"""Unit and API Tests for Deterministic Matching Engine."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.matching.education_matcher import EducationMatcher
from app.matching.experience_matcher import ExperienceMatcher
from app.matching.explanation import ExplanationGenerator
from app.matching.job_type_matcher import JobTypeMatcher
from app.matching.location_matcher import LocationMatcher
from app.matching.matcher import DeterministicMatchEngine
from app.matching.salary_matcher import SalaryMatcher
from app.matching.scorer import DecisionThresholds, ScoringConfig, ScoringWeights
from app.matching.semantic_matcher import PlaceholderSemanticMatcher
from app.matching.service import MatchService
from app.matching.skill_matcher import SkillMatcher
from app.matching.title_matcher import TitleMatcher
from app.models.job import Job
from app.models.match import JobMatch, MatchBatchResult, MatchBreakdown, MatchResult
from app.models.profile import CandidateProfileData, CareerPreferences, PersonalDetails, SkillItem


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def test_profile_data() -> CandidateProfileData:
    return CandidateProfileData(
        personal=PersonalDetails(
            name="Test Engineer",
            email="engineer@example.com",
            location="Kochi, Kerala, India",
        ),
        career=CareerPreferences(
            experience_years=3.0,
            preferred_roles=["AI ML Engineer", "Machine Learning Engineer", "Python Developer"],
            preferred_locations=["Kochi, Kerala", "Remote", "Bangalore"],
            remote_preference="any",
            job_type="full-time",
            salary_min=1000000,
            salary_max=2000000,
            education_degree="B.Tech",
            education_field="Computer Science",
        ),
        skills=[
            SkillItem(skill="Python", importance="critical", weight=1.5, years_experience=3.0),
            SkillItem(skill="Machine Learning", importance="critical", weight=1.5, years_experience=3.0),
            SkillItem(skill="PyTorch", importance="high", weight=1.2, years_experience=2.5),
            SkillItem(skill="FastAPI", importance="high", weight=1.2, years_experience=2.0),
            SkillItem(skill="Docker", importance="medium", weight=1.0, years_experience=2.0),
            SkillItem(skill="PostgreSQL", importance="medium", weight=1.0, years_experience=2.0),
        ],
    )


@pytest.fixture
def sample_job() -> Job:
    return Job(
        id=uuid4(),
        source="linkedin",
        external_id="li-9999",
        title="AI/ML Engineer - Python & PyTorch",
        company="Global AI Labs",
        location="Kochi, Kerala, India",
        description="We are seeking an AI/ML Engineer with strong skills in Python, PyTorch, Machine Learning, and Docker. 2-4 years experience required.",
        url="https://linkedin.com/jobs/view/li-9999",
        salary_min=1200000,
        salary_max=1800000,
        experience_text="2-4 years",
        job_type="full-time",
        remote=False,
        easy_apply=True,
        raw_data={},
        created_at="2026-08-17T12:00:00Z",
    )


# ---------------------------------------------------------------------------
# Sub-Matcher Unit Tests
# ---------------------------------------------------------------------------


def test_skill_matcher(test_profile_data, sample_job):
    """Test skill extraction, weighting, and coverage calculation."""
    matcher = SkillMatcher()
    score, matched, missing, diag = matcher.match(sample_job, test_profile_data)

    assert score >= 80.0
    assert "Python" in matched
    assert "PyTorch" in matched
    assert "Machine Learning" in matched
    assert diag["matched_count"] >= 3


def test_skill_matcher_missing_skills(test_profile_data):
    """Test skill matcher when job demands skills missing from candidate."""
    job = Job(
        id=uuid4(),
        source="indeed",
        external_id="in-123",
        title="Rust & Go Systems Engineer",
        company="Low Level Tech",
        location="Remote",
        description="Must have expert skills in Rust, Go, Kubernetes, C++.",
        url="https://indeed.com/view/123",
        created_at="2026-08-17T12:00:00Z",
    )
    matcher = SkillMatcher()
    score, matched, missing, _ = matcher.match(job, test_profile_data)

    assert score < 50.0
    assert len(missing) > 0


def test_title_matcher():
    """Test title similarity with exact matches, aliases, and unrelated roles."""
    matcher = TitleMatcher()
    preferred = ["AI ML Engineer", "Machine Learning Engineer"]

    # Exact match
    score, role = matcher.match("AI ML Engineer", preferred)
    assert score == 100.0

    # Strong alias/overlap match
    score_alias, _ = matcher.match("Senior Machine Learning Engineer", preferred)
    assert score_alias >= 85.0

    # Weak/unrelated match
    score_unrelated, _ = matcher.match("Civil Construction Supervisor", preferred)
    assert score_unrelated == 0.0


def test_experience_matcher():
    """Test experience range parsing and scoring."""
    matcher = ExperienceMatcher()

    # In range (candidate has 3.0 yrs, job asks 2-4 yrs)
    score, min_y, max_y, status = matcher.match(3.0, "2-4 years", "2-4 years experience")
    assert score == 100.0
    assert status == "exact_match"
    assert min_y == 2.0
    assert max_y == 4.0

    # Underqualified (candidate has 1.0 yr, job asks 3+ yrs)
    score_under, _, _, status_under = matcher.match(1.0, "3+ years", "Requires at least 3 years")
    assert score_under <= 60.0
    assert status_under == "underqualified"

    # Fresher match (candidate has 0.5 yrs, job says fresher)
    score_fresher, _, _, status_fresher = matcher.match(0.5, "Freshers welcome", "Entry level opening")
    assert score_fresher == 100.0
    assert status_fresher == "fresher_match"

    # Unspecified requirement
    score_unspec, _, _, status_unspec = matcher.match(3.0, None, "Great team environment")
    assert score_unspec is None
    assert status_unspec == "unspecified"

    # 15+ years phrase extraction
    score_15, min_15, max_15, status_15 = matcher.match(
        3.0,
        None,
        "Principal Layout Engineer. Requires 15+ years hands-on custom layout experience with Electrical/VLSI background.",
    )
    assert min_15 == 15.0
    assert score_15 == 0.0
    assert status_15 == "extreme_mismatch"


def test_location_matcher():
    """Test location proximity and remote handling."""
    matcher = LocationMatcher()

    # Exact match
    score_exact, _ = matcher.match("Kochi, Kerala, India", False, "Kochi, Kerala", ["Kochi, Kerala"])
    assert score_exact == 100.0

    # Remote job match
    score_remote, _ = matcher.match("Anywhere", True, "Kochi, Kerala", ["Kochi"], remote_preference="any")
    assert score_remote == 100.0

    # Alternate preferred location
    score_pref, _ = matcher.match("Bangalore, Karnataka", False, "Kochi, Kerala", ["Bangalore", "Kochi"])
    assert score_pref >= 85.0

    # Location mismatch
    score_mismatch, _ = matcher.match("Berlin, Germany", False, "Kochi, Kerala", ["Kochi"])
    assert score_mismatch <= 40.0


def test_salary_matcher():
    """Test salary overlap and missing salary handling."""
    matcher = SalaryMatcher()

    # Overlapping salary
    score_overlap, compat, _ = matcher.match(1200000, 1800000, 1000000, 2000000)
    assert score_overlap == 100.0
    assert compat is True

    # Job undisclosed salary -> returns None (excluded and redistributed, not awarded 80 pts)
    score_unknown, compat_u, status_u = matcher.match(None, None, 1000000, 2000000)
    assert score_unknown is None
    assert compat_u is True
    assert status_u == "salary_unknown"

    # Under candidate minimum
    score_under, compat_d, _ = matcher.match(400000, 600000, 1000000, 2000000)
    assert score_under <= 50.0
    assert compat_d is False


def test_education_and_job_type_matchers():
    """Test education hierarchy and employment type matching."""
    edu_matcher = EducationMatcher()
    edu_score, _ = edu_matcher.match("Requires Bachelor's in CS", "B.Tech", "Computer Science")
    assert edu_score == 100.0

    # No degree required -> returns None (not awarded 90 pts)
    edu_none, status_e = edu_matcher.match("Python programmer needed", "B.Tech", "Computer Science")
    assert edu_none is None
    assert status_e == "no_degree_requirement_detected"

    type_matcher = JobTypeMatcher()
    type_score, _ = type_matcher.match("Full-time", "full-time")
    assert type_score == 100.0

    # Unspecified job type -> returns None (not awarded 85 pts)
    type_none, status_t = type_matcher.match(None, "full-time")
    assert type_none is None
    assert status_t == "job_type_unspecified"


# ---------------------------------------------------------------------------
# Scoring Configuration & Explanation Tests
# ---------------------------------------------------------------------------


def test_scoring_weights_and_decision_thresholds():
    """Verify weighted score calculation and threshold mapping."""
    config = ScoringConfig()
    breakdown = MatchBreakdown(
        skills=100.0,
        title=100.0,
        experience=100.0,
        location=100.0,
        education=100.0,
        salary=100.0,
        job_type=100.0,
    )
    score, decision = config.calculate_final_score(breakdown)
    assert score == 100.0
    assert decision == "excellent"

    low_breakdown = MatchBreakdown(
        skills=40.0,
        title=40.0,
        experience=40.0,
        location=40.0,
        education=50.0,
        salary=50.0,
        job_type=50.0,
    )
    low_score, low_decision = config.calculate_final_score(low_breakdown)
    assert low_score < 60.0
    assert low_decision == "reject"


def test_explanation_generator():
    """Test natural language explanation formatting."""
    gen = ExplanationGenerator()
    breakdown = MatchBreakdown(skills=90.0, title=100.0, experience=100.0, location=100.0, education=90.0, salary=80.0, job_type=100.0)
    summary, reasons, warnings = gen.generate(
        final_score=94.5,
        decision="excellent",
        breakdown=breakdown,
        matched_skills=["Python", "PyTorch", "FastAPI"],
        missing_skills=["Kubernetes"],
        job_title="Senior AI Engineer",
        matched_role="AI ML Engineer",
        exp_status="exact_match",
        min_years=3.0,
        candidate_years=3.0,
        loc_status="exact_current_location",
        salary_status="salary_unknown",
    )

    assert "Match Score: 94.5/100" in summary
    assert len(reasons) >= 3
    assert "Missing 1 requested skill" in warnings[0]


@pytest.mark.asyncio
async def test_semantic_matcher_placeholder():
    """Verify SemanticMatcher placeholder returns None and zero API calls."""
    placeholder = PlaceholderSemanticMatcher()
    fit = await placeholder.evaluate_job_fit("AI Engineer", "Description", "Candidate", ["Python"])
    assert fit is None
    emb = await placeholder.generate_embedding("Test")
    assert emb == []


# ---------------------------------------------------------------------------
# Deterministic Engine & Service Tests
# ---------------------------------------------------------------------------


def test_deterministic_engine_consistency(sample_job, test_profile_data):
    """Verify that same inputs produce the EXACT same score every run."""
    engine = DeterministicMatchEngine()
    profile_id = uuid4()

    result1 = engine.evaluate_match(sample_job, test_profile_data, profile_id)
    result2 = engine.evaluate_match(sample_job, test_profile_data, profile_id)

    assert result1.final_score == result2.final_score
    assert result1.decision == result2.decision
    assert result1.matched_skills == result2.matched_skills
    assert result1.final_score >= 80.0


# ---------------------------------------------------------------------------
# API Endpoints Tests
# ---------------------------------------------------------------------------


def test_api_run_matching(client: TestClient):
    """Test POST /matches/run endpoint."""
    profile_id = uuid4()
    mock_batch = MatchBatchResult(
        profile_id=profile_id,
        jobs_processed=10,
        matches_created=10,
        matches_updated=0,
        top_matches=[],
    )

    with patch.object(MatchService, "match_jobs", return_value=mock_batch):
        response = client.post("/matches/run", json={"profile_id": str(profile_id), "limit": 10})
        assert response.status_code == 200
        data = response.json()
        assert data["jobs_processed"] == 10
        assert data["matches_created"] == 10


def test_api_get_matches(client: TestClient):
    """Test GET /matches endpoint."""
    profile_id = uuid4()
    job_id = uuid4()
    mock_matches = [
        JobMatch(
            id=uuid4(),
            job_id=job_id,
            profile_id=profile_id,
            match_score=92.5,
            skill_score=95.0,
            title_score=100.0,
            decision="excellent",
            reason="Strong skill alignment.",
            created_at="2026-08-17T12:00:00Z",
        )
    ]

    with patch.object(MatchService, "list_matches", return_value=(mock_matches, 1)):
        response = client.get(f"/matches?profile_id={profile_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["match_score"] == 92.5
        assert data[0]["llm_score"] is None


# ---------------------------------------------------------------------------
# Phase 5.3.1 Specific Regression & Gating Tests
# ---------------------------------------------------------------------------


def test_micron_false_positive_regression(test_profile_data):
    """Real-world regression test: Micron Principal Layout Engineer with AI keywords.

    Job requires 15+ years hands-on custom layout experience (VLSI/Electrical).
    Job description mentions AI/ML/Agentic AI/Python terminology.
    Candidate is a junior AI/ML engineer (3 years experience).
    Stored job has experience_text=None, salary=None, job_type=None.

    Expected:
    - High keyword overlap MUST NOT make it a recommended match.
    - Title/domain mismatch detected (title_score = 0.0).
    - 15+ years experience requirement extracted from description.
    - Experience mismatch detected (deficit = 12 years -> extreme_mismatch).
    - Hard qualification gates enforce decision = 'reject'.
    - Final score < 50.0.
    """
    engine = DeterministicMatchEngine()
    profile_id = uuid4()

    micron_job = Job(
        id=uuid4(),
        source="indeed",
        external_id="in-micron-layout-001",
        title="Principal Layout Engineer-DPG-LPDDR",
        company="Micron Technology",
        location="Bangalore, Karnataka, India",
        description="""
        Job Description:
        Principal Layout Engineer in DPG-LPDDR group.
        Requires 15+ years of hands-on custom layout experience with deep semiconductor background.
        Expertise in Electrical/Electronics/VLSI engineering.
        Familiarity with Python, Artificial Intelligence, Machine Learning, and Agentic AI workflows for CAD tooling is a plus.
        """,
        url="https://in.indeed.com/viewjob?jk=micron001",
        salary_min=None,
        salary_max=None,
        experience_text=None,
        job_type=None,
        remote=False,
        created_at="2026-08-18T12:00:00Z",
    )

    result = engine.evaluate_match(micron_job, test_profile_data, profile_id)

    # Assertions
    assert result.decision == "reject"
    assert result.final_score < 50.0
    assert result.title_score <= 20.0
    assert result.breakdown.salary is None  # Not awarded 80
    assert result.breakdown.job_type is None  # Not awarded 85
    assert result.breakdown.education is None  # Not awarded 90
    assert any("experience mismatch" in w.lower() or "deficit" in w.lower() for w in result.warnings)
    assert any("title" in w.lower() or "domain" in w.lower() for w in result.warnings)


def test_missing_salary_jobtype_does_not_award_points(test_profile_data):
    """Verify that missing salary, job_type, and education do not award optimistic scores."""
    engine = DeterministicMatchEngine()
    profile_id = uuid4()

    job_no_meta = Job(
        id=uuid4(),
        source="indeed",
        external_id="in-nometa-002",
        title="Machine Learning Engineer",
        company="Tech Innovators",
        location="Kochi, Kerala",
        description="Looking for a Machine Learning Engineer with Python and PyTorch skills. 2-4 years experience.",
        url="https://in.indeed.com/viewjob?jk=nometa002",
        salary_min=None,
        salary_max=None,
        experience_text=None,
        job_type=None,
        created_at="2026-08-18T12:00:00Z",
    )

    result = engine.evaluate_match(job_no_meta, test_profile_data, profile_id)

    assert result.breakdown.salary is None
    assert result.breakdown.job_type is None
    assert result.breakdown.education is None
    # Weights for missing fields are redistributed, and active dimensions determine the score
    assert result.final_score >= 80.0
    assert result.decision in ("excellent", "strong_match")


def test_experience_mismatch_gate_rejects_extreme_deficit(test_profile_data):
    """Verify experience mismatch gate rejects jobs requiring vastly more experience."""
    engine = DeterministicMatchEngine()
    profile_id = uuid4()

    senior_job = Job(
        id=uuid4(),
        source="indeed",
        external_id="in-sr-director-003",
        title="Director of AI Engineering",
        company="Enterprise AI",
        location="Kochi, Kerala",
        description="Requires at least 12 years of industry experience leading AI teams. Python, Machine Learning.",
        url="https://in.indeed.com/viewjob?jk=sr003",
        created_at="2026-08-18T12:00:00Z",
    )

    result = engine.evaluate_match(senior_job, test_profile_data, profile_id)
    assert result.decision == "reject"
    assert result.final_score <= 45.0


def test_reasonable_stretch_experience_remains_eligible(test_profile_data):
    """Verify reasonable stretch (e.g. 4 years required vs candidate 3 years) remains eligible."""
    engine = DeterministicMatchEngine()
    profile_id = uuid4()

    stretch_job = Job(
        id=uuid4(),
        source="indeed",
        external_id="in-stretch-004",
        title="Machine Learning Engineer",
        company="Smart AI Labs",
        location="Kochi, Kerala",
        description="Requires 4 years of experience with Python, Machine Learning, PyTorch.",
        url="https://in.indeed.com/viewjob?jk=stretch004",
        created_at="2026-08-18T12:00:00Z",
    )

    result = engine.evaluate_match(stretch_job, test_profile_data, profile_id)
    assert result.decision in ("strong_match", "excellent", "review")
    assert result.final_score >= 70.0


def test_rejected_job_skips_nvidia_semantic_matching(test_profile_data):
    """Verify MatchService skips NVIDIA semantic matching for deterministically rejected jobs."""
    mock_semantic = MagicMock()
    mock_match_repo = MagicMock()
    mock_job_repo = MagicMock()
    mock_profile_svc = MagicMock()
    mock_profile_svc.get_profile.return_value = MagicMock(id=uuid4())
    mock_profile_svc.get_active_profile_data.return_value = test_profile_data

    rejected_job = Job(
        id=uuid4(),
        source="indeed",
        external_id="in-rej-005",
        title="Civil Structural Engineer",
        company="BuildCorp",
        location="Mumbai",
        description="Requires 15+ years experience in civil construction and AutoCAD.",
        url="https://in.indeed.com/viewjob?jk=rej005",
        created_at="2026-08-18T12:00:00Z",
    )
    mock_job_repo.get_job_by_id.return_value = rejected_job

    service = MatchService(
        semantic_matcher=mock_semantic,
        match_repository=mock_match_repo,
        job_repository=mock_job_repo,
        profile_service=mock_profile_svc,
    )

    result = service.match_job(job_id=rejected_job.id, use_semantic=True)
    # Since rejected by deterministic gates, semantic_matcher should NOT be called
    mock_semantic.evaluate_semantic_match.assert_not_called()
    assert result is not None
    assert result.decision == "reject"
    assert result.llm_score is None


def test_match_discovered_jobs_batch_path(test_profile_data, sample_job):
    """Verify match_discovered_jobs loads profile ONCE and processes multiple jobs."""
    mock_semantic = MagicMock()
    mock_match_repo = MagicMock()
    mock_profile_svc = MagicMock()
    profile_id = uuid4()
    mock_profile_svc.get_profile.return_value = MagicMock(id=profile_id)
    mock_profile_svc.get_active_profile_data.return_value = test_profile_data

    service = MatchService(
        semantic_matcher=mock_semantic,
        match_repository=mock_match_repo,
        profile_service=mock_profile_svc,
    )

    job1 = sample_job
    job2 = Job(
        id=uuid4(),
        source="indeed",
        external_id="in-job2-006",
        title="Python Developer",
        company="FastTech",
        location="Kochi, Kerala",
        description="Python backend developer with FastAPI and PostgreSQL.",
        url="https://in.indeed.com/viewjob?jk=job2006",
        created_at="2026-08-18T12:00:00Z",
    )

    results = service.match_discovered_jobs([job1, job2], profile_id=profile_id, use_semantic=False)

    assert len(results) == 2
    # Loaded profile exactly once
    mock_profile_svc.get_profile.assert_called_once_with(profile_id)
    mock_profile_svc.get_active_profile_data.assert_called_once_with(profile_id)
    assert mock_match_repo.upsert_match.call_count == 2
