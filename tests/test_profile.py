"""Unit and API Tests for Candidate Profile System."""

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.models.profile import (
    CandidateProfileData,
    CareerPreferences,
    PersonalDetails,
    Profile,
    ProfileCreate,
    ProfileUpdate,
    Skill,
    SkillCreate,
    SkillItem,
)
from app.profile.profile_loader import ProfileLoader, normalize_skill_name, normalize_string
from app.profile.repository import ProfileRepository
from app.profile.service import ProfileService


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_profile_data() -> CandidateProfileData:
    return CandidateProfileData(
        personal=PersonalDetails(
            name="Jane Doe",
            email="jane.doe@example.com",
            phone="+91 9876543210",
            location="Kochi, Kerala, India",
        ),
        career=CareerPreferences(
            experience_years=3.5,
            preferred_roles=["AI ML Engineer", "Machine Learning Engineer"],
            preferred_locations=["Kochi, Kerala", "Remote"],
            remote_preference="any",
            job_type="full-time",
            salary_min=1000000,
            salary_max=2000000,
            education_degree="B.Tech",
            education_field="Computer Science",
        ),
        skills=[
            SkillItem(skill="Python", importance="critical", years_experience=3.5),
            SkillItem(skill="PyTorch", importance="high", years_experience=2.5),
            SkillItem(skill="FastAPI", importance="high", years_experience=2.0),
        ],
    )


# ---------------------------------------------------------------------------
# Profile Normalization & Loader Tests
# ---------------------------------------------------------------------------


def test_skill_normalization_and_aliases():
    """Verify that skill aliases, case variations, and whitespace are normalized cleanly."""
    assert normalize_skill_name("  python  ") == "Python"
    assert normalize_skill_name("PYTHON") == "Python"
    assert normalize_skill_name("py") == "Python"
    assert normalize_skill_name("ml") == "Machine Learning"
    assert normalize_skill_name("ai") == "Artificial Intelligence"
    assert normalize_skill_name("k8s") == "Kubernetes"
    assert normalize_skill_name("postgres") == "PostgreSQL"
    assert normalize_skill_name("ts") == "TypeScript"
    assert normalize_skill_name("node") == "Node.js"
    assert normalize_skill_name("CustomTool") == "CustomTool"


def test_string_normalization():
    """Verify whitespace stripping and normalization."""
    assert normalize_string("  Kochi,   Kerala  ") == "Kochi, Kerala"
    assert normalize_string("") == ""
    assert normalize_string(None) == ""


def test_profile_loader_reads_json(tmp_path):
    """Test that ProfileLoader correctly parses and validates a JSON profile file."""
    profile_json = {
        "personal": {
            "name": "Alex Smith",
            "email": "alex@example.com",
            "location": "Kochi, Kerala",
        },
        "career": {
            "experience_years": 4.0,
            "preferred_roles": ["AI Engineer", "ML Engineer"],
            "preferred_locations": ["Kochi", "Remote"],
        },
        "skills": [
            {"skill": "py", "importance": "critical"},
            {"skill": "pytorch", "importance": "high"},
        ],
        "experience": [],
        "education": [],
    }

    test_file = tmp_path / "candidate_profile.json"
    test_file.write_text(json.dumps(profile_json), encoding="utf-8")

    loader = ProfileLoader(file_path=str(test_file))
    profile_data = loader.load_and_normalize()

    assert profile_data.personal.name == "Alex Smith"
    assert profile_data.personal.email == "alex@example.com"
    assert profile_data.career.experience_years == 4.0
    assert len(profile_data.skills) == 2
    assert profile_data.skills[0].skill == "Python"
    assert profile_data.skills[1].skill == "PyTorch"


# ---------------------------------------------------------------------------
# Profile Repository & Service Tests
# ---------------------------------------------------------------------------


def test_profile_repository_mocked():
    """Test ProfileRepository CRUD operations with mocked Supabase client."""
    mock_client = MagicMock()
    repo = ProfileRepository(client=mock_client)

    profile_id = uuid4()
    mock_data = [{
        "id": str(profile_id),
        "name": "Jane Doe",
        "email": "jane@example.com",
        "experience_years": 3.0,
        "preferred_locations": ["Kochi"],
        "preferred_roles": ["AI Engineer"],
        "created_at": "2026-08-17T12:00:00Z",
        "updated_at": "2026-08-17T12:00:00Z",
    }]

    mock_client.table().select().eq().limit().execute.return_value = MagicMock(data=mock_data)
    mock_client.table().select().eq().order().execute.return_value = MagicMock(data=[])

    profile = repo.get_profile_by_id(profile_id)
    assert profile is not None
    assert profile.name == "Jane Doe"
    assert profile.email == "jane@example.com"


def test_profile_service_sync_flow(sample_profile_data):
    """Test ProfileService synchronizing candidate profile to database."""
    mock_repo = MagicMock(spec=ProfileRepository)
    mock_loader = MagicMock(spec=ProfileLoader)

    mock_loader.load_and_normalize.return_value = sample_profile_data
    created_profile = Profile(
        id=uuid4(),
        name=sample_profile_data.personal.name,
        email=sample_profile_data.personal.email,
        experience_years=sample_profile_data.career.experience_years,
        created_at="2026-08-17T12:00:00Z",
        updated_at="2026-08-17T12:00:00Z",
        skills=[],
    )
    mock_repo.sync_profile_from_data.return_value = created_profile

    service = ProfileService(repository=mock_repo, loader=mock_loader)
    result = service.sync_local_profile_to_database()

    assert result is not None
    assert result.name == "Jane Doe"
    mock_repo.sync_profile_from_data.assert_called_once_with(sample_profile_data)


# ---------------------------------------------------------------------------
# API Endpoints Tests
# ---------------------------------------------------------------------------


def test_api_get_profile(client: TestClient):
    """Test GET /profile endpoint."""
    test_id = uuid4()
    mock_profile = Profile(
        id=test_id,
        name="Candidate Name",
        email="candidate@example.com",
        created_at="2026-08-17T12:00:00Z",
        updated_at="2026-08-17T12:00:00Z",
        skills=[],
    )

    with patch.object(ProfileService, "get_profile", return_value=mock_profile):
        response = client.get("/profile")
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "candidate@example.com"
        assert data["id"] == str(test_id)


def test_api_create_sync_profile(client: TestClient):
    """Test POST /profile sync endpoint."""
    test_id = uuid4()
    mock_profile = Profile(
        id=test_id,
        name="Synced Candidate",
        email="synced@example.com",
        created_at="2026-08-17T12:00:00Z",
        updated_at="2026-08-17T12:00:00Z",
        skills=[],
    )

    with patch.object(ProfileService, "sync_local_profile_to_database", return_value=mock_profile):
        response = client.post("/profile")
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Synced Candidate"


def test_api_get_skills(client: TestClient):
    """Test GET /profile/skills endpoint."""
    profile_id = uuid4()
    mock_profile = Profile(
        id=profile_id,
        name="Candidate Name",
        email="candidate@example.com",
        created_at="2026-08-17T12:00:00Z",
        updated_at="2026-08-17T12:00:00Z",
        skills=[],
    )
    mock_skills = [
        Skill(id=uuid4(), profile_id=profile_id, skill="Python", importance="critical", created_at="2026-08-17T12:00:00Z"),
        Skill(id=uuid4(), profile_id=profile_id, skill="PyTorch", importance="high", created_at="2026-08-17T12:00:00Z"),
    ]

    with patch.object(ProfileService, "get_profile", return_value=mock_profile):
        with patch.object(ProfileService, "get_skills", return_value=mock_skills):
            response = client.get("/profile/skills")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["skill"] == "Python"
