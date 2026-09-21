"""Unit and Integration Tests for LinkedIn Outreach Agent (Phase 6.2).

Tests:
1. Keyword and hiring-intent matching
2. Lead relevance scoring (role similarity, skills, hiring intent, experience, location)
3. Public email acceptance vs guessed/dummy email rejection
4. Lead deduplication via canonical post_url
5. Candidate profile alignment and skill highlighting
6. Email draft generation and personalization
7. Resume PDF path validation (missing vs existing)
8. Safety send gate (OUTREACH_AUTO_SEND=False strictly blocks transmission)
9. Opt-out and DO_NOT_CONTACT handling
10. Budget and rate limit guard enforcement
11. Apify discovery provider lifecycle, token verification, and error handling
12. Malformed dataset items and partial discovery failures
13. FastAPI /linkedin endpoints
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import app
from app.config.settings import Settings
from app.database.repositories.linkedin_lead_repository import LinkedInLeadRepository
from app.models.profile import (
    CandidateProfileData,
    CareerPreferences,
    PersonalDetails,
    SkillItem,
)
from app.outreach.linkedin.agent import LinkedInOutreachAgent
from app.outreach.linkedin.budget_guard import LinkedInBudgetGuard
from app.outreach.linkedin.config import LinkedInKeywordsConfig
from app.outreach.linkedin.discovery_provider import (
    LinkedInApifyDiscoveryProvider,
    classify_lead_type,
    extract_public_email,
    normalize_post_url,
)
from app.outreach.linkedin.email_generator import (
    LinkedInEmailGenerator,
    ResumeNotFoundError,
)
from app.outreach.linkedin.email_service import (
    DraftOnlyEmailProvider,
    DuplicateOutreachError,
    LeadOptedOutError,
    LinkedInOutreachEmailService,
    OutreachSafetyError,
)
from app.outreach.linkedin.models import (
    DraftStatus,
    LeadStatus,
    LeadType,
    LinkedInDiscoverRequest,
    LinkedInLead,
    LinkedInLeadBase,
    LinkedInLeadCreate,
    LinkedInOutreachDraft,
)
from app.outreach.linkedin.relevance_engine import LinkedInLeadRelevanceEngine


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_candidate_profile() -> CandidateProfileData:
    """Fixture providing candidate profile."""
    return CandidateProfileData(
        personal=PersonalDetails(
            name="Rithin Rakesh",
            email="rithinrakesh2002@gmail.com",
            phone="+91 81118 55550",
            location="Kannur, Kerala, India",
        ),
        career=CareerPreferences(
            experience_years=0.8,
            preferred_roles=[
                "AI Engineer",
                "Machine Learning Engineer",
                "Data Scientist",
                "Generative AI",
            ],
            preferred_locations=["Bangalore", "Kochi", "Remote", "India"],
        ),
        skills=[
            SkillItem(skill="Python"),
            SkillItem(skill="Machine Learning"),
            SkillItem(skill="FastAPI"),
            SkillItem(skill="PyTorch"),
            SkillItem(skill="Docker"),
            SkillItem(skill="SQL"),
        ],
    )



@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    """Fixture providing isolated settings with temporary paths."""
    resume_file = tmp_path / "resume.pdf"
    resume_file.write_bytes(b"%PDF-1.4 Mock resume content")
    state_file = tmp_path / "linkedin_budget_state.json"
    keywords_file = tmp_path / "linkedin_keywords.json"

    s = Settings(
        LINKEDIN_OUTREACH_ENABLED=True,
        LINKEDIN_MAX_POSTS_PER_RUN=5,
        LINKEDIN_MAX_APIFY_RUNS_PER_DAY=2,
        LINKEDIN_MAX_LEADS_PER_RUN=5,
        LINKEDIN_MAX_DRAFTS_PER_RUN=3,
        LINKEDIN_MAX_FUTURE_SENDS_PER_DAY=5,
        RESUME_PDF_PATH=str(resume_file),
        OUTREACH_AUTO_SEND=False,
        LINKEDIN_KEYWORDS_CONFIG_PATH=str(keywords_file),
    )
    return s


@pytest.fixture
def mock_repository(tmp_path: Path) -> LinkedInLeadRepository:
    """Fixture providing repository backed by temporary JSON storage."""
    leads_file = tmp_path / "test_leads.json"
    drafts_file = tmp_path / "test_drafts.json"
    return LinkedInLeadRepository(
        supabase_client=None,
        leads_file_path=leads_file,
        drafts_file_path=drafts_file,
    )


# -----------------------------------------------------------------------------
# 1. KEYWORDS AND HIRING-INTENT MATCHING
# -----------------------------------------------------------------------------

def test_keyword_matching_and_intent_detection(tmp_path: Path):
    cfg_file = tmp_path / "keywords.json"
    cfg_file.write_text('{"role_keywords": ["AI Engineer", "ML Engineer"], "hiring_intent_keywords": ["hiring", "openings"]}', encoding="utf-8")
    config = LinkedInKeywordsConfig(config_path=str(cfg_file))

    post_snippet = "We are hiring an AI Engineer to join our Bangalore team. Exciting openings!"
    matched = config.find_matched_keywords(post_snippet)

    assert "AI Engineer" in matched
    assert "hiring" in matched
    assert "openings" in matched

    queries = config.build_search_queries(max_queries=1)
    assert len(queries) == 1
    assert "AI Engineer" in queries[0]
    assert "hiring" in queries[0]


def test_classify_lead_type():
    assert classify_lead_type("We're hiring an AI engineer", "") == LeadType.HIRING_POST
    assert classify_lead_type("Can refer candidates for ML role, DM me", "") == LeadType.REFERRAL_POST
    assert classify_lead_type("Looking for Python developers", "Senior Recruiter at TechCorp") == LeadType.RECRUITER_POST
    assert classify_lead_type("Check out our job opening", "Software Engineer") == LeadType.JOB_OPPORTUNITY
    assert classify_lead_type("Excited to announce our series A funding round", "CEO") == LeadType.GENERAL_PROFESSIONAL


# -----------------------------------------------------------------------------
# 2. PUBLIC EMAIL ACCEPTANCE VS GUESSED EMAIL REJECTION
# -----------------------------------------------------------------------------

def test_public_email_extraction_accepts_valid_email():
    sample_text = "Please reach out to careers@deepmind.com or hr-recruiting@acme.ai for immediate consideration."
    email, source = extract_public_email(sample_text)
    assert email == "careers@deepmind.com"
    assert source == "POST_TEXT"


def test_public_email_extraction_rejects_dummy_and_invalid_emails():
    # Placeholder / dummy domains must be rejected
    assert extract_public_email("Contact us at test@example.com")[0] is None
    assert extract_public_email("Reach out at user@company.com")[0] is None
    assert extract_public_email("Invalid format email@domain.invalid")[0] is None
    assert extract_public_email("No email here, just name (at) firm dot com")[0] is None
    # Image asset name falsely matching email regex
    assert extract_public_email("View screenshot at banner@2x.png")[0] is None


# -----------------------------------------------------------------------------
# 3. CANONICAL POST URL DEDUPLICATION
# -----------------------------------------------------------------------------

def test_normalize_post_url():
    raw_1 = "https://www.linkedin.com/feed/update/urn:li:activity:71234567890/?utm_source=share&utm_medium=member_desktop"
    assert normalize_post_url(raw_1) == "https://www.linkedin.com/feed/update/urn:li:activity:71234567890"

    raw_2 = "https://www.linkedin.com/posts/acme_ai-engineer-hiring-activity-71234567890-xyz/"
    assert normalize_post_url(raw_2) == "https://www.linkedin.com/posts/acme_ai-engineer-hiring-activity-71234567890-xyz"

    assert normalize_post_url(None, "urn:li:activity:987654321") == "https://www.linkedin.com/feed/update/urn:li:activity:987654321"


# -----------------------------------------------------------------------------
# 4. RELEVANCE SCORING ENGINE
# -----------------------------------------------------------------------------

def test_relevance_engine_high_match(mock_candidate_profile: CandidateProfileData):
    engine = LinkedInLeadRelevanceEngine(candidate_profile=mock_candidate_profile)

    lead = LinkedInLeadBase(
        post_url="https://www.linkedin.com/feed/update/urn:li:activity:1",
        post_text="We're hiring a Junior AI Engineer / ML Engineer in Bangalore or Remote. Required skills: Python, Machine Learning, PyTorch, FastAPI. Email hiring@tensorlabs.ai",
        author_headline="Technical Recruiter at TensorLabs",
        lead_type=LeadType.RECRUITER_POST,
        contact_email="hiring@tensorlabs.ai",
    )

    score, details = engine.calculate_relevance(lead)
    assert score >= 80.0
    assert details["role_similarity"]["score"] >= 80.0
    assert "python" in details["skill_overlap"]["matched_skills"]
    assert details["author_authority"]["score"] == 100.0

    assert details["experience_fit"]["score"] == 100.0


def test_relevance_engine_low_match(mock_candidate_profile: CandidateProfileData):
    engine = LinkedInLeadRelevanceEngine(candidate_profile=mock_candidate_profile)

    lead = LinkedInLeadBase(
        post_url="https://www.linkedin.com/feed/update/urn:li:activity:2",
        post_text="Looking for a Senior Principal C++ Embedded Firmware Architect with 15+ years experience in Munich, Germany.",
        author_headline="VP of Hardware",
        lead_type=LeadType.JOB_OPPORTUNITY,
    )


    score, details = engine.calculate_relevance(lead)
    assert score < 45.0
    assert details["experience_fit"]["score"] <= 40.0


# -----------------------------------------------------------------------------
# 5. RESUME VALIDATION AND DRAFT GENERATION
# -----------------------------------------------------------------------------

def test_resume_path_validation_success(mock_settings: Settings, mock_candidate_profile: CandidateProfile):
    generator = LinkedInEmailGenerator(settings=mock_settings, candidate_profile=mock_candidate_profile)
    res_path = generator.validate_resume_path()
    assert res_path.is_file()


def test_resume_path_validation_fails_when_missing(mock_candidate_profile: CandidateProfile, tmp_path: Path):
    s = Settings(RESUME_PDF_PATH=str(tmp_path / "non_existent_resume.pdf"))
    generator = LinkedInEmailGenerator(settings=s, candidate_profile=mock_candidate_profile)
    with pytest.raises(ResumeNotFoundError):
        generator.validate_resume_path()


def test_email_draft_generation(mock_settings: Settings, mock_candidate_profile: CandidateProfile):
    generator = LinkedInEmailGenerator(settings=mock_settings, candidate_profile=mock_candidate_profile)

    lead = LinkedInLead(
        id=uuid4(),
        post_url="https://www.linkedin.com/feed/update/urn:li:activity:100",
        post_text="We're hiring an AI Engineer at Acme Labs. Contact hiring@acmelabs.ai",
        author_name="Sarah Connor",
        company="Acme Labs",
        contact_email="hiring@acmelabs.ai",
        matched_keywords=["AI Engineer", "hiring"],
    )

    draft = generator.generate_draft(lead)

    assert draft.recipient_email == "hiring@acmelabs.ai"
    assert "Sarah Connor" in draft.body_text
    assert "Acme Labs" in draft.body_text
    assert "Python" in draft.body_text
    assert draft.status == DraftStatus.DRAFT
    assert draft.attachment_verified is True
    assert "resume.pdf" in draft.attachment_path


# -----------------------------------------------------------------------------
# 6. SAFETY SEND GATE & DRAFT ONLY PROVIDER
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safety_send_gate_blocks_transmission(mock_settings: Settings):
    mock_settings.OUTREACH_AUTO_SEND = False
    provider = DraftOnlyEmailProvider(settings=mock_settings)

    draft = LinkedInOutreachDraft(
        id=uuid4(),
        lead_id=uuid4(),
        recipient_email="recruiter@acme.ai",
        recipient_name="Recruiter",
        subject="Inquiry",
        body_text="Test body",
        attachment_path="data/profile/resume.pdf",
    )

    with pytest.raises(OutreachSafetyError) as exc_info:
        await provider.send_email(draft)

    assert "OUTREACH_AUTO_SEND is False" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 7. REPOSITORY DEDUPLICATION & OPT-OUT
# -----------------------------------------------------------------------------

def test_lead_repository_deduplication_and_opt_out(mock_repository: LinkedInLeadRepository):
    url = "https://www.linkedin.com/feed/update/urn:li:activity:555"

    lead_c = LinkedInLeadCreate(
        post_url=url,
        post_text="Hiring AI engineers!",
        contact_email="lead@company.ai",
    )

    # 1. Create lead
    l1 = mock_repository.create_lead(lead_c)
    assert l1.post_url == url

    # 2. Re-create same lead -> returns existing
    l2 = mock_repository.create_lead(lead_c)
    assert l1.id == l2.id

    # 3. Create draft
    draft = LinkedInOutreachDraft(
        id=uuid4(),
        lead_id=l1.id,
        recipient_email="lead@company.ai",
        subject="Test",
        body_text="Test",
        attachment_path="resume.pdf",
    )
    mock_repository.create_draft(draft)
    assert mock_repository.has_contacted_email("lead@company.ai") is True

    # 4. Opt-out handling
    service = LinkedInOutreachEmailService(repository=mock_repository)
    service.mark_lead_skipped(l1.id, opt_out=True)

    opted_lead = mock_repository.get_lead_by_id(l1.id)
    assert opted_lead.status == LeadStatus.DO_NOT_CONTACT

    # Attempting draft on opted-out lead raises LeadOptedOutError
    with pytest.raises(LeadOptedOutError):
        service.prepare_draft_for_lead(l1.id)


# -----------------------------------------------------------------------------
# 8. BUDGET GUARD LIMITS
# -----------------------------------------------------------------------------

def test_budget_guard_daily_limits(tmp_path: Path):
    s = Settings(
        LINKEDIN_OUTREACH_ENABLED=True,
        LINKEDIN_MAX_POSTS_PER_RUN=5,
        LINKEDIN_MAX_APIFY_RUNS_PER_DAY=2,
    )
    guard = LinkedInBudgetGuard(settings=s, state_file_path=str(tmp_path / "budget.json"))

    # Run 1 allowed
    allowed, clamped, _ = guard.can_execute_discovery(10)
    assert allowed is True
    assert clamped == 5  # Clamped to max_posts_per_run
    guard.record_discovery_run(posts_fetched=5, leads_created=3)

    # Run 2 allowed
    allowed, _, _ = guard.can_execute_discovery(3)
    assert allowed is True
    guard.record_discovery_run(posts_fetched=3, leads_created=2)

    # Run 3 blocked (exceeds daily 2 runs limit)
    allowed, _, reason = guard.can_execute_discovery(3)
    assert allowed is False
    assert reason == "DAILY_APIFY_RUN_LIMIT_REACHED"


# -----------------------------------------------------------------------------
# 9. DISCOVERY PROVIDER WITH MOCKED APIFY ACTOR
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discovery_provider_mocked_apify_run(mock_settings: Settings, tmp_path: Path):
    guard = LinkedInBudgetGuard(settings=mock_settings, state_file_path=str(tmp_path / "b.json"))
    provider = LinkedInApifyDiscoveryProvider(settings=mock_settings, budget_guard=guard)

    # Mock token validation and Actor HTTP calls
    provider.verify_token_validity = AsyncMock(return_value=True)
    provider._get_api_token = MagicMock(return_value="mock_token_123")

    mock_items = [
        {
            "id": "urn:li:activity:7111222333",
            "url": "https://www.linkedin.com/feed/update/urn:li:activity:7111222333?src=share",
            "content": "Exciting news! We are hiring an AI/ML Engineer in Bangalore. Send resume to talent@nextgen.ai",
            "author": {
                "name": "Alex Mercer",
                "headline": "Head of Talent Acquisition at NextGen",
                "linkedinUrl": "https://www.linkedin.com/in/alexmercer",
            },
            "postedAt": "2026-09-20T10:00:00Z",
        },
        {
            # Malformed item without content
            "id": "urn:li:activity:empty",
            "content": "",
        },
    ]

    mock_client = AsyncMock()
    mock_run_resp = MagicMock(status_code=201)
    mock_run_resp.json.return_value = {"data": {"id": "run_abc", "defaultDatasetId": "dataset_xyz"}}

    mock_poll_resp = MagicMock(status_code=200)
    mock_poll_resp.json.return_value = {"data": {"status": "SUCCEEDED"}}

    mock_data_resp = MagicMock(status_code=200)
    mock_data_resp.json.return_value = mock_items

    mock_client.post.return_value = mock_run_resp
    mock_client.get.side_effect = [mock_poll_resp, mock_data_resp]

    with patch("httpx.AsyncClient") as mock_http_client:
        mock_http_client.return_value.__aenter__.return_value = mock_client
        leads, diag = await provider.run_post_search(max_posts=5)

    assert len(leads) == 1
    assert leads[0].contact_email == "talent@nextgen.ai"
    assert leads[0].lead_type == LeadType.RECRUITER_POST
    assert diag.posts_returned == 2
    assert diag.leads_created == 1
    assert diag.leads_rejected == 1
    assert diag.emails_found == 1


# -----------------------------------------------------------------------------
# 10. FASTAPI /LINKEDIN ENDPOINTS TEST
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fastapi_linkedin_endpoints(mock_settings: Settings, mock_repository: LinkedInLeadRepository):
    # Seed lead in repository
    lead = mock_repository.create_lead(
        LinkedInLeadCreate(
            post_url="https://www.linkedin.com/feed/update/urn:li:activity:9999",
            post_text="Hiring GenAI Engineer at Frontier. Contact hr@frontier.ai",
            author_name="John Doe",
            company="Frontier",
            contact_email="hr@frontier.ai",
            contact_email_verified=True,
            relevance_score=85.0,
        )
    )

    agent = LinkedInOutreachAgent(
        settings=mock_settings,
        repository=mock_repository,
    )

    with patch("app.api.routers.linkedin.get_agent", return_value=agent):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. GET /linkedin/leads
            resp = await client.get("/linkedin/leads")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) >= 1
            assert data[0]["post_url"] == "https://www.linkedin.com/feed/update/urn:li:activity:9999"

            # 2. GET /linkedin/leads/{lead_id}
            resp_lead = await client.get(f"/linkedin/leads/{lead.id}")
            assert resp_lead.status_code == 200
            assert resp_lead.json()["contact_email"] == "hr@frontier.ai"

            # 3. POST /linkedin/leads/{lead_id}/draft
            resp_draft = await client.post(f"/linkedin/leads/{lead.id}/draft")
            assert resp_draft.status_code == 200
            draft_json = resp_draft.json()
            assert draft_json["recipient_email"] == "hr@frontier.ai"
            assert draft_json["status"] == "DRAFT"

            # 4. GET /linkedin/leads/{lead_id}/draft
            resp_preview = await client.get(f"/linkedin/leads/{lead.id}/draft")
            assert resp_preview.status_code == 200
            assert resp_preview.json()["id"] == draft_json["id"]

            # 5. POST /linkedin/leads/{lead_id}/skip
            resp_skip = await client.post(f"/linkedin/leads/{lead.id}/skip?opt_out=true")
            assert resp_skip.status_code == 200
            assert resp_skip.json()["status"] == "DO_NOT_CONTACT"

            # 6. POST /linkedin/test (Diagnostics)
            resp_diag = await client.post("/linkedin/test")
            assert resp_diag.status_code == 200
            diag_json = resp_diag.json()
            assert "outreach_enabled" in diag_json
            assert diag_json["outreach_auto_send"] is False
