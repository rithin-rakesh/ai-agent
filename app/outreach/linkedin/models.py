"""LinkedIn Outreach Domain Models and Schemas (Phase 6.2).

Defines entities, enums, and request/response models for:
- LinkedIn Leads (post text, author metadata, hiring intent, public contact info)
- Outreach Email Drafts (personalized messages, resume attachment validation, send safety)
- Discovery Diagnostics and Query Filters
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class LeadType(str, Enum):
    """Categorization of LinkedIn opportunity posts."""
    HIRING_POST = "HIRING_POST"
    REFERRAL_POST = "REFERRAL_POST"
    RECRUITER_POST = "RECRUITER_POST"
    JOB_OPPORTUNITY = "JOB_OPPORTUNITY"
    GENERAL_PROFESSIONAL = "GENERAL_PROFESSIONAL"


class LeadStatus(str, Enum):
    """Lifecycle state of an extracted LinkedIn lead."""
    NEW = "NEW"
    QUALIFIED = "QUALIFIED"
    DRAFTED = "DRAFTED"
    SKIPPED = "SKIPPED"
    CONTACTED = "CONTACTED"
    DO_NOT_CONTACT = "DO_NOT_CONTACT"


class DraftStatus(str, Enum):
    """Status of an outreach draft message."""
    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
    SENT = "SENT"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class LinkedInLeadBase(BaseModel):
    """Base fields shared by lead creation and reading."""
    post_url: str = Field(..., description="Canonical URL of the LinkedIn post")
    post_text: str = Field(..., description="Full text content of the post")
    author_name: Optional[str] = Field(default=None, description="Name of the post author")
    author_profile_url: Optional[str] = Field(default=None, description="LinkedIn profile URL of the author")
    author_headline: Optional[str] = Field(default=None, description="Headline or title of the post author")
    company: Optional[str] = Field(default=None, description="Detected employer or company name")
    company_url: Optional[str] = Field(default=None, description="LinkedIn company page URL")
    published_at: Optional[datetime] = Field(default=None, description="Post publication timestamp")
    matched_keywords: List[str] = Field(default_factory=list, description="Keywords identified in post/author")
    relevance_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Calculated relevance score (0-100)")
    relevance_details: Dict[str, Any] = Field(default_factory=dict, description="Factor breakdown of relevance score")
    lead_type: LeadType = Field(default=LeadType.JOB_OPPORTUNITY, description="Classification of the lead post")
    contact_email: Optional[str] = Field(default=None, description="Publicly listed email address, if any")
    contact_email_source: Optional[str] = Field(default=None, description="Origin of public email: POST_TEXT, AUTHOR_BIO, or None")
    contact_email_verified: bool = Field(default=False, description="Whether email has valid RFC syntax and is non-placeholder")
    status: LeadStatus = Field(default=LeadStatus.NEW, description="Current workflow status")


class LinkedInLeadCreate(LinkedInLeadBase):
    """Model for creating a new LinkedIn lead."""
    pass


class LinkedInLeadUpdate(BaseModel):
    """Model for updating an existing LinkedIn lead."""
    status: Optional[LeadStatus] = None
    relevance_score: Optional[float] = None
    relevance_details: Optional[Dict[str, Any]] = None
    contact_email: Optional[str] = None
    contact_email_source: Optional[str] = None
    contact_email_verified: Optional[bool] = None
    last_contacted_at: Optional[datetime] = None


class LinkedInLead(LinkedInLeadBase):
    """Full persistent LinkedIn lead model."""
    id: UUID = Field(default_factory=uuid4, description="Unique lead identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_contacted_at: Optional[datetime] = Field(default=None, description="Timestamp of last outreach attempt")

    class Config:
        from_attributes = True


class LinkedInOutreachDraft(BaseModel):
    """Personalized outreach email draft with resume attachment metadata."""
    id: UUID = Field(default_factory=uuid4, description="Unique draft identifier")
    lead_id: UUID = Field(..., description="Referenced LinkedIn lead ID")
    recipient_email: str = Field(..., description="Target contact email address")
    recipient_name: str = Field(default="Hiring Team", description="Target recipient contact name")
    subject: str = Field(..., description="Professional email subject line")
    body_text: str = Field(..., description="Personalized body text of outreach email")
    attachment_path: str = Field(..., description="Absolute or relative path to resume PDF")
    attachment_verified: bool = Field(default=False, description="Whether the resume PDF exists on disk")
    personalization_fields: Dict[str, Any] = Field(default_factory=dict, description="Values used in email template")
    generation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata on generator, prompt, timestamps")
    status: DraftStatus = Field(default=DraftStatus.DRAFT, description="Status of the draft")
    sent_at: Optional[datetime] = Field(default=None, description="Timestamp of transmission if sent")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        from_attributes = True


class LinkedInDiscoveryDiagnostics(BaseModel):
    """Structured diagnostic summary returned by discovery cycles."""
    provider: str = Field(default="apify")
    actor: str = Field(default="harvestapi~linkedin-post-search")
    queries_requested: int = 0
    posts_returned: int = 0
    posts_deduplicated: int = 0
    leads_created: int = 0
    leads_rejected: int = 0
    emails_found: int = 0
    errors: List[str] = Field(default_factory=list)


class LinkedInDiscoverRequest(BaseModel):
    """Request payload to trigger LinkedIn post discovery."""
    search_queries: Optional[List[str]] = Field(
        default=None,
        description="Optional explicit search queries. If omitted, built dynamically from keywords config",
    )
    max_posts: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Maximum posts to scrape in this run (defaults to settings.LINKEDIN_MAX_POSTS_PER_RUN)",
    )
    date_posted: str = Field(
        default="past-week",
        description="Apify date filter: 'past-24h', 'past-week', 'past-month'",
    )
    minimum_relevance_score: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Minimum score required for post to be marked QUALIFIED",
    )
    auto_generate_drafts: bool = Field(
        default=False,
        description="Whether to immediately generate email drafts for qualified leads with verified emails",
    )


class LinkedInDiscoverResponse(BaseModel):
    """Response payload returned by POST /linkedin/discover."""
    status: str = Field(..., description="Overall discovery execution status: success, partial, or failed")
    leads_count: int = 0
    qualified_count: int = 0
    diagnostics: LinkedInDiscoveryDiagnostics
    leads: List[LinkedInLead] = Field(default_factory=list)
