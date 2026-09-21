"""Data models for AI Job Application Agent."""

from app.models.profile import (
    CandidateProfileData,
    CareerPreferences,
    EducationItem,
    ExperienceItem,
    PersonalDetails,
    Profile,
    ProfileBase,
    ProfileCreate,
    ProfileUpdate,
    Skill,
    SkillBase,
    SkillCreate,
    SkillItem,
)
from app.models.job import (
    Job,
    JobBase,
    JobCreate,
    JobListResponse,
    JobSearchRequest,
    JobSearchResponse,
    JobStats,
    SourceStatus,
)
from app.models.match import (
    JobMatch,
    JobMatchBase,
    JobMatchCreate,
    MatchBatchResult,
    MatchBreakdown,
    MatchResult,
    MatchRunRequest,
)
from app.models.application import (
    Application,
    ApplicationBase,
    ApplicationCreate,
    ApplicationUpdate,
    ApplicationAnswer,
    ApplicationAnswerBase,
    ApplicationAnswerCreate,
    AutomationLog,
    AutomationLogBase,
    AutomationLogCreate,
)

from app.models.llm import (
    LLMHealthResponse,
    ReasoningOutput,
    SemanticMatchEvaluation,
)

__all__ = [
    # Candidate Profile Data & DB Models
    "CandidateProfileData",
    "PersonalDetails",
    "CareerPreferences",
    "SkillItem",
    "ExperienceItem",
    "EducationItem",
    "Profile",
    "ProfileBase",
    "ProfileCreate",
    "ProfileUpdate",
    "Skill",
    "SkillBase",
    "SkillCreate",
    # Job
    "Job",
    "JobBase",
    "JobCreate",
    "JobListResponse",
    "JobSearchRequest",
    "JobSearchResponse",
    "JobStats",
    "SourceStatus",
    # Match & Engine Models
    "JobMatch",
    "JobMatchBase",
    "JobMatchCreate",
    "MatchBreakdown",
    "MatchResult",
    "MatchRunRequest",
    "MatchBatchResult",
    # LLM & Semantic Models
    "LLMHealthResponse",
    "ReasoningOutput",
    "SemanticMatchEvaluation",
    # Application & Logs
    "Application",
    "ApplicationBase",
    "ApplicationCreate",
    "ApplicationUpdate",
    "ApplicationAnswer",
    "ApplicationAnswerBase",
    "ApplicationAnswerCreate",
    "AutomationLog",
    "AutomationLogBase",
    "AutomationLogCreate",
]
