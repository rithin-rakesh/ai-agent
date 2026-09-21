"""Application Settings module using Pydantic Settings.

Manages environment variables, validation, and secret protection.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=(str(ENV_FILE_PATH), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


    # Supabase Configuration
    SUPABASE_URL: str = Field(
        default="",
        description="The Supabase project URL (e.g. https://xyz.supabase.co)",
    )

    @field_validator("SUPABASE_URL", mode="before")
    @classmethod
    def validate_supabase_url(cls, v: Any) -> str:
        """Normalize Supabase URL by trimming trailing slashes and /rest/v1 paths."""
        if isinstance(v, str) and v.strip():
            cleaned = v.strip().rstrip("/")
            if cleaned.endswith("/rest/v1"):
                cleaned = cleaned[:-8].rstrip("/")
            return cleaned
        return v if isinstance(v, str) else ""
    SUPABASE_ANON_KEY: str = Field(
        default="",
        description="The Supabase anonymous / public API key",
    )
    SUPABASE_SERVICE_ROLE_KEY: Optional[SecretStr] = Field(
        default=None,
        description="The Supabase service-role secret key (never printed/logged)",
    )

    # Application Configuration
    APP_ENV: str = Field(
        default="development",
        description="Application environment: development, staging, production, test",
    )
    APP_HOST: str = Field(
        default="0.0.0.0",
        description="Host interface to bind FastAPI server to",
    )
    APP_PORT: int = Field(
        default=8000,
        description="Port for FastAPI server",
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    # JobSpy MCP Configuration
    JOBSPY_ENABLED: bool = Field(
        default=True,
        description="Whether JobSpy integration is enabled",
    )
    JOBSPY_HOST: str = Field(
        default="127.0.0.1",
        description="JobSpy MCP server host",
    )
    JOBSPY_PORT: int = Field(
        default=9423,
        description="JobSpy MCP server port",
    )
    JOBSPY_TIMEOUT_SECONDS: int = Field(
        default=120,
        description="Timeout in seconds for JobSpy queries per source",
    )

    # Candidate Profile Configuration
    CANDIDATE_PROFILE_PATH: str = Field(
        default="data/profile/candidate_profile.json",
        description="Relative or absolute path to the local candidate profile JSON file",
    )

    # NVIDIA AI Configuration (Phase 4 Semantic Matching)
    NVIDIA_BASE_URL: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="NVIDIA NIM API base URL",
    )

    @field_validator("NVIDIA_BASE_URL", mode="before")
    @classmethod
    def validate_nvidia_base_url(cls, v: Any) -> str:
        """Normalize NVIDIA base URL by trimming trailing slashes."""
        if isinstance(v, str) and v.strip():
            return v.strip().rstrip("/")
        return "https://integrate.api.nvidia.com/v1"

    NVIDIA_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="NVIDIA NIM API key (protected, never logged)",
    )
    NVIDIA_REASONING_MODEL: str = Field(
        default="nvidia/nemotron-3.5-lightning-30b-a3b",
        description="NVIDIA reasoning model identifier",
    )
    NVIDIA_EMBEDDING_MODEL: str = Field(
        default="nvidia/nemotron-3-embed-1b",
        description="NVIDIA embedding model identifier",
    )
    NVIDIA_REQUEST_TIMEOUT_SECONDS: int = Field(
        default=60,
        description="Timeout in seconds for NVIDIA API requests",
    )

    # Apify Configuration (Glassdoor & Web Scraper Actors)
    APIFY_API_TOKEN: Optional[SecretStr] = Field(
        default=None,
        description="Apify API token for web scraper Actors (protected, never logged)",
    )
    APIFY_GLASSDOOR_ENABLED: bool = Field(
        default=False,
        description="Whether Apify Glassdoor discovery is enabled (budget-guarded)",
    )
    APIFY_GLASSDOOR_ACTOR_ID: str = Field(
        default="valig~glassdoor-jobs-scraper",
        description="Apify Actor ID for Glassdoor scraping",
    )
    APIFY_GLASSDOOR_MAX_RESULTS_PER_RUN: int = Field(
        default=20,
        description="Maximum results allowed per individual Apify run",
    )
    APIFY_GLASSDOOR_MAX_RESULTS_PER_MONTH: int = Field(
        default=500,
        description="Hard monthly cap on total results scraped via Apify (500 results/month)",
    )
    APIFY_GLASSDOOR_COST_PER_RESULT: Optional[float] = Field(
        default=0.0004,
        description="Configurable cost in USD per scraped job result ($0.0004 for valig, $0.40/1k, None if unknown)",
    )
    APIFY_GLASSDOOR_MAX_RUNS_PER_DAY: int = Field(
        default=20,
        description="Maximum allowed Apify Actor discovery runs per calendar day",
    )
    APIFY_GLASSDOOR_MAX_CONCURRENCY: int = Field(
        default=1,
        description="Maximum concurrent Apify discovery runs permitted (default 1 to protect budget)",
    )
    APIFY_GLASSDOOR_TIMEOUT_SECONDS: int = Field(
        default=120,
        description="Maximum timeout in seconds for Apify Actor execution and polling",
    )
    APIFY_GLASSDOOR_FALLBACK_TO_JOBSPY: bool = Field(
        default=False,
        description="Whether to fall back to JobSpy for Glassdoor (disabled: Glassdoor is Apify only)",
    )
    APIFY_ACTOR_FAILURE_THRESHOLD: int = Field(
        default=2,
        description="Consecutive Actor failure threshold before circuit opens",
    )
    APIFY_ACTOR_COOLDOWN_MINUTES: int = Field(
        default=60,
        description="Cooldown in minutes before re-attempting a tripped Actor circuit",
    )

    @property
    def apify_token_value(self) -> Optional[str]:
        """Return the raw, sanitized Apify token string from settings or environment.

        Strips whitespace, newlines, carriage returns, quotes, and accidental 'Bearer ' prefix.
        Never exposes the secret token value.
        """
        raw: Optional[str] = None
        if self.APIFY_API_TOKEN is not None:
            raw = self.APIFY_API_TOKEN.get_secret_value()
        else:
            import os
            raw = os.getenv("APIFY_API_TOKEN")

        if not raw:
            return None


        # Sanitize whitespace, newlines, carriage returns
        cleaned = raw.strip().strip("\r\n\t ").strip("'\"").strip()
        # Strip accidental 'Bearer ' or 'bearer ' prefixes
        while cleaned.lower().startswith("bearer "):
            cleaned = cleaned[7:].strip().strip("'\"").strip()

        return cleaned if cleaned else None


    SEMANTIC_MATCHING_ENABLED: bool = Field(
        default=True,
        description="Whether semantic matching with NVIDIA is enabled",
    )
    DETERMINISTIC_SEMANTIC_THRESHOLD: float = Field(
        default=60.0,
        description="Minimum deterministic score (0-100) required to trigger LLM reasoning",
    )
    DETERMINISTIC_WEIGHT: float = Field(
        default=0.70,
        description="Composite score weight for deterministic match (70%)",
    )
    EMBEDDING_WEIGHT: float = Field(
        default=0.20,
        description="Composite score weight for embedding similarity (20%)",
    )
    REASONING_WEIGHT: float = Field(
        default=0.10,
        description="Composite score weight for reasoning model evaluation (10%)",
    )

    # Browser Automation Configuration (Phase 5)
    PLAYWRIGHT_HEADLESS: bool = Field(
        default=False,
        description="Whether to run browser in headless mode (default visible for manual login)",
    )
    PLAYWRIGHT_BROWSER: str = Field(
        default="chromium",
        description="Browser engine to launch (chromium, firefox, webkit)",
    )
    PLAYWRIGHT_TIMEOUT_MS: int = Field(
        default=30000,
        description="Default Playwright action and navigation timeout in milliseconds",
    )
    PLAYWRIGHT_TRACE: bool = Field(
        default=False,
        description="Whether to record Playwright execution traces for debugging",
    )
    INDEED_BROWSER_PROFILE_PATH: str = Field(
        default="browser_sessions/indeed",
        description="Path for persistent Indeed browser session profile data",
    )
    SCREENSHOTS_PATH: str = Field(
        default="screenshots",
        description="Directory path for saving verification screenshots",
    )
    PLAYWRIGHT_ARTIFACTS_PATH: str = Field(
        default="playwright-artifacts",
        description="Directory path for saving Playwright trace zip artifacts",
    )

    @property
    def jobspy_api_url(self) -> str:
        """Return the base URL for the JobSpy MCP server API."""
        return f"http://{self.JOBSPY_HOST}:{self.JOBSPY_PORT}/api"

    @property
    def jobspy_health_url(self) -> str:
        """Return the health URL for the JobSpy MCP server."""
        return f"http://{self.JOBSPY_HOST}:{self.JOBSPY_PORT}/health"

    @property
    def is_production(self) -> bool:
        """Return True if running in production environment."""
        return self.APP_ENV.lower() == "production"

    @property
    def is_test(self) -> bool:
        """Return True if running in test environment."""
        return self.APP_ENV.lower() == "test"

    # Automation Question & Final Submit Configuration (Phase 5.8)
    ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS: bool = Field(
        default=False,
        description="Whether to attempt progression when required questions are unknown and unanswered",
    )
    AUTO_SUBMIT_ENABLED: bool = Field(
        default=False,
        description="Whether automated job application may perform final Submit (default False stops at SUBMISSION_READY)",
    )
    CANDIDATE_ANSWERS_FILE_PATH: str = Field(
        default="data/candidate_answers.json",
        description="Path to persistent Candidate Answer Bank JSON storage",
    )
    APPLICATION_DELAY_SECONDS: int = Field(
        default=5,
        description="Delay between consecutive job applications in seconds",
    )
    APPLICATION_REQUIRE_EASY_APPLY: bool = Field(
        default=True,
        description="When True, only jobs supporting Easy Apply (Indeed Apply / Glassdoor Easy Apply) are application-eligible",
    )
    MAX_APPLICATIONS_PER_DAY: int = Field(
        default=10,
        description="Maximum total job applications allowed per calendar day across all platforms",
    )
    MAX_INDEED_APPLICATIONS_PER_DAY: int = Field(
        default=5,
        description="Maximum Indeed applications allowed per calendar day",
    )
    MAX_GLASSDOOR_APPLICATIONS_PER_DAY: int = Field(
        default=5,
        description="Maximum Glassdoor applications allowed per calendar day",
    )
    MAX_JOB_AGE_DAYS: int = Field(
        default=7,
        description="Maximum age of job postings in days before considered stale",
    )
    COOLDOWN_SECONDS_BETWEEN_APPLICATIONS: int = Field(
        default=15,
        description="Cooldown interval in seconds between consecutive browser applications",
    )
    ALLOW_EXTERNAL_APPLY: bool = Field(
        default=False,
        description="Whether to allow applying to external company site jobs",
    )

    # LinkedIn Outreach Configuration (Phase 6.2)
    LINKEDIN_OUTREACH_ENABLED: bool = Field(
        default=True,
        description="Whether LinkedIn outreach discovery and lead management is enabled",
    )
    LINKEDIN_POST_SEARCH_ACTOR_ID: str = Field(
        default="harvestapi~linkedin-post-search",
        description="Apify Actor ID for searching LinkedIn posts",
    )
    LINKEDIN_PROFILE_POSTS_ACTOR_ID: str = Field(
        default="harvestapi~linkedin-profile-posts",
        description="Apify Actor ID for profile enrichment posts",
    )
    LINKEDIN_MAX_POSTS_PER_RUN: int = Field(
        default=10,
        description="Maximum posts retrieved per individual LinkedIn discovery run",
    )
    LINKEDIN_MAX_APIFY_RUNS_PER_DAY: int = Field(
        default=5,
        description="Maximum allowed Apify Actor discovery runs per calendar day for LinkedIn",
    )
    LINKEDIN_MAX_LEADS_PER_RUN: int = Field(
        default=10,
        description="Maximum qualified leads created per discovery run",
    )
    LINKEDIN_MAX_DRAFTS_PER_RUN: int = Field(
        default=5,
        description="Maximum outreach email drafts generated per run",
    )
    LINKEDIN_MAX_FUTURE_SENDS_PER_DAY: int = Field(
        default=10,
        description="Maximum future email sends permitted per calendar day (when sending is enabled)",
    )
    RESUME_PDF_PATH: str = Field(
        default="data/profile/resume.pdf",
        description="Path to the candidate resume PDF file for outreach attachments",
    )
    OUTREACH_AUTO_SEND: bool = Field(
        default=False,
        description="Safety gate: whether outreach emails may be sent automatically (default False for draft-only mode)",
    )
    LINKEDIN_KEYWORDS_CONFIG_PATH: str = Field(
        default="data/linkedin_keywords.json",
        description="Path to custom LinkedIn search keywords configuration JSON",
    )


    def get_service_role_key(self) -> Optional[str]:
        """Safely retrieve the service role key as plain string for database client use.

        Returns None if not configured.
        """
        if self.SUPABASE_SERVICE_ROLE_KEY is None:
            return None
        return self.SUPABASE_SERVICE_ROLE_KEY.get_secret_value()

    def get_nvidia_api_key(self) -> Optional[str]:
        """Safely retrieve the NVIDIA API key as plain string for client headers.

        Returns None if not configured.
        """
        if self.NVIDIA_API_KEY is None:
            return None
        return self.NVIDIA_API_KEY.get_secret_value()

    def resolve_path(self, relative_or_absolute_path: str) -> Path:
        """Resolve a path relative to the project root directory."""
        path = Path(relative_or_absolute_path)
        if path.is_absolute():
            return path
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / path

    def get_indeed_profile_path(self) -> Path:
        """Return the resolved absolute path for the Indeed persistent browser profile."""
        return self.resolve_path(self.INDEED_BROWSER_PROFILE_PATH)

    def get_screenshots_path(self) -> Path:
        """Return the resolved absolute path for screenshots."""
        return self.resolve_path(self.SCREENSHOTS_PATH)

    def get_playwright_artifacts_path(self) -> Path:
        """Return the resolved absolute path for Playwright artifacts/traces."""
        return self.resolve_path(self.PLAYWRIGHT_ARTIFACTS_PATH)

    def get_resume_pdf_path(self) -> Path:
        """Return the resolved absolute path for the candidate resume PDF."""
        return self.resolve_path(self.RESUME_PDF_PATH)



@lru_cache()
def get_settings() -> Settings:
    """Provide a cached instance of application settings."""
    return Settings()
