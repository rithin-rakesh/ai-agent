"""LinkedIn Outreach Keywords and Search Configuration (Phase 6.2).

Manages configurable keyword groups for:
- Role keywords (AI Engineer, Machine Learning, Data Science, etc.)
- Hiring-intent keywords (hiring, looking for, vacancy, referral, etc.)
- Location and author title indicators

Supports dynamic overrides from data/linkedin_keywords.json or environment variables.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

DEFAULT_ROLE_KEYWORDS: List[str] = [
    "AI Engineer",
    "AI/ML",
    "Machine Learning",
    "ML Engineer",
    "Data Scientist",
    "Data Science",
    "Generative AI",
    "GenAI",
    "Agentic AI",
    "Artificial Intelligence",
]

DEFAULT_HIRING_INTENT_KEYWORDS: List[str] = [
    "hiring",
    "we're hiring",
    "looking for",
    "opening",
    "openings",
    "vacancy",
    "job",
    "jobs",
    "recruiting",
    "recruiter",
    "referral",
    "internship",
    "intern",
    "opportunity",
    "join our team",
]

DEFAULT_TARGET_LOCATIONS: List[str] = [
    "India",
    "Bangalore",
    "Bengaluru",
    "Kochi",
    "Kerala",
    "Chennai",
    "Hyderabad",
    "Remote",
]

DEFAULT_AUTHOR_INDICATORS: Dict[str, List[str]] = {
    "recruiter": ["recruiter", "talent acquisition", "sourcer", "staffing", "ta partner"],
    "hiring_manager": ["engineering manager", "founder", "co-founder", "cto", "ceo", "director", "head of ai", "lead", "vp"],
    "employee_referral": ["teammate", "colleague", "dm me", "reach out", "send resume"],
}


class LinkedInKeywordsConfig:
    """Manages role keywords, hiring intent keywords, and query construction."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.settings = get_settings()
        self.config_path = Path(config_path or getattr(self.settings, "LINKEDIN_KEYWORDS_CONFIG_PATH", "data/linkedin_keywords.json"))
        self.role_keywords = list(DEFAULT_ROLE_KEYWORDS)
        self.hiring_intent_keywords = list(DEFAULT_HIRING_INTENT_KEYWORDS)
        self.target_locations = list(DEFAULT_TARGET_LOCATIONS)
        self.author_indicators = dict(DEFAULT_AUTHOR_INDICATORS)
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from JSON file or environment variables if present."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data.get("role_keywords"), list) and data["role_keywords"]:
                    self.role_keywords = [str(k).strip() for k in data["role_keywords"] if str(k).strip()]
                if isinstance(data.get("hiring_intent_keywords"), list) and data["hiring_intent_keywords"]:
                    self.hiring_intent_keywords = [str(k).strip() for k in data["hiring_intent_keywords"] if str(k).strip()]
                if isinstance(data.get("target_locations"), list) and data["target_locations"]:
                    self.target_locations = [str(k).strip() for k in data["target_locations"] if str(k).strip()]
                if isinstance(data.get("author_title_indicators"), dict):
                    self.author_indicators = data["author_title_indicators"]
                logger.info("Loaded custom LinkedIn keywords from %s", self.config_path)
            except Exception as exc:
                logger.warning("Failed to load LinkedIn keywords config from %s: %s", self.config_path, exc)

        # Environment variable overrides (comma-separated strings)
        env_roles = os.getenv("LINKEDIN_ROLE_KEYWORDS")
        if env_roles:
            self.role_keywords = [r.strip() for r in env_roles.split(",") if r.strip()]

        env_hiring = os.getenv("LINKEDIN_HIRING_KEYWORDS")
        if env_hiring:
            self.hiring_intent_keywords = [h.strip() for h in env_hiring.split(",") if h.strip()]

    def build_search_queries(self, max_queries: int = 3) -> List[str]:
        """Construct boolean search queries combining role keywords and hiring intent.

        Example:
            '("AI Engineer" OR "Machine Learning") AND ("hiring" OR "we\'re hiring" OR "opening")'
        """
        queries: List[str] = []
        # Chunk roles to avoid exceeding query length limits
        chunk_size = max(2, (len(self.role_keywords) + max_queries - 1) // max_queries)
        intent_sample = self.hiring_intent_keywords[:6]
        intent_expr = " OR ".join(f'"{kw}"' if " " in kw else kw for kw in intent_sample)

        for i in range(0, len(self.role_keywords), chunk_size):
            roles_chunk = self.role_keywords[i:i + chunk_size]
            role_expr = " OR ".join(f'"{kw}"' if " " in kw else kw for kw in roles_chunk)
            q = f"({role_expr}) AND ({intent_expr})"
            queries.append(q)
            if len(queries) >= max_queries:
                break

        return queries or ['("AI Engineer" OR "Machine Learning") AND (hiring OR "we\'re hiring")']

    def find_matched_keywords(self, text: str) -> List[str]:
        """Identify which role and hiring intent keywords match in a given text snippet."""
        if not text:
            return []
        low = text.lower()
        matched: List[str] = []
        for kw in self.role_keywords:
            if kw.lower() in low:
                matched.append(kw)
        for kw in self.hiring_intent_keywords:
            if kw.lower() in low:
                matched.append(kw)
        return matched


_default_keywords_config: Optional[LinkedInKeywordsConfig] = None


def get_keywords_config() -> LinkedInKeywordsConfig:
    """Retrieve singleton instance of LinkedInKeywordsConfig."""
    global _default_keywords_config
    if _default_keywords_config is None:
        _default_keywords_config = LinkedInKeywordsConfig()
    return _default_keywords_config
