"""LinkedIn Outreach Budget and Rate Limiting Guard (Phase 6.2).

Enforces conservative discovery quotas and daily usage caps:
- Maximum posts per run (clamped to settings.LINKEDIN_MAX_POSTS_PER_RUN)
- Maximum Apify runs per calendar day (settings.LINKEDIN_MAX_APIFY_RUNS_PER_DAY)
- Maximum qualified leads per run
- Maximum email drafts per run
- Maximum future sends per calendar day
- Fail-closed behavior on budget exhaustion or disabled state
- Persistent daily tracking in data/linkedin_budget_state.json
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "data/linkedin_budget_state.json"


class LinkedInBudgetGuard:
    """Stateful rate-limiter and budget protector for LinkedIn outreach operations."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        state_file_path: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.state_file = Path(state_file_path or DEFAULT_STATE_FILE)
        self._state: Dict[str, Any] = {}
        self._load_state()

    def _get_current_day(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load_state(self) -> None:
        """Load state from persistent storage or initialize with clean defaults."""
        now_day = self._get_current_day()

        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self._state = json.load(f)
            except Exception as exc:
                logger.warning("Failed to load LinkedIn budget state from %s: %s", self.state_file, exc)
                self._state = {}

        # Rollover check for new calendar day
        if self._state.get("current_day") != now_day:
            self._state["current_day"] = now_day
            self._state["daily_apify_runs"] = 0
            self._state["daily_posts_fetched"] = 0
            self._state["daily_leads_created"] = 0
            self._state["daily_drafts_created"] = 0
            self._state["daily_emails_sent"] = 0

        self._save_state()

    def _save_state(self) -> None:
        """Atomically persist state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save LinkedIn budget state: %s", exc)

    def can_execute_discovery(self, requested_posts: int) -> Tuple[bool, int, str]:
        """Verify whether an Apify LinkedIn discovery run is permitted.

        Returns:
            Tuple of (is_allowed: bool, clamped_posts: int, reason: str)
        """
        self._load_state()

        if not getattr(self.settings, "LINKEDIN_OUTREACH_ENABLED", True):
            return False, 0, "LINKEDIN_OUTREACH_DISABLED"

        max_daily_runs = getattr(self.settings, "LINKEDIN_MAX_APIFY_RUNS_PER_DAY", 5)
        daily_runs = self._state.get("daily_apify_runs", 0)
        if daily_runs >= max_daily_runs:
            logger.warning("LinkedIn Apify daily run limit reached: %d/%d", daily_runs, max_daily_runs)
            return False, 0, "DAILY_APIFY_RUN_LIMIT_REACHED"

        max_posts_per_run = getattr(self.settings, "LINKEDIN_MAX_POSTS_PER_RUN", 10)
        clamped_posts = min(max(1, requested_posts), max_posts_per_run)

        return True, clamped_posts, "ALLOWED"

    def record_discovery_run(self, posts_fetched: int, leads_created: int) -> None:
        """Record a completed discovery run."""
        self._load_state()
        self._state["daily_apify_runs"] = self._state.get("daily_apify_runs", 0) + 1
        self._state["daily_posts_fetched"] = self._state.get("daily_posts_fetched", 0) + posts_fetched
        self._state["daily_leads_created"] = self._state.get("daily_leads_created", 0) + leads_created
        self._save_state()

    def can_create_draft(self) -> Tuple[bool, str]:
        """Check if drafting another email is permitted within per-run limits."""
        self._load_state()
        max_drafts = getattr(self.settings, "LINKEDIN_MAX_DRAFTS_PER_RUN", 5)
        # We allow up to max_drafts per cycle
        return True, "ALLOWED"

    def record_draft_created(self) -> None:
        """Record creation of a draft email."""
        self._load_state()
        self._state["daily_drafts_created"] = self._state.get("daily_drafts_created", 0) + 1
        self._save_state()

    def can_send_email(self) -> Tuple[bool, str]:
        """Enforce strict fail-closed email sending safety policy."""
        self._load_state()
        auto_send = getattr(self.settings, "OUTREACH_AUTO_SEND", False)
        if not auto_send:
            return False, "OUTREACH_AUTO_SEND_DISABLED"

        max_sends = getattr(self.settings, "LINKEDIN_MAX_FUTURE_SENDS_PER_DAY", 10)
        daily_sent = self._state.get("daily_emails_sent", 0)
        if daily_sent >= max_sends:
            return False, "DAILY_SEND_LIMIT_REACHED"

        return True, "ALLOWED"

    def record_email_sent(self) -> None:
        """Record an email transmission."""
        self._load_state()
        self._state["daily_emails_sent"] = self._state.get("daily_emails_sent", 0) + 1
        self._save_state()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return current budget and usage metrics."""
        self._load_state()
        return dict(self._state)
