"""Apify Budget Guard module.

Strictly enforces discovery budget limits:
- Pricing model: $4.00 per 1,000 results ($0.004 per result)
- Hard monthly cap: 500 results per month ($2.00 max monthly spend)
- Per-run result clamping and daily run frequency caps
- Fail-closed behavior on budget exhaustion or disabled state
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "data/apify_budget_state.json"


class ApifyBudgetGuard:
    """Stateful budget guard protecting against unexpected Apify scraping expenditure."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        state_file_path: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.state_file = Path(state_file_path or DEFAULT_STATE_FILE)
        self._state: Dict[str, Any] = {}
        self._load_state()

    def _get_current_month(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _get_current_day(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load_state(self) -> None:
        """Load budget state from persistent storage or initialize defaults."""
        now_month = self._get_current_month()
        now_day = self._get_current_day()

        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self._state = json.load(f)
            except Exception as exc:
                logger.warning("Failed to load budget state file (%s), reinitializing: %s", self.state_file, exc)
                self._state = {}

        # Monthly rollover check
        if self._state.get("current_month") != now_month:
            self._state["current_month"] = now_month
            self._state["monthly_results_count"] = 0
            self._state["monthly_runs_count"] = 0
            self._state["monthly_estimated_cost"] = 0.0

        # Daily rollover check
        if self._state.get("current_day") != now_day:
            self._state["current_day"] = now_day
            self._state["daily_runs_count"] = 0

        self._save_state()

    def _save_state(self) -> None:
        """Persist state to disk safely."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save budget state: %s", exc)

    def can_execute(self, requested_results: int) -> Tuple[bool, int, str]:
        """Evaluate if an Apify discovery run is allowed under current budget constraints.

        Args:
            requested_results: Desired result count for the query

        Returns:
            Tuple of (is_allowed: bool, clamped_results: int, reason_code: str)
        """
        self._load_state()

        if not self.settings.APIFY_GLASSDOOR_ENABLED:
            logger.info("Apify Glassdoor discovery is disabled by configuration")
            return False, 0, "APIFY_GLASSDOOR_DISABLED"

        # Check daily runs limit
        daily_runs = self._state.get("daily_runs_count", 0)
        if daily_runs >= self.settings.APIFY_GLASSDOOR_MAX_RUNS_PER_DAY:
            logger.warning("Daily Apify run limit reached: %d/%d", daily_runs, self.settings.APIFY_GLASSDOOR_MAX_RUNS_PER_DAY)
            return False, 0, "APIFY_DAILY_LIMIT_REACHED"

        # Check monthly results limit (e.g. 500)
        current_results = self._state.get("monthly_results_count", 0)
        max_monthly = self.settings.APIFY_GLASSDOOR_MAX_RESULTS_PER_MONTH
        remaining_quota = max_monthly - current_results

        if remaining_quota <= 0:
            logger.warning(
                "Monthly Apify budget limit reached: %d/%d results used ($%.2f spent)",
                current_results,
                max_monthly,
                self._state.get("monthly_estimated_cost", 0.0),
            )
            return False, 0, "APIFY_BUDGET_LIMIT_REACHED"

        # Clamp requested results to per-run max and remaining monthly quota
        per_run_max = self.settings.APIFY_GLASSDOOR_MAX_RESULTS_PER_RUN
        clamped = min(requested_results, per_run_max, remaining_quota)

        if clamped <= 0:
            return False, 0, "APIFY_BUDGET_LIMIT_REACHED"

        return True, clamped, "OK"

    def record_run(
        self,
        results_returned: int,
        actual_cost: Optional[float] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Record usage metrics after an Apify Actor run completes.

        Args:
            results_returned: Number of job posts returned
            actual_cost: Actual USD charge reported by Apify if available
            run_id: Apify run identifier
        """
        self._load_state()

        cost_per_item = self.settings.APIFY_GLASSDOOR_COST_PER_RESULT
        run_cost = actual_cost if actual_cost is not None else (results_returned * cost_per_item)

        self._state["monthly_results_count"] = self._state.get("monthly_results_count", 0) + results_returned
        self._state["monthly_runs_count"] = self._state.get("monthly_runs_count", 0) + 1
        self._state["daily_runs_count"] = self._state.get("daily_runs_count", 0) + 1
        self._state["monthly_estimated_cost"] = round(self._state.get("monthly_estimated_cost", 0.0) + run_cost, 4)
        self._state["last_run_at"] = datetime.now(timezone.utc).isoformat()
        if run_id:
            self._state["last_run_id"] = run_id

        self._save_state()
        logger.info(
            "Apify run recorded: +%d results (month total: %d/%d), run cost: $%.4f (month spend: $%.4f)",
            results_returned,
            self._state["monthly_results_count"],
            self.settings.APIFY_GLASSDOOR_MAX_RESULTS_PER_MONTH,
            run_cost,
            self._state["monthly_estimated_cost"],
        )

    def get_budget_status(self) -> Dict[str, Any]:
        """Return a structured summary of budget usage and limits."""
        self._load_state()
        current_results = self._state.get("monthly_results_count", 0)
        max_monthly = self.settings.APIFY_GLASSDOOR_MAX_RESULTS_PER_MONTH
        remaining = max(0, max_monthly - current_results)

        return {
            "enabled": self.settings.APIFY_GLASSDOOR_ENABLED,
            "monthly_results_count": current_results,
            "monthly_results_limit": max_monthly,
            "remaining_monthly_quota": remaining,
            "monthly_runs_count": self._state.get("monthly_runs_count", 0),
            "daily_runs_count": self._state.get("daily_runs_count", 0),
            "daily_runs_limit": self.settings.APIFY_GLASSDOOR_MAX_RUNS_PER_DAY,
            "max_results_per_run": self.settings.APIFY_GLASSDOOR_MAX_RESULTS_PER_RUN,
            "cost_per_result_usd": self.settings.APIFY_GLASSDOOR_COST_PER_RESULT,
            "estimated_monthly_cost_usd": self._state.get("monthly_estimated_cost", 0.0),
            "current_month": self._state.get("current_month"),
            "current_day": self._state.get("current_day"),
            "last_run_at": self._state.get("last_run_at"),
            "last_run_id": self._state.get("last_run_id"),
        }
