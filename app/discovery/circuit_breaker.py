"""Actor Circuit Breaker for Apify Providers.

Prevents repeated execution against failing or broken Actors:
- Permanently blacklists known broken actors (orgupdate~glassdoor-jobs-scraper -> ACTOR_UNAVAILABLE)
- Tracks consecutive execution failures per Actor
- Trips to ACTOR_CIRCUIT_OPEN after APIFY_ACTOR_FAILURE_THRESHOLD consecutive failures
- Enforces APIFY_ACTOR_COOLDOWN_MINUTES cooldown before entering HALF_OPEN trial state
- Persists state to data/apify_circuit_breaker.json across process restarts
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Set, Tuple

from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "data/apify_circuit_breaker.json"

PERMANENTLY_UNAVAILABLE_ACTORS: Set[str] = {
    "orgupdate~glassdoor-jobs-scraper",
    "orgupdate/glassdoor-jobs-scraper",
}


class ActorCircuitBreaker:
    """Manages circuit breaker state per Apify Actor."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        state_file_path: Optional[str] = None,
        failure_threshold: Optional[int] = None,
        cooldown_minutes: Optional[int] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.state_file_path = state_file_path or DEFAULT_STATE_FILE
        self.failure_threshold = (
            failure_threshold
            if failure_threshold is not None
            else getattr(self.settings, "APIFY_ACTOR_FAILURE_THRESHOLD", 2)
        )
        self.cooldown_seconds = (
            (cooldown_minutes if cooldown_minutes is not None else getattr(self.settings, "APIFY_ACTOR_COOLDOWN_MINUTES", 60))
            * 60
        )
        self._actors: Dict[str, Dict[str, Any]] = {}
        self._load_state()

    def _normalize_actor_id(self, actor_id: str) -> str:
        return actor_id.replace("/", "~").strip()

    def _load_state(self) -> None:
        if not os.path.exists(self.state_file_path):
            return
        try:
            with open(self.state_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self._actors = data.get("actors", {})
        except Exception as exc:
            logger.warning("Could not load circuit breaker state from %s: %s", self.state_file_path, exc)

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.state_file_path)), exist_ok=True)
            with open(self.state_file_path, "w", encoding="utf-8") as f:
                json.dump({"actors": self._actors, "updated_at": time.time()}, f, indent=2)
        except Exception as exc:
            logger.warning("Could not persist circuit breaker state to %s: %s", self.state_file_path, exc)

    def is_actor_permanently_unavailable(self, actor_id: str) -> bool:
        """Check if an Actor is in the permanent blacklist."""
        norm = self._normalize_actor_id(actor_id)
        return norm in PERMANENTLY_UNAVAILABLE_ACTORS or actor_id in PERMANENTLY_UNAVAILABLE_ACTORS

    def can_execute(self, actor_id: str) -> Tuple[bool, str]:
        """Check whether execution is permitted for the given Actor.

        Returns:
            Tuple[bool, str]: (is_allowed, reason)
            If rejected, reason is either 'ACTOR_UNAVAILABLE' or 'ACTOR_CIRCUIT_OPEN'.
        """
        norm_id = self._normalize_actor_id(actor_id)

        # 1. Check permanent blacklist
        if self.is_actor_permanently_unavailable(norm_id):
            logger.warning("Execution blocked: Actor '%s' is permanently marked ACTOR_UNAVAILABLE", norm_id)
            return False, "ACTOR_UNAVAILABLE"

        # 2. Check circuit state
        actor_state = self._actors.get(norm_id)
        if not actor_state:
            return True, "CLOSED"

        state = actor_state.get("state", "CLOSED")
        if state == "OPEN":
            tripped_at = actor_state.get("tripped_at", 0)
            elapsed = time.time() - tripped_at
            if elapsed < self.cooldown_seconds:
                remaining_min = int((self.cooldown_seconds - elapsed) / 60)
                logger.warning(
                    "Execution blocked: Actor '%s' circuit is OPEN (tripped %ds ago, cooldown %dm remaining). Reason: %s",
                    norm_id,
                    int(elapsed),
                    remaining_min,
                    actor_state.get("last_error"),
                )
                return False, "ACTOR_CIRCUIT_OPEN"
            else:
                logger.info("Actor '%s' circuit transitioned from OPEN to HALF_OPEN after cooldown", norm_id)
                actor_state["state"] = "HALF_OPEN"
                self._save_state()
                return True, "HALF_OPEN"

        return True, state

    def record_success(self, actor_id: str) -> None:
        """Record successful completion of an Actor run, closing the circuit."""
        norm_id = self._normalize_actor_id(actor_id)
        actor_state = self._actors.setdefault(norm_id, {})
        actor_state["state"] = "CLOSED"
        actor_state["consecutive_failures"] = 0
        actor_state["last_success_at"] = time.time()
        self._save_state()
        logger.info("Actor '%s' run succeeded; circuit state is CLOSED", norm_id)

    def record_failure(self, actor_id: str, error_reason: str) -> bool:
        """Record execution failure for an Actor.

        Returns:
            bool: True if this failure caused the circuit to trip OPEN, False otherwise.
        """
        norm_id = self._normalize_actor_id(actor_id)
        actor_state = self._actors.setdefault(norm_id, {"consecutive_failures": 0, "state": "CLOSED"})
        actor_state["consecutive_failures"] = actor_state.get("consecutive_failures", 0) + 1
        actor_state["last_failure_at"] = time.time()
        actor_state["last_error"] = str(error_reason)[:300]

        failures = actor_state["consecutive_failures"]
        tripped = False
        if failures >= self.failure_threshold:
            actor_state["state"] = "OPEN"
            actor_state["tripped_at"] = time.time()
            tripped = True
            logger.error(
                "Actor '%s' circuit TRIPPED OPEN after %d consecutive failures (threshold=%d). Error: %s",
                norm_id,
                failures,
                self.failure_threshold,
                error_reason,
            )
        else:
            logger.warning(
                "Actor '%s' failure recorded (%d/%d before trip): %s",
                norm_id,
                failures,
                self.failure_threshold,
                error_reason,
            )

        self._save_state()
        return tripped

    def get_status(self, actor_id: Optional[str] = None) -> Dict[str, Any]:
        """Return circuit breaker diagnostic status for an Actor or all tracked actors."""
        if actor_id:
            norm_id = self._normalize_actor_id(actor_id)
            is_unavail = self.is_actor_permanently_unavailable(norm_id)
            state_info = self._actors.get(norm_id, {"state": "CLOSED", "consecutive_failures": 0})
            effective_state = "UNAVAILABLE" if is_unavail else state_info.get("state", "CLOSED")
            return {
                "actor_id": norm_id,
                "state": effective_state,
                "is_permanently_unavailable": is_unavail,
                "consecutive_failures": state_info.get("consecutive_failures", 0),
                "failure_threshold": self.failure_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "last_error": state_info.get("last_error"),
                "tripped_at": state_info.get("tripped_at"),
            }

        return {
            "failure_threshold": self.failure_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "tracked_actors": {
                aid: {
                    **info,
                    "is_permanently_unavailable": self.is_actor_permanently_unavailable(aid),
                }
                for aid, info in self._actors.items()
            },
        }

    def reset(self, actor_id: Optional[str] = None) -> None:
        """Reset circuit breaker state (primarily for tests)."""
        if actor_id:
            norm_id = self._normalize_actor_id(actor_id)
            self._actors.pop(norm_id, None)
        else:
            self._actors.clear()
        self._save_state()
