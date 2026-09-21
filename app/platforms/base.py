"""Base platform adapter abstraction.

Defines the contract for job platform browser automation adapters with
first-class policy enforcement before every browser action.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional

from app.automation.session_manager import SessionStatus
from app.platforms.policy import (
    AutomationAction,
    PlatformAutomationPolicy,
    PolicyEvaluationResult,
    get_platform_policy,
)

logger = logging.getLogger(__name__)


class PlatformAdapter(ABC):
    """Abstract base class for platform-specific browser interaction and session adapters."""

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """The identifier of the platform (e.g. 'indeed', 'linkedin')."""
        pass

    def get_policy(self) -> PlatformAutomationPolicy:
        """Fetch the active automation policy for this platform."""
        return get_platform_policy(self.platform_name)

    def can_navigate(self) -> bool:
        """Whether navigation to job URLs is allowed by policy."""
        return self.get_policy().is_action_allowed(AutomationAction.NAVIGATE)

    def can_inspect(self) -> bool:
        """Whether read-only DOM and page inspection is allowed by policy."""
        return self.get_policy().is_action_allowed(AutomationAction.INSPECT_JOB)

    def can_detect_forms(self) -> bool:
        """Whether application form detection is allowed by policy."""
        return self.get_policy().is_action_allowed(AutomationAction.DETECT_FORM)

    def can_fill_form(self) -> bool:
        """Whether automated form filling is allowed by policy."""
        return self.get_policy().is_action_allowed(AutomationAction.FILL_FORM)

    def can_generate_answers(self) -> bool:
        """Whether AI answer generation is allowed by policy."""
        return self.get_policy().is_action_allowed(AutomationAction.GENERATE_ANSWERS)

    def can_submit(self) -> bool:
        """Whether final application submission is allowed by policy."""
        return self.get_policy().is_action_allowed(AutomationAction.SUBMIT_APPLICATION)

    def evaluate_action(self, action: AutomationAction) -> PolicyEvaluationResult:
        """Evaluate whether a proposed automation action is permitted."""
        return self.get_policy().evaluate_action(action)

    @abstractmethod
    async def get_session_status(self) -> SessionStatus:
        """Inspect the current persistent session and return structured status."""
        pass

    @abstractmethod
    async def open_job(self, url: str) -> Dict[str, Any]:
        """Navigate to a job posting URL and return basic non-sensitive page metadata.

        Strictly governed by platform automation policy.
        """
        pass
