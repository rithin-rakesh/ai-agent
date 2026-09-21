"""Platform Router for Multi-Agent Orchestration (Phase 5.5).

Routes application execution to platform-specific agents (Indeed, and future platforms).
Enforces platform boundaries and safely rejects unconfigured job boards or external ATS.
"""

import logging
from typing import Any, Dict, Optional, Type
from app.agents.models import IndeedAgentRunRequest

logger = logging.getLogger(__name__)


class PlatformRouter:
    """Platform-neutral router directing job application tasks to target platform agents."""

    SUPPORTED_PLATFORMS = {"indeed", "glassdoor"}

    def __init__(self) -> None:
        self._registry: Dict[str, Any] = {}
        self._apply_service_registry: Dict[str, Any] = {}

    def register_platform(self, platform_name: str, agent_instance: Any) -> None:
        """Register a platform agent adapter instance."""
        self._registry[platform_name.lower().strip()] = agent_instance
        logger.info("Registered platform agent for: %s", platform_name)

    def register_apply_service(self, platform_name: str, service_instance: Any) -> None:
        """Register a platform apply service instance."""
        self._apply_service_registry[platform_name.lower().strip()] = service_instance
        logger.info("Registered platform apply service for: %s", platform_name)

    def is_platform_supported(self, platform: str) -> bool:
        """Check if a platform is currently supported by an active agent adapter."""
        if not platform:
            return False
        return platform.lower().strip() in self.SUPPORTED_PLATFORMS or platform.lower().strip() in self._registry

    def get_agent_for_platform(self, platform: str) -> Any:
        """Retrieve registered platform agent for target platform.

        Raises:
            ValueError: If platform is unsupported.
        """
        p = (platform or "").lower().strip()
        if not self.is_platform_supported(p):
            raise ValueError(
                f"Platform '{platform}' is not supported. "
                f"Supported platforms: {sorted(list(self.SUPPORTED_PLATFORMS))}"
            )

        if p in self._registry:
            return self._registry[p]

        if p == "indeed":
            from app.agents.indeed_agent import IndeedApplicationAgent
            agent = IndeedApplicationAgent()
            self._registry["indeed"] = agent
            return agent

        if p == "glassdoor":
            from app.automation.glassdoor.apply_service import get_glassdoor_apply_service
            service = get_glassdoor_apply_service()
            self._registry["glassdoor"] = service
            return service

        raise ValueError(f"Platform adapter for '{platform}' is not configured.")

    def get_apply_service_for_platform(self, platform: str) -> Any:
        """Retrieve the platform-specific apply service instance for direct in-process calls.

        Raises:
            ValueError: If platform is unsupported.
        """
        p = (platform or "").lower().strip()
        if not self.is_platform_supported(p):
            raise ValueError(
                f"Platform '{platform}' is not supported. "
                f"Supported platforms: {sorted(list(self.SUPPORTED_PLATFORMS))}"
            )

        if p in self._apply_service_registry:
            return self._apply_service_registry[p]

        if p == "indeed":
            from app.automation.indeed.apply_service import IndeedApplyService
            service = IndeedApplyService()
            self._apply_service_registry["indeed"] = service
            return service

        if p == "glassdoor":
            from app.automation.glassdoor.apply_service import GlassdoorApplyService
            service = GlassdoorApplyService()
            self._apply_service_registry["glassdoor"] = service
            return service

        raise ValueError(f"Apply service for platform '{platform}' is not configured.")

