"""Indeed Platform Adapter.

Implements PlatformAdapter for Indeed browser interactions, session checking,
job page inspection, and application flow detection governed strictly by platform-safe automation policy.
"""

import logging
from typing import Any, Dict, Optional
from playwright.async_api import BrowserContext

from app.automation.session_manager import SessionStatus
from app.config.settings import Settings, get_settings
from app.platforms.base import PlatformAdapter
from app.platforms.indeed.inspector import IndeedJobInspectionResult, IndeedJobInspector
from app.platforms.indeed.session import IndeedSessionManager
from app.platforms.policy import ActionState, AutomationAction

logger = logging.getLogger(__name__)


class IndeedPlatformAdapter(PlatformAdapter):
    """Platform adapter for Indeed job portal interactions."""

    def __init__(
        self,
        session_manager: Optional[IndeedSessionManager] = None,
        inspector: Optional[IndeedJobInspector] = None,
    ) -> None:
        self.session_manager = session_manager or IndeedSessionManager()
        settings = getattr(self.session_manager, "settings", None)
        if not isinstance(settings, Settings):
            settings = get_settings()

        self.inspector = inspector or IndeedJobInspector(
            settings=settings,
            session_manager=self.session_manager,
        )

    @property
    def platform_name(self) -> str:
        return "indeed"

    async def get_session_status(self) -> SessionStatus:
        """Inspect and return the current Indeed session status."""
        return await self.session_manager.check_session_status()

    async def inspect_job(
        self,
        url: str,
        save_screenshot: bool = True,
        headless: Optional[bool] = None,
    ) -> IndeedJobInspectionResult:
        """Inspect an Indeed job posting URL, detect application flow, and discover accessible forms.

        Strictly governed by platform-safe automation policy.
        """
        return await self.inspector.inspect_job(
            job_url=url,
            save_screenshot=save_screenshot,
            headless=headless,
        )

    async def open_job(self, url: str) -> Dict[str, Any]:
        """Navigate to an Indeed job posting URL and extract non-sensitive page metadata.

        Strictly checks platform policy and does NOT click Apply or submit any forms.
        """
        # Policy verification before automation action
        decision = self.evaluate_action(AutomationAction.NAVIGATE)
        if not decision.allowed:
            logger.warning("Navigation to '%s' blocked by policy: %s", url, decision.reason)
            return {
                "platform": self.platform_name,
                "requested_url": url,
                "status": decision.action_state.value,
                "allowed": False,
                "reason": decision.reason,
                "allowed_until": decision.allowed_until,
                "requires_user_action": decision.requires_user_action,
            }

        logger.info("Opening Indeed job URL: %s", url)
        context: Optional[BrowserContext] = None

        try:
            settings = getattr(self.session_manager, "settings", None)
            if not isinstance(settings, Settings):
                settings = get_settings()

            context = await self.session_manager.playwright_manager.launch_persistent_context(
                user_data_dir=settings.get_indeed_profile_path(),
                headless=settings.PLAYWRIGHT_HEADLESS,
            )
            page = context.pages[0] if context.pages else await context.new_page()

            await page.goto(url, wait_until="domcontentloaded", timeout=settings.PLAYWRIGHT_TIMEOUT_MS)
            page_title = await page.title()
            current_url = page.url

            # Basic metadata extraction without form interaction
            h1 = await page.query_selector("h1")
            heading_text = await h1.inner_text() if h1 else page_title

            return {
                "platform": self.platform_name,
                "requested_url": url,
                "final_url": current_url,
                "title": heading_text.strip(),
                "page_title": page_title.strip(),
                "status": "opened",
                "allowed": True,
                "policy_level": self.get_policy().automation_level.value,
                "allowed_until": self.get_policy().allowed_until,
            }
        finally:
            if context:
                await self.session_manager.playwright_manager.close_context(context)
