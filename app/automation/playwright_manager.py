"""Playwright Browser Manager.

Provides a clean abstraction around Playwright persistent browser contexts,
lifecycle management, realistic viewport/user-agent configuration, and trace capture.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from playwright.async_api import BrowserContext, Playwright, async_playwright

from app.automation.browser_profiles import BrowserProfileManager
from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Standard modern desktop user agent to avoid headless flag issues
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class PlaywrightManager:
    """Centralized lifecycle manager for Playwright browser contexts."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        profile_manager: Optional[BrowserProfileManager] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.profile_manager = profile_manager or BrowserProfileManager()
        self._playwright: Optional[Playwright] = None
        self._active_contexts: List[BrowserContext] = []

    async def _ensure_playwright(self) -> Playwright:
        """Start the Playwright driver if not already active."""
        if self._playwright is None:
            self._playwright = await async_playwright().start()
            logger.debug("Playwright driver initialized.")
        return self._playwright

    async def launch_persistent_context(
        self,
        user_data_dir: Path,
        headless: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
        trace: Optional[bool] = None,
        extra_launch_args: Optional[List[str]] = None,
    ) -> BrowserContext:
        """Launch a persistent browser context using the specified user data directory.

        Preserves cookies, local storage, and session tokens across runs.
        """
        playwright = await self._ensure_playwright()

        # Ensure directory exists
        profile_dir = self.profile_manager.ensure_profile_dir(user_data_dir)

        is_headless = headless if headless is not None else self.settings.PLAYWRIGHT_HEADLESS
        operation_timeout = timeout_ms if timeout_ms is not None else self.settings.PLAYWRIGHT_TIMEOUT_MS
        enable_trace = trace if trace is not None else self.settings.PLAYWRIGHT_TRACE

        # Standard Chromium launch flags for stable automation
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-default-browser-check",
        ]
        if extra_launch_args:
            launch_args.extend(extra_launch_args)

        logger.info(
            "Launching persistent context at '%s' (headless=%s, browser=%s)",
            profile_dir,
            is_headless,
            self.settings.PLAYWRIGHT_BROWSER,
        )

        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=is_headless,
            user_agent=_DEFAULT_USER_AGENT,
            viewport={"width": 1280, "height": 800},
            locale="en-US",
            timeout=operation_timeout,
            args=launch_args,
        )

        context.set_default_timeout(operation_timeout)
        context.set_default_navigation_timeout(operation_timeout)

        # Optional tracing for debugging
        if enable_trace:
            logger.debug("Playwright tracing started for persistent context.")
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        self._active_contexts.append(context)
        return context

    async def close_context(
        self,
        context: BrowserContext,
        trace_output_path: Optional[Path] = None,
    ) -> None:
        """Cleanly stop tracing (if enabled) and close a persistent browser context."""
        try:
            if trace_output_path:
                trace_output_path.parent.mkdir(parents=True, exist_ok=True)
                await context.tracing.stop(path=str(trace_output_path))
                logger.info("Saved Playwright trace artifact to: %s", trace_output_path)
            elif self.settings.PLAYWRIGHT_TRACE:
                artifacts_dir = self.settings.get_playwright_artifacts_path()
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                default_trace = artifacts_dir / "session_trace.zip"
                await context.tracing.stop(path=str(default_trace))
        except Exception as exc:
            logger.warning("Error stopping trace during context closure: %s", exc)

        try:
            await context.close()
            logger.debug("Persistent browser context closed.")
        except Exception as exc:
            logger.warning("Error closing browser context: %s", exc)
        finally:
            if context in self._active_contexts:
                self._active_contexts.remove(context)

    async def shutdown(self) -> None:
        """Close all active contexts and terminate the Playwright driver."""
        for ctx in list(self._active_contexts):
            await self.close_context(ctx)

        if self._playwright is not None:
            try:
                await self._playwright.stop()
                logger.info("Playwright driver terminated.")
            except Exception as exc:
                logger.warning("Error stopping Playwright driver: %s", exc)
            finally:
                self._playwright = None
