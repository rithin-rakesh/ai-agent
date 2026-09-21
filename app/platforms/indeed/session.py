"""Indeed Session Manager.

Manages persistent Playwright browser context, session verification,
manual login workflow coordination, and non-sensitive screenshot generation for Indeed.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple
from playwright.async_api import BrowserContext, Page

from app.automation.playwright_manager import PlaywrightManager
from app.automation.session_manager import SessionState, SessionStatus
from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Primary Indeed navigation URLs
INDEED_HOME_URL = "https://www.indeed.com"
INDEED_SIGNIN_URL = "https://secure.indeed.com/account/login"

# Selectors that indicate an active authenticated user session
AUTHENTICATED_SELECTORS = [
    'button[data-gnav-element-name="AccountMenu"]',
    'button#AccountMenu',
    'a[data-gnav-element-name="Profile"]',
    'a[href*="/myjobs"]',
    'a[href*="/account/profile"]',
    'button[aria-label*="Account" i]',
    'button[aria-label*="Profile" i]',
    'span[data-testid="account-menu"]',
    'a[href*="/account/settings"]',
]

# Selectors that indicate the user is currently logged out
UNAUTHENTICATED_SELECTORS = [
    'a[data-gnav-element-name="SignIn"]',
    'a[href*="secure.indeed.com/auth"]',
    'a[href*="secure.indeed.com/account/login"]',
    'a[href*="secure.indeed.com/account/"]',
    'a:has-text("Sign in")',
    'a:has-text("Sign In")',
    'button:has-text("Sign in")',
]


class IndeedSessionManager:
    """Manages the lifecycle and verification of Indeed persistent browser sessions."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        playwright_manager: Optional[PlaywrightManager] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.playwright_manager = playwright_manager or PlaywrightManager(settings=self.settings)

    @property
    def profile_path(self) -> Path:
        """Return the resolved persistent profile path for Indeed."""
        return self.settings.get_indeed_profile_path()

    @property
    def screenshot_dir(self) -> Path:
        """Return the directory for Indeed screenshots."""
        path = self.settings.get_screenshots_path() / "indeed"
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def detect_auth_state(self, page: Page) -> Tuple[bool, Optional[str]]:
        """Inspect page DOM for authenticated and unauthenticated indicators.

        Returns:
            Tuple[bool, Optional[str]]: (is_authenticated, detected_element_info)
        """
        # 1. Check authenticated selectors
        for selector in AUTHENTICATED_SELECTORS:
            try:
                elem = await page.query_selector(selector)
                if elem and await elem.is_visible():
                    logger.debug("Detected authenticated indicator via selector: %s", selector)
                    return True, selector
            except Exception:
                continue

        # 2. Check unauthenticated sign-in buttons
        for selector in UNAUTHENTICATED_SELECTORS:
            try:
                elem = await page.query_selector(selector)
                if elem and await elem.is_visible():
                    logger.debug("Detected unauthenticated sign-in indicator via selector: %s", selector)
                    return False, selector
            except Exception:
                continue

        # 3. Check page title/URL heuristics as fallback
        current_url = page.url.lower()
        if "secure.indeed.com/auth" in current_url or "account/login" in current_url:
            return False, "login_url"

        # Default to false if uncertain
        return False, None

    async def check_session_status(self, headless: Optional[bool] = None) -> SessionStatus:
        """Perform a check of the persistent profile's authentication status."""
        is_headless = headless if headless is not None else True
        context: Optional[BrowserContext] = None

        try:
            context = await self.playwright_manager.launch_persistent_context(
                user_data_dir=self.profile_path,
                headless=is_headless,
            )
            page = context.pages[0] if context.pages else await context.new_page()

            logger.info("Navigating to Indeed home to inspect session status...")
            await page.goto(INDEED_HOME_URL, wait_until="domcontentloaded", timeout=self.settings.PLAYWRIGHT_TIMEOUT_MS)
            await asyncio.sleep(2)  # Brief wait for client-side hydration

            is_auth, indicator = await self.detect_auth_state(page)

            status = SessionState.AUTHENTICATED if is_auth else SessionState.LOGIN_REQUIRED
            msg = "Authenticated session active" if is_auth else "No active login detected; manual sign-in required"

            return SessionStatus(
                platform="indeed",
                browser=self.settings.PLAYWRIGHT_BROWSER,
                profile_path=str(self.profile_path),
                authenticated=is_auth,
                status=status,
                message=msg,
            )
        except Exception as exc:
            logger.error("Failed to verify Indeed session status: %s", exc)
            return SessionStatus(
                platform="indeed",
                browser=self.settings.PLAYWRIGHT_BROWSER,
                profile_path=str(self.profile_path),
                authenticated=False,
                status=SessionState.SESSION_ERROR,
                message=f"Session verification error: {str(exc)}",
            )
        finally:
            if context:
                await self.playwright_manager.close_context(context)

    async def open_indeed_home(
        self,
        headless: Optional[bool] = None,
        save_screenshot: bool = True,
    ) -> Tuple[SessionStatus, Optional[Path]]:
        """Launch the persistent browser, navigate to Indeed home, capture screenshot, and return status."""
        is_headless = headless if headless is not None else self.settings.PLAYWRIGHT_HEADLESS
        screenshot_path: Optional[Path] = None
        context: Optional[BrowserContext] = None

        try:
            context = await self.playwright_manager.launch_persistent_context(
                user_data_dir=self.profile_path,
                headless=is_headless,
            )
            page = context.pages[0] if context.pages else await context.new_page()

            logger.info("Opening Indeed homepage: %s", INDEED_HOME_URL)
            await page.goto(INDEED_HOME_URL, wait_until="domcontentloaded", timeout=self.settings.PLAYWRIGHT_TIMEOUT_MS)
            await asyncio.sleep(2)

            is_auth, _ = await self.detect_auth_state(page)

            if save_screenshot:
                screenshot_path = self.screenshot_dir / "indeed_home.png"
                await page.screenshot(path=str(screenshot_path), full_page=False)
                logger.info("Captured non-sensitive homepage screenshot: %s", screenshot_path)

            status = SessionState.AUTHENTICATED if is_auth else SessionState.LOGIN_REQUIRED
            msg = "Authenticated Indeed session" if is_auth else "Indeed loaded. Manual login required."

            return SessionStatus(
                platform="indeed",
                browser=self.settings.PLAYWRIGHT_BROWSER,
                profile_path=str(self.profile_path),
                authenticated=is_auth,
                status=status,
                message=msg,
            ), screenshot_path
        finally:
            if context:
                await self.playwright_manager.close_context(context)

    async def run_manual_login_workflow(
        self,
        timeout_seconds: int = 180,
    ) -> SessionStatus:
        """Interactive workflow: opens visible browser to allow the user to log in manually.

        Strictly does NOT automate credentials, CAPTCHA, or 2FA.
        Waits for the user to complete login in the visible browser window, then persists profile.
        """
        logger.info("Starting manual Indeed login workflow (visible browser, timeout: %ds)...", timeout_seconds)
        context = await self.playwright_manager.launch_persistent_context(
            user_data_dir=self.profile_path,
            headless=False,
        )
        page = context.pages[0] if context.pages else await context.new_page()

        try:
            logger.info("Navigating to Indeed sign-in page...")
            await page.goto(INDEED_SIGNIN_URL, wait_until="domcontentloaded")

            logger.info(
                "==============================================================\n"
                " ACTION REQUIRED: Please log in to your Indeed account in the\n"
                " visible browser window. Complete any CAPTCHA or 2FA prompts.\n"
                " Waiting up to %d seconds for authenticated session...\n"
                "==============================================================",
                timeout_seconds,
            )

            start_time = asyncio.get_event_loop().time()
            is_auth = False

            while asyncio.get_event_loop().time() - start_time < timeout_seconds:
                await asyncio.sleep(3)
                is_auth, indicator = await self.detect_auth_state(page)
                if is_auth:
                    logger.info("Authenticated session detected! (Indicator: %s)", indicator)
                    break

            # Capture non-sensitive screenshot after completion
            screenshot_path = self.screenshot_dir / "indeed_home.png"
            await page.screenshot(path=str(screenshot_path), full_page=False)

            status = SessionState.AUTHENTICATED if is_auth else SessionState.LOGIN_REQUIRED
            msg = "Manual login succeeded and session profile persisted" if is_auth else "Manual login timed out before authentication was detected"

            return SessionStatus(
                platform="indeed",
                browser=self.settings.PLAYWRIGHT_BROWSER,
                profile_path=str(self.profile_path),
                authenticated=is_auth,
                status=status,
                message=msg,
            )
        finally:
            await self.playwright_manager.close_context(context)
