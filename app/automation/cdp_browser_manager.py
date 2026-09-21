"""Shared Playwright CDP Browser Manager (Phase 6.0.1).

Coordinates authoritative Chrome DevTools Protocol (CDP) attachment on 127.0.0.1:9222
across both Indeed and Glassdoor automation flows without launching secondary browser instances,
without requiring Windows OS foreground window focus, and supporting empty/newtab page reuse.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from app.automation.glassdoor.config import is_live_cdp_allowed

logger = logging.getLogger(__name__)

DEFAULT_CDP_HOST = "127.0.0.1"
DEFAULT_CDP_PORT = 9222
DEFAULT_TIMEOUT_SECONDS = 15


async def _maybe_await(val: Any) -> Any:
    """Safely await coroutines or return raw sync values."""
    if asyncio.iscoroutine(val):
        return await val
    if hasattr(val, "__await__"):
        return await val
    return val


class PlaywrightCDPBrowserManager:
    """Centralized Playwright CDP browser manager for multi-platform automation."""

    def __init__(
        self,
        host: str = DEFAULT_CDP_HOST,
        port: int = DEFAULT_CDP_PORT,
        cdp_url: Optional[str] = None,
    ) -> None:
        if cdp_url:
            from urllib.parse import urlparse
            self.endpoint_url = cdp_url
            parsed = urlparse(cdp_url)
            self.host = parsed.hostname or host
            self.port = parsed.port or port
        else:
            self.host = host
            self.port = port
            self.endpoint_url = f"http://{self.host}:{self.port}"
        self._playwright: Optional[Any] = None
        self._browser: Optional[Any] = None
        self._context: Optional[Any] = None
        self._active_page: Optional[Any] = None

    def _is_page_valid(self, page: Any) -> bool:
        """Check if a Playwright page instance is active and not closed."""
        if page is None:
            return False
        try:
            if hasattr(page, "is_closed") and callable(page.is_closed):
                return not page.is_closed()
            return True
        except Exception:
            return False

    async def connect_to_cdp(
        self,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> Tuple[bool, Optional[Any], Dict[str, Any]]:
        """Connect to the running Chrome/Edge instance over CDP without launching extra browsers.

        Returns:
            Tuple of:
                - success: bool
                - authoritative_page: Optional[Page]
                - diagnostics: Dict[str, Any]
        """
        diag: Dict[str, Any] = {
            "cdp_endpoint": self.endpoint_url,
            "cdp_connect_attempted": True,
            "cdp_connected": False,
            "browser_session_verified": False,
            "contexts_found": 0,
            "pages_found": 0,
            "page_urls": [],
            "authoritative_page_url": None,
            "second_browser_launched": False,
        }

        # Safety Check: Fail closed during test runs unless explicitly allowed or mocked
        live_allowed = is_live_cdp_allowed()
        if "PYTEST_CURRENT_TEST" in os.environ and not live_allowed:
            raise RuntimeError("Live CDP connection attempted during isolated test execution")

        if not live_allowed:
            logger.debug("Live CDP connection is disabled by configuration (ALLOW_LIVE_CDP=False).")
            diag["error"] = "Live CDP connection is disabled by configuration (ALLOW_LIVE_CDP=False)."
            return False, None, diag

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            diag["error"] = "Playwright package is not installed."
            return False, None, diag

        logger.info("Connecting Playwright Async API over CDP to '%s'...", self.endpoint_url)

        try:
            if self._playwright is None:
                self._playwright = await async_playwright().start()

            if self._browser is None or (hasattr(self._browser, "is_connected") and not self._browser.is_connected()):
                self._browser = await self._playwright.chromium.connect_over_cdp(
                    self.endpoint_url,
                    timeout=timeout_seconds * 1000,
                )

            diag["cdp_connected"] = True

            # Enumerate contexts
            contexts = getattr(self._browser, "contexts", [])
            diag["contexts_found"] = len(contexts)
            if not contexts:
                diag["error"] = "Connected over CDP but no browser contexts were found."
                return False, None, diag

            self._context = contexts[0]

            # Enumerate pages
            pages = [p for p in getattr(self._context, "pages", []) if self._is_page_valid(p)]
            diag["pages_found"] = len(pages)
            page_urls = []
            for p in pages:
                try:
                    p_u = getattr(p, "url", "")
                    if p_u:
                        page_urls.append(p_u)
                except Exception:
                    pass
            diag["page_urls"] = page_urls

            # Select or create page
            auth_page = pages[-1] if pages else None
            if auth_page is None:
                auth_page = await _maybe_await(self._context.new_page())
                pages = [auth_page]
                diag["pages_found"] = 1
                try:
                    u = getattr(auth_page, "url", "")
                    if u:
                        diag["page_urls"] = [u]
                except Exception:
                    pass

            self._active_page = auth_page
            auth_url = getattr(auth_page, "url", "") if auth_page else None
            diag["authoritative_page_url"] = auth_url
            diag["browser_session_verified"] = True

            logger.info("Attached Playwright to authoritative page: '%s'", auth_url)
            return True, auth_page, diag

        except Exception as exc:
            logger.warning("Failed to connect Playwright over CDP (%s): %s", self.endpoint_url, exc)
            diag["error"] = str(exc)
            diag["browser_session_verified"] = False
            return False, None, diag

    async def resolve_or_navigate_page(
        self,
        target_url: str,
        platform: str = "indeed",
        expected_id: Optional[str] = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> Tuple[bool, Optional[Any], Dict[str, Any]]:
        """Find an existing page in the CDP session or navigate the current page to target_url.

        Ensures NO secondary browser or window is launched, and supports:
        1. Matching existing tab by target URL or job ID (e.g. Indeed jk / Glassdoor jl).
        2. Reusing open tab (including chrome://newtab/ and about:blank).
        3. Creating a new tab in the same context if none exist.
        """
        # Connect if needed
        if self._browser is None or (hasattr(self._browser, "is_connected") and not self._browser.is_connected()):
            connected, auth_page, diag = await _maybe_await(self.connect_to_cdp(timeout_seconds=timeout_seconds))
            if not connected or not self._browser:
                diag["requested_job_url"] = target_url
                if expected_id:
                    diag["expected_jk" if platform == "indeed" else "expected_id"] = expected_id
                return False, None, diag

        contexts = getattr(self._browser, "contexts", [])
        if not contexts:
            diag = {
                "cdp_endpoint": self.endpoint_url,
                "cdp_connect_attempted": True,
                "cdp_connected": True,
                "browser_session_verified": False,
                "contexts_found": 0,
                "pages_found": 0,
                "page_urls": [],
                "authoritative_page_url": None,
                "requested_job_url": target_url,
                "second_browser_launched": False,
                "error": "Connected over CDP but no browser contexts were found.",
            }
            if expected_id:
                diag["expected_jk" if platform == "indeed" else "expected_id"] = expected_id
            return False, None, diag

        self._context = contexts[0]
        pages = [p for p in getattr(self._context, "pages", []) if self._is_page_valid(p)]
        page_urls = [getattr(p, "url", "") for p in pages if hasattr(p, "url")]
        matched_page = None

        target_clean = target_url.split("?")[0].rstrip("/")

        # 1. Look for existing page with matching URL or listing ID / jk
        for p in pages:
            try:
                if not self._is_page_valid(p):
                    continue
                p_url = getattr(p, "url", "") or ""
                p_url_clean = p_url.split("?")[0].rstrip("/")
                if target_clean in p_url_clean or p_url_clean in target_clean:
                    matched_page = p
                    break
                if expected_id and expected_id.lower() in p_url.lower():
                    matched_page = p
                    break
            except Exception:
                continue

        # 2. Look for existing page of the same platform (e.g. indeed.com or glassdoor.com)
        if matched_page is None:
            domain_keyword = "indeed" if platform == "indeed" else "glassdoor"
            for p in pages:
                try:
                    if not self._is_page_valid(p):
                        continue
                    p_url = getattr(p, "url", "").lower()
                    if domain_keyword in p_url and "smartapply" not in p_url:
                        matched_page = p
                        break
                except Exception:
                    continue

        # 3. Look for reusable tab (chrome://newtab/, about:blank, or generic web tab)
        if matched_page is None:
            for p in pages:
                try:
                    if not self._is_page_valid(p):
                        continue
                    p_url = getattr(p, "url", "").lower()
                    if "localhost:8000" not in p_url and "127.0.0.1:8000" not in p_url and "/docs" not in p_url and "openapi.json" not in p_url:
                        matched_page = p
                        break
                except Exception:
                    continue

        # 4. Use existing open page or create new tab in same CDP context
        if matched_page is None:
            open_pages = [p for p in pages if self._is_page_valid(p)]
            if open_pages:
                matched_page = open_pages[-1]
            else:
                matched_page = await _maybe_await(self._context.new_page())
                pages = [matched_page]

        self._active_page = matched_page

        try:
            if hasattr(matched_page, "bring_to_front") and callable(matched_page.bring_to_front):
                await _maybe_await(matched_page.bring_to_front())
        except Exception:
            pass

        # Navigate if not already on the target URL or expected ID
        page_url = getattr(matched_page, "url", "") or ""
        needs_nav = True
        if target_clean in page_url.split("?")[0].rstrip("/"):
            if not expected_id or (expected_id.lower() in page_url.lower()):
                needs_nav = False

        if needs_nav:
            logger.info("Navigating authoritative CDP page from '%s' to '%s'", page_url, target_url)
            try:
                try:
                    if hasattr(matched_page, "on"):
                        matched_page.on("dialog", lambda dialog: asyncio.create_task(dialog.accept()))
                    if hasattr(matched_page, "evaluate") and callable(matched_page.evaluate):
                        await _maybe_await(matched_page.evaluate("""() => {
                            window.onbeforeunload = null;
                            window.addEventListener('beforeunload', (e) => {
                                delete e['returnValue'];
                            }, { capture: true });
                        }"""))
                except Exception:
                    pass
                if hasattr(matched_page, "goto") and callable(matched_page.goto):
                    await _maybe_await(matched_page.goto(target_url, wait_until="domcontentloaded", timeout=timeout_seconds * 1000))
                await asyncio.sleep(1.0)
                page_url = getattr(matched_page, "url", "") or ""
            except Exception as exc:
                logger.warning("CDP page navigation warning: %s", exc)

        page_title = ""
        try:
            if hasattr(matched_page, "title") and callable(matched_page.title):
                page_title = await _maybe_await(matched_page.title())
        except Exception:
            pass

        # Check resolved ID in URL
        resolved_id = None
        if platform == "indeed":
            from app.automation.indeed.url_validator import extract_indeed_jk
            resolved_id = extract_indeed_jk(page_url)
        elif platform == "glassdoor":
            from app.automation.glassdoor.url_validator import extract_glassdoor_job_id
            resolved_id = extract_glassdoor_job_id(page_url)

        diagnostics: Dict[str, Any] = {
            "cdp_endpoint": self.endpoint_url,
            "cdp_connect_attempted": True,
            "cdp_connected": True,
            "browser_session_verified": True,
            "contexts_found": len(contexts),
            "pages_found": len(pages),
            "page_urls": page_urls,
            "authoritative_page_url": page_url,
            "requested_job_url": target_url,
            "second_browser_launched": False,
            "cdp_page_title": page_title,
        }
        if platform == "indeed":
            diagnostics["expected_jk"] = expected_id
            diagnostics["resolved_jk"] = resolved_id
        else:
            diagnostics["expected_id"] = expected_id
            diagnostics["resolved_id"] = resolved_id

        return True, matched_page, diagnostics
