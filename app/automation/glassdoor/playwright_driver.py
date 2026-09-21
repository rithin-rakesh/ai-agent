"""Playwright CDP Driver for Hosted Application DOM Automation (Async API).

Handles Chrome DevTools Protocol (CDP) connection to existing browser sessions,
page resolution/navigation without launching secondary browsers, semantic DOM inspection,
dynamic form-state detection, question extraction, truthful input filling with scrolling,
progression action activation (scroll_into_view_if_needed), and exact Submit detection.
"""

import asyncio
import inspect
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from app.automation.forms.models import ApplicationQuestion, QuestionInputType
from app.automation.forms.question_normalizer import normalize_question_key
from app.automation.glassdoor.config import (
    ALREADY_APPLIED_EXACT_NAMES,
    APPLY_ANYWAY_BUTTON_NAMES,
    BLACKLISTED_PROGRESSION_BUTTON_NAMES,
    CDP_CONNECT_TIMEOUT_SECONDS,
    CDP_HOST,
    CDP_REMOTE_DEBUGGING_PORT,
    CONTINUE_BUTTON_NAMES,
    CONTINUE_ON_REQUIREMENTS_WARNING,
    EASY_APPLY_BUTTON_NAMES,
    EXACT_SUBMIT_NAMES,
    NEXT_BUTTON_NAMES,
    REVIEW_BUTTON_NAMES,
    SMARTAPPLY_HOST_DOMAINS,
    SPA_TRANSITION_POLL_INTERVAL_SECONDS,
    SPA_TRANSITION_TIMEOUT_SECONDS,
    SURVEY_FIELD_IDENTIFIERS,
    SURVEY_HEADING_PATTERNS,
    SURVEY_SUBTEXT_PATTERNS,
)
from app.automation.glassdoor.models import GlassdoorAutomationState
from app.automation.glassdoor.state_detector import (
    ALREADY_APPLIED_PATTERNS,
    GlassdoorStateDetector,
)

logger = logging.getLogger(__name__)


def _normalize_text(text: Optional[str]) -> str:
    """Normalize text for whitespace, casing, and Unicode apostrophes."""
    if not text:
        return ""
    cleaned = (
        str(text)
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace("´", "'")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
    )
    return re.sub(r"\s+", " ", cleaned.lower().strip())


def _get_all_frames_and_page(target_page: Any) -> List[Any]:
    """Retrieve list containing top-level page and all unique subframes for DOM scanning."""
    contexts = [target_page]
    if hasattr(target_page, "frames"):
        try:
            for fr in target_page.frames:
                if fr not in contexts:
                    contexts.append(fr)
        except Exception:
            pass
    return contexts


async def _maybe_await(val: Any) -> Any:
    """Await val if it is a coroutine or awaitable, otherwise return it directly."""
    if inspect.isawaitable(val):
        return await val
    return val

class AsyncHybridResult:
    """Hybrid wrapper allowing driver methods to be awaited in async pipelines
    or resolved synchronously in legacy sync unit tests.
    """

    def __init__(self, coro_fn: Any, *args: Any, **kwargs: Any) -> None:
        self._coro_fn = coro_fn
        self._args = args
        self._kwargs = kwargs
        self._coro: Any = None

    def __await__(self) -> Any:
        if self._coro is None:
            self._coro = self._coro_fn(*self._args, **self._kwargs)
        return self._coro.__await__()


def async_hybrid(coro_fn: Any) -> Any:
    """Decorator converting an async method into an AsyncHybridResult in event loops or direct result synchronously."""
    import functools

    @functools.wraps(coro_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return AsyncHybridResult(coro_fn, *args, **kwargs)
        else:
            return asyncio.run(coro_fn(*args, **kwargs))

    return wrapper


class GlassdoorPlaywrightDriver:
    """Playwright CDP-based automation driver for hosted application pages (Async API)."""

    def __init__(
        self,
        host: str = CDP_HOST,
        port: int = CDP_REMOTE_DEBUGGING_PORT,
    ) -> None:
        self.host = host
        self.port = port
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self.job_page = None
        self.application_page = None
        self._active_application_page = None
        self._previous_state_signature = None
        self._step_history = []
        self._pages_before_easy_apply = None
        self._last_progression_diagnostics: Dict[str, Any] = {}

    def reset_navigation_state(self) -> None:
        """Reset transient Python navigation state before a new application run.

        Clears page references and loop state while preserving live CDP browser/context connection.
        """
        self.job_page = None
        self.application_page = None
        self._page = None
        self._active_application_page = None
        self._previous_state_signature = None
        self._step_history = []
        self._pages_before_easy_apply = None
        self._last_progression_diagnostics = {}

    def _is_page_valid(self, page: Optional[Any]) -> bool:
        """Check if a Playwright page reference is valid, non-None, and not closed."""
        if page is None:
            return False
        try:
            if hasattr(page, "is_closed") and callable(page.is_closed):
                res = page.is_closed()
                if isinstance(res, bool):
                    return not res
            return True
        except Exception:
            return False

    @property
    def page(self) -> Optional[Any]:
        """Active Playwright Page object (prefers application_page, then job_page, then _page)."""
        return self.application_page or self.job_page or self._page

    @page.setter
    def page(self, val: Optional[Any]) -> None:
        self._page = val

    @async_hybrid
    async def connect_to_cdp(
        self,
        timeout_seconds: int = CDP_CONNECT_TIMEOUT_SECONDS,
    ) -> Tuple[bool, Optional[Any], Dict[str, Any]]:
        """Authoritatively connect to running Chrome instance over CDP.

        Enumerates contexts, pages, page URLs, identifies authoritative page,
        and sets browser_session_verified = True without requiring OS active-window focus.

        Returns:
            Tuple: (success: bool, authoritative_page: Optional[Page], diagnostics: Dict[str, Any])
        """
        import os
        from app.automation.glassdoor.config import is_live_cdp_allowed

        endpoint_url = f"http://{self.host}:{self.port}"
        diag: Dict[str, Any] = {
            "cdp_endpoint": endpoint_url,
            "cdp_connect_attempted": True,
            "cdp_connected": False,
            "browser_session_verified": False,
            "contexts_found": 0,
            "pages_found": 0,
            "page_urls": [],
            "authoritative_page_url": None,
            "second_browser_launched": False,
        }

        # Safety Check: Fail closed during test runs unless explicitly allowed
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

        logger.info("Connecting Playwright Async API over CDP to '%s'...", endpoint_url)

        try:
            if self._playwright is None:
                self._playwright = await async_playwright().start()

            if self._browser is None or (hasattr(self._browser, "is_connected") and not self._browser.is_connected()):
                self._browser = await self._playwright.chromium.connect_over_cdp(
                    endpoint_url,
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

            # Identify authoritative page
            auth_page = None
            if hasattr(self, "_find_active_application_page") and callable(self._find_active_application_page):
                auth_page = await _maybe_await(self._find_active_application_page())

            # If CDP connects successfully but only chrome://newtab/ exists, that is STILL a valid CDP session
            if auth_page is None and pages:
                auth_page = pages[-1]

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

            self._page = auth_page
            self.job_page = auth_page
            auth_url = getattr(auth_page, "url", "") if auth_page else None
            diag["authoritative_page_url"] = auth_url
            diag["browser_session_verified"] = True

            try:
                logger.info("Attached Playwright to authoritative page: '%s'", auth_url)
            except Exception:
                pass

            return True, auth_page, diag

        except Exception as exc:
            logger.warning("Failed to connect Playwright over CDP (%s): %s", endpoint_url, exc)
            diag["error"] = str(exc)
            diag["browser_session_verified"] = False
            return False, None, diag

    @async_hybrid
    async def connect_cdp(
        self,
        timeout_seconds: int = CDP_CONNECT_TIMEOUT_SECONDS,
    ) -> Tuple[bool, Optional[str]]:
        """Connect to the running Chrome instance over CDP without launching secondary browsers.

        Delegates to authoritative connect_to_cdp().
        Returns:
            Tuple: (success: bool, error_message: Optional[str])
        """
        ok, page, diag = await _maybe_await(self.connect_to_cdp(timeout_seconds=timeout_seconds))
        if ok:
            return True, None
        return False, diag.get("error") or "Failed to connect over CDP"

    @async_hybrid
    async def resolve_or_navigate_page(
        self,
        target_url: str,
        timeout_seconds: int = 15,
    ) -> Tuple[bool, Optional[Any], Dict[str, Any]]:
        """Find an existing page in the CDP session or navigate the current page to target_url.

        Ensures NO secondary browser or window is launched, and discards stale/closed pages.
        """
        if self._browser is None or (hasattr(self._browser, "is_connected") and not self._browser.is_connected()):
            connected, auth_page, diag = await _maybe_await(self.connect_to_cdp(timeout_seconds=timeout_seconds))
            if not connected or not self._browser:
                diag["requested_job_url"] = target_url
                return False, None, diag

        contexts = getattr(self._browser, "contexts", [])
        if not contexts:
            return False, None, {
                "cdp_endpoint": f"http://{self.host}:{self.port}",
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

        self._context = contexts[0]
        pages = [p for p in getattr(self._context, "pages", []) if self._is_page_valid(p)]
        page_urls = [getattr(p, "url", "") for p in pages if hasattr(p, "url")]
        matched_page = None

        # Extract target job id / jl parameter if present
        target_clean = target_url.split("?")[0].rstrip("/")
        from app.automation.glassdoor.url_validator import extract_glassdoor_job_id
        target_jl = extract_glassdoor_job_id(target_url)

        # 1. Look for existing page with matching URL or listing ID
        for p in pages:
            try:
                if not self._is_page_valid(p):
                    continue
                p_url = getattr(p, "url", "").split("?")[0].rstrip("/")
                if target_clean in p_url or p_url in target_clean:
                    matched_page = p
                    break
                if target_jl and f"jl={target_jl}" in getattr(p, "url", ""):
                    matched_page = p
                    break
            except Exception:
                continue

        # 2. Look for existing Glassdoor listing page (exclude smartapply application tab if starting fresh)
        if matched_page is None:
            for p in pages:
                try:
                    if not self._is_page_valid(p):
                        continue
                    p_url = getattr(p, "url", "").lower()
                    if "glassdoor" in p_url and "smartapply" not in p_url:
                        matched_page = p
                        break
                except Exception:
                    continue

        # 3. Look for reusable non-docs/non-swagger tab (including chrome://newtab/)
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

        self.job_page = matched_page
        self.application_page = None  # Reset any stale application page from earlier runs
        self._page = matched_page

        try:
            if hasattr(matched_page, "bring_to_front") and callable(matched_page.bring_to_front):
                await _maybe_await(matched_page.bring_to_front())
        except Exception:
            pass

        # Navigate if not already on the target URL
        page_url = ""
        try:
            page_url = getattr(matched_page, "url", "") or ""
            if target_clean not in page_url:
                logger.info("Navigating authoritative CDP page from '%s' to '%s'", page_url, target_url)
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

        diagnostics = {
            "cdp_endpoint": f"http://{self.host}:{self.port}",
            "cdp_connect_attempted": True,
            "cdp_connected": True,
            "browser_session_verified": True,
            "contexts_found": len(contexts),
            "pages_found": len(pages),
            "page_urls": page_urls,
            "authoritative_page_url": page_url or target_url,
            "requested_job_url": target_url,
            "job_page_id": getattr(matched_page, "id", "page-0"),
            "job_page_url": page_url or target_url,
            "cdp_page_id": getattr(matched_page, "id", "page-0"),
            "cdp_page_url": page_url or target_url,
            "cdp_page_title": page_title,
            "second_browser_launched": False,
        }

        return True, matched_page, diagnostics

    @async_hybrid
    async def find_exact_easy_apply_button(self, page: Optional[Any] = None) -> Optional[Any]:
        """Locate visible, enabled Easy Apply button or link in the DOM."""
        target_page = page or self.job_page or self._page
        if not target_page:
            return None

        try:
            for name in EASY_APPLY_BUTTON_NAMES:
                pattern = re.compile(rf"^\s*{re.escape(name)}\s*$", re.IGNORECASE)
                # Priority 1: Button role with exact name
                btn = target_page.get_by_role("button", name=pattern)
                cnt = await _maybe_await(btn.count()) if hasattr(btn, "count") else 0
                if isinstance(cnt, int) and cnt > 0:
                    for idx in range(cnt):
                        cand = btn.nth(idx) if hasattr(btn, "nth") else btn
                        if await self._is_in_excluded_container(cand):
                            continue
                        is_vis = await _maybe_await(cand.is_visible()) if hasattr(cand, "is_visible") else True
                        is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                        if is_vis and is_en:
                            text = (await _maybe_await(cand.text_content()) or "").strip()
                            if GlassdoorStateDetector.is_easy_apply_button(text):
                                return cand

                # Priority 2: Link role with exact name
                link = target_page.get_by_role("link", name=pattern)
                l_cnt = await _maybe_await(link.count()) if hasattr(link, "count") else 0
                if isinstance(l_cnt, int) and l_cnt > 0:
                    for idx in range(l_cnt):
                        cand = link.nth(idx) if hasattr(link, "nth") else link
                        if await self._is_in_excluded_container(cand):
                            continue
                        is_vis = await _maybe_await(cand.is_visible()) if hasattr(cand, "is_visible") else True
                        is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                        if is_vis and is_en:
                            text = (await _maybe_await(cand.text_content()) or "").strip()
                            if GlassdoorStateDetector.is_easy_apply_button(text):
                                return cand

                # Priority 3: Interactive elements (button, a) by text / aria-label / data-test
                custom_loc = target_page.locator(
                    f"button:has-text('{name}'), a:has-text('{name}'), button[aria-label*='{name}' i], a[aria-label*='{name}' i], button[data-test*='easy-apply' i], a[data-test*='easy-apply' i]"
                )
                c_cnt = await _maybe_await(custom_loc.count()) if hasattr(custom_loc, "count") else 0
                if isinstance(c_cnt, int) and c_cnt > 0:
                    for idx in range(c_cnt):
                        cand = custom_loc.nth(idx) if hasattr(custom_loc, "nth") else custom_loc
                        if await self._is_in_excluded_container(cand):
                            continue
                        is_vis = await _maybe_await(cand.is_visible()) if hasattr(cand, "is_visible") else True
                        is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                        if is_vis and is_en:
                            text = (await _maybe_await(cand.text_content()) or "").strip()
                            aria = (await _maybe_await(cand.get_attribute("aria-label")) or "").strip()
                            combined = f"{text} {aria}".strip()
                            if GlassdoorStateDetector.is_easy_apply_button(combined) or (
                                "easy" in combined.lower() and "apply" in combined.lower() and not GlassdoorStateDetector.is_external_apply(combined)
                            ):
                                return cand
        except Exception as exc:
            logger.warning("Error finding Easy Apply button via Playwright: %s", exc)

        return None

    @async_hybrid
    async def click_easy_apply(self, page: Optional[Any] = None) -> Tuple[bool, Optional[str]]:
        """Click the Easy Apply button via Playwright DOM after scrolling into view and verifying visible/enabled."""
        btn = await _maybe_await(self.find_exact_easy_apply_button(page))
        if btn is None:
            return False, "Easy Apply button not found or not enabled"

        try:
            logger.info("Found verified Easy Apply button via Playwright DOM. Scrolling into view and clicking once...")
            try:
                if hasattr(btn, "scroll_into_view_if_needed"):
                    await _maybe_await(btn.scroll_into_view_if_needed())
            except Exception:
                pass
            if hasattr(btn, "wait_for"):
                await _maybe_await(btn.wait_for(state="visible", timeout=3000))
            is_en = await _maybe_await(btn.is_enabled()) if hasattr(btn, "is_enabled") else True
            if not is_en:
                return False, "Easy Apply button is disabled"
            await _maybe_await(btn.click())
            return True, None
        except Exception as exc:
            logger.warning("Playwright Easy Apply click failed: %s", exc)
            return False, str(exc)

    @async_hybrid
    async def _is_in_excluded_container(self, elem: Any) -> bool:
        """Check if an element is inside an excluded section like sidebars, recommendations, or search results."""
        try:
            if not elem or not hasattr(elem, "evaluate"):
                return False
            is_excluded = await _maybe_await(elem.evaluate("""el => {
                if (!el) return false;
                const excluded = el.closest('aside, footer, nav, [data-test*="recommended" i], [data-test*="similar" i], .similarJobs, [data-test*="history" i], [data-test*="activity" i], .search-results, [data-test="job-search-results"], [data-test="job-link"], [class*="JobCard_trackingLink"], [class*="JobCard_jobCardWrapper"], [data-test="job-card-wrapper"]');
                return excluded !== null;
            }"""))
            return bool(is_excluded)
        except Exception:
            return False

    @async_hybrid
    async def check_already_applied(self, page: Optional[Any] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """Inspect the resolved Glassdoor job page for explicit Already Applied evidence.

        Requires strong visible evidence specifically attached to the current listing action area:
        - Status badge / button / pill with exact text: 'Applied', 'You applied', 'Already applied', etc.
        - [data-test="job-applied"] or similar explicit application status component.

        Explicitly excludes:
        - Sidebars and recommendation widgets ('Jobs you've applied to', 'similar jobs', etc.)
        - Search filter badges ('Applied filters')
        - General text containing 'applied' in descriptions or footer notes.

        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (is_applied, signal_text, locator_description)
        """
        target_page = page or self.job_page or self._page
        if not target_page:
            return False, None, None

        try:
            # 1. Check exact buttons / links with role="button" or role="link"
            for name in ALREADY_APPLIED_EXACT_NAMES:
                pattern = re.compile(rf"^\s*{re.escape(name)}\s*$", re.IGNORECASE)

                # Check buttons
                try:
                    btn = target_page.get_by_role("button", name=pattern)
                    cnt = await _maybe_await(btn.count()) if hasattr(btn, "count") else 0
                    if isinstance(cnt, int) and cnt > 0:
                        for idx in range(cnt):
                            cand = btn.nth(idx) if hasattr(btn, "nth") else btn
                            is_vis = await _maybe_await(cand.is_visible()) if hasattr(cand, "is_visible") else False
                            if is_vis:
                                raw_txt = await _maybe_await(cand.inner_text()) if hasattr(cand, "inner_text") else name
                                if not await _maybe_await(self._is_in_excluded_container(cand)):
                                    return True, str(raw_txt).strip(), f"button[name='{name}']"
                except Exception:
                    pass

                # Check links/status pills
                try:
                    link = target_page.get_by_role("link", name=pattern)
                    l_cnt = await _maybe_await(link.count()) if hasattr(link, "count") else 0
                    if isinstance(l_cnt, int) and l_cnt > 0:
                        for idx in range(l_cnt):
                            cand = link.nth(idx) if hasattr(link, "nth") else link
                            is_vis = await _maybe_await(cand.is_visible()) if hasattr(cand, "is_visible") else False
                            if is_vis:
                                raw_txt = await _maybe_await(cand.inner_text()) if hasattr(cand, "inner_text") else name
                                if not await _maybe_await(self._is_in_excluded_container(cand)):
                                    return True, str(raw_txt).strip(), f"link[name='{name}']"
                except Exception:
                    pass

            # 2. Check explicit data-test or semantic status badge selectors
            status_selectors = [
                '[data-test="job-applied"]',
                '[data-test="applied-status"]',
                '[data-test="application-status"]',
                '[data-testid*="applied" i]',
                '.job-applied',
                '.application-status-applied',
            ]
            for sel in status_selectors:
                try:
                    loc = target_page.locator(sel)
                    s_cnt = await _maybe_await(loc.count()) if hasattr(loc, "count") else 0
                    if isinstance(s_cnt, int) and s_cnt > 0:
                        for idx in range(s_cnt):
                            cand = loc.nth(idx) if hasattr(loc, "nth") else loc
                            is_vis = await _maybe_await(cand.is_visible()) if hasattr(cand, "is_visible") else False
                            if is_vis:
                                raw_txt = await _maybe_await(cand.inner_text()) if hasattr(cand, "inner_text") else ""
                                norm_txt = _normalize_text(raw_txt)
                                if any(n in norm_txt for n in ALREADY_APPLIED_EXACT_NAMES):
                                    if not await _maybe_await(self._is_in_excluded_container(cand)):
                                        return True, str(raw_txt).strip(), sel
                except Exception:
                    pass

            # 3. Check Job Header / Action Container for explicit status pills
            header_locators = [
                '[data-test="job-view-header"]',
                '[data-test="job-detail-header"]',
                '[data-test="job-view-action-container"]',
                '.job-view-actions',
                '#JobDetails',
                '.JobDetails',
            ]
            for h_sel in header_locators:
                try:
                    h_loc = target_page.locator(h_sel)
                    h_cnt = await _maybe_await(h_loc.count()) if hasattr(h_loc, "count") else 0
                    if isinstance(h_cnt, int) and h_cnt > 0:
                        h_elem = h_loc.first if hasattr(h_loc, "first") else h_loc
                        for name in ALREADY_APPLIED_EXACT_NAMES:
                            p_loc = h_elem.locator(f"span:has-text('{name}'), div:has-text('{name}'), p:has-text('{name}')")
                            p_cnt = await _maybe_await(p_loc.count()) if hasattr(p_loc, "count") else 0
                            if isinstance(p_cnt, int) and p_cnt > 0:
                                for idx in range(p_cnt):
                                    p_cand = p_loc.nth(idx) if hasattr(p_loc, "nth") else p_loc
                                    is_vis = await _maybe_await(p_cand.is_visible()) if hasattr(p_cand, "is_visible") else False
                                    if is_vis:
                                        p_txt = await _maybe_await(p_cand.inner_text()) if hasattr(p_cand, "inner_text") else ""
                                        p_norm = _normalize_text(p_txt)
                                        if p_norm in ALREADY_APPLIED_EXACT_NAMES or any(re.search(pat, p_norm) for pat in ALREADY_APPLIED_PATTERNS):
                                            return True, str(p_txt).strip(), f"{h_sel} -> {name}"
                except Exception:
                    pass

        except Exception as exc:
            logger.debug("Error during check_already_applied: %s", exc)

        return False, None, None

    @async_hybrid
    async def resolve_application_page(
        self,
        pages_before: Optional[set] = None,
        timeout_seconds: int = 15,
    ) -> Optional[Any]:
        """Detect and adopt the active hosted application page (supporting both same-tab navigation and newly opened tabs)."""
        if not self._context:
            return self.application_page or self.job_page or self.page

        start_time = time.time()
        poll_interval = 0.5

        while time.time() - start_time < timeout_seconds:
            all_pages = getattr(self._context, "pages", [])

            # Filter candidate pages (excluding Swagger / docs / localhost)
            candidate_pages = []
            for p in all_pages:
                try:
                    p_url = getattr(p, "url", "").lower()
                    if "localhost:8000" in p_url or "127.0.0.1:8000" in p_url or "/docs" in p_url or "/redoc" in p_url:
                        continue
                    candidate_pages.append(p)
                except Exception:
                    continue

            # 1. Check newly opened tabs
            if pages_before is not None:
                new_tabs = [p for p in candidate_pages if p not in pages_before]
                for p in new_tabs:
                    try:
                        p_url = getattr(p, "url", "").lower()
                        if any(domain in p_url for domain in SMARTAPPLY_HOST_DOMAINS):
                            self.application_page = p
                            self.page = p
                            try:
                                await _maybe_await(p.bring_to_front())
                            except Exception:
                                pass
                            logger.info("Adopted newly opened application tab (SmartApply URL): %s", p.url)
                            return p

                        # Check DOM state
                        state, _ = await self.detect_page_state(p)
                        if state in (
                            GlassdoorAutomationState.RESUME_STEP,
                            GlassdoorAutomationState.QUESTIONS_STEP,
                            GlassdoorAutomationState.REQUIREMENTS_WARNING,
                            GlassdoorAutomationState.REVIEW_STEP,
                            GlassdoorAutomationState.SUBMISSION_READY,
                        ):
                            self.application_page = p
                            self.page = p
                            try:
                                await _maybe_await(p.bring_to_front())
                            except Exception:
                                pass
                            logger.info("Adopted newly opened application tab with DOM state %s: %s", state.value, p.url)
                            return p
                    except Exception:
                        continue

            # 2. Check existing application_page
            if self.application_page and self.application_page in candidate_pages:
                try:
                    p_url = getattr(self.application_page, "url", "").lower()
                    if any(domain in p_url for domain in SMARTAPPLY_HOST_DOMAINS):
                        return self.application_page
                    res_state = await _maybe_await(self.detect_page_state(self.application_page))
                    state = res_state[0] if isinstance(res_state, tuple) and len(res_state) > 0 else res_state
                    if state != GlassdoorAutomationState.UNKNOWN:
                        return self.application_page
                except Exception:
                    pass

            # 3. Check all candidate pages for SmartApply domain
            for p in candidate_pages:
                try:
                    p_url = getattr(p, "url", "").lower()
                    if any(domain in p_url for domain in SMARTAPPLY_HOST_DOMAINS):
                        self.application_page = p
                        self.page = p
                        try:
                            await _maybe_await(p.bring_to_front())
                        except Exception:
                            pass
                        logger.info("Adopted page matching SmartApply domain: %s", p.url)
                        return p
                except Exception:
                    continue

            # 4. Check candidate pages for Application DOM states
            for p in candidate_pages:
                try:
                    res_state = await _maybe_await(self.detect_page_state(p))
                    state = res_state[0] if isinstance(res_state, tuple) and len(res_state) > 0 else res_state
                    if state in (
                        GlassdoorAutomationState.RESUME_STEP,
                        GlassdoorAutomationState.QUESTIONS_STEP,
                        GlassdoorAutomationState.REQUIREMENTS_WARNING,
                        GlassdoorAutomationState.REVIEW_STEP,
                        GlassdoorAutomationState.SUBMISSION_READY,
                    ):
                        self.application_page = p
                        self.page = p
                        try:
                            await _maybe_await(p.bring_to_front())
                        except Exception:
                            pass
                        logger.info("Adopted page with application DOM state %s: %s", state.value, p.url)
                        return p
                except Exception:
                    continue

            # 5. Check if job_page transitioned in-place
            if self.job_page and self.job_page in candidate_pages:
                try:
                    j_url = getattr(self.job_page, "url", "").lower()
                    if any(domain in j_url for domain in SMARTAPPLY_HOST_DOMAINS):
                        self.application_page = self.job_page
                        self.page = self.job_page
                        logger.info("Adopted same-tab transition on job_page: %s", self.job_page.url)
                        return self.job_page
                except Exception:
                    pass

            await asyncio.sleep(poll_interval)

        return self.application_page or self.job_page or self.page

    @async_hybrid
    async def resolve_paused_application_page(
        self,
        session_data: Dict[str, Any],
        timeout_seconds: int = 5,
    ) -> Tuple[Optional[Any], str, Dict[str, Any]]:
        """Dedicated resolver for HITL resume: rediscover the paused application tab from browser state.

        Resolution Priority:
        1. Question signature / unresolved keys match (strongest evidence)
        2. Exact stored application_page_url match
        3. Stored application host + supported application DOM state
        4. Any candidate page in a supported application DOM state (QUESTIONS_STEP, RESUME_STEP, etc.)
        5. SmartApply domain match fallback

        Returns:
            Tuple: (page: Optional[Any], recovery_method: str, diagnostics: Dict[str, Any])
        """
        import urllib.parse

        if self._browser is None or (hasattr(self._browser, "is_connected") and not self._browser.is_connected()):
            connected, err = await _maybe_await(self.connect_cdp(timeout_seconds=timeout_seconds))
            if not connected or not self._browser:
                return None, "cdp_failed", {"error": err or "Failed to connect over CDP"}

        contexts = getattr(self._browser, "contexts", [])
        if not contexts:
            return None, "no_contexts", {"error": "Connected over CDP but no browser contexts were found."}

        self._context = contexts[0]
        all_pages = getattr(self._context, "pages", [])

        # Filter candidate pages (ignore Swagger / docs / localhost)
        candidate_pages = []
        for p in all_pages:
            try:
                if not self._is_page_valid(p):
                    continue
                p_url = getattr(p, "url", "").lower()
                if "localhost:8000" in p_url or "127.0.0.1:8000" in p_url or "/docs" in p_url or "/redoc" in p_url:
                    continue
                candidate_pages.append(p)
            except Exception:
                continue

        stored_url = session_data.get("application_page_url", "")
        stored_clean_url = stored_url.split("?")[0].rstrip("/").lower() if stored_url else ""
        stored_host = (session_data.get("application_page_host") or session_data.get("application_host") or "").lower()
        if not stored_host and stored_url:
            try:
                stored_host = urllib.parse.urlparse(stored_url).netloc.lower()
            except Exception:
                pass

        stored_unresolved = session_data.get("unresolved_questions") or []
        stored_keys = set(session_data.get("unresolved_question_keys") or [])
        if not stored_keys and stored_unresolved:
            for q in stored_unresolved:
                if isinstance(q, dict) and q.get("normalized_key"):
                    stored_keys.add(q["normalized_key"])
                elif isinstance(q, str):
                    stored_keys.add(q)

        pages_inspected: List[Dict[str, Any]] = []
        page_details: List[Tuple[Any, str, str, GlassdoorAutomationState, List[str]]] = []

        for p in candidate_pages:
            try:
                p_url = getattr(p, "url", "")
                p_title = ""
                try:
                    if hasattr(p, "title") and callable(p.title):
                        raw_title = p.title()
                        if inspect.isawaitable(raw_title):
                            p_title = await raw_title
                        elif isinstance(raw_title, str):
                            p_title = raw_title
                except Exception:
                    pass
                p_host = urllib.parse.urlparse(p_url).netloc.lower()

                # Detect state on page
                det_res = await _maybe_await(self.detect_page_state(p))
                det_state = det_res[0] if isinstance(det_res, tuple) and len(det_res) > 0 else (det_res if isinstance(det_res, GlassdoorAutomationState) else GlassdoorAutomationState.UNKNOWN)

                # Extract question keys on page
                extracted_qs = []
                if det_state == GlassdoorAutomationState.QUESTIONS_STEP or not det_state:
                    extracted_qs = await _maybe_await(self.extract_questions(p))
                q_keys = [q.normalized_key for q in (extracted_qs or []) if hasattr(q, "normalized_key")]

                pages_inspected.append({
                    "url": p_url,
                    "title": p_title,
                    "host": p_host,
                    "detected_state": det_state.value if hasattr(det_state, "value") else str(det_state),
                    "question_keys": q_keys,
                })
                page_details.append((p, p_url, p_host, det_state, q_keys))
            except Exception as exc:
                logger.debug("Error inspecting candidate page during resume: %s", exc)

        recovered_page = None
        recovery_method = "none"

        # 1. Question signature / unresolved question keys match
        if stored_keys:
            for p, p_url, p_host, det_state, q_keys in page_details:
                if set(q_keys) & stored_keys:
                    recovered_page = p
                    recovery_method = "question_signature"
                    logger.info("Recovered paused application page via question signature (%s): %s", q_keys, p_url)
                    break

        # 2. Exact stored URL match
        if recovered_page is None and stored_clean_url:
            for p, p_url, p_host, det_state, q_keys in page_details:
                p_clean = p_url.split("?")[0].rstrip("/").lower()
                if p_clean == stored_clean_url or (len(stored_clean_url) > 10 and stored_clean_url in p_clean):
                    recovered_page = p
                    recovery_method = "exact_stored_url"
                    logger.info("Recovered paused application page via exact stored URL: %s", p_url)
                    break

        # 3. Stored host + supported application state
        if recovered_page is None and stored_host:
            for p, p_url, p_host, det_state, q_keys in page_details:
                if stored_host in p_host and det_state in (
                    GlassdoorAutomationState.QUESTIONS_STEP,
                    GlassdoorAutomationState.RESUME_STEP,
                    GlassdoorAutomationState.REQUIREMENTS_WARNING,
                    GlassdoorAutomationState.REVIEW_STEP,
                    GlassdoorAutomationState.SUBMISSION_READY,
                ):
                    recovered_page = p
                    recovery_method = "stored_host_and_state"
                    logger.info("Recovered paused application page via stored host '%s' and state %s: %s", stored_host, det_state.value, p_url)
                    break

        # 4. Any candidate page in a supported application state
        if recovered_page is None:
            for p, p_url, p_host, det_state, q_keys in page_details:
                if det_state in (
                    GlassdoorAutomationState.QUESTIONS_STEP,
                    GlassdoorAutomationState.RESUME_STEP,
                    GlassdoorAutomationState.REQUIREMENTS_WARNING,
                    GlassdoorAutomationState.REVIEW_STEP,
                    GlassdoorAutomationState.SUBMISSION_READY,
                ):
                    recovered_page = p
                    recovery_method = "application_state_fallback"
                    logger.info("Recovered paused application page via DOM state %s: %s", det_state.value, p_url)
                    break

        # 5. SmartApply domain match fallback
        if recovered_page is None:
            for p, p_url, p_host, det_state, q_keys in page_details:
                if any(dom in p_url.lower() for dom in SMARTAPPLY_HOST_DOMAINS):
                    recovered_page = p
                    recovery_method = "smartapply_domain_fallback"
                    logger.info("Recovered paused application page via SmartApply domain: %s", p_url)
                    break

        diagnostics = {
            "cdp_connect_attempted": True,
            "cdp_connected": True,
            "browser_session_verified": True,
            "stored_application_page_url": stored_url,
            "stored_application_host": stored_host,
            "pages_inspected": pages_inspected,
            "application_page_recovered": recovered_page is not None,
            "application_page_recovery_method": recovery_method,
            "current_application_page_url": getattr(recovered_page, "url", "") if recovered_page else "",
        }

        if recovered_page is not None:
            self.application_page = recovered_page
            self._page = recovered_page
            try:
                if hasattr(recovered_page, "bring_to_front") and callable(recovered_page.bring_to_front):
                    await _maybe_await(recovered_page.bring_to_front())
            except Exception:
                pass

        return recovered_page, recovery_method, diagnostics

    @async_hybrid
    async def _find_active_application_page(self) -> Optional[Any]:
        """Locate the tab corresponding to the hosted application or active page."""
        if not self._context:
            return None

        pages = getattr(self._context, "pages", [])
        if not pages:
            return None

        # Priority 1: Match SmartApply host domains
        for p in pages:
            try:
                url = getattr(p, "url", "").lower()
                if any(domain in url for domain in SMARTAPPLY_HOST_DOMAINS):
                    return p
            except Exception:
                continue

        # Priority 2: Match Glassdoor domain
        for p in pages:
            try:
                url = getattr(p, "url", "").lower()
                if "glassdoor" in url and ("job-listing" in url or "job" in url):
                    return p
            except Exception:
                continue

        # Priority 3: Return the most recent page with a valid HTTP URL (ignoring localhost / docs)
        for p in reversed(pages):
            try:
                p_url = getattr(p, "url", "")
                if p_url.startswith("http") and "docs" not in p_url.lower():
                    return p
            except Exception:
                continue

        return pages[-1] if pages else None

    @async_hybrid
    async def close(self) -> None:
        """Clean up Playwright CDP connection resources."""
        try:
            if self._browser:
                await _maybe_await(self._browser.close())
                self._browser = None
            if self._playwright:
                await _maybe_await(self._playwright.stop())
                self._playwright = None
            self._page = None
            self._context = None
            logger.debug("Playwright CDP connection closed.")
        except Exception as exc:
            logger.debug("Error while closing Playwright CDP connection: %s", exc)

    # ==========================================================================
    # State Machine & Challenge Detection
    # ==========================================================================

    @async_hybrid
    async def detect_page_state(self, page: Optional[Any] = None) -> Tuple[GlassdoorAutomationState, Dict[str, Any]]:
        """Inspect the current DOM state of the hosted application page.

        Strict Detection Priority (Requirement R):
        1. REAL CAPTCHA / login / MFA
        2. exact final Submit -> SUBMISSION_READY
        3. requirements warning ("Apply anyway") -> REQUIREMENTS_WARNING
        4. editable employer questions -> QUESTIONS_STEP
        5. existing resume selection -> RESUME_STEP
        6. review summary -> REVIEW_STEP
        7. action controls / unknown
        """
        target_page = page or self._page
        if not target_page:
            return GlassdoorAutomationState.UNKNOWN, {"reason": "No active Playwright page"}

        try:
            current_url = getattr(target_page, "url", "") or ""
        except Exception:
            current_url = ""

        diagnostics: Dict[str, Any] = {
            "current_url": current_url,
            "is_smartapply": any(d in current_url for d in SMARTAPPLY_HOST_DOMAINS),
        }

        # Collect visible actions for diagnostics across all frames
        vis_actions = await _maybe_await(self.get_visible_actions(target_page))
        diagnostics["visible_actions"] = vis_actions

        # ----------------------------------------------------------------------
        # Priority 1: Blocking / Challenge Detection (Strong interactive evidence only)
        # ----------------------------------------------------------------------
        blocking_state, blocking_reason = await _maybe_await(self._detect_blocking_in_dom(target_page))
        if blocking_state is not None:
            diagnostics["blocking_reason"] = blocking_reason
            return blocking_state, diagnostics

        # ----------------------------------------------------------------------
        # Priority 2: Exact Submit Button Check (HIGHEST APPLICATION STATE PRIORITY)
        # ----------------------------------------------------------------------
        submit_locator = await _maybe_await(self.find_exact_submit_button(target_page))
        if submit_locator is not None:
            diagnostics["submit_button_detected"] = True
            return GlassdoorAutomationState.SUBMISSION_READY, diagnostics

        # ----------------------------------------------------------------------
        # Priority 3: Requirements Warning Step Check ("Apply anyway" / "Don't meet requirements")
        # ----------------------------------------------------------------------
        is_warning, reqs_not_met, reqs_text = await _maybe_await(self._detect_requirements_warning(target_page))
        if is_warning:
            diagnostics["requirements_warning_detected"] = True
            diagnostics["requirements_not_met"] = reqs_not_met
            diagnostics["requirements_text"] = reqs_text
            apply_btn = await _maybe_await(self.find_exact_apply_anyway_button(target_page))
            diagnostics["apply_anyway_found"] = apply_btn is not None
            return GlassdoorAutomationState.REQUIREMENTS_WARNING, diagnostics

        # ----------------------------------------------------------------------
        # Priority 4: Optional Survey / Interstitial Step Check ("Help Indeed learn more...")
        # ----------------------------------------------------------------------
        is_survey, survey_info = await _maybe_await(self._detect_survey_step(target_page))
        if is_survey:
            diagnostics["survey_step_detected"] = True
            diagnostics["survey_info"] = survey_info
            return GlassdoorAutomationState.OPTIONAL_SURVEY_STEP, diagnostics

        # ----------------------------------------------------------------------
        # Priority 5: Questions Step Check (Editable employer questions)
        # ----------------------------------------------------------------------
        questions = await _maybe_await(self.extract_questions(target_page))
        if questions:
            diagnostics["questions_count"] = len(questions)
            return GlassdoorAutomationState.QUESTIONS_STEP, diagnostics

        # ----------------------------------------------------------------------
        # Priority 6: Resume Step Check
        # ----------------------------------------------------------------------
        if await _maybe_await(self._is_resume_step(target_page)):
            diagnostics["resume_step_detected"] = True
            return GlassdoorAutomationState.RESUME_STEP, diagnostics

        # ----------------------------------------------------------------------
        # Priority 7: Review Step Check
        # ----------------------------------------------------------------------
        if await _maybe_await(self._is_review_step(target_page)):
            diagnostics["review_step_detected"] = True
            return GlassdoorAutomationState.REVIEW_STEP, diagnostics

        # ----------------------------------------------------------------------
        # Priority 8: Action Control Check (Continue / Next / Review)
        # ----------------------------------------------------------------------
        action_name, _ = await _maybe_await(self.find_progression_action(target_page))
        if action_name:
            diagnostics["progression_action"] = action_name
            if action_name == "review":
                return GlassdoorAutomationState.REVIEW_AVAILABLE, diagnostics
            elif action_name == "continue":
                return GlassdoorAutomationState.CONTINUE_AVAILABLE, diagnostics
            elif action_name == "next":
                return GlassdoorAutomationState.NEXT_AVAILABLE, diagnostics
            return GlassdoorAutomationState.APPLICATION_MODAL, diagnostics

        return GlassdoorAutomationState.APPLICATION_MODAL, diagnostics

    @async_hybrid
    async def get_visible_actions(self, page: Optional[Any] = None) -> List[str]:
        """Extract visible action button and link labels across all frames."""
        target_page = page or self._page
        if not target_page:
            return []

        targets = _get_all_frames_and_page(target_page)
        actions: List[str] = []
        seen = set()

        for tgt in targets:
            try:
                loc = tgt.locator("button, a, [role='button'], [role='link'], input[type='submit'], input[type='button']")
                cnt = await _maybe_await(loc.count()) if hasattr(loc, "count") else 0
                if isinstance(cnt, int):
                    for idx in range(min(cnt, 30)):
                        elem = loc.nth(idx) if hasattr(loc, "nth") else loc
                        try:
                            is_vis = await _maybe_await(elem.is_visible()) if hasattr(elem, "is_visible") else True
                            if not is_vis:
                                continue
                            raw_txt = ""
                            if hasattr(elem, "inner_text"):
                                raw_txt = await _maybe_await(elem.inner_text())
                            if not raw_txt and hasattr(elem, "get_attribute"):
                                raw_txt = (await _maybe_await(elem.get_attribute("value"))) or (await _maybe_await(elem.get_attribute("aria-label"))) or ""
                            txt = str(raw_txt).strip()
                            txt = re.sub(r"\s+", " ", txt).strip()
                            if txt and len(txt) <= 60 and txt.lower() not in seen:
                                seen.add(txt.lower())
                                actions.append(txt)
                        except Exception:
                            continue
            except Exception:
                continue

        return actions

    @async_hybrid
    async def _detect_blocking_in_dom(self, page: Any) -> Tuple[Optional[GlassdoorAutomationState], Optional[str]]:
        """Detect interactive challenges, logins, or MFA widgets in DOM while ignoring benign footers."""
        try:
            # Check for active CAPTCHA challenge iframes or interactive checkboxes
            captcha_selectors = [
                'iframe[src*="recaptcha/api2/bframe"]',
                'iframe[src*="recaptcha/enterprise/bframe"]',
                'iframe[src*="hcaptcha.com"]',
                'iframe[src*="challenges.cloudflare.com"]',
                'div.g-recaptcha[data-sitekey]',
                'div.h-captcha',
                '#cf-turnstile',
            ]
            for sel in captcha_selectors:
                try:
                    loc = page.locator(sel)
                    cnt = await _maybe_await(loc.count()) if hasattr(loc, "count") else 0
                    if isinstance(cnt, int) and cnt > 0:
                        first_loc = loc.first if hasattr(loc, "first") else loc
                        is_vis = await _maybe_await(first_loc.is_visible()) if hasattr(first_loc, "is_visible") else False
                        if is_vis is True:
                            return GlassdoorAutomationState.CAPTCHA_OR_CHALLENGE, f"Active CAPTCHA widget visible: {sel}"
                except Exception:
                    pass

            # Check for interactive challenge text in modal/page body
            # EXCLUDE benign footer text: "This site is protected by reCAPTCHA..."
            try:
                body_elem = page.locator("body")
                raw_text = await _maybe_await(body_elem.inner_text()) if hasattr(body_elem, "inner_text") else ""
                if isinstance(raw_text, str) and raw_text.strip():
                    blocking_info = GlassdoorStateDetector.detect_blocking_state(raw_text)
                    if blocking_info:
                        return blocking_info[0], blocking_info[1]
            except Exception:
                pass

            # Check for Login / Password inputs
            try:
                pwd_input = page.locator('input[type="password"]')
                pwd_cnt = await _maybe_await(pwd_input.count()) if hasattr(pwd_input, "count") else 0
                if isinstance(pwd_cnt, int) and pwd_cnt > 0:
                    first_pwd = pwd_input.first if hasattr(pwd_input, "first") else pwd_input
                    is_vis = await _maybe_await(first_pwd.is_visible()) if hasattr(first_pwd, "is_visible") else False
                    if is_vis is True:
                        return GlassdoorAutomationState.LOGIN_REQUIRED, "Password input field visible."
            except Exception:
                pass

            # Check for MFA / OTP inputs
            try:
                otp_input = page.locator('input[autocomplete="one-time-code"], input[name*="otp" i], input[name*="code" i]')
                otp_cnt = await _maybe_await(otp_input.count()) if hasattr(otp_input, "count") else 0
                if isinstance(otp_cnt, int) and otp_cnt > 0:
                    first_otp = otp_input.first if hasattr(otp_input, "first") else otp_input
                    is_vis = await _maybe_await(first_otp.is_visible()) if hasattr(first_otp, "is_visible") else False
                    if is_vis is True:
                        return GlassdoorAutomationState.MFA_REQUIRED, "One-time passcode input field visible."
            except Exception:
                pass

        except Exception as exc:
            logger.debug("Error inspecting blocking state in DOM: %s", exc)

        return None, None

    @async_hybrid
    async def find_exact_submit_button(self, page: Optional[Any] = None) -> Optional[Any]:
        """Locate visible, enabled button whose normalized text is EXACTLY in EXACT_SUBMIT_NAMES."""
        target_page = page or self._page
        if not target_page:
            return None

        for name in EXACT_SUBMIT_NAMES:
            try:
                # 1. Look by role="button"
                pattern = re.compile(rf"^\s*{re.escape(name)}\s*$", re.IGNORECASE)
                btn = target_page.get_by_role("button", name=pattern)
                cnt = await _maybe_await(btn.count()) if hasattr(btn, "count") else 0
                if isinstance(cnt, int) and cnt > 0:
                    cand = btn.first if hasattr(btn, "first") else btn
                    is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                    if is_en:
                        return cand

                # 2. Look by input[type="submit"]
                submit_input = target_page.locator(f'input[type="submit"][value="{name}" i]')
                s_cnt = await _maybe_await(submit_input.count()) if hasattr(submit_input, "count") else 0
                if isinstance(s_cnt, int) and s_cnt > 0:
                    cand = submit_input.first if hasattr(submit_input, "first") else submit_input
                    is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                    if is_en:
                        return cand

                # 3. Look by button locator with text
                loc_btn = target_page.locator(f"button:has-text('{name}')")
                b_cnt = await _maybe_await(loc_btn.count()) if hasattr(loc_btn, "count") else 0
                if isinstance(b_cnt, int) and b_cnt > 0:
                    for i in range(b_cnt):
                        cand = loc_btn.nth(i) if hasattr(loc_btn, "nth") else loc_btn
                        is_vis = await _maybe_await(cand.is_visible()) if hasattr(cand, "is_visible") else True
                        is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                        if is_vis and is_en:
                            return cand
            except Exception:
                continue

        return None

    @async_hybrid
    async def _is_resume_step(self, page: Any) -> bool:
        """Check if current page is the resume selection step."""
        try:
            url = getattr(page, "url", "").lower()
            if "resume-selection" in url or "indeedapply/form/resume" in url:
                return True

            # Check for resume heading / radio card
            headings = page.locator("h1, h2, h3, legend, [role='heading']")
            h_cnt = await _maybe_await(headings.count()) if hasattr(headings, "count") else 0
            if isinstance(h_cnt, int):
                for i in range(min(h_cnt, 10)):
                    h_elem = headings.nth(i) if hasattr(headings, "nth") else headings
                    h_raw = await _maybe_await(h_elem.inner_text()) if hasattr(h_elem, "inner_text") else ""
                    h_text = _normalize_text(h_raw)
                    if "select a resume" in h_text or "choose a resume" in h_text or "resume" in h_text:
                        # Verify presence of resume radio or file list
                        r_loc = page.locator("input[type='radio'][name*='resume' i], [data-testid*='resume' i], .resume-item")
                        r_cnt = await _maybe_await(r_loc.count()) if hasattr(r_loc, "count") else 0
                        if isinstance(r_cnt, int) and r_cnt > 0:
                            return True
        except Exception:
            pass
        return False

    @async_hybrid
    async def _is_review_step(self, page: Any) -> bool:
        """Check if current page is the application review step."""
        try:
            url = getattr(page, "url", "").lower()
            if "/review" in url or "indeedapply/form/review" in url:
                return True

            headings = page.locator("h1, h2, h3, [role='heading']")
            h_cnt = await _maybe_await(headings.count()) if hasattr(headings, "count") else 0
            if isinstance(h_cnt, int):
                for i in range(min(h_cnt, 5)):
                    h_elem = headings.nth(i) if hasattr(headings, "nth") else headings
                    h_raw = await _maybe_await(h_elem.inner_text()) if hasattr(h_elem, "inner_text") else ""
                    h_text = _normalize_text(h_raw)
                    if "review your application" in h_text or "review application" in h_text:
                        return True
        except Exception:
            pass
        return False

    @async_hybrid
    async def _detect_requirements_warning(self, page: Any) -> Tuple[bool, List[str], str]:
        """Detect whether the page is an employer requirements warning page.

        Signals:
        - Signal A: Heading or prominent text containing:
          "It looks like you don't meet these employer requirements" (tolerant of straight/curly apostrophes, whitespace)
          or "You may not hear back from the employer based on your responses"
        - Signal B: Visible "Apply anyway" action combined with "Return to job search" action/text.
        - Signal C: Requirement text containing "(Required)" / "(Mandatory)" combined with "Apply anyway" action.

        Returns:
            Tuple: (is_warning: bool, requirements_not_met: List[str], requirements_text: str)
        """
        target_page = page or self._page
        if not target_page:
            return False, [], ""

        try:
            targets = _get_all_frames_and_page(target_page)
            warning_heading_found = False
            warning_patterns = [
                "it looks like you don't meet these employer requirements",
                "it looks like you do not meet these employer requirements",
                "don't meet these employer requirements",
                "do not meet these employer requirements",
                "don't meet employer requirements",
                "do not meet employer requirements",
                "you may not hear back from the employer based on your responses",
            ]

            # 1. Search headings and text across all frames
            for tgt in targets:
                try:
                    text_elems = tgt.locator("h1, h2, h3, h4, h5, [role='heading'], p, div")
                    t_cnt = await _maybe_await(text_elems.count()) if hasattr(text_elems, "count") else 0
                    if isinstance(t_cnt, int):
                        for idx in range(min(t_cnt, 25)):
                            elem = text_elems.nth(idx) if hasattr(text_elems, "nth") else text_elems
                            raw = await _maybe_await(elem.inner_text()) if hasattr(elem, "inner_text") else ""
                            norm = _normalize_text(raw)
                            if any(pat in norm for pat in warning_patterns):
                                warning_heading_found = True
                                break
                except Exception:
                    continue
                if warning_heading_found:
                    break

            # 2. Check for "Apply anyway" action across frames
            has_apply_anyway = (await _maybe_await(self.find_exact_apply_anyway_button(target_page))) is not None

            # 3. Check for "Return to job search" action/text across frames
            has_return_to_search = False
            for tgt in targets:
                try:
                    ret_loc1 = tgt.get_by_role("button", name=re.compile(r"return\s+to\s+job\s+search", re.I))
                    ret_loc2 = tgt.get_by_role("link", name=re.compile(r"return\s+to\s+job\s+search", re.I))
                    ret_loc3 = tgt.locator("button:has-text('Return to job search'), a:has-text('Return to job search'), [role='button']:has-text('Return to job search'), [role='link']:has-text('Return to job search')")
                    c1 = await _maybe_await(ret_loc1.count()) if hasattr(ret_loc1, "count") else 0
                    c2 = await _maybe_await(ret_loc2.count()) if hasattr(ret_loc2, "count") else 0
                    c3 = await _maybe_await(ret_loc3.count()) if hasattr(ret_loc3, "count") else 0
                    if c1 > 0 or c2 > 0 or c3 > 0:
                        has_return_to_search = True
                        break
                except Exception:
                    continue

            # 4. Extract unmet requirements text across frames
            requirements_not_met: List[str] = []
            for tgt in targets:
                try:
                    items = tgt.locator("ul li, ol li, [data-testid*='requirement' i], div[class*='requirement' i] p, div[class*='Requirement' i], [role='listitem']")
                    i_cnt = await _maybe_await(items.count()) if hasattr(items, "count") else 0
                    if isinstance(i_cnt, int):
                        for i_idx in range(min(i_cnt, 15)):
                            item_elem = items.nth(i_idx) if hasattr(items, "nth") else items
                            item_raw = await _maybe_await(item_elem.inner_text()) if hasattr(item_elem, "inner_text") else ""
                            item_txt = item_raw.strip()
                            norm_item = _normalize_text(item_txt)
                            if item_txt and not any(kw in norm_item for kw in ["return to job search", "apply anyway", "don't meet", "you may not hear back", "employer requirements"]):
                                if item_txt not in requirements_not_met:
                                    requirements_not_met.append(item_txt)
                except Exception:
                    continue

            if not requirements_not_met:
                for tgt in targets:
                    try:
                        body_elem = tgt.locator("body")
                        body_text = await _maybe_await(body_elem.inner_text()) if hasattr(body_elem, "inner_text") else ""
                        if body_text:
                            matches = re.findall(r"([A-Za-z0-9\s/,-]+:\s*\d+[\s\w]*?(?:\(Required\)|\(Mandatory\))?)", body_text, re.IGNORECASE)
                            for m in matches:
                                m_clean = m.strip()
                                if m_clean and ":" in m_clean and m_clean not in requirements_not_met:
                                    requirements_not_met.append(m_clean)
                    except Exception:
                        continue

            has_requirement_text = bool(requirements_not_met)
            is_warning = warning_heading_found or (has_apply_anyway and has_return_to_search) or (has_apply_anyway and has_requirement_text)

            if is_warning:
                req_summary = "; ".join(requirements_not_met)
                return True, requirements_not_met, req_summary

        except Exception as exc:
            logger.debug("Error checking requirements warning in DOM: %s", exc)

        return False, [], ""

    @async_hybrid
    async def find_exact_apply_anyway_button(self, page: Optional[Any] = None) -> Optional[Any]:
        """Locate visible, enabled Apply Anyway button, link, or clickable control across all frames."""
        target_page = page or self._page
        if not target_page:
            return None

        targets = _get_all_frames_and_page(target_page)
        for tgt in targets:
            for name in APPLY_ANYWAY_BUTTON_NAMES:
                pattern = re.compile(rf"^\s*{re.escape(name)}\s*$", re.IGNORECASE)

                # Priority 1: Button role with exact name
                try:
                    btn = tgt.get_by_role("button", name=pattern)
                    cnt = await _maybe_await(btn.count()) if hasattr(btn, "count") else 0
                    if isinstance(cnt, int) and cnt > 0:
                        for idx in range(cnt):
                            cand = btn.nth(idx) if hasattr(btn, "nth") else btn
                            is_vis = await _maybe_await(cand.is_visible()) if hasattr(cand, "is_visible") else True
                            is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                            if is_vis and is_en:
                                return cand
                except Exception:
                    pass

                # Priority 2: Link role with exact name
                try:
                    link = tgt.get_by_role("link", name=pattern)
                    l_cnt = await _maybe_await(link.count()) if hasattr(link, "count") else 0
                    if isinstance(l_cnt, int) and l_cnt > 0:
                        for idx in range(l_cnt):
                            cand = link.nth(idx) if hasattr(link, "nth") else link
                            is_vis = await _maybe_await(cand.is_visible()) if hasattr(cand, "is_visible") else True
                            is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                            if is_vis and is_en:
                                return cand
                except Exception:
                    pass

                # Priority 3: Locator by button, link, role=button, role=link with text
                try:
                    custom_loc = tgt.locator(f"button:has-text('{name}'), a:has-text('{name}'), [role='button']:has-text('{name}'), [role='link']:has-text('{name}')")
                    c_cnt = await _maybe_await(custom_loc.count()) if hasattr(custom_loc, "count") else 0
                    if isinstance(c_cnt, int) and c_cnt > 0:
                        for idx in range(c_cnt):
                            cand = custom_loc.nth(idx) if hasattr(custom_loc, "nth") else custom_loc
                            is_vis = await _maybe_await(cand.is_visible()) if hasattr(cand, "is_visible") else True
                            is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                            if is_vis and is_en:
                                return cand
                except Exception:
                    pass

                # Priority 4: Verified clickable text control (span, div, p, strong, em)
                try:
                    text_cands = tgt.locator("span, div, p, strong, em, b").filter(has_text=pattern)
                    t_cnt = await _maybe_await(text_cands.count()) if hasattr(text_cands, "count") else 0
                    if isinstance(t_cnt, int) and t_cnt > 0:
                        for idx in range(t_cnt):
                            cand = text_cands.nth(idx) if hasattr(text_cands, "nth") else text_cands
                            is_vis = await _maybe_await(cand.is_visible()) if hasattr(cand, "is_visible") else True
                            if is_vis:
                                return cand
                except Exception:
                    pass

        return None

    @async_hybrid
    async def click_apply_anyway(self, page: Optional[Any] = None) -> Tuple[bool, Optional[str]]:
        """Click the verified Apply Anyway button or link once after scrolling into view."""
        btn = await _maybe_await(self.find_exact_apply_anyway_button(page))
        if btn is None:
            return False, "Apply anyway button not found or not enabled"

        try:
            logger.info("Clicking verified 'Apply anyway' control once...")
            if hasattr(btn, "scroll_into_view_if_needed"):
                await _maybe_await(btn.scroll_into_view_if_needed())
            if hasattr(btn, "wait_for"):
                await _maybe_await(btn.wait_for(state="visible", timeout=3000))
            is_en = await _maybe_await(btn.is_enabled()) if hasattr(btn, "is_enabled") else True
            if not is_en:
                return False, "Apply anyway button is disabled"
            await _maybe_await(btn.click())
            target_page = page or self._page
            if target_page and hasattr(target_page, "wait_for_load_state"):
                try:
                    await _maybe_await(target_page.wait_for_load_state("domcontentloaded", timeout=3000))
                except Exception:
                    pass
                await asyncio.sleep(0.5)
            return True, None
        except Exception as exc:
            logger.warning("Failed to click Apply anyway: %s", exc)
            return False, str(exc)

    # ==========================================================================
    # Survey Interstitial Detection & SPA Transition Waiting
    # ==========================================================================

    @async_hybrid
    async def _detect_survey_step(self, page: Any) -> Tuple[bool, Dict[str, Any]]:
        """Detect Indeed SmartApply survey/interstitial step (e.g. 'Help Indeed learn more about why you're applying').

        Signals:
        - Prominent heading matching 'help indeed learn more about why you're applying' or 'why you're applying'
        - Subtext matching 'we won't share your response with the employer'
        - Textarea / input with label or placeholder containing 'Reason for applying'

        Returns:
            Tuple: (is_survey: bool, survey_info: Dict[str, Any])
        """
        target_page = page or self._page
        if not target_page:
            return False, {}

        try:
            targets = _get_all_frames_and_page(target_page)
            survey_heading_found = False
            survey_subtext_found = False
            survey_field_found = False
            primary_heading = ""
            subtext_str = ""
            field_label = "Reason for applying"
            field_is_req = False

            for tgt in targets:
                # 1. Search headings
                try:
                    headings = tgt.locator("h1, h2, h3, h4, [role='heading']")
                    h_cnt = await _maybe_await(headings.count()) if hasattr(headings, "count") else 0
                    if isinstance(h_cnt, int):
                        for idx in range(min(h_cnt, 10)):
                            h_elem = headings.nth(idx) if hasattr(headings, "nth") else headings
                            raw_h = await _maybe_await(h_elem.inner_text()) if hasattr(h_elem, "inner_text") else ""
                            norm_h = _normalize_text(raw_h)
                            if any(pat in norm_h for pat in SURVEY_HEADING_PATTERNS):
                                survey_heading_found = True
                                primary_heading = raw_h.strip()
                                break
                except Exception:
                    pass

                # 2. Search subtext/paragraphs
                try:
                    paras = tgt.locator("p, div, span")
                    p_cnt = await _maybe_await(paras.count()) if hasattr(paras, "count") else 0
                    if isinstance(p_cnt, int):
                        for idx in range(min(p_cnt, 20)):
                            p_elem = paras.nth(idx) if hasattr(paras, "nth") else paras
                            raw_p = await _maybe_await(p_elem.inner_text()) if hasattr(p_elem, "inner_text") else ""
                            norm_p = _normalize_text(raw_p)
                            if any(pat in norm_p for pat in SURVEY_SUBTEXT_PATTERNS):
                                survey_subtext_found = True
                                subtext_str = raw_p.strip()
                                break
                except Exception:
                    pass

                # 3. Search survey field (Reason for applying textarea / input)
                try:
                    textareas = tgt.locator("textarea, input[type='text']")
                    t_cnt = await _maybe_await(textareas.count()) if hasattr(textareas, "count") else 0
                    if isinstance(t_cnt, int):
                        for idx in range(min(t_cnt, 10)):
                            ta = textareas.nth(idx) if hasattr(textareas, "nth") else textareas
                            lbl = await _maybe_await(self._find_label_for_input(tgt, ta))
                            lbl_norm = _normalize_text(lbl)
                            if any(f_id in lbl_norm for f_id in SURVEY_FIELD_IDENTIFIERS):
                                survey_field_found = True
                                field_label = lbl or "Reason for applying"
                                # Check required status
                                is_req = False
                                is_req_attr = await _maybe_await(ta.get_attribute("required")) if hasattr(ta, "get_attribute") else None
                                is_aria_req = await _maybe_await(ta.get_attribute("aria-required")) if hasattr(ta, "get_attribute") else None
                                if is_req_attr is True or (isinstance(is_req_attr, str) and is_req_attr.lower() in ("true", "required", "")):
                                    is_req = True
                                elif is_aria_req is True or (isinstance(is_aria_req, str) and is_aria_req.lower() == "true"):
                                    is_req = True
                                elif "required" in lbl_norm or "*" in (lbl or ""):
                                    is_req = True
                                field_is_req = is_req
                                break
                except Exception:
                    pass

                if survey_heading_found or (survey_subtext_found and survey_field_found):
                    survey_info = {
                        "heading": primary_heading or "Help Indeed learn more about why you're applying",
                        "subtext": subtext_str or "We won’t share your response with the employer.",
                        "field_name": "reason_for_applying",
                        "field_label": field_label,
                        "required": field_is_req,
                    }
                    return True, survey_info

        except Exception as exc:
            logger.debug("Error checking survey step in DOM: %s", exc)

        return False, {}

    @async_hybrid
    async def capture_dom_snapshot(self, page: Optional[Any] = None) -> Dict[str, Any]:
        """Capture lightweight DOM state signature to detect client-side SPA transitions."""
        target_page = page or self._page
        if not target_page:
            return {}

        snapshot: Dict[str, Any] = {
            "url": getattr(target_page, "url", "") or "",
            "primary_heading": "",
            "is_warning": False,
            "is_survey": False,
            "has_submit": False,
            "visible_actions": [],
            "questions_count": 0,
        }

        try:
            # 1. Primary heading
            headings = target_page.locator("h1, h2, h3, [role='heading']")
            h_cnt = await _maybe_await(headings.count()) if hasattr(headings, "count") else 0
            if isinstance(h_cnt, int) and h_cnt > 0:
                h0 = headings.first if hasattr(headings, "first") else headings
                snapshot["primary_heading"] = _normalize_text(await _maybe_await(h0.inner_text()) if hasattr(h0, "inner_text") else "")

            # 2. Warning check
            is_w, _, _ = await _maybe_await(self._detect_requirements_warning(target_page))
            snapshot["is_warning"] = is_w

            # 3. Survey check
            is_s, _ = await _maybe_await(self._detect_survey_step(target_page))
            snapshot["is_survey"] = is_s

            # 4. Submit check
            sub_btn = await _maybe_await(self.find_exact_submit_button(target_page))
            snapshot["has_submit"] = sub_btn is not None

            # 5. Visible actions
            snapshot["visible_actions"] = await _maybe_await(self.get_visible_actions(target_page))

        except Exception as exc:
            logger.debug("Error capturing DOM snapshot: %s", exc)

        return snapshot

    @async_hybrid
    async def wait_for_spa_transition(
        self,
        page: Optional[Any] = None,
        previous_snapshot: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = SPA_TRANSITION_TIMEOUT_SECONDS,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Bounded polling waiting for client-side SPA DOM state mutation after a progression action."""
        target_page = page or self._page
        if not target_page or not previous_snapshot:
            await asyncio.sleep(0.5)
            new_snap = await _maybe_await(self.capture_dom_snapshot(target_page))
            return True, "default_delay", new_snap

        start_time = time.time()
        poll_interval = SPA_TRANSITION_POLL_INTERVAL_SECONDS
        prev_url = previous_snapshot.get("url", "")
        prev_heading = previous_snapshot.get("primary_heading", "")
        prev_warning = previous_snapshot.get("is_warning", False)
        prev_survey = previous_snapshot.get("is_survey", False)
        prev_actions = set(previous_snapshot.get("visible_actions", []))

        # Initial small delay for client React render loop
        await asyncio.sleep(0.25)

        while (time.time() - start_time) < timeout_seconds:
            curr_snap = await _maybe_await(self.capture_dom_snapshot(target_page))
            curr_url = curr_snap.get("url", "")
            curr_heading = curr_snap.get("primary_heading", "")
            curr_warning = curr_snap.get("is_warning", False)
            curr_survey = curr_snap.get("is_survey", False)
            curr_has_submit = curr_snap.get("has_submit", False)
            curr_actions = set(curr_snap.get("visible_actions", []))

            # 1. URL changed
            if curr_url and prev_url and curr_url.split("?")[0] != prev_url.split("?")[0]:
                logger.info("SPA transition detected: URL changed from %s to %s", prev_url, curr_url)
                return True, "url_changed", curr_snap

            # 2. Requirements warning disappeared
            if prev_warning and not curr_warning:
                logger.info("SPA transition detected: Requirements warning disappeared")
                return True, "warning_disappeared", curr_snap

            # 3. New survey appeared
            if not prev_survey and curr_survey:
                logger.info("SPA transition detected: Optional survey step appeared")
                return True, "survey_appeared", curr_snap

            # 4. Heading changed
            if curr_heading and prev_heading and curr_heading != prev_heading:
                logger.info("SPA transition detected: Heading changed from '%s' to '%s'", prev_heading, curr_heading)
                return True, "heading_changed", curr_snap

            # 5. Submit button appeared
            if curr_has_submit:
                logger.info("SPA transition detected: Submit button appeared")
                return True, "submit_appeared", curr_snap

            # 6. Visible actions changed meaningfully
            if curr_actions != prev_actions and bool(curr_actions):
                logger.info("SPA transition detected: Visible actions changed (%s -> %s)", prev_actions, curr_actions)
                return True, "actions_changed", curr_snap

            await asyncio.sleep(poll_interval)

        # Timeout reached: return latest snapshot
        final_snap = await _maybe_await(self.capture_dom_snapshot(target_page))
        return False, "transition_timeout", final_snap

    # ==========================================================================
    # Question Extraction & Truthful Answering
    # ==========================================================================

    @async_hybrid
    async def extract_questions(self, page: Optional[Any] = None) -> List[ApplicationQuestion]:
        """Extract all visible employer input questions using semantic DOM relationships.

        Splits individual fields independently (e.g. Current CTC, Expected CTC, Notice Period).
        """
        target_page = page or self._page
        if not target_page:
            return []

        # If on Review page, do not treat summary items as editable questions
        if await _maybe_await(self._is_review_step(target_page)):
            return []

        questions: List[ApplicationQuestion] = []
        seen_keys = set()

        try:
            # 1. Fieldsets / Radio Groups
            fieldsets = target_page.locator("fieldset, [role='radiogroup']")
            fs_cnt = await _maybe_await(fieldsets.count()) if hasattr(fieldsets, "count") else 0
            if isinstance(fs_cnt, int):
                for i in range(fs_cnt):
                    fs = fieldsets.nth(i) if hasattr(fieldsets, "nth") else fieldsets
                    try:
                        if hasattr(fs, "is_hidden") and await _maybe_await(fs.is_hidden()):
                            continue
                    except Exception:
                        pass

                    legend = fs.locator("legend, [role='heading']").first if hasattr(fs.locator("legend, [role='heading']"), "first") else fs.locator("legend, [role='heading']")
                    leg_cnt = await _maybe_await(legend.count()) if hasattr(legend, "count") else 0
                    legend_text = (await _maybe_await(legend.inner_text())).strip() if (isinstance(leg_cnt, int) and leg_cnt > 0 and hasattr(legend, "inner_text")) else ""
                    if not legend_text and hasattr(fs, "get_attribute"):
                        legend_text = (await _maybe_await(fs.get_attribute("aria-label"))) or ""

                    if not legend_text:
                        continue

                    # Check if it contains radios or checkboxes
                    radios = fs.locator("input[type='radio']")
                    r_cnt = await _maybe_await(radios.count()) if hasattr(radios, "count") else 0
                    if isinstance(r_cnt, int) and r_cnt > 0:
                        options = []
                        current_val = None
                        for r_idx in range(r_cnt):
                            r_elem = radios.nth(r_idx) if hasattr(radios, "nth") else radios
                            r_id = await _maybe_await(r_elem.get_attribute("id")) if hasattr(r_elem, "get_attribute") else ""
                            opt_label = ""
                            if r_id:
                                lbl = target_page.locator(f"label[for='{r_id}']")
                                l_cnt = await _maybe_await(lbl.count()) if hasattr(lbl, "count") else 0
                                if isinstance(l_cnt, int) and l_cnt > 0:
                                    first_lbl = lbl.first if hasattr(lbl, "first") else lbl
                                    opt_label = (await _maybe_await(first_lbl.inner_text())).strip() if hasattr(first_lbl, "inner_text") else ""
                            if not opt_label and hasattr(r_elem, "get_attribute"):
                                opt_label = (await _maybe_await(r_elem.get_attribute("value"))) or ""
                            if opt_label:
                                options.append(opt_label)
                            if hasattr(r_elem, "is_checked") and await _maybe_await(r_elem.is_checked()):
                                current_val = opt_label

                        norm_key = normalize_question_key(legend_text)
                        if norm_key not in seen_keys:
                            seen_keys.add(norm_key)
                            req_attr_cnt = await _maybe_await(fs.locator("[aria-required='true']").count()) if hasattr(fs, "locator") else 0
                            is_req = "required" in legend_text.lower() or req_attr_cnt > 0 or "*" in legend_text
                            fs_name = (await _maybe_await(fs.get_attribute("name"))) if hasattr(fs, "get_attribute") else norm_key
                            questions.append(
                                ApplicationQuestion(
                                    text=legend_text,
                                    normalized_key=norm_key,
                                    input_type=QuestionInputType.RADIO,
                                    required=is_req,
                                    current_value=current_val,
                                    options=options,
                                    field_name=fs_name or norm_key,
                                )
                            )
                        continue

                    # Check for checkboxes inside fieldset
                    checkboxes = fs.locator("input[type='checkbox']")
                    c_cnt = await _maybe_await(checkboxes.count()) if hasattr(checkboxes, "count") else 0
                    if isinstance(c_cnt, int) and c_cnt > 1:
                        options = []
                        for c_idx in range(c_cnt):
                            c_elem = checkboxes.nth(c_idx) if hasattr(checkboxes, "nth") else checkboxes
                            c_id = await _maybe_await(c_elem.get_attribute("id")) if hasattr(c_elem, "get_attribute") else ""
                            opt_label = ""
                            if c_id:
                                lbl = target_page.locator(f"label[for='{c_id}']")
                                l_cnt = await _maybe_await(lbl.count()) if hasattr(lbl, "count") else 0
                                if isinstance(l_cnt, int) and l_cnt > 0:
                                    first_lbl = lbl.first if hasattr(lbl, "first") else lbl
                                    opt_label = (await _maybe_await(first_lbl.inner_text())).strip() if hasattr(first_lbl, "inner_text") else ""
                            if not opt_label and hasattr(c_elem, "get_attribute"):
                                opt_label = (await _maybe_await(c_elem.get_attribute("value"))) or ""
                            if opt_label:
                                options.append(opt_label)

                        norm_key = normalize_question_key(legend_text)
                        if norm_key not in seen_keys:
                            seen_keys.add(norm_key)
                            req_attr_cnt = await _maybe_await(fs.locator("[aria-required='true']").count()) if hasattr(fs, "locator") else 0
                            is_req = "required" in legend_text.lower() or req_attr_cnt > 0 or "*" in legend_text
                            fs_name = (await _maybe_await(fs.get_attribute("name"))) if hasattr(fs, "get_attribute") else norm_key
                            questions.append(
                                ApplicationQuestion(
                                    text=legend_text,
                                    normalized_key=norm_key,
                                    input_type=QuestionInputType.CHECKBOX_MULTI,
                                    required=is_req,
                                    options=options,
                                    field_name=fs_name or norm_key,
                                )
                            )
                        continue

            # 2. Standard Individual Inputs (text, number, email, textarea, select, standalone checkbox)
            inputs = target_page.locator(
                "input[type='text'], input[type='number'], input[type='email'], input[type='tel'], "
                "input:not([type]), textarea, select, [role='combobox'], input[type='checkbox']"
            )
            inp_cnt = await _maybe_await(inputs.count()) if hasattr(inputs, "count") else 0
            if isinstance(inp_cnt, int):
                for i in range(inp_cnt):
                    inp = inputs.nth(i) if hasattr(inputs, "nth") else inputs
                    try:
                        if hasattr(inp, "is_enabled") and not await _maybe_await(inp.is_enabled()):
                            continue
                    except Exception:
                        pass

                    inp_type_attr = ((await _maybe_await(inp.get_attribute("type"))) or "text").lower() if hasattr(inp, "get_attribute") else "text"
                    inp_name = (await _maybe_await(inp.get_attribute("name"))) or "" if hasattr(inp, "get_attribute") else ""
                    if inp_name in ("q", "search", "location"):
                        continue

                    label_text = await _maybe_await(self._find_label_for_input(target_page, inp))
                    if not label_text:
                        continue

                    norm_key = normalize_question_key(label_text)
                    if norm_key in seen_keys or any(sf in norm_key for sf in SURVEY_FIELD_IDENTIFIERS):
                        continue
                    seen_keys.add(norm_key)

                    tag_name = (await _maybe_await(inp.evaluate("el => el.tagName.toLowerCase()"))) if hasattr(inp, "evaluate") else "input"
                    req_attr = (await _maybe_await(inp.get_attribute("required"))) if hasattr(inp, "get_attribute") else None
                    aria_req = (await _maybe_await(inp.get_attribute("aria-required"))) if hasattr(inp, "get_attribute") else None
                    is_req = (
                        req_attr is not None
                        or aria_req == "true"
                        or "required" in label_text.lower()
                        or "*" in label_text
                    )
                    current_val = (await _maybe_await(inp.input_value())) if (tag_name in ("input", "textarea", "select") and hasattr(inp, "input_value")) else None

                    q_type = QuestionInputType.TEXT
                    options: List[str] = []

                    if tag_name == "textarea":
                        q_type = QuestionInputType.TEXT
                    elif tag_name == "select":
                        q_type = QuestionInputType.DROPDOWN
                        opts = inp.locator("option")
                        o_cnt = await _maybe_await(opts.count()) if hasattr(opts, "count") else 0
                        if isinstance(o_cnt, int):
                            for o_idx in range(o_cnt):
                                opt_elem = opts.nth(o_idx) if hasattr(opts, "nth") else opts
                                opt_raw = (await _maybe_await(opt_elem.inner_text())).strip() if hasattr(opt_elem, "inner_text") else ""
                                if opt_raw and opt_raw.lower() not in ("select", "choose", "please select", "select an option"):
                                    options.append(opt_raw)
                    elif inp_type_attr == "number" or "years of experience" in label_text.lower() or "(lpa)" in label_text.lower() or "period" in label_text.lower():
                        q_type = QuestionInputType.NUMBER
                    elif inp_type_attr == "checkbox":
                        q_type = QuestionInputType.CHECKBOX
                    elif hasattr(inp, "get_attribute") and (await _maybe_await(inp.get_attribute("role"))) == "combobox":
                        q_type = QuestionInputType.DROPDOWN

                    questions.append(
                        ApplicationQuestion(
                            text=label_text,
                            normalized_key=norm_key,
                            input_type=q_type,
                            required=is_req,
                            current_value=current_val,
                            options=options,
                            field_name=inp_name or norm_key,
                        )
                    )

        except Exception as exc:
            logger.warning("Error extracting questions from DOM: %s", exc)

        return questions

    @async_hybrid
    async def _find_label_for_input(self, page: Any, inp: Any) -> str:
        """Find the visible question label associated with an input element."""
        try:
            # Priority A: aria-label
            if hasattr(inp, "get_attribute"):
                aria_label = await _maybe_await(inp.get_attribute("aria-label"))
                if aria_label:
                    return aria_label.strip()

            # Priority B: aria-labelledby
            if hasattr(inp, "get_attribute"):
                aria_labelledby = await _maybe_await(inp.get_attribute("aria-labelledby"))
                if aria_labelledby:
                    lbl_elem = page.locator(f"#{aria_labelledby}")
                    cnt = await _maybe_await(lbl_elem.count()) if hasattr(lbl_elem, "count") else 0
                    if isinstance(cnt, int) and cnt > 0:
                        first_lbl = lbl_elem.first if hasattr(lbl_elem, "first") else lbl_elem
                        return (await _maybe_await(first_lbl.inner_text())).strip()

            # Priority C: label[for=id]
            if hasattr(inp, "get_attribute"):
                inp_id = await _maybe_await(inp.get_attribute("id"))
                if inp_id:
                    lbl = page.locator(f"label[for='{inp_id}']")
                    cnt = await _maybe_await(lbl.count()) if hasattr(lbl, "count") else 0
                    if isinstance(cnt, int) and cnt > 0:
                        first_lbl = lbl.first if hasattr(lbl, "first") else lbl
                        return (await _maybe_await(first_lbl.inner_text())).strip()

            # Priority D: Parent label
            if hasattr(inp, "locator"):
                parent_label = inp.locator("xpath=ancestor::label")
                cnt = await _maybe_await(parent_label.count()) if hasattr(parent_label, "count") else 0
                if isinstance(cnt, int) and cnt > 0:
                    first_lbl = parent_label.first if hasattr(parent_label, "first") else parent_label
                    return (await _maybe_await(first_lbl.inner_text())).strip()

            # Priority E: Question container heading/label
            if hasattr(inp, "locator"):
                container = inp.locator(
                    "xpath=ancestor::div[contains(@class, 'Question') or contains(@data-testid, 'question') or contains(@class, 'field')]//label "
                    "| xpath=ancestor::div[contains(@class, 'Question') or contains(@data-testid, 'question') or contains(@class, 'field')]//span"
                )
                cnt = await _maybe_await(container.count()) if hasattr(container, "count") else 0
                if isinstance(cnt, int) and cnt > 0:
                    first_c = container.first if hasattr(container, "first") else container
                    txt = (await _maybe_await(first_c.inner_text())).strip()
                    if txt:
                        return txt

            # Priority F: Preceding label or heading sibling
            if hasattr(inp, "locator"):
                prev_label = inp.locator("xpath=preceding-sibling::label | preceding-sibling::span | preceding-sibling::p")
                cnt = await _maybe_await(prev_label.count()) if hasattr(prev_label, "count") else 0
                if isinstance(cnt, int) and cnt > 0:
                    last_p = prev_label.last if hasattr(prev_label, "last") else prev_label
                    txt = (await _maybe_await(last_p.inner_text())).strip()
                    if txt:
                        return txt

            if hasattr(inp, "get_attribute"):
                ph = await _maybe_await(inp.get_attribute("placeholder"))
                if ph:
                    return ph.strip()

        except Exception:
            pass

        return ""

    @async_hybrid
    async def fill_question(
        self,
        page: Any,
        question: ApplicationQuestion,
        resolved_value: Any,
    ) -> bool:
        """Fill a single question control with the resolved value, scrolling into view first."""
        if resolved_value is None:
            return False

        try:
            q_text = question.text
            norm_key = question.normalized_key

            # 1. Text / Number / Email / Textarea
            if question.input_type in (QuestionInputType.TEXT, QuestionInputType.NUMBER):
                val_str = str(resolved_value)
                loc = page.get_by_label(re.compile(re.escape(q_text[:30]), re.IGNORECASE))
                l_cnt = await _maybe_await(loc.count()) if hasattr(loc, "count") else 0
                if l_cnt == 0 and question.field_name:
                    loc = page.locator(f"input[name='{question.field_name}'], textarea[name='{question.field_name}']")
                    l_cnt = await _maybe_await(loc.count()) if hasattr(loc, "count") else 0

                if isinstance(l_cnt, int) and l_cnt > 0:
                    target_elem = loc.first if hasattr(loc, "first") else loc
                    try:
                        if hasattr(target_elem, "scroll_into_view_if_needed"):
                            await _maybe_await(target_elem.scroll_into_view_if_needed())
                        if hasattr(target_elem, "focus"):
                            await _maybe_await(target_elem.focus())
                    except Exception:
                        pass
                    if hasattr(target_elem, "fill"):
                        await _maybe_await(target_elem.fill(val_str))
                    # Verify filled value
                    try:
                        if hasattr(target_elem, "input_value"):
                            actual = await _maybe_await(target_elem.input_value())
                            if actual != val_str:
                                await _maybe_await(target_elem.fill(val_str))
                    except Exception:
                        pass
                    return True

            # 2. Dropdown / Select / Combobox
            elif question.input_type == QuestionInputType.DROPDOWN:
                target_val = str(resolved_value).strip()
                select_loc = page.get_by_label(re.compile(re.escape(q_text[:30]), re.IGNORECASE))
                s_cnt = await _maybe_await(select_loc.count()) if hasattr(select_loc, "count") else 0
                if isinstance(s_cnt, int) and s_cnt > 0:
                    target_elem = select_loc.first if hasattr(select_loc, "first") else select_loc
                    try:
                        if hasattr(target_elem, "scroll_into_view_if_needed"):
                            await _maybe_await(target_elem.scroll_into_view_if_needed())
                    except Exception:
                        pass
                    tag = (await _maybe_await(target_elem.evaluate("el => el.tagName.toLowerCase()"))) if hasattr(target_elem, "evaluate") else "select"
                    if tag == "select":
                        if hasattr(target_elem, "select_option"):
                            await _maybe_await(target_elem.select_option(label=target_val))
                        return True
                    else:
                        # Combobox click
                        if hasattr(target_elem, "click"):
                            await _maybe_await(target_elem.click())
                        opt = page.get_by_role("option", name=re.compile(re.escape(target_val), re.IGNORECASE))
                        o_cnt = await _maybe_await(opt.count()) if hasattr(opt, "count") else 0
                        if isinstance(o_cnt, int) and o_cnt > 0:
                            first_opt = opt.first if hasattr(opt, "first") else opt
                            if hasattr(first_opt, "click"):
                                await _maybe_await(first_opt.click())
                            return True

            # 3. Radio
            elif question.input_type == QuestionInputType.RADIO:
                target_opt = str(resolved_value).strip()
                radio = page.get_by_role("radio", name=re.compile(rf"^{re.escape(target_opt)}$", re.IGNORECASE))
                r_cnt = await _maybe_await(radio.count()) if hasattr(radio, "count") else 0
                if r_cnt == 0:
                    radio = page.get_by_label(re.compile(rf"^{re.escape(target_opt)}$", re.IGNORECASE))
                    r_cnt = await _maybe_await(radio.count()) if hasattr(radio, "count") else 0

                if isinstance(r_cnt, int) and r_cnt > 0:
                    target_elem = radio.first if hasattr(radio, "first") else radio
                    try:
                        if hasattr(target_elem, "scroll_into_view_if_needed"):
                            await _maybe_await(target_elem.scroll_into_view_if_needed())
                    except Exception:
                        pass
                    if hasattr(target_elem, "check"):
                        await _maybe_await(target_elem.check())
                    elif hasattr(target_elem, "click"):
                        await _maybe_await(target_elem.click())
                    # Read-back verification with safe single retry
                    try:
                        if hasattr(target_elem, "is_checked"):
                            is_chk = await _maybe_await(target_elem.is_checked())
                            if not is_chk:
                                logger.warning("Read-back verification mismatch for radio '%s'. Performing safe single retry.", target_opt)
                                if hasattr(target_elem, "click"):
                                    await _maybe_await(target_elem.click())
                                is_chk = await _maybe_await(target_elem.is_checked())
                            if not is_chk:
                                logger.error("Read-back verification failed for radio '%s' after retry.", target_opt)
                                return False
                    except Exception as exc:
                        logger.warning("Error during radio read-back verification: %s", exc)
                    return True

            # 4. Checkbox
            elif question.input_type == QuestionInputType.CHECKBOX:
                if bool(resolved_value):
                    cb = page.get_by_label(re.compile(re.escape(q_text[:30]), re.IGNORECASE))
                    c_cnt = await _maybe_await(cb.count()) if hasattr(cb, "count") else 0
                    if isinstance(c_cnt, int) and c_cnt > 0:
                        target_elem = cb.first if hasattr(cb, "first") else cb
                        try:
                            if hasattr(target_elem, "scroll_into_view_if_needed"):
                                await _maybe_await(target_elem.scroll_into_view_if_needed())
                        except Exception:
                            pass
                        if hasattr(target_elem, "check"):
                            await _maybe_await(target_elem.check())
                        elif hasattr(target_elem, "click"):
                            await _maybe_await(target_elem.click())
                        # Read-back verification with safe single retry
                        try:
                            if hasattr(target_elem, "is_checked"):
                                is_chk = await _maybe_await(target_elem.is_checked())
                                if not is_chk:
                                    logger.warning("Read-back verification mismatch for checkbox '%s'. Performing safe single retry.", q_text)
                                    if hasattr(target_elem, "click"):
                                        await _maybe_await(target_elem.click())
                                    is_chk = await _maybe_await(target_elem.is_checked())
                                if not is_chk:
                                    logger.error("Read-back verification failed for checkbox '%s' after retry.", q_text)
                                    return False
                        except Exception as exc:
                            logger.warning("Error during checkbox read-back verification: %s", exc)
                        return True

            # 5. Multi-Checkbox
            elif question.input_type == QuestionInputType.CHECKBOX_MULTI:
                selected_options = resolved_value if isinstance(resolved_value, list) else [str(resolved_value)]
                for opt in selected_options:
                    cb = page.get_by_role("checkbox", name=re.compile(re.escape(str(opt)), re.IGNORECASE))
                    c_cnt = await _maybe_await(cb.count()) if hasattr(cb, "count") else 0
                    if c_cnt == 0:
                        cb = page.get_by_label(re.compile(re.escape(str(opt)), re.IGNORECASE))
                        c_cnt = await _maybe_await(cb.count()) if hasattr(cb, "count") else 0
                    if isinstance(c_cnt, int) and c_cnt > 0:
                        target_elem = cb.first if hasattr(cb, "first") else cb
                        try:
                            if hasattr(target_elem, "scroll_into_view_if_needed"):
                                await _maybe_await(target_elem.scroll_into_view_if_needed())
                        except Exception:
                            pass
                        if hasattr(target_elem, "check"):
                            await _maybe_await(target_elem.check())
                        elif hasattr(target_elem, "click"):
                            await _maybe_await(target_elem.click())
                return True

        except Exception as exc:
            logger.warning("Failed to fill question '%s' with value '%s': %s", question.text, resolved_value, exc)

        return False

    @async_hybrid
    async def read_question_value(self, page: Optional[Any], question: ApplicationQuestion) -> Optional[Any]:
        """Read the current user-entered or selected DOM value for a question."""
        target_page = page or self._page
        if not target_page or not question:
            return None

        try:
            q_text = question.text
            norm_key = question.normalized_key

            # 1. Radio
            if question.input_type == QuestionInputType.RADIO:
                fieldsets = target_page.locator("fieldset, [role='radiogroup']")
                fs_cnt = await _maybe_await(fieldsets.count()) if hasattr(fieldsets, "count") else 0
                if isinstance(fs_cnt, int) and fs_cnt > 0:
                    for idx in range(fs_cnt):
                        fs = fieldsets.nth(idx) if hasattr(fieldsets, "nth") else fieldsets
                        legend = fs.locator("legend, [role='heading']").first if hasattr(fs.locator("legend, [role='heading']"), "first") else fs.locator("legend, [role='heading']")
                        leg_cnt = await _maybe_await(legend.count()) if hasattr(legend, "count") else 0
                        leg_text = (await _maybe_await(legend.inner_text())).strip() if (isinstance(leg_cnt, int) and leg_cnt > 0 and hasattr(legend, "inner_text")) else ""
                        if not leg_text and hasattr(fs, "get_attribute"):
                            leg_text = (await _maybe_await(fs.get_attribute("aria-label"))) or ""
                        if _normalize_text(leg_text) == _normalize_text(q_text) or normalize_question_key(leg_text) == norm_key:
                            radios = fs.locator("input[type='radio']")
                            r_cnt = await _maybe_await(radios.count()) if hasattr(radios, "count") else 0
                            if isinstance(r_cnt, int) and r_cnt > 0:
                                for r_idx in range(r_cnt):
                                    r = radios.nth(r_idx) if hasattr(radios, "nth") else radios
                                    is_chk = await _maybe_await(r.is_checked()) if hasattr(r, "is_checked") else False
                                    if is_chk is True:
                                        r_id = (await _maybe_await(r.get_attribute("id"))) if hasattr(r, "get_attribute") else ""
                                        if r_id:
                                            lbl = target_page.locator(f"label[for='{r_id}']")
                                            lbl_cnt = await _maybe_await(lbl.count()) if hasattr(lbl, "count") else 0
                                            if isinstance(lbl_cnt, int) and lbl_cnt > 0:
                                                first_lbl = lbl.first if hasattr(lbl, "first") else lbl
                                                return (await _maybe_await(first_lbl.inner_text())).strip()
                                        return (await _maybe_await(r.get_attribute("value"))) or "" if hasattr(r, "get_attribute") else ""

                # Fallback: check by field name or role
                radios = target_page.locator(f"input[type='radio'][name='{question.field_name}']")
                r_cnt = await _maybe_await(radios.count()) if hasattr(radios, "count") else 0
                if isinstance(r_cnt, int) and r_cnt > 0:
                    for r_idx in range(r_cnt):
                        r = radios.nth(r_idx) if hasattr(radios, "nth") else radios
                        is_chk = await _maybe_await(r.is_checked()) if hasattr(r, "is_checked") else False
                        if is_chk is True:
                            r_id = (await _maybe_await(r.get_attribute("id"))) if hasattr(r, "get_attribute") else ""
                            if r_id:
                                lbl = target_page.locator(f"label[for='{r_id}']")
                                lbl_cnt = await _maybe_await(lbl.count()) if hasattr(lbl, "count") else 0
                                if isinstance(lbl_cnt, int) and lbl_cnt > 0:
                                    first_lbl = lbl.first if hasattr(lbl, "first") else lbl
                                    return (await _maybe_await(first_lbl.inner_text())).strip()
                            return (await _maybe_await(r.get_attribute("value"))) or "" if hasattr(r, "get_attribute") else ""

            # 2. Checkbox / Multi-checkbox
            elif question.input_type in (QuestionInputType.CHECKBOX, QuestionInputType.CHECKBOX_MULTI):
                checked_vals = []
                fieldsets = target_page.locator("fieldset, [role='group']")
                fs_cnt = await _maybe_await(fieldsets.count()) if hasattr(fieldsets, "count") else 0
                if isinstance(fs_cnt, int) and fs_cnt > 0:
                    for idx in range(fs_cnt):
                        fs = fieldsets.nth(idx) if hasattr(fieldsets, "nth") else fieldsets
                        legend = fs.locator("legend, [role='heading']").first if hasattr(fs.locator("legend, [role='heading']"), "first") else fs.locator("legend, [role='heading']")
                        leg_cnt = await _maybe_await(legend.count()) if hasattr(legend, "count") else 0
                        leg_text = (await _maybe_await(legend.inner_text())).strip() if (isinstance(leg_cnt, int) and leg_cnt > 0 and hasattr(legend, "inner_text")) else ""
                        if not leg_text and hasattr(fs, "get_attribute"):
                            leg_text = (await _maybe_await(fs.get_attribute("aria-label"))) or ""
                        if _normalize_text(leg_text) == _normalize_text(q_text) or normalize_question_key(leg_text) == norm_key:
                            cbs = fs.locator("input[type='checkbox']")
                            c_cnt = await _maybe_await(cbs.count()) if hasattr(cbs, "count") else 0
                            if isinstance(c_cnt, int) and c_cnt > 0:
                                for c_idx in range(c_cnt):
                                    cb = cbs.nth(c_idx) if hasattr(cbs, "nth") else cbs
                                    is_chk = await _maybe_await(cb.is_checked()) if hasattr(cb, "is_checked") else False
                                    if is_chk is True:
                                        c_id = (await _maybe_await(cb.get_attribute("id"))) if hasattr(cb, "get_attribute") else ""
                                        opt_label = ""
                                        if c_id:
                                            lbl = target_page.locator(f"label[for='{c_id}']")
                                            lbl_cnt = await _maybe_await(lbl.count()) if hasattr(lbl, "count") else 0
                                            if isinstance(lbl_cnt, int) and lbl_cnt > 0:
                                                first_lbl = lbl.first if hasattr(lbl, "first") else lbl
                                                opt_label = (await _maybe_await(first_lbl.inner_text())).strip()
                                        if not opt_label and hasattr(cb, "get_attribute"):
                                            opt_label = (await _maybe_await(cb.get_attribute("value"))) or ""
                                        if opt_label:
                                            checked_vals.append(opt_label)
                                if checked_vals:
                                    return checked_vals if question.input_type == QuestionInputType.CHECKBOX_MULTI else checked_vals[0]

                # Standalone checkbox
                loc = target_page.get_by_label(re.compile(re.escape(q_text[:30]), re.IGNORECASE))
                l_cnt = await _maybe_await(loc.count()) if hasattr(loc, "count") else 0
                if isinstance(l_cnt, int) and l_cnt > 0:
                    first_loc = loc.first if hasattr(loc, "first") else loc
                    return await _maybe_await(first_loc.is_checked()) if hasattr(first_loc, "is_checked") else False

            # 3. Dropdown / Select / Combobox
            elif question.input_type == QuestionInputType.DROPDOWN:
                loc = target_page.get_by_label(re.compile(re.escape(q_text[:30]), re.IGNORECASE))
                l_cnt = await _maybe_await(loc.count()) if hasattr(loc, "count") else 0
                if isinstance(l_cnt, int) and l_cnt > 0:
                    elem = loc.first if hasattr(loc, "first") else loc
                    tag = (await _maybe_await(elem.evaluate("el => el.tagName.toLowerCase()"))) if hasattr(elem, "evaluate") else "select"
                    if tag == "select":
                        val = (await _maybe_await(elem.input_value())) if hasattr(elem, "input_value") else ""
                        if val:
                            opt = elem.locator(f"option[value='{val}']")
                            opt_cnt = await _maybe_await(opt.count()) if hasattr(opt, "count") else 0
                            if isinstance(opt_cnt, int) and opt_cnt > 0:
                                first_opt = opt.first if hasattr(opt, "first") else opt
                                return (await _maybe_await(first_opt.inner_text())).strip()
                            return val
                    else:
                        txt = (await _maybe_await(elem.inner_text())).strip() if hasattr(elem, "inner_text") else ""
                        if txt and txt.lower() not in ("select", "choose", "please select", "select an option"):
                            return txt
                        return (await _maybe_await(elem.input_value())) if hasattr(elem, "input_value") else None

            # 4. Text / Number / Email / Textarea
            else:
                loc = target_page.get_by_label(re.compile(re.escape(q_text[:30]), re.IGNORECASE))
                l_cnt = await _maybe_await(loc.count()) if hasattr(loc, "count") else 0
                if (not isinstance(l_cnt, int) or l_cnt == 0) and question.field_name:
                    loc = target_page.locator(f"input[name='{question.field_name}'], textarea[name='{question.field_name}']")
                    l_cnt = await _maybe_await(loc.count()) if hasattr(loc, "count") else 0

                if isinstance(l_cnt, int) and l_cnt > 0:
                    elem = loc.first if hasattr(loc, "first") else loc
                    val = (await _maybe_await(elem.input_value())).strip() if hasattr(elem, "input_value") else ""
                    placeholder = (await _maybe_await(elem.get_attribute("placeholder"))) or "" if hasattr(elem, "get_attribute") else ""
                    if val and val != placeholder:
                        if question.input_type == QuestionInputType.NUMBER:
                            try:
                                return int(val) if val.isdigit() else float(val)
                            except ValueError:
                                return val
                        return val

        except Exception as exc:
            logger.warning("Error reading question value for '%s': %s", question.text, exc)

        return None

    @async_hybrid
    async def read_all_questions_values(self, page: Optional[Any], questions: List[ApplicationQuestion]) -> Dict[str, Any]:
        """Read all current user-entered DOM values for a list of questions."""
        target_page = page or self._page
        results: Dict[str, Any] = {}
        for q in questions:
            val = await _maybe_await(self.read_question_value(target_page, q))
            if val is not None and str(val).strip():
                results[q.normalized_key] = val
        return results

    # ==========================================================================
    # Resume & Progression Actions (Scrolling into view)
    # ==========================================================================

    @async_hybrid
    async def handle_resume_step(self, page: Optional[Any] = None) -> Tuple[bool, str, Optional[str]]:
        """Verify resume selection on the resume step without uploading or replacing."""
        target_page = page or self._page
        if not target_page:
            return False, "unverified", "No active page"

        try:
            # Check if a resume radio or card is already selected
            checked_radio = target_page.locator("input[type='radio'][checked], input[type='radio']:checked, [aria-checked='true']")
            cnt = await _maybe_await(checked_radio.count()) if hasattr(checked_radio, "count") else 0
            if isinstance(cnt, int) and cnt > 0:
                logger.info("Default resume already selected on Resume step.")
                return True, "default_selected", None

            # Select first available resume option
            resume_options = target_page.locator("input[type='radio'][name*='resume' i], [data-testid*='resume' i]")
            r_cnt = await _maybe_await(resume_options.count()) if hasattr(resume_options, "count") else 0
            if isinstance(r_cnt, int) and r_cnt > 0:
                first_r = resume_options.first if hasattr(resume_options, "first") else resume_options
                if hasattr(first_r, "scroll_into_view_if_needed"):
                    await _maybe_await(first_r.scroll_into_view_if_needed())
                if hasattr(first_r, "click"):
                    await _maybe_await(first_r.click())
                logger.info("Selected first available resume option.")
                return True, "selected", None

            return True, "not_needed", None
        except Exception as exc:
            logger.warning("Error during resume step handling: %s", exc)
            return False, "unverified", str(exc)

    @async_hybrid
    async def scroll_element_into_view_deep(self, target_elem: Any, page: Optional[Any] = None) -> Dict[str, Any]:
        """Perform multi-tier robust scrolling to ensure target_elem is visible in usable viewport."""
        diag: Dict[str, Any] = {
            "scroll_into_view_attempted": False,
            "scrollable_ancestor_detected": False,
            "visible_before_scroll": False,
            "visible_after_scroll": False,
            "element_bounds": None,
        }
        if target_elem is None:
            return diag

        try:
            if hasattr(target_elem, "is_visible") and callable(target_elem.is_visible):
                diag["visible_before_scroll"] = bool(await _maybe_await(target_elem.is_visible()))
        except Exception:
            pass

        # Tier 1: Standard Playwright scroll_into_view_if_needed
        diag["scroll_into_view_attempted"] = True
        try:
            if hasattr(target_elem, "scroll_into_view_if_needed") and callable(target_elem.scroll_into_view_if_needed):
                await _maybe_await(target_elem.scroll_into_view_if_needed(timeout=2000))
                await asyncio.sleep(0.15)
        except Exception as exc:
            logger.debug("scroll_into_view_if_needed note: %s", exc)

        # Check if visible after Tier 1
        is_vis = False
        try:
            if hasattr(target_elem, "is_visible") and callable(target_elem.is_visible):
                is_vis = bool(await _maybe_await(target_elem.is_visible()))
        except Exception:
            pass

        # Tier 2: DOM element.scrollIntoView evaluate
        if not is_vis:
            try:
                if hasattr(target_elem, "evaluate") and callable(target_elem.evaluate):
                    await _maybe_await(
                        target_elem.evaluate(
                            """el => el.scrollIntoView({
                                behavior: 'instant',
                                block: 'center',
                                inline: 'nearest'
                            })"""
                        )
                    )
                    await asyncio.sleep(0.15)
                    is_vis = bool(await _maybe_await(target_elem.is_visible()))
            except Exception as exc:
                logger.debug("DOM scrollIntoView note: %s", exc)

        # Tier 3: Walk up DOM to find and scroll nearest scrollable container
        try:
            if hasattr(target_elem, "evaluate") and callable(target_elem.evaluate):
                res_ancestor = await _maybe_await(
                    target_elem.evaluate(
                        """el => {
                            let curr = el.parentElement;
                            let foundScrollable = false;
                            while (curr) {
                                const style = window.getComputedStyle(curr);
                                const overflowY = style.overflowY;
                                const isScrollable = (overflowY === 'auto' || overflowY === 'scroll') && curr.scrollHeight > curr.clientHeight;
                                if (isScrollable) {
                                    foundScrollable = true;
                                    curr.scrollTop = el.offsetTop - (curr.clientHeight / 2);
                                    break;
                                }
                                curr = curr.parentElement;
                            }
                            return foundScrollable;
                        }"""
                    )
                )
                diag["scrollable_ancestor_detected"] = bool(res_ancestor)
                if res_ancestor:
                    await asyncio.sleep(0.15)
        except Exception as exc:
            logger.debug("Ancestor scroll note: %s", exc)

        # Final check: visibility and bounding box
        diag["visible_after_scroll"] = is_vis
        try:
            if hasattr(target_elem, "bounding_box") and callable(target_elem.bounding_box):
                diag["element_bounds"] = await _maybe_await(target_elem.bounding_box())
        except Exception:
            pass

        return diag

    @async_hybrid
    async def find_progression_action(self, page: Optional[Any] = None) -> Tuple[Optional[str], Optional[Any]]:
        """Locate enabled progression action (Priority: Review -> Continue -> Next).

        Supports searching top-level DOM and any subframes / iframes.
        Explicitly blacklists 'Return to job search' and 'Exit'.
        """
        target_page = page or self._page
        if not target_page:
            return None, None

        targets = [target_page]
        try:
            if hasattr(target_page, "frames"):
                for f in target_page.frames:
                    if f not in targets:
                        targets.append(f)
        except Exception:
            pass

        # Priority 1: Review
        for tgt in targets:
            for name in REVIEW_BUTTON_NAMES:
                try:
                    pattern = re.compile(rf"^\s*{re.escape(name)}\s*$", re.IGNORECASE)
                    btn = tgt.get_by_role("button", name=pattern)
                    cnt = await _maybe_await(btn.count()) if hasattr(btn, "count") else 0
                    if isinstance(cnt, int) and cnt > 0:
                        for idx in range(cnt):
                            cand = getattr(btn, "first", btn) if idx == 0 else (btn.nth(idx) if hasattr(btn, "nth") else btn)
                            is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                            if is_en:
                                try:
                                    raw_txt = (await _maybe_await(cand.inner_text())).strip().lower() if hasattr(cand, "inner_text") else ""
                                    if any(bl in raw_txt for bl in BLACKLISTED_PROGRESSION_BUTTON_NAMES):
                                        continue
                                except Exception:
                                    pass
                                return "review", cand
                except Exception:
                    continue

        # Priority 2: Continue
        for tgt in targets:
            for name in CONTINUE_BUTTON_NAMES:
                try:
                    pattern = re.compile(rf"^\s*{re.escape(name)}\s*$", re.IGNORECASE)
                    btn = tgt.get_by_role("button", name=pattern)
                    cnt = await _maybe_await(btn.count()) if hasattr(btn, "count") else 0
                    if isinstance(cnt, int) and cnt > 0:
                        for idx in range(cnt):
                            cand = getattr(btn, "first", btn) if idx == 0 else (btn.nth(idx) if hasattr(btn, "nth") else btn)
                            is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                            if is_en:
                                try:
                                    raw_txt = (await _maybe_await(cand.inner_text())).strip().lower() if hasattr(cand, "inner_text") else ""
                                    if any(bl in raw_txt for bl in BLACKLISTED_PROGRESSION_BUTTON_NAMES):
                                        continue
                                except Exception:
                                    pass
                                return "continue", cand

                    # Fallback text locator
                    custom_loc = tgt.locator(f"button:has-text('{name}'), input[type='submit'][value*='{name}' i]")
                    c_cnt = await _maybe_await(custom_loc.count()) if hasattr(custom_loc, "count") else 0
                    if isinstance(c_cnt, int) and c_cnt > 0:
                        for idx in range(c_cnt):
                            cand = getattr(custom_loc, "first", custom_loc) if idx == 0 else (custom_loc.nth(idx) if hasattr(custom_loc, "nth") else custom_loc)
                            is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                            if is_en:
                                try:
                                    raw_txt = (await _maybe_await(cand.inner_text())).strip().lower() if hasattr(cand, "inner_text") else ""
                                    if any(bl in raw_txt for bl in BLACKLISTED_PROGRESSION_BUTTON_NAMES):
                                        continue
                                except Exception:
                                    pass
                                return "continue", cand
                except Exception:
                    continue

        # Priority 3: Next
        for tgt in targets:
            for name in NEXT_BUTTON_NAMES:
                try:
                    pattern = re.compile(rf"^\s*{re.escape(name)}\s*$", re.IGNORECASE)
                    btn = tgt.get_by_role("button", name=pattern)
                    cnt = await _maybe_await(btn.count()) if hasattr(btn, "count") else 0
                    if isinstance(cnt, int) and cnt > 0:
                        for idx in range(cnt):
                            cand = getattr(btn, "first", btn) if idx == 0 else (btn.nth(idx) if hasattr(btn, "nth") else btn)
                            is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                            if is_en:
                                try:
                                    raw_txt = (await _maybe_await(cand.inner_text())).strip().lower() if hasattr(cand, "inner_text") else ""
                                    if any(bl in raw_txt for bl in BLACKLISTED_PROGRESSION_BUTTON_NAMES):
                                        continue
                                except Exception:
                                    pass
                                return "next", cand

                    custom_loc = tgt.locator(f"button:has-text('{name}'), input[type='submit'][value*='{name}' i]")
                    c_cnt = await _maybe_await(custom_loc.count()) if hasattr(custom_loc, "count") else 0
                    if isinstance(c_cnt, int) and c_cnt > 0:
                        for idx in range(c_cnt):
                            cand = getattr(custom_loc, "first", custom_loc) if idx == 0 else (custom_loc.nth(idx) if hasattr(custom_loc, "nth") else custom_loc)
                            is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                            if is_en:
                                try:
                                    raw_txt = (await _maybe_await(cand.inner_text())).strip().lower() if hasattr(cand, "inner_text") else ""
                                    if any(bl in raw_txt for bl in BLACKLISTED_PROGRESSION_BUTTON_NAMES):
                                        continue
                                except Exception:
                                    pass
                                return "next", cand
                            is_en = await _maybe_await(cand.is_enabled()) if hasattr(cand, "is_enabled") else True
                            if is_en:
                                try:
                                    raw_txt = (await _maybe_await(cand.inner_text())).strip().lower() if hasattr(cand, "inner_text") else ""
                                    if any(bl in raw_txt for bl in BLACKLISTED_PROGRESSION_BUTTON_NAMES):
                                        continue
                                except Exception:
                                    pass
                                return "next", cand
                except Exception:
                    continue

        return None, None

    @async_hybrid
    async def activate_progress_button(self, page: Optional[Any] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """Locate exact progression action, deep scroll into view, verify visible/enabled, click once, and wait for state change."""
        target_page = page or self._page
        action_name, action_btn = await _maybe_await(self.find_progression_action(target_page))

        # Track full progression diagnostics
        has_btn = action_btn is not None
        btn_count = 1 if has_btn else 0
        scroll_diag = await _maybe_await(self.scroll_element_into_view_deep(action_btn, target_page)) if has_btn else {}

        is_enabled = False
        if has_btn:
            try:
                is_enabled = bool(await _maybe_await(action_btn.is_enabled())) if hasattr(action_btn, "is_enabled") else True
            except Exception:
                is_enabled = True

        iframe_found = False
        try:
            if target_page and hasattr(target_page, "frames") and len(target_page.frames) > 1:
                iframe_found = True
        except Exception:
            pass

        self._last_progression_diagnostics = {
            "resume_continue_locator_found": has_btn,
            "resume_continue_count": btn_count,
            "resume_continue_visible_before_scroll": scroll_diag.get("visible_before_scroll", False),
            "resume_continue_visible_after_scroll": scroll_diag.get("visible_after_scroll", has_btn),
            "resume_continue_enabled": is_enabled,
            "resume_continue_bounds": scroll_diag.get("element_bounds"),
            "scroll_into_view_attempted": scroll_diag.get("scroll_into_view_attempted", False),
            "scrollable_ancestor_detected": scroll_diag.get("scrollable_ancestor_detected", False),
            "iframe_detected": iframe_found,
            "resume_continue_clicked": False,
        }

        if action_btn is None:
            return False, None, "No enabled progression button found (Continue/Next/Review)."

        if not is_enabled:
            return False, action_name, f"Progression button '{action_name}' is disabled."

        try:
            logger.info("Clicking verified progression button '%s' once...", action_name)
            try:
                if hasattr(action_btn, "wait_for") and callable(action_btn.wait_for):
                    await _maybe_await(action_btn.wait_for(state="visible", timeout=3000))
            except Exception:
                pass

            if hasattr(action_btn, "click"):
                await _maybe_await(action_btn.click())
            self._last_progression_diagnostics["resume_continue_clicked"] = True

            # Wait for observable state transition or network settling
            if target_page and hasattr(target_page, "wait_for_load_state"):
                try:
                    await _maybe_await(target_page.wait_for_load_state("domcontentloaded", timeout=3000))
                except Exception:
                    pass
                await asyncio.sleep(0.8)

            return True, action_name, None
        except Exception as exc:
            logger.warning("Failed to activate progression button '%s': %s", action_name, exc)
            return False, action_name, str(exc)

    @async_hybrid
    async def click_progression_action(self, page: Optional[Any] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """Backward-compatible alias for activate_progress_button."""
        return await _maybe_await(self.activate_progress_button(page))

    @async_hybrid
    async def click_final_submit(self, page: Optional[Any] = None) -> Tuple[bool, Optional[str]]:
        """Click the exact final submit button strictly once (authorized for apply endpoint only)."""
        submit_btn = await _maybe_await(self.find_exact_submit_button(page))
        if submit_btn is None:
            return False, "Exact submit button not found or not enabled."

        try:
            logger.info("Clicking verified final Submit application button strictly ONCE.")
            if hasattr(submit_btn, "scroll_into_view_if_needed"):
                await _maybe_await(submit_btn.scroll_into_view_if_needed())
            if hasattr(submit_btn, "click"):
                await _maybe_await(submit_btn.click())
            target_page = page or self._page
            if target_page and hasattr(target_page, "wait_for_load_state"):
                try:
                    await _maybe_await(target_page.wait_for_load_state("networkidle", timeout=5000))
                except Exception:
                    pass
                await asyncio.sleep(1.0)
            return True, None
        except Exception as exc:
            return False, str(exc)

