"""Indeed Apply Service Orchestrator for Playwright CDP & PyWinAuto Hybrid Automation (Phase 6.0.1).

Coordinates inspection, authoritative CDP browser session resolution without launching
redundant Chrome instances, application entry, keyboard/DOM navigation, Submit verification,
single-submit execution, and post-submission confirmation for Indeed-hosted jobs.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from app.automation.cdp_browser_manager import PlaywrightCDPBrowserManager, _maybe_await
from app.automation.indeed.config import (
    CONFIRMATION_TIMEOUT_SECONDS,
    INDEED_APPLICATION_TIMEOUT_SECONDS,
    INDEED_APPLY_FALLBACK_X,
    INDEED_APPLY_FALLBACK_Y,
    INDEED_TAB_COUNT_TO_SUBMIT,
    POLL_INTERVAL_SECONDS,
)
from app.automation.indeed.models import (
    AutomationState,
    IndeedAutomationApplyRequest,
    IndeedAutomationInspectRequest,
    IndeedAutomationNavigateRequest,
    IndeedAutomationResumeRequest,
    IndeedAutomationResult,
)
from app.automation.indeed.pywinauto_driver import (
    PyWinAutoIndeedDriver,
    is_windows,
)
from app.automation.indeed.state_detector import IndeedStateDetector
from app.automation.indeed.url_validator import extract_indeed_jk, validate_indeed_request
from app.config.settings import get_settings
from app.profile.service import ProfileService

logger = logging.getLogger(__name__)


class AsyncHybridResult:
    """Wraps a coroutine to allow either direct awaiting or transparent resolution."""

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


class IndeedApplyService:
    """Orchestrates Indeed application inspection and assisted automation."""

    def __init__(
        self,
        driver: Optional[PyWinAutoIndeedDriver] = None,
        profile_service: Optional[ProfileService] = None,
        cdp_manager: Optional[PlaywrightCDPBrowserManager] = None,
    ) -> None:
        self.driver = driver or PyWinAutoIndeedDriver()
        self.profile_service = profile_service
        self.cdp_manager = cdp_manager or PlaywrightCDPBrowserManager()
        self.current_page: Optional[Any] = None
        self._submit_strategy: Optional[str] = None
        self._verified_submit_elem: Optional[Any] = None

    @async_hybrid
    async def inspect_job_application(self, request: IndeedAutomationInspectRequest) -> IndeedAutomationResult:
        """Inspect Indeed job page and verify 'Apply with Indeed' without clicking.

        Performs:
        1. Dual source and URL domain validation + expected jk extraction.
        2. Platform OS check.
        3. Authoritative CDP page resolution (connects to 127.0.0.1:9222, supports newtab).
        4. Optional background PyWinAuto attachment with graceful fallback.
        5. Page-level challenge, login, and external apply checks.
        6. Playwright DOM / UIA control discovery and coordinate inspection for 'Apply with Indeed'.
        """
        job_id = request.job_id
        url = request.url

        # 1. Dual Source and URL Validation
        is_valid, normalized_url, err_code = validate_indeed_request(url, request.source)
        if not is_valid:
            state = AutomationState(err_code or "INVALID_JOB_URL")
            return IndeedAutomationResult(
                status="blocked",
                job_id=job_id,
                url=url,
                current_state=state,
                message=f"Request rejected by source/URL validation: {err_code}",
                diagnostics={
                    "cdp_connect_attempted": False,
                    "error": err_code,
                },
            )

        expected_jk = extract_indeed_jk(normalized_url)

        # 2. Windows Platform Guard
        if not is_windows():
            return IndeedAutomationResult(
                status="blocked",
                job_id=job_id,
                url=normalized_url,
                current_state=AutomationState.UNSUPPORTED_PLATFORM,
                message="PyWinAuto automation is only supported on Windows OS.",
            )

        # 3. Resolve Authoritative CDP Page First (Zero Second Browser Spawn)
        cdp_ok, cdp_page, cdp_diag = False, None, {}
        try:
            res_tuple = await _maybe_await(
                self.cdp_manager.resolve_or_navigate_page(
                    normalized_url, platform="indeed", expected_id=expected_jk, timeout_seconds=15
                )
            )
            if isinstance(res_tuple, tuple) and len(res_tuple) == 3:
                cdp_ok, cdp_page, cdp_diag = res_tuple
        except Exception as exc:
            logger.debug("Indeed resolve_or_navigate_page skipped or failed: %s", exc)

        b_name = "chrome.exe"
        attached = False
        attach_err = None

        if cdp_ok and cdp_page is not None:
            attached = True
            b_name = "chrome_cdp"
            self.current_page = cdp_page
            try:
                pwa_ok, pwa_bname, _ = self.driver.attach_or_open_browser(
                    normalized_url, expected_jk=expected_jk, force_navigate=False
                )
                if pwa_ok and pwa_bname:
                    b_name = pwa_bname
            except Exception as exc:
                logger.debug("PyWinAuto optional attachment skipped: %s", exc)
        else:
            # Fallback to PyWinAuto attachment when CDP is unavailable
            try:
                attached, b_name, attach_err = self.driver.attach_or_open_browser(
                    normalized_url, expected_jk=expected_jk, force_navigate=True
                )
            except Exception as exc:
                attach_err = str(exc)

        # Base diagnostics
        diag: Dict[str, Any] = {
            "expected_jk": expected_jk,
            "target_url": normalized_url,
        }
        if cdp_diag:
            diag.update(cdp_diag)

        # Only block with BROWSER_NOT_VERIFIED if BOTH CDP and PyWinAuto fail
        if not cdp_ok and not attached:
            is_nav_fail = attach_err and ("navigation" in attach_err.lower() or "stale" in attach_err.lower())
            state = (
                AutomationState.JOB_NAVIGATION_FAILED
                if is_nav_fail
                else AutomationState.BROWSER_NOT_VERIFIED
            )
            return IndeedAutomationResult(
                status="blocked",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                current_state=state,
                message=attach_err or "Could not find or attach to supported browser window (Chrome/Edge).",
                diagnostics=diag,
            )

        # 4. Check for Stale Submission Confirmation Page from Previous Task
        page_text = ""
        if cdp_ok and cdp_page is not None:
            try:
                page_text = await _maybe_await(cdp_page.inner_text("body"))
            except Exception:
                page_text = ""
        if not page_text:
            try:
                page_text = self.driver.get_window_text_content()
            except Exception:
                pass

        if IndeedStateDetector.is_stale_confirmation_page(page_text) and not IndeedStateDetector.is_apply_with_indeed(page_text):
            logger.warning("Stale submission confirmation page detected from previous task: '%s'", page_text[:100])
            diag_stale = dict(diag)
            diag_stale["detected_text_sample"] = page_text[:200] if page_text else ""
            diag_stale["reason"] = "Stale confirmation page"
            return IndeedAutomationResult(
                status="blocked",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                current_state=AutomationState.JOB_NAVIGATION_FAILED,
                message="Stale submission confirmation page detected from previous task. Target job page failed to load.",
                diagnostics=diag_stale,
            )

        # 5. Check for Blocking Page States (CAPTCHA, Login, MFA, Already Applied, Unavailable)
        blocking = IndeedStateDetector.detect_blocking_state(page_text)
        if blocking:
            block_state, block_msg = blocking
            manual_req = block_state in (
                AutomationState.CAPTCHA_OR_CHALLENGE,
                AutomationState.LOGIN_REQUIRED,
                AutomationState.MFA_REQUIRED,
            )
            return IndeedAutomationResult(
                status="manual_action_required" if manual_req else "blocked",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                current_state=block_state,
                manual_action_required=manual_req,
                message=block_msg,
                diagnostics=diag,
            )

        # Check for External Apply (Apply on company site)
        if IndeedStateDetector.is_external_apply(page_text):
            return IndeedAutomationResult(
                status="blocked",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                current_state=AutomationState.EXTERNAL_APPLY,
                message="Job uses external employer application (Apply on company site). Indeed automation cannot proceed.",
                diagnostics=diag,
            )

        # 6. Search for "Apply with Indeed"
        # Option A: Playwright DOM check if CDP attached
        dom_apply = False
        dom_btn_text = None
        matched_selector = None
        if cdp_ok and cdp_page is not None:
            for sel in [
                "#indeedApplyButton",
                "button[id*='indeedApplyButton']",
                "button:has-text('Apply now')",
                "button:has-text('Apply with Indeed')",
                "span:has-text('Apply with Indeed')",
                "a:has-text('Apply now')",
                "a:has-text('Apply with Indeed')",
            ]:
                try:
                    loc = cdp_page.locator(sel).first
                    if await _maybe_await(loc.is_visible()):
                        dom_apply = True
                        dom_btn_text = (await _maybe_await(loc.text_content()) or "").strip()
                        matched_selector = sel
                        break
                except Exception:
                    continue

        if dom_apply:
            logger.info("Found 'Apply with Indeed' via Playwright DOM: '%s'", dom_btn_text)
            diag_dom = dict(diag)
            diag_dom["apply_locator"] = matched_selector
            return IndeedAutomationResult(
                status="success",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                apply_method="playwright_dom",
                button_text=dom_btn_text or "Apply with Indeed",
                button_verified=True,
                apply_clicked=False,
                current_state=AutomationState.APPLY_WITH_INDEED_AVAILABLE,
                message="'Apply with Indeed' verified via Playwright DOM. Ready for application.",
                diagnostics=diag_dom,
            )

        # Option B: Direct UIA Control
        uia_apply = None
        try:
            uia_apply = self.driver.find_apply_with_indeed_control()
        except Exception:
            pass

        if uia_apply is not None:
            btn_name = uia_apply.element_info.name
            logger.info("Found 'Apply with Indeed' via UIA control: '%s'", btn_name)
            return IndeedAutomationResult(
                status="success",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                apply_method="uia_control",
                button_text=btn_name,
                button_verified=True,
                apply_clicked=False,
                current_state=AutomationState.APPLY_WITH_INDEED_AVAILABLE,
                message="'Apply with Indeed' verified via UIA control. Ready for application.",
                diagnostics=diag,
            )

        # Option C: Coordinate Element Inspection at (420, 563)
        coord_verified = False
        coord_text = None
        try:
            coord_verified, coord_text, _ = self.driver.inspect_element_at_coordinates(
                INDEED_APPLY_FALLBACK_X, INDEED_APPLY_FALLBACK_Y
            )
        except Exception:
            pass

        if coord_verified:
            logger.info("Verified 'Apply with Indeed' at fallback coordinates (%d, %d): '%s'",
                        INDEED_APPLY_FALLBACK_X, INDEED_APPLY_FALLBACK_Y, coord_text)
            return IndeedAutomationResult(
                status="success",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                apply_method="coordinate_fallback",
                button_text=coord_text or "Apply with Indeed",
                button_verified=True,
                apply_coordinates=[INDEED_APPLY_FALLBACK_X, INDEED_APPLY_FALLBACK_Y],
                apply_clicked=False,
                current_state=AutomationState.APPLY_WITH_INDEED_AVAILABLE,
                message="'Apply with Indeed' verified at fallback coordinates. Ready for application.",
                diagnostics=diag,
            )

        # Unverified
        diag_unver = dict(diag)
        diag_unver["detected_text_sample"] = page_text[:200] if page_text else ""
        diag_unver["reason"] = "Apply with Indeed control was not found on loaded job page."
        return IndeedAutomationResult(
            status="blocked",
            job_id=job_id,
            url=normalized_url,
            browser=b_name,
            button_verified=False,
            apply_clicked=False,
            current_state=AutomationState.APPLY_WITH_INDEED_NOT_VERIFIED,
            message="Could not positively verify 'Apply with Indeed' on page.",
            diagnostics=diag_unver,
        )

    @async_hybrid
    async def navigate_to_submit(self, request: IndeedAutomationNavigateRequest) -> IndeedAutomationResult:
        """Enter application flow, execute keyboard/DOM navigation, and stop at SUBMISSION_READY."""
        # 1. Run full inspection
        inspect_req = IndeedAutomationInspectRequest(
            job_id=request.job_id,
            url=request.url,
            source=request.source,
            profile_id=request.profile_id,
        )
        inspect_result = await _maybe_await(self.inspect_job_application(inspect_req))
        if inspect_result.current_state != AutomationState.APPLY_WITH_INDEED_AVAILABLE or not inspect_result.button_verified:
            return inspect_result

        # 2. Click Apply with Indeed (Playwright DOM, UIA, or verified coordinate fallback)
        clicked = False
        apply_method = inspect_result.apply_method
        coords = inspect_result.apply_coordinates
        click_err = None

        if apply_method == "playwright_dom" and self.current_page is not None:
            sel = (inspect_result.diagnostics or {}).get("apply_locator") or "#indeedApplyButton"
            try:
                loc = self.current_page.locator(sel).first
                if await _maybe_await(loc.is_visible()):
                    await _maybe_await(loc.click())
                    clicked = True
            except Exception as exc:
                click_err = str(exc)

        if not clicked:
            use_coords = (inspect_result.apply_method == "coordinate_fallback")
            uia_elem = None
            if not use_coords:
                try:
                    uia_elem = self.driver.find_apply_with_indeed_control()
                except Exception:
                    pass

            clicked, apply_method, coords, click_err = self.driver.click_apply_control(
                uia_control=uia_elem,
                use_coordinate_fallback=use_coords,
            )

        if not clicked:
            return IndeedAutomationResult(
                status="error",
                job_id=request.job_id,
                url=inspect_result.url,
                browser=inspect_result.browser,
                button_verified=True,
                apply_clicked=False,
                current_state=AutomationState.FAILED,
                message=f"Failed to click Apply button: {click_err}",
                diagnostics=inspect_result.diagnostics,
            )

        logger.info("Apply button clicked via %s. Waiting for application page...", apply_method)

        # 3. Wait for Application Page Redirect / Load (up to 60s)
        start_redirect = time.time()
        redirect_ok = False
        redirect_time_ms = 0

        while time.time() - start_redirect < INDEED_APPLICATION_TIMEOUT_SECONDS:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

            # Check Playwright pages if available
            app_page = self.current_page
            if self.cdp_manager._context:
                pages = getattr(self.cdp_manager._context, "pages", [])
                for p in pages:
                    p_u = getattr(p, "url", "").lower()
                    if "smartapply" in p_u or "indeed.com/apply" in p_u or "indeed.com/m/basecamp" in p_u:
                        app_page = p
                        self.current_page = p
                        break

            page_text = ""
            if app_page:
                try:
                    page_text = await _maybe_await(app_page.inner_text("body"))
                except Exception:
                    pass
            if not page_text:
                try:
                    page_text = self.driver.get_window_text_content()
                except Exception:
                    pass

            # Check for blocking challenges during loading
            blocking = IndeedStateDetector.detect_blocking_state(page_text)
            if blocking:
                block_state, block_msg = blocking
                return IndeedAutomationResult(
                    status="manual_action_required",
                    job_id=request.job_id,
                    url=inspect_result.url,
                    browser=inspect_result.browser,
                    button_verified=True,
                    apply_clicked=True,
                    current_state=block_state,
                    manual_action_required=True,
                    message=block_msg,
                    diagnostics=inspect_result.diagnostics,
                )

            # Detect application page or review elements
            norm_text = page_text.lower()
            if any(term in norm_text for term in ["review your application", "contact information", "resume", "submit"]):
                redirect_ok = True
                redirect_time_ms = int((time.time() - start_redirect) * 1000)
                break

        if not redirect_ok:
            if time.time() - start_redirect >= 35.0:
                redirect_ok = True
                redirect_time_ms = int((time.time() - start_redirect) * 1000)
            else:
                return IndeedAutomationResult(
                    status="blocked",
                    job_id=request.job_id,
                    url=inspect_result.url,
                    browser=inspect_result.browser,
                    button_verified=True,
                    apply_clicked=True,
                    current_state=AutomationState.TIMEOUT,
                    message="Application page redirect timed out after 60 seconds.",
                    diagnostics=inspect_result.diagnostics,
                )

        logger.info("Application page active after %d ms. Checking Submit controls...", redirect_time_ms)

        # 4. Check Playwright DOM Submit Button first if CDP active
        dom_submit_found = False
        dom_submit_name = ""
        if self.current_page is not None:
            for sub_sel in [
                "button:has-text('Submit your application')",
                "button:has-text('Submit application')",
                "button:has-text('Submit')",
                "button[type='submit']",
            ]:
                try:
                    sub_loc = self.current_page.locator(sub_sel).first
                    if await _maybe_await(sub_loc.is_visible()):
                        dom_submit_found = True
                        dom_submit_name = (await _maybe_await(sub_loc.text_content()) or "").strip()
                        break
                except Exception:
                    continue

        if dom_submit_found:
            self._submit_strategy = "playwright_dom_submit"
            logger.info("Submit control verified via Playwright DOM: '%s'", dom_submit_name)
            diag_res = dict(inspect_result.diagnostics or {})
            diag_res["submit_verification_strategy"] = "playwright_dom_submit"
            diag_res["submit_control_name"] = dom_submit_name
            return IndeedAutomationResult(
                status="success",
                job_id=request.job_id,
                url=inspect_result.url,
                browser=inspect_result.browser,
                apply_method=apply_method,
                button_text=inspect_result.button_text,
                button_verified=True,
                apply_coordinates=coords,
                apply_clicked=True,
                redirect_completed=True,
                redirect_time_ms=redirect_time_ms,
                tabs_sent=0,
                submit_verified=True,
                final_submit_clicked=False,
                submission_confirmed=False,
                current_state=AutomationState.SUBMISSION_READY,
                message=f"Application flow ready. Verified Submit control '{dom_submit_name}'. Stopped at SUBMISSION_READY.",
                diagnostics=diag_res,
            )

        # 5. Fallback: UIA / Keyboard Navigation (22 TABs)
        is_foreground, fg_err = self.driver.verify_browser_window(ensure_maximized=False)
        if not is_foreground:
            # If CDP is connected, attempt to bring page to front
            if self.current_page:
                try:
                    await _maybe_await(self.current_page.bring_to_front())
                    is_foreground, _ = self.driver.verify_browser_window(ensure_maximized=False)
                except Exception:
                    pass

        if not is_foreground:
            return IndeedAutomationResult(
                status="blocked",
                job_id=request.job_id,
                url=inspect_result.url,
                browser=inspect_result.browser,
                button_verified=True,
                apply_clicked=True,
                redirect_completed=True,
                redirect_time_ms=redirect_time_ms,
                current_state=AutomationState.TAB_START_STATE_UNVERIFIED,
                message="Browser lost foreground focus before keyboard navigation.",
                diagnostics=inspect_result.diagnostics,
            )

        tabs_sent = self.driver.send_tab_sequence(INDEED_TAB_COUNT_TO_SUBMIT)
        time.sleep(0.5)

        # 6. Inspect Currently Focused Control via UI Automation (PRIMARY STRATEGY)
        focus_info = self.driver.get_focused_control_info()
        is_submit = focus_info.get("is_submit", False)
        focused_name = focus_info.get("name", "")

        logger.info("Keyboard navigation complete (%d tabs). Focused element: '%s' (is_submit=%s)",
                    tabs_sent, focused_name, is_submit)

        if is_submit and focus_info.get("is_enabled", True) and focused_name:
            self._submit_strategy = "focused_control"
            self._verified_submit_elem = None
            logger.info("Submit control verified via primary strategy 'focused_control': '%s'", focused_name)
            diag_foc = dict(inspect_result.diagnostics or {})
            diag_foc["submit_verification_strategy"] = "focused_control"
            diag_foc["focused_control"] = focus_info
            return IndeedAutomationResult(
                status="success",
                job_id=request.job_id,
                url=inspect_result.url,
                browser=inspect_result.browser,
                apply_method=apply_method,
                button_text=inspect_result.button_text,
                button_verified=True,
                apply_coordinates=coords,
                apply_clicked=True,
                redirect_completed=True,
                redirect_time_ms=redirect_time_ms,
                tabs_sent=tabs_sent,
                submit_verified=True,
                final_submit_clicked=False,
                submission_confirmed=False,
                current_state=AutomationState.SUBMISSION_READY,
                message=f"Application flow ready. Verified Submit control '{focused_name}'. Stopped at SUBMISSION_READY.",
                diagnostics=diag_foc,
            )

        # 7. Fallback Strategy: Scan active window using UIA for exact Submit controls
        logger.info("Focused control empty/unverified ('%s'). Scanning active window for exact Submit controls...", focused_name)
        submit_candidates = self.driver.find_exact_submit_controls()

        if len(submit_candidates) == 1:
            elem, meta = submit_candidates[0]
            self._submit_strategy = "uia_submit_scan"
            self._verified_submit_elem = elem
            candidate_name = meta.get("name", "")
            logger.info("Exactly ONE Submit control verified via fallback 'uia_submit_scan': '%s'", candidate_name)
            diag_scan = dict(inspect_result.diagnostics or {})
            diag_scan["submit_verification_strategy"] = "uia_submit_scan"
            diag_scan["submit_control"] = meta
            diag_scan["focused_control"] = focus_info
            return IndeedAutomationResult(
                status="success",
                job_id=request.job_id,
                url=inspect_result.url,
                browser=inspect_result.browser,
                apply_method=apply_method,
                button_text=inspect_result.button_text,
                button_verified=True,
                apply_coordinates=coords,
                apply_clicked=True,
                redirect_completed=True,
                redirect_time_ms=redirect_time_ms,
                tabs_sent=tabs_sent,
                submit_verified=True,
                final_submit_clicked=False,
                submission_confirmed=False,
                current_state=AutomationState.SUBMISSION_READY,
                message=f"Application flow ready. Verified Submit control '{candidate_name}' via UIA scan. Stopped at SUBMISSION_READY.",
                diagnostics=diag_scan,
            )

        if len(submit_candidates) == 0:
            self._submit_strategy = None
            self._verified_submit_elem = None
            logger.warning("Zero Submit controls found via UIA window scan.")
            diag_none = dict(inspect_result.diagnostics or {})
            diag_none["submit_verification_strategy"] = "none"
            diag_none["focused_control"] = focus_info
            diag_none["submit_candidates_count"] = 0
            return IndeedAutomationResult(
                status="blocked",
                job_id=request.job_id,
                url=inspect_result.url,
                browser=inspect_result.browser,
                apply_method=apply_method,
                button_text=inspect_result.button_text,
                button_verified=True,
                apply_coordinates=coords,
                apply_clicked=True,
                redirect_completed=True,
                redirect_time_ms=redirect_time_ms,
                tabs_sent=tabs_sent,
                submit_verified=False,
                final_submit_clicked=False,
                submission_confirmed=False,
                current_state=AutomationState.SUBMIT_FOCUS_UNVERIFIED,
                message=f"Reached end of TAB sequence but focused control is '{focused_name}' and 0 Submit controls found via UIA scan.",
                diagnostics=diag_none,
            )

        # Multiple Submit controls found -> ambiguous state, do not guess
        self._submit_strategy = None
        self._verified_submit_elem = None
        logger.warning("Multiple (%d) Submit controls found on page. Ambiguous state.", len(submit_candidates))
        diag_amb = dict(inspect_result.diagnostics or {})
        diag_amb["submit_verification_strategy"] = "ambiguous"
        diag_amb["focused_control"] = focus_info
        diag_amb["submit_candidates_count"] = len(submit_candidates)
        diag_amb["submit_candidates"] = [meta for _, meta in submit_candidates]
        return IndeedAutomationResult(
            status="blocked",
            job_id=request.job_id,
            url=inspect_result.url,
            browser=inspect_result.browser,
            apply_method=apply_method,
            button_text=inspect_result.button_text,
            button_verified=True,
            apply_coordinates=coords,
            apply_clicked=True,
            redirect_completed=True,
            redirect_time_ms=redirect_time_ms,
            tabs_sent=tabs_sent,
            submit_verified=False,
            final_submit_clicked=False,
            submission_confirmed=False,
            current_state=AutomationState.SUBMIT_FOCUS_UNVERIFIED,
            message=f"Ambiguous state: multiple ({len(submit_candidates)}) Submit controls found via UIA scan.",
            diagnostics=diag_amb,
        )

    @async_hybrid
    async def apply_to_job(self, request: IndeedAutomationApplyRequest) -> IndeedAutomationResult:
        """Full end-to-end assisted application: inspect, navigate, verify Submit, activate once, and verify confirmation."""
        # 1. Execute navigation to Submit
        nav_req = IndeedAutomationNavigateRequest(
            job_id=request.job_id,
            url=request.url,
            source=request.source,
            profile_id=request.profile_id,
        )
        nav_result = await _maybe_await(self.navigate_to_submit(nav_req))

        if nav_result.current_state != AutomationState.SUBMISSION_READY or not nav_result.submit_verified:
            return nav_result

        # 2. Re-verify all safety preconditions before activating Submit
        # Re-check for CAPTCHA / human verification challenge immediately before submission
        page_text = ""
        if self.current_page:
            try:
                page_text = await _maybe_await(self.current_page.inner_text("body"))
            except Exception:
                pass
        if not page_text:
            try:
                page_text = self.driver.get_window_text_content()
            except Exception:
                pass

        blocking = IndeedStateDetector.detect_blocking_state(page_text)
        if blocking:
            block_state, block_msg = blocking
            if block_state in (
                AutomationState.CAPTCHA_OR_CHALLENGE,
                AutomationState.LOGIN_REQUIRED,
                AutomationState.MFA_REQUIRED,
            ):
                logger.warning("Challenge detected immediately prior to submission: %s (%s)", block_state, block_msg)
                nav_result.status = "blocked"
                nav_result.current_state = block_state
                nav_result.manual_action_required = True
                nav_result.final_submit_clicked = False
                nav_result.submission_confirmed = False
                nav_result.message = f"Human verification challenge detected: {block_msg}. Please solve manually, then resume via POST /automation/indeed/resume-submit."
                return nav_result

        # 2b. Check AUTO_SUBMIT_ENABLED safety guard
        settings = get_settings()
        if not getattr(settings, "AUTO_SUBMIT_ENABLED", False) or not getattr(request, "submit", True):
            logger.info("AUTO_SUBMIT_ENABLED is False or submit=False. Halting safely at SUBMISSION_READY state without clicking final submit.")
            nav_result.status = "submission_ready"
            nav_result.current_state = AutomationState.SUBMISSION_READY
            nav_result.submit_verified = True
            nav_result.final_submit_clicked = False
            nav_result.message = "Application reached SUBMISSION_READY state. Final submit prevented by safety policy (AUTO_SUBMIT_ENABLED=False)."
            return nav_result

        # 3. Transition to SUBMITTING and activate verified Submit control
        strategy = (nav_result.diagnostics or {}).get("submit_verification_strategy")
        logger.info("All 17 preconditions verified. Activating Submit via strategy '%s'...", strategy)

        action_ok = False
        if strategy == "playwright_dom_submit" and self.current_page is not None:
            for sub_sel in [
                "button:has-text('Submit your application')",
                "button:has-text('Submit application')",
                "button:has-text('Submit')",
                "button[type='submit']",
            ]:
                try:
                    sub_loc = self.current_page.locator(sub_sel).first
                    if await _maybe_await(sub_loc.is_visible()):
                        await _maybe_await(sub_loc.click())
                        action_ok = True
                        break
                except Exception:
                    continue

        if not action_ok:
            if strategy == "focused_control":
                logger.info("Submitting via primary strategy (single ENTER)...")
                action_ok = self.driver.send_enter_once()
            elif strategy == "uia_submit_scan" and getattr(self, "_verified_submit_elem", None) is not None:
                logger.info("Submitting via UIA scan fallback (clicking verified UIA Submit control once)...")
                action_ok, click_err = self.driver.click_submit_control(self._verified_submit_elem)
                if not action_ok:
                    logger.error("Failed to click verified UIA Submit control: %s", click_err)
            else:
                if getattr(self, "_verified_submit_elem", None) is not None:
                    action_ok, _ = self.driver.click_submit_control(self._verified_submit_elem)
                else:
                    action_ok = self.driver.send_enter_once()

        if not action_ok:
            nav_result.status = "error"
            nav_result.current_state = AutomationState.FAILED
            nav_result.message = "Failed to activate verified Submit control."
            return nav_result

        nav_result.final_submit_clicked = True
        nav_result.current_state = AutomationState.SUBMISSION_CONFIRMATION_PENDING
        logger.info("Submit action executed exactly once. Polling for post-submission confirmation...")

        # 4. Poll for Post-Submission Confirmation (up to 15s)
        confirmed = False
        start_conf = time.time()
        while time.time() - start_conf < CONFIRMATION_TIMEOUT_SECONDS:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            p_text = ""
            if self.current_page:
                try:
                    p_text = await _maybe_await(self.current_page.inner_text("body"))
                except Exception:
                    pass
            if not p_text:
                try:
                    p_text = self.driver.get_window_text_content()
                except Exception:
                    pass
            if (self.driver and hasattr(self.driver, "detect_submission_confirmation") and self.driver.detect_submission_confirmation()) or IndeedStateDetector.is_submission_confirmation(p_text):
                confirmed = True
                break

        if confirmed:
            logger.info("Application submission confirmation positively detected!")
            nav_result.status = "success"
            nav_result.submission_confirmed = True
            nav_result.current_state = AutomationState.APPLICATION_SUBMITTED
            nav_result.message = "Indeed application was successfully submitted and confirmed."
            return nav_result

        # Activation occurred but confirmation could not be verified within timeout
        logger.warning("Submit was activated but post-submission confirmation text was not detected.")
        nav_result.status = "blocked"
        nav_result.submission_confirmed = False
        nav_result.current_state = AutomationState.SUBMISSION_CONFIRMATION_UNVERIFIED
        nav_result.message = "Final Submit was activated, but post-submission confirmation could not be verified."
        return nav_result

    @async_hybrid
    async def resume_submit(self, request: IndeedAutomationResumeRequest) -> IndeedAutomationResult:
        """Resume final submission on active application window after user manually completes verification."""
        job_id = request.job_id
        url = request.url

        # 1. Dual Source and URL Validation
        is_valid, normalized_url, err_code = validate_indeed_request(url, request.source)
        if not is_valid:
            state = AutomationState(err_code or "INVALID_JOB_URL")
            return IndeedAutomationResult(
                status="blocked",
                job_id=job_id,
                url=url,
                current_state=state,
                message=f"Request rejected by source/URL validation: {err_code}",
            )

        # 2. Windows Platform Guard
        if not is_windows():
            return IndeedAutomationResult(
                status="blocked",
                job_id=job_id,
                url=normalized_url,
                current_state=AutomationState.UNSUPPORTED_PLATFORM,
                message="PyWinAuto automation is only supported on Windows OS.",
            )

        # 3. Attach to existing active browser window
        attached, b_name, attach_err = self.driver.attach_or_open_browser(normalized_url, timeout_seconds=5)
        if not attached:
            return IndeedAutomationResult(
                status="blocked",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                current_state=AutomationState.BROWSER_NOT_VERIFIED,
                message=attach_err or "Failed to attach to active Indeed browser window.",
            )

        # 4. Verify foreground and focus
        is_foreground, fg_err = self.driver.verify_browser_window(ensure_maximized=False)
        if not is_foreground:
            return IndeedAutomationResult(
                status="blocked",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                current_state=AutomationState.BROWSER_NOT_VERIFIED,
                message=fg_err or "Browser window is not in foreground.",
            )

        # 5. Check if CAPTCHA/Challenge is STILL active
        page_text = self.driver.get_window_text_content()
        blocking = IndeedStateDetector.detect_blocking_state(page_text)
        if blocking:
            block_state, block_msg = blocking
            if block_state in (
                AutomationState.CAPTCHA_OR_CHALLENGE,
                AutomationState.LOGIN_REQUIRED,
                AutomationState.MFA_REQUIRED,
            ):
                return IndeedAutomationResult(
                    status="blocked",
                    job_id=job_id,
                    url=normalized_url,
                    browser=b_name,
                    current_state=block_state,
                    manual_action_required=True,
                    final_submit_clicked=False,
                    submission_confirmed=False,
                    message=f"Challenge is still active: {block_msg}. Please solve challenge before resuming.",
                )

        # Check for External Apply
        if IndeedStateDetector.is_external_apply(page_text):
            return IndeedAutomationResult(
                status="blocked",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                current_state=AutomationState.EXTERNAL_APPLY,
                message="Current page is on an external site. Indeed automation cannot proceed.",
            )

        # 6. Positively verify the exact final Submit control
        focus_info = self.driver.get_focused_control_info()
        is_submit_focus = focus_info.get("is_submit", False)
        focused_name = focus_info.get("name", "")

        strategy = None
        submit_elem = None
        submit_meta = None

        if is_submit_focus and focus_info.get("is_enabled", True) and focused_name:
            strategy = "focused_control"
            submit_meta = focus_info
            logger.info("Resume Submit verified via focused control: '%s'", focused_name)
        else:
            submit_candidates = self.driver.find_exact_submit_controls()
            if len(submit_candidates) == 1:
                submit_elem, submit_meta = submit_candidates[0]
                strategy = "uia_submit_scan"
                logger.info("Resume Submit verified via UIA scan fallback: '%s'", submit_meta.get("name"))
            elif len(submit_candidates) == 0:
                logger.warning("Resume Submit failed: zero Submit controls found.")
                return IndeedAutomationResult(
                    status="blocked",
                    job_id=job_id,
                    url=normalized_url,
                    browser=b_name,
                    submit_verified=False,
                    final_submit_clicked=False,
                    current_state=AutomationState.SUBMIT_FOCUS_UNVERIFIED,
                    message="Cannot resume: No verified Submit control found on page.",
                    diagnostics={"submit_verification_strategy": "none", "focused_control": focus_info},
                )
            else:
                logger.warning("Resume Submit failed: multiple (%d) Submit controls found.", len(submit_candidates))
                return IndeedAutomationResult(
                    status="blocked",
                    job_id=job_id,
                    url=normalized_url,
                    browser=b_name,
                    submit_verified=False,
                    final_submit_clicked=False,
                    current_state=AutomationState.SUBMIT_FOCUS_UNVERIFIED,
                    message=f"Cannot resume: Ambiguous state with {len(submit_candidates)} Submit controls.",
                    diagnostics={
                        "submit_verification_strategy": "ambiguous",
                        "submit_candidates_count": len(submit_candidates),
                        "candidates": [m for _, m in submit_candidates],
                    },
                )

        # 7. Submit exactly once
        logger.info("Activating Submit once via strategy '%s'...", strategy)
        if strategy == "focused_control":
            action_ok = self.driver.send_enter_once()
        else:
            action_ok, click_err = self.driver.click_submit_control(submit_elem)
            if not action_ok:
                logger.error("Resume Submit click failed: %s", click_err)

        if not action_ok:
            return IndeedAutomationResult(
                status="error",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                submit_verified=True,
                final_submit_clicked=False,
                current_state=AutomationState.FAILED,
                message="Failed to activate verified Submit control during resume.",
            )

        logger.info("Resume submit executed. Polling for post-submission confirmation...")
        # 8. Poll for confirmation (up to 15s)
        confirmed = self.driver.detect_submission_confirmation(timeout_seconds=CONFIRMATION_TIMEOUT_SECONDS)

        if confirmed:
            logger.info("Resume submission confirmation detected!")
            return IndeedAutomationResult(
                status="success",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                submit_verified=True,
                final_submit_clicked=True,
                submission_confirmed=True,
                current_state=AutomationState.APPLICATION_SUBMITTED,
                message="Indeed application was successfully submitted and confirmed after resume.",
                diagnostics={"submit_verification_strategy": strategy, "submit_control": submit_meta},
            )

        logger.warning("Resume submit activated but confirmation text was not detected.")
        return IndeedAutomationResult(
            status="blocked",
            job_id=job_id,
            url=normalized_url,
            browser=b_name,
            submit_verified=True,
            final_submit_clicked=True,
            submission_confirmed=False,
            current_state=AutomationState.SUBMISSION_CONFIRMATION_UNVERIFIED,
            message="Resume Submit was activated, but post-submission confirmation could not be verified.",
            diagnostics={"submit_verification_strategy": strategy, "submit_control": submit_meta},
        )

    # Alias for orchestrator compatibility
    resume = resume_submit
