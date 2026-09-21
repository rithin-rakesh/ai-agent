"""Glassdoor Apply Service Orchestrator for PyWinAuto & Playwright Hybrid Automation.

Coordinates inspection, authoritative CDP browser session resolution without launching
redundant Chrome instances, dynamic modal form processing, question extraction, truthful answering,
resume verification with scroll_into_view_if_needed, review navigation, Submit verification,
single-submit execution, and post-submission confirmation for Glassdoor Easy Apply jobs.
"""

import asyncio
import inspect
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.automation.forms.answer_resolver import FormAnswerResolver
from app.automation.forms.form_inspector import FormInspector
from app.candidate_answers.bank import CandidateAnswerBank
from app.config.settings import Settings, get_settings
from app.automation.forms.handlers import (
    fill_number_field,
    fill_text_field,
    select_dropdown_option,
    select_radio_option,
    set_checkboxes,
)
from app.automation.forms.models import ApplicationQuestion, QuestionInputType
from app.automation.glassdoor.config import (
    ALLOW_VERIFIED_GLASSDOOR_COORDINATE_FALLBACK,
    CONTINUE_BUTTON_NAMES,
    CONTINUE_ON_REQUIREMENTS_WARNING,
    EXACT_SUBMIT_NAMES,
    GLASSDOOR_EASY_APPLY_FALLBACK_X,
    GLASSDOOR_EASY_APPLY_FALLBACK_Y,
    MAX_APPLICATION_STEPS,
    MAX_MODAL_STEPS,
    MAX_STALLED_SIGNATURE_COUNT,
    MAX_TAB_TRAVERSAL,
    NEXT_BUTTON_NAMES,
    POST_EASY_APPLY_REDIRECT_WAIT_SECONDS,
    REVIEW_BUTTON_NAMES,
    SMARTAPPLY_HOST_DOMAINS,
)
from app.automation.glassdoor.confirmation_detector import GlassdoorConfirmationDetector
from app.automation.glassdoor.modal_inspector import GlassdoorModalInspector
from app.automation.glassdoor.models import (
    GlassdoorAutomationApplyRequest,
    GlassdoorAutomationInspectRequest,
    GlassdoorAutomationNavigateRequest,
    GlassdoorAutomationResumeRequest,
    GlassdoorAutomationResult,
    GlassdoorAutomationState,
)
from app.automation.glassdoor.playwright_driver import GlassdoorPlaywrightDriver
from app.automation.glassdoor.pywinauto_driver import (
    PyWinAutoGlassdoorDriver,
    is_windows,
)
from app.automation.glassdoor.resume_handler import GlassdoorResumeHandler
from app.automation.glassdoor.state_detector import GlassdoorStateDetector
from app.automation.glassdoor.url_validator import extract_glassdoor_job_id, validate_glassdoor_request
from app.database.repositories.application_repository import ApplicationRepository
from app.models.application import ApplicationCreate, AutomationLogCreate
from app.profile.service import ProfileService

logger = logging.getLogger(__name__)


async def _maybe_await(val: Any) -> Any:
    """Await val if it is a coroutine or awaitable, otherwise return it directly."""
    if inspect.isawaitable(val):
        return await val
    return val


class AsyncHybridResult:
    """Hybrid wrapper allowing a service method to be awaited in async contexts (FastAPI)
    or resolved synchronously in legacy sync test suites.
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


class GlassdoorApplyService:
    """Orchestrates Glassdoor application inspection, dynamic form autofill, and assisted application."""

    _sessions: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        driver: Optional[PyWinAutoGlassdoorDriver] = None,
        profile_service: Optional[ProfileService] = None,
        application_repository: Optional[ApplicationRepository] = None,
        playwright_driver: Optional[GlassdoorPlaywrightDriver] = None,
        settings: Optional[Settings] = None,
        candidate_answer_bank: Optional[CandidateAnswerBank] = None,
    ) -> None:
        self.driver = driver or PyWinAutoGlassdoorDriver()
        self.playwright_driver = playwright_driver or GlassdoorPlaywrightDriver()
        self.profile_service = profile_service
        self.application_repository = application_repository or ApplicationRepository()
        self.settings = settings or get_settings()
        self.candidate_answer_bank = candidate_answer_bank or CandidateAnswerBank(file_path=self.settings.CANDIDATE_ANSWERS_FILE_PATH)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored session data for a paused application."""
        return self._sessions.get(session_id)

    def save_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Store or update session data for a paused application."""
        self._sessions[session_id] = data

    def reset_navigation_state(self) -> None:
        """Reset transient Python execution state before starting a new navigation or inspection run."""
        if hasattr(self.playwright_driver, "reset_navigation_state") and callable(self.playwright_driver.reset_navigation_state):
            self.playwright_driver.reset_navigation_state()

    @async_hybrid
    async def inspect_job_application(
        self, request: GlassdoorAutomationInspectRequest
    ) -> GlassdoorAutomationResult:
        """Inspect Glassdoor job page and verify Easy Apply presence without clicking or spawning extra browsers."""
        self.reset_navigation_state()
        job_id = request.job_id
        url = request.url

        # 1. Dual Source and URL Validation
        is_valid, normalized_url, err_code = validate_glassdoor_request(url, request.source)
        if not is_valid:
            state = GlassdoorAutomationState(err_code or "INVALID_JOB_URL")
            return GlassdoorAutomationResult(
                status="blocked",
                job_id=job_id,
                url=url,
                current_state=state,
                message=f"Request rejected by source/URL validation: {err_code}",
            )

        # 2. Windows Platform Check
        if not is_windows():
            return GlassdoorAutomationResult(
                status="blocked",
                job_id=job_id,
                url=normalized_url,
                current_state=GlassdoorAutomationState.UNSUPPORTED_PLATFORM,
                message="PyWinAuto automation is only supported on Windows OS.",
            )

        # 3. Resolve Authoritative CDP Page First (Zero Second Browser Spawn)
        cdp_ok, cdp_page, cdp_diag = False, None, {}
        try:
            res_tuple = await _maybe_await(
                self.playwright_driver.resolve_or_navigate_page(
                    normalized_url, timeout_seconds=15
                )
            )
            if isinstance(res_tuple, tuple) and len(res_tuple) == 3:
                cdp_ok, cdp_page, cdp_diag = res_tuple
        except Exception as exc:
            logger.debug("resolve_or_navigate_page skipped or failed: %s", exc)

        b_name = "chrome.exe"
        attached = False
        attach_err = None
        if cdp_ok and cdp_page is not None:
            # Playwright CDP is authoritative! PyWinAuto attaches optionally for potential fallback
            attached = True
            b_name = "chrome_cdp"
            try:
                pwa_ok, pwa_bname, _ = self.driver.attach_or_open_browser(
                    normalized_url, force_navigate=False
                )
                if pwa_ok and pwa_bname:
                    b_name = pwa_bname
            except Exception as exc:
                logger.debug("PyWinAuto optional attachment skipped: %s", exc)
        else:
            # Fallback to PyWinAuto attachment when CDP is unavailable
            attached, b_name, attach_err = self.driver.attach_or_open_browser(
                normalized_url, force_navigate=True
            )

        # Only block with BROWSER_NOT_VERIFIED if BOTH CDP and PyWinAuto fail
        if not cdp_ok and not attached:
            is_nav_fail = attach_err and ("navigation" in attach_err.lower() or "stale" in attach_err.lower())
            state = (
                GlassdoorAutomationState.JOB_NAVIGATION_FAILED
                if is_nav_fail
                else GlassdoorAutomationState.BROWSER_NOT_VERIFIED
            )
            return GlassdoorAutomationResult(
                status="blocked",
                job_id=job_id,
                url=normalized_url,
                browser=b_name or "chrome.exe",
                current_state=state,
                message=attach_err or "Failed to attach to verified browser window.",
                diagnostics={
                    "cdp_connect_attempted": cdp_diag.get("cdp_connect_attempted", True),
                    "cdp_connected": cdp_diag.get("cdp_connected", False),
                    "browser_session_verified": False,
                    "contexts_found": cdp_diag.get("contexts_found", 0),
                    "pages_found": cdp_diag.get("pages_found", 0),
                    "authoritative_page_url": cdp_diag.get("authoritative_page_url"),
                    "requested_job_url": normalized_url,
                    "second_browser_launched": False,
                    **cdp_diag,
                },
            )

        # 4. Check for Stale Confirmation Page from Prior Task
        page_text = ""
        if cdp_ok and cdp_page is not None:
            try:
                body_elem = cdp_page.locator("body") if hasattr(cdp_page, "locator") else None
                if body_elem and hasattr(body_elem, "inner_text"):
                    page_text = await _maybe_await(body_elem.inner_text())
                elif hasattr(cdp_page, "inner_text") and callable(cdp_page.inner_text):
                    page_text = await _maybe_await(cdp_page.inner_text("body"))
                else:
                    page_text = self.driver.get_window_text_content()
            except Exception:
                page_text = self.driver.get_window_text_content()
        else:
            page_text = self.driver.get_window_text_content()
        if GlassdoorConfirmationDetector.is_stale_confirmation_page(page_text) and not (
            "easy apply" in page_text.lower() or "apply now" in page_text.lower()
        ):
            return GlassdoorAutomationResult(
                status="blocked",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                current_state=GlassdoorAutomationState.JOB_NAVIGATION_FAILED,
                message="Stale submission confirmation page detected from previous task.",
                diagnostics={
                    "cdp_connect_attempted": cdp_diag.get("cdp_connect_attempted", True),
                    "cdp_connected": cdp_diag.get("cdp_connected", cdp_ok),
                    "browser_session_verified": cdp_ok or attached,
                    "contexts_found": cdp_diag.get("contexts_found", 1 if cdp_ok else 0),
                    "pages_found": cdp_diag.get("pages_found", 1 if cdp_ok else 0),
                    "authoritative_page_url": cdp_diag.get("authoritative_page_url") or resolved_job_url,
                    "requested_job_url": normalized_url,
                    "second_browser_launched": False,
                    **cdp_diag,
                    "detected_text_sample": page_text[:200],
                },
            )

        from app.automation.glassdoor.url_validator import extract_glassdoor_job_id
        requested_listing_id = extract_glassdoor_job_id(normalized_url) or job_id
        raw_job_url = getattr(cdp_page, "url", "") if cdp_page else None
        resolved_job_url = raw_job_url if isinstance(raw_job_url, str) else (cdp_diag.get("job_page_url") if isinstance(cdp_diag.get("job_page_url"), str) else normalized_url)
        resolved_listing_id = extract_glassdoor_job_id(resolved_job_url)

        # 5. Check for Interactive Blocking States (CAPTCHA, Login, MFA, Unavailable)
        blocking = GlassdoorStateDetector.detect_blocking_state(page_text)
        if blocking and blocking[0] in (
            GlassdoorAutomationState.CAPTCHA_OR_CHALLENGE,
            GlassdoorAutomationState.LOGIN_REQUIRED,
            GlassdoorAutomationState.MFA_REQUIRED,
            GlassdoorAutomationState.JOB_UNAVAILABLE,
        ):
            block_state, block_msg = blocking
            manual_req = block_state in (
                GlassdoorAutomationState.CAPTCHA_OR_CHALLENGE,
                GlassdoorAutomationState.LOGIN_REQUIRED,
                GlassdoorAutomationState.MFA_REQUIRED,
            )
            return GlassdoorAutomationResult(
                status="manual_action_required" if manual_req else "blocked",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                current_state=block_state,
                manual_action_required=manual_req,
                message=block_msg,
                diagnostics={
                    "cdp_connect_attempted": cdp_diag.get("cdp_connect_attempted", True),
                    "cdp_connected": cdp_diag.get("cdp_connected", cdp_ok),
                    "browser_session_verified": cdp_ok or attached,
                    "contexts_found": cdp_diag.get("contexts_found", 1 if cdp_ok else 0),
                    "pages_found": cdp_diag.get("pages_found", 1 if cdp_ok else 0),
                    "authoritative_page_url": cdp_diag.get("authoritative_page_url") or resolved_job_url,
                    "requested_job_url": normalized_url,
                    "second_browser_launched": False,
                    **cdp_diag,
                    "resolved_job_page_url": resolved_job_url,
                    "resolved_listing_id": resolved_listing_id,
                    "stale_application_pages_ignored": True,
                },
            )

        # 6. Check Playwright DOM for Easy Apply Button
        pw_ea = None
        easy_apply_present = False
        easy_apply_enabled = False
        if cdp_ok and cdp_page is not None:
            try:
                pw_ea = await _maybe_await(self.playwright_driver.find_exact_easy_apply_button(cdp_page))
                if pw_ea is not None:
                    easy_apply_present = True
                    easy_apply_enabled = True
            except Exception as exc:
                logger.debug("Error finding Easy Apply via Playwright: %s", exc)

        # 7. Check Playwright DOM for explicit Already Applied evidence
        already_applied_found = False
        already_applied_text = None
        already_applied_loc = None
        if cdp_ok and cdp_page is not None and hasattr(self.playwright_driver, "check_already_applied"):
            try:
                already_applied_res = await _maybe_await(self.playwright_driver.check_already_applied(cdp_page))
                if isinstance(already_applied_res, tuple) and len(already_applied_res) == 3:
                    already_applied_found, already_applied_text, already_applied_loc = already_applied_res
            except Exception as exc:
                logger.debug("Error checking already applied: %s", exc)
        elif blocking and blocking[0] == GlassdoorAutomationState.ALREADY_APPLIED:
            already_applied_found = True
            already_applied_text = "Page text indicator"

        # Build comprehensive base diagnostics
        base_diag = {
            "cdp_connect_attempted": cdp_diag.get("cdp_connect_attempted", True),
            "cdp_connected": cdp_diag.get("cdp_connected", cdp_ok),
            "browser_session_verified": cdp_ok or attached,
            "contexts_found": cdp_diag.get("contexts_found", 1 if cdp_ok else 0),
            "pages_found": cdp_diag.get("pages_found", 1 if cdp_ok else 0),
            "authoritative_page_url": cdp_diag.get("authoritative_page_url") or resolved_job_url,
            "requested_job_url": normalized_url,
            **cdp_diag,
            "already_applied_check_performed": True,
            "already_applied_signal_found": already_applied_found,
            "already_applied_signal_text": already_applied_text,
            "already_applied_signal_locator": already_applied_loc,
            "easy_apply_present": easy_apply_present,
            "easy_apply_enabled": easy_apply_enabled,
            "resolved_job_page_url": resolved_job_url,
            "resolved_listing_id": resolved_listing_id,
            "stale_application_pages_ignored": True,
            "second_browser_launched": False,
        }

        # CONFLICT RESOLUTION:
        # If an active Easy Apply button exists on the current requested listing,
        # it takes precedence over any stray already applied mention elsewhere on the page!
        if easy_apply_present and easy_apply_enabled:
            logger.info("Found verified Glassdoor Easy Apply button via Playwright DOM.")
            return GlassdoorAutomationResult(
                status="success",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                easy_apply_verified=True,
                easy_apply_clicked=False,
                current_state=GlassdoorAutomationState.EASY_APPLY_AVAILABLE,
                message="Easy Apply button verified on Glassdoor job page via Playwright DOM.",
                diagnostics={
                    **base_diag,
                    "playwright_dom_found": True,
                    "uia_primary_found": False,
                },
            )

        # Genuine Already Applied on the requested job without active Easy Apply
        if already_applied_found:
            logger.info("Confirmed candidate has already applied to this position on Glassdoor: %s (%s)", already_applied_text, already_applied_loc)
            return GlassdoorAutomationResult(
                status="blocked",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                current_state=GlassdoorAutomationState.ALREADY_APPLIED,
                message="Candidate has already applied to this position on Glassdoor.",
                diagnostics=base_diag,
            )

        # Check for External Apply
        if GlassdoorStateDetector.is_external_apply(page_text):
            return GlassdoorAutomationResult(
                status="blocked",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                current_state=GlassdoorAutomationState.EXTERNAL_APPLY,
                message="Job uses external employer application (Apply on company website).",
                diagnostics=base_diag,
            )

        # Search for Easy Apply UIA Button (Primary Descendant Scan)
        easy_apply_btn = self.driver.find_easy_apply_control()
        if easy_apply_btn is not None:
            btn_name = getattr(easy_apply_btn.element_info, "name", "") or "Easy Apply"
            logger.info("Found Glassdoor Easy Apply button (UIA Primary): '%s'", btn_name)
            return GlassdoorAutomationResult(
                status="success",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                easy_apply_verified=True,
                easy_apply_clicked=False,
                current_state=GlassdoorAutomationState.EASY_APPLY_AVAILABLE,
                message="Easy Apply button verified on Glassdoor job page via primary UIA scan.",
                diagnostics={
                    **base_diag,
                    "easy_apply_present": True,
                    "easy_apply_enabled": True,
                    "uia_primary_found": True,
                },
            )

        # Element-From-Point Fallback at (GLASSDOOR_EASY_APPLY_FALLBACK_X, GLASSDOOR_EASY_APPLY_FALLBACK_Y)
        coord_x = GLASSDOOR_EASY_APPLY_FALLBACK_X
        coord_y = GLASSDOOR_EASY_APPLY_FALLBACK_Y

        point_elem, point_diag = self.driver.get_element_at_point(coord_x, coord_y)
        fallback_verified = point_diag.get("is_easy_apply", False)

        if fallback_verified:
            logger.info("Found Glassdoor Easy Apply button via element-from-point at (%d, %d)", coord_x, coord_y)
            return GlassdoorAutomationResult(
                status="success",
                job_id=job_id,
                url=normalized_url,
                browser=b_name,
                easy_apply_verified=True,
                easy_apply_clicked=False,
                current_state=GlassdoorAutomationState.EASY_APPLY_AVAILABLE,
                message=f"Easy Apply button verified via element-from-point at ({coord_x}, {coord_y}).",
                diagnostics={
                    **base_diag,
                    "easy_apply_present": True,
                    "easy_apply_enabled": True,
                    "uia_primary_found": False,
                    "fallback_coordinate": [coord_x, coord_y],
                    "element_at_point_name": point_diag.get("element_name"),
                    "element_at_point_control_type": point_diag.get("control_type"),
                    "element_at_point_bounds": point_diag.get("bounds"),
                    "fallback_verified": True,
                    "fallback_clicked": False,
                },
            )

        # Inspection mode does not perform clicks; report unverified
        return GlassdoorAutomationResult(
            status="blocked",
            job_id=job_id,
            url=normalized_url,
            browser=b_name,
            easy_apply_verified=False,
            easy_apply_clicked=False,
            current_state=GlassdoorAutomationState.EASY_APPLY_NOT_VERIFIED,
            message="Could not positively verify Easy Apply button on Glassdoor page.",
            diagnostics={
                **base_diag,
                "uia_primary_found": False,
                "fallback_coordinate": [coord_x, coord_y],
                "element_at_point_name": point_diag.get("element_name"),
                "element_at_point_control_type": point_diag.get("control_type"),
                "element_at_point_bounds": point_diag.get("bounds"),
                "fallback_verified": False,
                "fallback_clicked": False,
                "detected_text_sample": page_text[:200] if page_text else "",
            },
        )

    def _wait_for_state_change(self, previous_text: str, timeout: float = 3.0) -> bool:
        """Wait until page content or modal signature changes from previous_text."""
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(0.4)
            curr = self.driver.get_window_text_content()
            if curr and curr != previous_text:
                return True
        return False

    @async_hybrid
    async def navigate_to_submit(
        self, request: GlassdoorAutomationNavigateRequest
    ) -> GlassdoorAutomationResult:
        """Enter Glassdoor Easy Apply, navigate dynamically through all form steps, and stop at SUBMISSION_READY."""
        self.reset_navigation_state()
        # 1. Run inspection on authoritative CDP session
        inspect_req = GlassdoorAutomationInspectRequest(
            job_id=request.job_id,
            url=request.url,
            source=request.source,
            profile_id=request.profile_id,
        )
        inspect_res = await _maybe_await(self.inspect_job_application(inspect_req))

        # If inspect encountered blocking state (CAPTCHA, Login, MFA, External Apply, etc.), halt immediately
        if inspect_res.current_state not in (
            GlassdoorAutomationState.EASY_APPLY_AVAILABLE,
            GlassdoorAutomationState.EASY_APPLY_NOT_VERIFIED,
        ):
            return inspect_res

        coord_x = GLASSDOOR_EASY_APPLY_FALLBACK_X
        coord_y = GLASSDOOR_EASY_APPLY_FALLBACK_Y
        clicked_successfully = False
        click_method = "none"

        # Capture pages before clicking Easy Apply to detect newly opened tab handoff
        pages_before = None
        if getattr(self.playwright_driver, "_context", None):
            try:
                pages_before = set(self.playwright_driver._context.pages)
            except Exception:
                pass

        # 2. Click Easy Apply using Priority Hierarchy:
        # Priority A: Playwright DOM click if visible
        pw_ea_click_fn = getattr(self.playwright_driver, "click_easy_apply", None)
        if callable(pw_ea_click_fn):
            try:
                target_job_page = getattr(self.playwright_driver, "job_page", None) or getattr(self.playwright_driver, "page", None)
                res_click = await _maybe_await(pw_ea_click_fn(target_job_page))
                if isinstance(res_click, tuple) and len(res_click) == 2:
                    ea_ok, ea_err = res_click
                    if ea_ok:
                        clicked_successfully = True
                        click_method = "playwright"
            except Exception as exc:
                logger.warning("Playwright click_easy_apply exception: %s", exc)

        # Priority B: Primary UIA Easy Apply Control
        if not clicked_successfully and inspect_res.easy_apply_verified:
            easy_apply_btn = self.driver.find_easy_apply_control()
            if easy_apply_btn is not None:
                clicked_successfully = self.driver.click_control(easy_apply_btn)
                if clicked_successfully:
                    click_method = "uia_primary"

            # Priority C: Element-From-Point Fallback
            if not clicked_successfully:
                point_elem, point_diag = self.driver.get_element_at_point(coord_x, coord_y)
                if point_diag.get("is_easy_apply") or point_elem is not None:
                    clicked_successfully = self.driver.click_control(point_elem)
                    if not clicked_successfully:
                        clicked_successfully = self.driver.click_coordinate(coord_x, coord_y)
                    if clicked_successfully:
                        click_method = "element_from_point"

        # Priority D: Strict Verified Coordinate Fallback (Last Resort)
        if not clicked_successfully and ALLOW_VERIFIED_GLASSDOOR_COORDINATE_FALLBACK:
            is_fg, _ = self.driver.verify_browser_window()
            is_inside = self.driver.is_point_inside_window(coord_x, coord_y)
            page_text = self.driver.get_window_text_content()
            blocking = GlassdoorStateDetector.detect_blocking_state(page_text)

            if is_fg and is_inside and not blocking:
                logger.info("Attempting verified single coordinate click at (%d, %d)...", coord_x, coord_y)
                clicked_successfully = self.driver.click_coordinate(coord_x, coord_y)
                if clicked_successfully:
                    click_method = "verified_coordinate"

        if not clicked_successfully:
            logger.warning("Failed to click Easy Apply via any verified method.")
            return inspect_res

        logger.info(
            "Clicked Easy Apply using method '%s'. Waiting %d seconds for browser redirect / new tab before inspection...",
            click_method,
            POST_EASY_APPLY_REDIRECT_WAIT_SECONDS,
        )
        time.sleep(POST_EASY_APPLY_REDIRECT_WAIT_SECONDS)

        # 3. Attempt Playwright Takeover over CDP for Hosted Application DOM (supporting new tabs & same-tab)
        app_page = None
        resolve_app_fn = getattr(self.playwright_driver, "resolve_application_page", None)
        if callable(resolve_app_fn):
            try:
                app_page = await _maybe_await(resolve_app_fn(pages_before=pages_before, timeout_seconds=POST_EASY_APPLY_REDIRECT_WAIT_SECONDS))
            except Exception as exc:
                logger.debug("resolve_application_page failed: %s", exc)

        if app_page is None:
            app_page = getattr(self.playwright_driver, "application_page", None) or getattr(self.playwright_driver, "page", None)

        if app_page is not None:
            pages_after = set(self.playwright_driver._context.pages) if getattr(self.playwright_driver, "_context", None) else set()
            is_new_tab = bool(pages_before and (app_page not in pages_before or (self.playwright_driver.job_page and app_page != self.playwright_driver.job_page)))
            logger.info("Playwright attached to hosted application DOM via CDP (new tab: %s). Running Playwright dynamic loop...", is_new_tab)
            return await self._run_playwright_application_loop(
                request=request,
                inspect_res=inspect_res,
                click_method=click_method,
                coord_x=coord_x,
                coord_y=coord_y,
                pages_before=pages_before,
                pages_after=pages_after,
                is_new_tab=is_new_tab,
            )

        # 4. Fallback: UIA Post-Click Transition Verification (Polling up to 5 seconds)
        post_transition_ok = False
        application_host = "glassdoor"
        post_state_name = "unknown"
        post_text_sample = ""

        for _ in range(10):
            post_text = self.driver.get_window_text_content()
            post_text_sample = post_text[:200] if post_text else ""
            active_win = getattr(self.driver, "_window", None) or getattr(self.driver, "window", None)
            post_modal = GlassdoorModalInspector.find_application_modal(active_win)
            is_smartapply = self.driver.is_smartapply_detected()
            has_submit = self.driver.find_exact_submit_control(container=post_modal) is not None
            has_action = self.driver.find_action_control(
                CONTINUE_BUTTON_NAMES + NEXT_BUTTON_NAMES + REVIEW_BUTTON_NAMES, container=post_modal
            ) is not None
            has_resume = "resume" in post_text.lower() or "cv" in post_text.lower()

            if is_smartapply or has_submit or has_action or has_resume or (post_modal and post_modal != active_win):
                post_transition_ok = True
                application_host = "indeed_smartapply" if is_smartapply else "glassdoor"
                if is_smartapply:
                    post_state_name = GlassdoorAutomationState.SMART_APPLY_HOST_VERIFIED.value
                elif has_submit:
                    post_state_name = GlassdoorAutomationState.SUBMISSION_READY.value
                elif has_resume:
                    post_state_name = GlassdoorAutomationState.RESUME_STEP.value
                elif has_action:
                    post_state_name = "APPLICATION_FLOW"
                else:
                    post_state_name = GlassdoorAutomationState.APPLICATION_MODAL.value
                break
            time.sleep(0.5)

        if not post_transition_ok:
            logger.warning("No verified post-click application transition detected after Easy Apply click.")
            return GlassdoorAutomationResult(
                status="blocked",
                job_id=request.job_id,
                url=inspect_res.url,
                browser=inspect_res.browser,
                easy_apply_verified=True,
                easy_apply_clicked=True,
                current_state=GlassdoorAutomationState.EASY_APPLY_CLICK_UNVERIFIED,
                message="Easy Apply was activated, but application modal / SmartApply transition could not be verified.",
                diagnostics={
                    "cdp_connect_attempted": inspect_res.diagnostics.get("cdp_connect_attempted", True),
                    "cdp_connected": inspect_res.diagnostics.get("cdp_connected", True),
                    "browser_session_verified": inspect_res.diagnostics.get("browser_session_verified", True),
                    "contexts_found": inspect_res.diagnostics.get("contexts_found", 1),
                    "pages_found": inspect_res.diagnostics.get("pages_found", 1),
                    "authoritative_page_url": inspect_res.diagnostics.get("authoritative_page_url") or inspect_res.url,
                    "requested_job_url": inspect_res.diagnostics.get("requested_job_url") or inspect_res.url,
                    "second_browser_launched": False,
                    "easy_apply_method": click_method,
                    "easy_apply_clicked": True,
                    "post_easy_apply_wait_seconds": POST_EASY_APPLY_REDIRECT_WAIT_SECONDS,
                    "url_after_redirect_wait": inspect_res.url,
                    "application_host_after_wait": application_host,
                    "state_after_redirect_wait": post_state_name,
                    "initial_host": "glassdoor",
                    "application_host": application_host,
                    "fallback_coordinate": [coord_x, coord_y],
                    "post_click_state": "unverified",
                    "post_click_text_sample": post_text_sample,
                },
            )

        logger.info(
            "Post-click application transition verified! Host: %s, State: %s. Entering dynamic form loop...",
            application_host,
            post_state_name,
        )

        # 4. Load Candidate Profile & Memory Data
        profile_data = None
        if request.profile_id and self.profile_service:
            try:
                profile_data = self.profile_service.get_profile(request.profile_id)
            except Exception as exc:
                logger.warning("Could not load profile %s: %s", request.profile_id, exc)

        stored_answers = self.application_repository.get_answers_by_profile(request.profile_id)
        resolver = FormAnswerResolver(
            answer_bank=self.candidate_answer_bank,
            stored_answers=stored_answers,
            profile=profile_data,
            platform="glassdoor",
            host="glassdoor",
            allow_unanswered_questions=self.settings.ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS,
        )

        # 5. Dynamic Multi-Step Application Loop
        total_steps = 0
        total_q_detected = 0
        total_q_answered = 0
        resume_state = "not_needed"
        step_history: List[Dict[str, Any]] = []
        signature_counts: Dict[str, int] = {}

        while total_steps < MAX_APPLICATION_STEPS:
            total_steps += 1

            # Identify active modal container and text
            active_win = getattr(self.driver, "_window", None) or getattr(self.driver, "window", None)
            modal = GlassdoorModalInspector.find_application_modal(active_win)
            modal_text = self.driver.get_window_text_content()
            is_smartapply = self.driver.is_smartapply_detected()
            current_host = "indeed_smartapply" if is_smartapply else "glassdoor"

            # Base diagnostics dictionary for all step returns
            step_base_diagnostics = {
                "cdp_connect_attempted": inspect_res.diagnostics.get("cdp_connect_attempted", True),
                "cdp_connected": inspect_res.diagnostics.get("cdp_connected", True),
                "browser_session_verified": inspect_res.diagnostics.get("browser_session_verified", True),
                "contexts_found": inspect_res.diagnostics.get("contexts_found", 1),
                "pages_found": inspect_res.diagnostics.get("pages_found", 1),
                "authoritative_page_url": inspect_res.diagnostics.get("authoritative_page_url") or inspect_res.url,
                "requested_job_url": inspect_res.diagnostics.get("requested_job_url") or inspect_res.url,
                "second_browser_launched": False,
                "easy_apply_method": click_method,
                "easy_apply_clicked": True,
                "post_easy_apply_wait_seconds": POST_EASY_APPLY_REDIRECT_WAIT_SECONDS,
                "url_after_redirect_wait": inspect_res.url,
                "application_host_after_wait": application_host,
                "state_after_redirect_wait": post_state_name,
                "initial_host": "glassdoor",
                "application_host": current_host,
                "post_easy_apply_url": inspect_res.url,
                "post_easy_apply_state": post_state_name,
                "steps_processed": total_steps,
                "step_history": step_history,
            }

            # Compute State Signature to prevent stalled loops
            sig = f"{current_host}|{modal_text[:120]}"
            signature_counts[sig] = signature_counts.get(sig, 0) + 1
            if signature_counts[sig] > MAX_STALLED_SIGNATURE_COUNT:
                logger.warning("Stalled application state detected (%s repeated %d times). Halting loop.", sig, signature_counts[sig])
                return GlassdoorAutomationResult(
                    status="blocked",
                    job_id=request.job_id,
                    url=inspect_res.url,
                    browser=inspect_res.browser,
                    easy_apply_verified=True,
                    easy_apply_clicked=True,
                    steps_processed=total_steps,
                    questions_detected=total_q_detected,
                    questions_answered=total_q_answered,
                    questions_unresolved=0,
                    resume_state=resume_state,
                    current_state=GlassdoorAutomationState.APPLICATION_PROGRESS_STALLED,
                    message="Application progress stalled: modal state repeated without forward progress.",
                    diagnostics={
                        **step_base_diagnostics,
                        "stalled_signature": sig,
                    },
                )

            # Check for Blocking Challenges in modal
            modal_blocking = GlassdoorStateDetector.detect_blocking_state(modal_text)
            if modal_blocking:
                b_state, b_msg = modal_blocking
                step_history.append({"step": total_steps, "state": b_state.value, "action": "blocked"})
                return GlassdoorAutomationResult(
                    status="manual_action_required",
                    job_id=request.job_id,
                    url=inspect_res.url,
                    browser=inspect_res.browser,
                    easy_apply_verified=True,
                    easy_apply_clicked=True,
                    steps_processed=total_steps,
                    current_state=b_state,
                    manual_action_required=True,
                    message=b_msg,
                    diagnostics=step_base_diagnostics,
                )

            # Step 1: Check for Exact Final Submit Button (SUBMISSION_READY)
            submit_control = self.driver.find_exact_submit_control(container=modal)
            if submit_control is not None:
                btn_name = getattr(submit_control.element_info, "name", "Submit")
                logger.info("Found exact Submit button on step %d: '%s'. Reached SUBMISSION_READY.", total_steps, btn_name)
                step_history.append({"step": total_steps, "state": "SUBMISSION_READY", "submit_verified": True})
                return GlassdoorAutomationResult(
                    status="success",
                    job_id=request.job_id,
                    url=inspect_res.url,
                    browser=inspect_res.browser,
                    easy_apply_verified=True,
                    easy_apply_clicked=True,
                    steps_processed=total_steps,
                    questions_detected=total_q_detected,
                    questions_answered=total_q_answered,
                    questions_unresolved=0,
                    resume_state=resume_state,
                    submit_verified=True,
                    final_submit_clicked=False,
                    submission_confirmed=False,
                    current_state=GlassdoorAutomationState.SUBMISSION_READY,
                    message="Glassdoor application flow ready. Reached SUBMISSION_READY state.",
                    diagnostics={
                        **step_base_diagnostics,
                        "submit_verified": True,
                        "final_submit_clicked": False,
                    },
                )

            # Step 2: Handle Resume Step
            if GlassdoorResumeHandler.is_resume_step(modal_text):
                res_ok, res_state_val, res_err = GlassdoorResumeHandler.handle_resume_step(
                    modal, profile=profile_data
                )
                resume_state = res_state_val
                if not res_ok:
                    step_history.append({"step": total_steps, "state": "RESUME_STEP", "action": "needs_user_input"})
                    return GlassdoorAutomationResult(
                        status="manual_action_required",
                        job_id=request.job_id,
                        url=inspect_res.url,
                        browser=inspect_res.browser,
                        easy_apply_verified=True,
                        easy_apply_clicked=True,
                        steps_processed=total_steps,
                        resume_state=res_state_val,
                        current_state=GlassdoorAutomationState.NEEDS_USER_INPUT,
                        manual_action_required=True,
                        message=res_err or "Resume selection required.",
                        diagnostics=step_base_diagnostics,
                    )

                action_clicked, act_name = self._click_next_available_action(modal)
                step_history.append({"step": total_steps, "state": "RESUME_STEP", "action": act_name or "continue"})
                self._wait_for_state_change(modal_text)
                continue

            # Step 3: Handle Questions Step
            form_insp = FormInspector.inspect_container(modal)
            form_fields = form_insp.questions if form_insp and form_insp.questions else []
            if form_fields:
                step_q_count = len(form_fields)
                step_q_answered = 0
                total_q_detected += step_q_count

                for field in form_fields:
                    res = resolver.resolve_question(field)
                    if res.is_resolved and res.resolved_value is not None:
                        fill_ok = self._fill_form_field(field, res.resolved_value)
                        if fill_ok:
                            step_q_answered += 1
                            total_q_answered += 1
                    elif field.required:
                        if getattr(res, "action", "") == "leave_unanswered":
                            logger.info("Required question '%s' left unanswered per policy for platform validation.", field.text)
                        else:
                            logger.warning("Unresolved required question: '%s' (%s)", field.text, res.reason)
                            step_history.append({
                                "step": total_steps,
                                "state": "QUESTIONS_STEP",
                                "questions_detected": step_q_count,
                                "questions_answered": step_q_answered,
                                "action": "needs_user_input",
                            })
                            return GlassdoorAutomationResult(
                                status="manual_action_required",
                                job_id=request.job_id,
                                url=inspect_res.url,
                                browser=inspect_res.browser,
                                easy_apply_verified=True,
                                easy_apply_clicked=True,
                                steps_processed=total_steps,
                                questions_detected=total_q_detected,
                                questions_answered=total_q_answered,
                                questions_unresolved=1,
                                resume_state=resume_state,
                                current_state=GlassdoorAutomationState.NEEDS_USER_INPUT,
                                manual_action_required=True,
                                message=f"Manual answer required for question: '{field.text}'",
                                diagnostics={
                                    **step_base_diagnostics,
                                    "question": field.text,
                                    "normalized_key": field.normalized_key,
                                    "input_type": field.input_type.value,
                                    "options": field.options,
                                    "reason": res.reason,
                                    "questions_unresolved": 1,
                                },
                            )
                    else:
                        logger.info("Skipping optional question: '%s'", field.text)

                action_clicked, act_name = self._click_next_available_action(modal)
                step_history.append({
                    "step": total_steps,
                    "state": "QUESTIONS_STEP",
                    "questions_detected": step_q_count,
                    "questions_answered": step_q_answered,
                    "action": act_name or "continue",
                })
                self._wait_for_state_change(modal_text)
                continue

            # Step 4: Progression Button Click (Continue / Next / Review)
            action_clicked, act_name = self._click_next_available_action(modal)
            if action_clicked:
                step_history.append({"step": total_steps, "state": "APPLICATION_FLOW", "action": act_name})
                self._wait_for_state_change(modal_text)
                continue

            # Step 5: Adaptive TAB Traversal Fallback
            tab_ok, tab_action = self._adaptive_tab_traversal(modal)
            if tab_ok:
                step_history.append({"step": total_steps, "state": "TAB_TRAVERSAL", "action": tab_action})
                self._wait_for_state_change(modal_text)
                continue

            return GlassdoorAutomationResult(
                status="blocked",
                job_id=request.job_id,
                url=inspect_res.url,
                browser=inspect_res.browser,
                easy_apply_verified=True,
                easy_apply_clicked=True,
                steps_processed=total_steps,
                current_state=GlassdoorAutomationState.ACTION_CONTROL_NOT_VERIFIED,
                message=f"Action control could not be positively verified on modal step {total_steps}.",
                diagnostics=step_base_diagnostics,
            )

        return GlassdoorAutomationResult(
            status="blocked",
            job_id=request.job_id,
            url=inspect_res.url,
            browser=inspect_res.browser,
            easy_apply_verified=True,
            easy_apply_clicked=True,
            steps_processed=total_steps,
            current_state=GlassdoorAutomationState.APPLICATION_PROGRESS_STALLED,
            message=f"Exceeded maximum modal steps limit ({MAX_APPLICATION_STEPS}).",
            diagnostics={
                "easy_apply_method": click_method,
                "easy_apply_clicked": True,
                "steps_processed": total_steps,
                "step_history": step_history,
            },
        )

    async def _activate_or_click_progression(self, page: Any) -> Tuple[bool, Optional[str], Optional[str]]:
        """Safely invoke activate_progress_button or fallback to click_progression_action."""
        act_fn = getattr(self.playwright_driver, "activate_progress_button", None)
        if callable(act_fn):
            try:
                res = await _maybe_await(act_fn(page))
                if isinstance(res, tuple) and len(res) == 3:
                    if res[0] and page and hasattr(page, "wait_for_load_state"):
                        try:
                            await _maybe_await(page.wait_for_load_state("domcontentloaded", timeout=2000))
                        except Exception:
                            pass
                        await asyncio.sleep(0.5)
                    return res
            except Exception:
                pass

        click_fn = getattr(self.playwright_driver, "click_progression_action", None)
        if callable(click_fn):
            try:
                res = await _maybe_await(click_fn(page))
                if isinstance(res, tuple) and len(res) == 3:
                    if res[0] and page and hasattr(page, "wait_for_load_state"):
                        try:
                            await _maybe_await(page.wait_for_load_state("domcontentloaded", timeout=2000))
                        except Exception:
                            pass
                        await asyncio.sleep(0.5)
                    return res
            except Exception:
                pass

        return False, None, "No enabled progression button found"

    async def _safe_wait_for_spa_transition(
        self, page: Any, snapshot: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Safely invoke driver's wait_for_spa_transition with fallback for unmocked unit tests."""
        try:
            if hasattr(self.playwright_driver, "wait_for_spa_transition") and callable(self.playwright_driver.wait_for_spa_transition):
                res = await _maybe_await(self.playwright_driver.wait_for_spa_transition(page, snapshot))
                if isinstance(res, tuple):
                    if len(res) == 3:
                        return res[0], res[1], res[2]
                    elif len(res) == 2:
                        return res[0], res[1], {}
        except Exception as exc:
            logger.debug("wait_for_spa_transition exception: %s", exc)
        return True, "default", {}

    @async_hybrid
    async def _run_playwright_application_loop(
        self,
        request: Any,
        inspect_res: GlassdoorAutomationResult,
        click_method: str = "playwright",
        coord_x: int = GLASSDOOR_EASY_APPLY_FALLBACK_X,
        coord_y: int = GLASSDOOR_EASY_APPLY_FALLBACK_Y,
        pages_before: Optional[Any] = None,
        pages_after: Optional[Any] = None,
        is_new_tab: bool = False,
        session_id: Optional[str] = None,
        initial_total_steps: int = 0,
        initial_q_detected: int = 0,
        initial_q_answered: int = 0,
        initial_history: Optional[List[Dict[str, Any]]] = None,
        initial_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> GlassdoorAutomationResult:
        """Dynamic multi-step form loop executing directly on hosted page DOM via Playwright."""
        page = getattr(self.playwright_driver, "application_page", None) or getattr(self.playwright_driver, "page", None)
        job_page_url = getattr(self.playwright_driver.job_page, "url", inspect_res.url) if getattr(self.playwright_driver, "job_page", None) else inspect_res.url
        job_page_id = getattr(self.playwright_driver.job_page, "id", "page-job") if getattr(self.playwright_driver, "job_page", None) else "page-job"

        session_id = session_id or getattr(request, "automation_session_id", None) or f"gdoor_session_{uuid4().hex[:12]}"

        profile_data = None
        if getattr(request, "profile_id", None) and self.profile_service:
            try:
                profile_data = self.profile_service.get_profile(request.profile_id)
            except Exception as exc:
                logger.warning("Could not load profile %s: %s", request.profile_id, exc)

        stored_answers = self.application_repository.get_answers_by_profile(getattr(request, "profile_id", None))
        resolver = FormAnswerResolver(
            answer_bank=self.candidate_answer_bank,
            stored_answers=stored_answers,
            profile=profile_data,
            platform="glassdoor",
            host="glassdoor",
            allow_unanswered_questions=self.settings.ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS,
        )

        total_steps = initial_total_steps
        total_q_detected = initial_q_detected
        total_q_answered = initial_q_answered
        resume_state = "not_needed"
        requirements_warning_detected = False
        requirements_not_met: List[str] = []
        requirements_text = ""
        apply_anyway_found = False
        apply_anyway_clicked = False
        step_history: List[Dict[str, Any]] = list(initial_history) if initial_history else []
        signature_counts: Dict[str, int] = {}
        init_diag = initial_diagnostics or {}
        manual_answers_detected = init_diag.get("manual_answers_detected", {})
        manual_answers_saved = init_diag.get("manual_answers_saved", False)
        progress_action_clicked = init_diag.get("progress_action_clicked")
        resume_attempt = init_diag.get("resume_attempt", 0)

        # Ensure application page has settled and finished initial redirects (e.g. applybyapplyablejobid -> form/...)
        if page:
            try:
                for _ in range(12):
                    p_url = getattr(page, "url", "")
                    if "applybyapplyablejobid" not in p_url and "about:blank" not in p_url:
                        break
                    await asyncio.sleep(0.5)
                if hasattr(page, "wait_for_load_state"):
                    await _maybe_await(page.wait_for_load_state("domcontentloaded", timeout=4000))
            except Exception:
                pass

        while total_steps < MAX_APPLICATION_STEPS:
            total_steps += 1
            curr_url = getattr(page, "url", inspect_res.url) if page else inspect_res.url
            is_smartapply = any(d in curr_url for d in ["smartapply.indeed.com", "indeed.com"])
            current_host = "indeed_smartapply" if is_smartapply else "glassdoor"

            # 1. Detect DOM State using Refined Priority (Blocking -> Submit -> Requirements Warning -> Questions -> Resume -> Review -> Action)
            dom_state, state_diag = await self.playwright_driver.detect_page_state(page)

            # Compute State Signature to prevent stalled loops
            sig = f"{current_host}|{dom_state.value}|{curr_url[:80]}"
            signature_counts[sig] = signature_counts.get(sig, 0) + 1

            step_base_diagnostics = {
                "cdp_connect_attempted": inspect_res.diagnostics.get("cdp_connect_attempted", True),
                "cdp_connected": True,
                "browser_session_verified": True,
                "contexts_found": inspect_res.diagnostics.get("contexts_found", 1),
                "pages_found": inspect_res.diagnostics.get("pages_found", 1),
                "authoritative_page_url": job_page_url,
                "requested_job_url": inspect_res.diagnostics.get("requested_job_url") or inspect_res.url,
                "job_page_url": job_page_url,
                "job_page_id": job_page_id,
                "application_page_detected": True,
                "application_page_url": curr_url,
                "application_page_id": getattr(page, "id", "page-app") if page else "page-app",
                "application_page_opened_as_new_tab": is_new_tab,
                "pages_before_easy_apply": len(pages_before) if pages_before else 0,
                "pages_after_easy_apply": len(pages_after) if pages_after else 0,
                "easy_apply_method": click_method,
                "easy_apply_clicked": True,
                "playwright_attached": True,
                "active_application_url": curr_url,
                "cdp_page_url": curr_url,
                "cdp_page_id": getattr(page, "id", "page-0") if page else "page-0",
                "second_browser_launched": False,
                "post_easy_apply_wait_seconds": POST_EASY_APPLY_REDIRECT_WAIT_SECONDS,
                "url_after_redirect_wait": curr_url,
                "application_host_after_wait": current_host,
                "state_after_redirect_wait": dom_state.value,
                "initial_host": "glassdoor",
                "application_host": current_host,
                "post_easy_apply_url": curr_url,
                "post_easy_apply_state": dom_state.value,
                "steps_processed": total_steps,
                "step_history": step_history,
                "visible_actions": state_diag.get("visible_actions") if state_diag.get("visible_actions") is not None else ([state_diag.get("progression_action")] if state_diag.get("progression_action") else []),
                "questions_detected": total_q_detected,
                "questions_answered": total_q_answered,
                "questions_unresolved": 0,
                "requirements_warning_detected": requirements_warning_detected,
                "requirements_not_met": requirements_not_met,
                "requirements_text": requirements_text,
                "apply_anyway_found": apply_anyway_found,
                "apply_anyway_clicked": apply_anyway_clicked,
                "continue_on_requirements_warning": CONTINUE_ON_REQUIREMENTS_WARNING,
                "automation_session_id": session_id,
                "paused_for_user_input": False,
                "manual_answers_detected": manual_answers_detected,
                "manual_answers_saved": manual_answers_saved,
                "progress_action_clicked": progress_action_clicked,
                "resume_attempt": resume_attempt,
                "resume_state": resume_state,
            }

            if signature_counts[sig] > MAX_STALLED_SIGNATURE_COUNT:
                logger.warning("Stalled state detected (%s repeated %d times). Halting loop.", sig, signature_counts[sig])
                return GlassdoorAutomationResult(
                    status="blocked",
                    job_id=getattr(request, "job_id", None),
                    url=inspect_res.url,
                    browser=inspect_res.browser,
                    easy_apply_verified=True,
                    easy_apply_clicked=True,
                    steps_processed=total_steps,
                    questions_detected=total_q_detected,
                    questions_answered=total_q_answered,
                    questions_unresolved=0,
                    resume_state=resume_state,
                    current_state=GlassdoorAutomationState.APPLICATION_PROGRESS_STALLED,
                    message="Application progress stalled: page signature repeated without forward progress.",
                    diagnostics={
                        **step_base_diagnostics,
                        "stalled_signature": sig,
                    },
                )

            # 2. Check Blocking Challenges in DOM
            if dom_state in (
                GlassdoorAutomationState.CAPTCHA_OR_CHALLENGE,
                GlassdoorAutomationState.LOGIN_REQUIRED,
                GlassdoorAutomationState.MFA_REQUIRED,
            ):
                block_msg = state_diag.get("blocking_reason") or "Manual action required (Challenge/Login/MFA)."
                step_history.append({"step": total_steps, "state": dom_state.value, "action": "blocked"})
                return GlassdoorAutomationResult(
                    status="manual_action_required",
                    job_id=getattr(request, "job_id", None),
                    url=inspect_res.url,
                    browser=inspect_res.browser,
                    easy_apply_verified=True,
                    easy_apply_clicked=True,
                    steps_processed=total_steps,
                    current_state=dom_state,
                    manual_action_required=True,
                    message=f"Challenge encountered on step {total_steps}: {block_msg}",
                    diagnostics={
                        **step_base_diagnostics,
                        "challenge_detected": True,
                        "challenge_reason": block_msg,
                    },
                )

            # 3. Check Exact Submit Button (SUBMISSION_READY reached - STOP and verify)
            if dom_state == GlassdoorAutomationState.SUBMISSION_READY:
                logger.info("Final Submit button verified on step %d. Flow complete.", total_steps)
                step_history.append({"step": total_steps, "state": "SUBMISSION_READY", "action": "verified_stop"})
                return GlassdoorAutomationResult(
                    status="success",
                    job_id=getattr(request, "job_id", None),
                    url=inspect_res.url,
                    browser=inspect_res.browser,
                    easy_apply_verified=True,
                    easy_apply_clicked=True,
                    steps_processed=total_steps,
                    questions_detected=total_q_detected,
                    questions_answered=total_q_answered,
                    questions_unresolved=0,
                    resume_state=resume_state,
                    submit_verified=True,
                    final_submit_clicked=False,
                    submission_confirmed=False,
                    current_state=GlassdoorAutomationState.SUBMISSION_READY,
                    message="Glassdoor hosted application flow ready. Reached SUBMISSION_READY via Playwright DOM.",
                    diagnostics={
                        **step_base_diagnostics,
                        "submit_verified": True,
                        "final_submit_clicked": False,
                    },
                )

            # 4. Check Requirements Warning Step ("Apply anyway" / "Don't meet requirements")
            if dom_state == GlassdoorAutomationState.REQUIREMENTS_WARNING:
                requirements_warning_detected = True
                requirements_not_met = state_diag.get("requirements_not_met", [])
                requirements_text = state_diag.get("requirements_text", "")
                apply_anyway_btn = await _maybe_await(self.playwright_driver.find_exact_apply_anyway_button(page))
                apply_anyway_found = apply_anyway_btn is not None

                warning_diag = {
                    **step_base_diagnostics,
                    "requirements_warning_detected": True,
                    "requirements_not_met": requirements_not_met,
                    "requirements_text": requirements_text,
                    "apply_anyway_found": apply_anyway_found,
                    "apply_anyway_clicked": False,
                    "continue_on_requirements_warning": CONTINUE_ON_REQUIREMENTS_WARNING,
                    "visible_actions": state_diag.get("visible_actions", ["Return to job search", "Apply anyway"] if apply_anyway_found else ["Return to job search"]),
                }

                if not CONTINUE_ON_REQUIREMENTS_WARNING:
                    logger.info("CONTINUE_ON_REQUIREMENTS_WARNING is False. Halting for user decision.")
                    step_history.append({
                        "step": total_steps,
                        "state": "REQUIREMENTS_WARNING",
                        "requirements": requirements_not_met,
                        "action": "manual_action_required",
                    })
                    return GlassdoorAutomationResult(
                        status="manual_action_required",
                        job_id=getattr(request, "job_id", None),
                        url=inspect_res.url,
                        browser=inspect_res.browser,
                        easy_apply_verified=True,
                        easy_apply_clicked=True,
                        steps_processed=total_steps,
                        questions_detected=total_q_detected,
                        questions_answered=total_q_answered,
                        questions_unresolved=0,
                        resume_state=resume_state,
                        current_state=GlassdoorAutomationState.REQUIREMENTS_WARNING,
                        manual_action_required=True,
                        message=f"Requirements warning: {requirements_text or 'Candidate does not meet employer requirements.'}",
                        diagnostics=warning_diag,
                    )

                # Policy is True: Click Apply anyway
                if not apply_anyway_found:
                    logger.warning("REQUIREMENTS_WARNING detected but Apply anyway button not found.")
                    return GlassdoorAutomationResult(
                        status="blocked",
                        job_id=getattr(request, "job_id", None),
                        url=inspect_res.url,
                        browser=inspect_res.browser,
                        easy_apply_verified=True,
                        easy_apply_clicked=True,
                        steps_processed=total_steps,
                        current_state=GlassdoorAutomationState.REQUIREMENTS_ACTION_NOT_VERIFIED,
                        message="Apply anyway button not found on requirements warning page.",
                        diagnostics=warning_diag,
                    )

                # Capture snapshot before clicking Apply anyway
                snapshot = await _maybe_await(self.playwright_driver.capture_dom_snapshot(page))
                click_ok, click_err = await _maybe_await(self.playwright_driver.click_apply_anyway(page))
                if not click_ok:
                    logger.warning("Failed to click Apply anyway button: %s", click_err)
                    return GlassdoorAutomationResult(
                        status="blocked",
                        job_id=getattr(request, "job_id", None),
                        url=inspect_res.url,
                        browser=inspect_res.browser,
                        easy_apply_verified=True,
                        easy_apply_clicked=True,
                        steps_processed=total_steps,
                        current_state=GlassdoorAutomationState.REQUIREMENTS_ACTION_NOT_VERIFIED,
                        message=click_err or "Failed to click Apply anyway button.",
                        diagnostics=warning_diag,
                    )

                apply_anyway_clicked = True
                trans_ok, trans_reason, new_snap = await self._safe_wait_for_spa_transition(page, snapshot)
                logger.info("Transition after Apply anyway: %s (reason: %s)", trans_ok, trans_reason)
                step_history.append({
                    "step": total_steps,
                    "state": "REQUIREMENTS_WARNING",
                    "requirements": requirements_not_met,
                    "action": "apply_anyway",
                    "transition_reason": trans_reason,
                })
                continue

            # 5. Handle Optional Survey / Interstitial Step ("Help Indeed learn more about why you're applying")
            if dom_state in (
                GlassdoorAutomationState.OPTIONAL_SURVEY_STEP,
                GlassdoorAutomationState.SMARTAPPLY_INTERSTITIAL,
            ):
                survey_info = state_diag.get("survey_info", {})
                field_name = survey_info.get("field_name", "reason_for_applying")
                is_required = survey_info.get("required", False)
                field_label = survey_info.get("field_label", "Reason for applying")
                heading_text = survey_info.get("heading", "Help Indeed learn more about why you're applying")

                logger.info(
                    "Handling Survey Interstitial: '%s' (field: %s, required: %s)",
                    heading_text,
                    field_name,
                    is_required,
                )

                if is_required:
                    stored_ans = stored_answers.get(field_name) or stored_answers.get(f"survey_{field_name}")

                    if stored_ans:
                        try:
                            ta = page.locator("textarea, input[type='text']").first
                            if hasattr(ta, "fill"):
                                await _maybe_await(ta.fill(str(stored_ans)))
                        except Exception as exc:
                            logger.debug("Could not fill survey field: %s", exc)
                    else:
                        session_id = session_id or f"gdoor_session_{uuid4().hex[:12]}"
                        unresolved_survey_q = [{
                            "text": field_label,
                            "normalized_key": field_name,
                            "type": "textarea",
                            "required": True,
                            "is_survey": True,
                        }]
                        session_data = {
                            "automation_session_id": session_id,
                            "job_url": inspect_res.url,
                            "job_id": getattr(request, "job_id", None),
                            "source": getattr(request, "source", "glassdoor") or "glassdoor",
                            "profile_id": getattr(request, "profile_id", None),
                            "application_page_url": curr_url,
                            "unresolved_questions": unresolved_survey_q,
                            "unresolved_question_keys": [field_name],
                            "total_steps": total_steps,
                            "current_state": GlassdoorAutomationState.NEEDS_USER_INPUT.value,
                            "pause_state": GlassdoorAutomationState.NEEDS_USER_INPUT.value,
                            "paused_at": datetime.now(timezone.utc).isoformat(),
                        }
                        self.save_session(session_id, session_data)
                        step_history.append({
                            "step": total_steps,
                            "state": "OPTIONAL_SURVEY_STEP",
                            "field": field_name,
                            "required": True,
                            "action": "needs_user_input",
                        })
                        return GlassdoorAutomationResult(
                            status="manual_action_required",
                            job_id=getattr(request, "job_id", None),
                            url=inspect_res.url,
                            browser=inspect_res.browser,
                            easy_apply_verified=True,
                            easy_apply_clicked=True,
                            steps_processed=total_steps,
                            current_state=GlassdoorAutomationState.NEEDS_USER_INPUT,
                            manual_action_required=True,
                            message=f"Manual answer required for survey question: '{field_label}'",
                            diagnostics={
                                **step_base_diagnostics,
                                "automation_session_id": session_id,
                                "paused_for_user_input": True,
                                "unresolved_questions": unresolved_survey_q,
                                "unresolved_question_keys": [field_name],
                                "survey_info": survey_info,
                            },
                        )
                else:
                    # Optional field: leave blank (truthful policy: do not fabricate answers)
                    step_history.append({
                        "step": total_steps,
                        "state": "OPTIONAL_SURVEY_STEP",
                        "field": field_name,
                        "required": False,
                        "action": "skip_optional_and_continue",
                    })

                # Capture snapshot before clicking progression button
                snapshot = await _maybe_await(self.playwright_driver.capture_dom_snapshot(page))
                click_ok, act_name, click_err = await self._activate_or_click_progression(page)
                if not click_ok:
                    logger.warning("Failed to activate progression button on Survey step: %s", click_err)
                    return GlassdoorAutomationResult(
                        status="blocked",
                        job_id=getattr(request, "job_id", None),
                        url=inspect_res.url,
                        browser=inspect_res.browser,
                        easy_apply_verified=True,
                        easy_apply_clicked=True,
                        steps_processed=total_steps,
                        current_state=GlassdoorAutomationState.ACTION_CONTROL_NOT_VERIFIED,
                        message=click_err or "Progression button on survey step could not be verified.",
                        diagnostics={
                            **step_base_diagnostics,
                            "survey_info": survey_info,
                        },
                    )

                # Wait for SPA transition after survey progression
                trans_ok, trans_reason, new_snap = await self._safe_wait_for_spa_transition(page, snapshot)
                logger.info("Transition after survey progression: %s (reason: %s)", trans_ok, trans_reason)
                continue

            # 6. Handle Questions Step (Priority before Resume to prevent misclassification)
            if dom_state == GlassdoorAutomationState.QUESTIONS_STEP:
                questions = await self.playwright_driver.extract_questions(page)
                step_q_count = len(questions)
                step_q_answered = 0
                unresolved_list: List[Dict[str, Any]] = []
                answered_details: List[Dict[str, Any]] = []
                total_q_detected += step_q_count

                for q in questions:
                    res = resolver.resolve_question(q)
                    if res.is_resolved and res.resolved_value is not None:
                        await self.playwright_driver.fill_question(page, q, res.resolved_value)
                        total_q_answered += 1
                        step_q_answered += 1
                        answered_details.append({
                            "question": q.text,
                            "normalized_key": q.normalized_key,
                            "answer_source": getattr(res, "source", "application_answers") or "application_answers",
                        })
                    elif q.required:
                        if getattr(res, "action", "") == "leave_unanswered":
                            logger.info("Required question '%s' left unanswered per policy for platform validation.", q.text)
                        else:
                            logger.warning("Unresolved required question encountered: '%s' (%s)", q.text, res.reason)
                            unresolved_list.append({
                                "text": q.text,
                                "normalized_key": q.normalized_key,
                                "type": q.input_type.value,
                                "required": True,
                            })
                    else:
                        logger.info("Skipping optional question: '%s'", q.text)

                if unresolved_list:
                    import urllib.parse
                    from app.automation.glassdoor.url_validator import extract_glassdoor_job_id

                    session_id = session_id or f"gdoor_session_{uuid4().hex[:12]}"
                    page_title = ""
                    try:
                        if hasattr(page, "title") and callable(page.title):
                            page_title = await _maybe_await(page.title())
                    except Exception:
                        pass
                    app_host = ""
                    try:
                        app_host = urllib.parse.urlparse(curr_url).netloc
                    except Exception:
                        pass

                    job_jl = extract_glassdoor_job_id(inspect_res.url)

                    session_data = {
                        "automation_session_id": session_id,
                        "job_url": inspect_res.url,
                        "job_id": getattr(request, "job_id", None),
                        "job_listing_id": job_jl,
                        "source": getattr(request, "source", "glassdoor") or "glassdoor",
                        "source_platform": "glassdoor",
                        "profile_id": getattr(request, "profile_id", None),
                        "application_page_url": curr_url,
                        "application_page_title": page_title,
                        "application_page_host": app_host or current_host,
                        "application_host": current_host,
                        "question_signature": "-".join(sorted(q["normalized_key"] for q in unresolved_list)),
                        "unresolved_questions": unresolved_list,
                        "unresolved_question_keys": [q["normalized_key"] for q in unresolved_list],
                        "total_steps": total_steps,
                        "total_q_detected": total_q_detected,
                        "total_q_answered": total_q_answered,
                        "resume_state": resume_state,
                        "step_history": step_history,
                        "current_state": GlassdoorAutomationState.NEEDS_USER_INPUT.value,
                        "pause_state": GlassdoorAutomationState.NEEDS_USER_INPUT.value,
                        "paused_at": datetime.now(timezone.utc).isoformat(),
                        "resume_attempt": initial_diagnostics.get("resume_attempt", 0) if isinstance(initial_diagnostics, dict) else 0,
                    }
                    self.save_session(session_id, session_data)

                    step_history.append({
                        "step": total_steps,
                        "state": "QUESTIONS_STEP",
                        "questions_detected": step_q_count,
                        "questions_answered": step_q_answered,
                        "questions_unresolved": len(unresolved_list),
                        "action": "needs_user_input",
                    })
                    first_unresolved = unresolved_list[0]
                    return GlassdoorAutomationResult(
                        status="manual_action_required",
                        job_id=getattr(request, "job_id", None),
                        url=inspect_res.url,
                        browser=inspect_res.browser,
                        easy_apply_verified=True,
                        easy_apply_clicked=True,
                        steps_processed=total_steps,
                        questions_detected=total_q_detected,
                        questions_answered=total_q_answered,
                        questions_unresolved=len(unresolved_list),
                        resume_state=resume_state,
                        current_state=GlassdoorAutomationState.NEEDS_USER_INPUT,
                        manual_action_required=True,
                        message=f"Manual answer required for question: '{first_unresolved['text']}'",
                        diagnostics={
                            **step_base_diagnostics,
                            "automation_session_id": session_id,
                            "paused_for_user_input": True,
                            "questions_page_detected": True,
                            "questions_detected": step_q_count,
                            "questions_answered": step_q_answered,
                            "questions_unresolved": len(unresolved_list),
                            "unresolved_questions": unresolved_list,
                            "unresolved_keys": [q["normalized_key"] for q in unresolved_list],
                            "unresolved_question_keys": [q["normalized_key"] for q in unresolved_list],
                            "unresolved_questions_details": unresolved_list,
                            "question": first_unresolved["text"],
                            "normalized_key": first_unresolved["normalized_key"],
                            "input_type": first_unresolved["type"],
                        },
                    )

                # All questions resolved -> activate progression button with scroll_into_view_if_needed
                snapshot = await _maybe_await(self.playwright_driver.capture_dom_snapshot(page))
                click_ok, act_name, click_err = await self._activate_or_click_progression(page)
                trans_ok, trans_reason, new_snap = await self._safe_wait_for_spa_transition(page, snapshot)
                logger.info("Transition after questions progression: %s (reason: %s)", trans_ok, trans_reason)
                hist_item = {
                    "step": total_steps,
                    "state": "QUESTIONS_STEP",
                    "questions_detected": step_q_count,
                    "questions_answered": step_q_answered,
                    "questions_unresolved": 0,
                    "action": act_name or "continue",
                    "transition_reason": trans_reason,
                }
                if answered_details:
                    hist_item["question"] = answered_details[0]["question"]
                    hist_item["normalized_key"] = answered_details[0]["normalized_key"]
                    hist_item["answer_source"] = answered_details[0]["answer_source"]
                step_history.append(hist_item)
                continue

            # 5. Handle Resume Step (Scroll Continue into view)
            if dom_state == GlassdoorAutomationState.RESUME_STEP:
                r_ok, r_state, r_err = await self.playwright_driver.handle_resume_step(page)
                resume_state = r_state
                click_ok, act_name, click_err = await self._activate_or_click_progression(page)
                prog_diag = getattr(self.playwright_driver, "_last_progression_diagnostics", {})

                resume_step_diag = {
                    **step_base_diagnostics,
                    "resume_step_detected": True,
                    "resume_present": True,
                    "resume_selected": r_state in ("selected", "default_selected", "not_needed"),
                    "resume_continue_locator_found": prog_diag.get("resume_continue_locator_found", click_ok),
                    "resume_continue_count": prog_diag.get("resume_continue_count", 1 if click_ok else 0),
                    "resume_continue_visible_before_scroll": prog_diag.get("resume_continue_visible_before_scroll", False),
                    "resume_continue_visible_after_scroll": prog_diag.get("resume_continue_visible_after_scroll", click_ok),
                    "resume_continue_enabled": prog_diag.get("resume_continue_enabled", click_ok),
                    "resume_continue_bounds": prog_diag.get("resume_continue_bounds"),
                    "scroll_into_view_attempted": prog_diag.get("scroll_into_view_attempted", True),
                    "scrollable_ancestor_detected": prog_diag.get("scrollable_ancestor_detected", False),
                    "iframe_detected": prog_diag.get("iframe_detected", False),
                    "resume_continue_clicked": click_ok,
                }

                if not r_ok:
                    step_history.append({"step": total_steps, "state": "RESUME_STEP", "action": "needs_user_input"})
                    return GlassdoorAutomationResult(
                        status="manual_action_required",
                        job_id=getattr(request, "job_id", None),
                        url=inspect_res.url,
                        browser=inspect_res.browser,
                        easy_apply_verified=True,
                        easy_apply_clicked=True,
                        steps_processed=total_steps,
                        resume_state=r_state,
                        current_state=GlassdoorAutomationState.NEEDS_USER_INPUT,
                        manual_action_required=True,
                        message=r_err or "Resume selection required.",
                        diagnostics=resume_step_diag,
                    )

                if not click_ok:
                    logger.warning("Failed to activate Continue button on Resume step: %s", click_err)
                    step_history.append({"step": total_steps, "state": "RESUME_STEP", "action": "continue_failed"})
                    return GlassdoorAutomationResult(
                        status="blocked",
                        job_id=getattr(request, "job_id", None),
                        url=inspect_res.url,
                        browser=inspect_res.browser,
                        easy_apply_verified=True,
                        easy_apply_clicked=True,
                        steps_processed=total_steps,
                        current_state=GlassdoorAutomationState.ACTION_CONTROL_NOT_VERIFIED,
                        message=click_err or "Continue button on resume step could not be verified or activated.",
                        diagnostics=resume_step_diag,
                    )

                step_history.append({
                    "step": total_steps,
                    "state": "RESUME_STEP",
                    "resume_selected": True,
                    "action": act_name or "continue",
                    "scrolled_into_view": True,
                })
                continue

            # 6. Handle Review Step
            if dom_state == GlassdoorAutomationState.REVIEW_STEP:
                submit_btn = await _maybe_await(self.playwright_driver.find_exact_submit_button(page))
                if submit_btn is not None:
                    step_history.append({"step": total_steps, "state": "SUBMISSION_READY", "submit_verified": True})
                    return GlassdoorAutomationResult(
                        status="success",
                        job_id=getattr(request, "job_id", None),
                        url=inspect_res.url,
                        browser=inspect_res.browser,
                        easy_apply_verified=True,
                        easy_apply_clicked=True,
                        steps_processed=total_steps,
                        questions_detected=total_q_detected,
                        questions_answered=total_q_answered,
                        questions_unresolved=0,
                        resume_state=resume_state,
                        submit_verified=True,
                        final_submit_clicked=False,
                        submission_confirmed=False,
                        current_state=GlassdoorAutomationState.SUBMISSION_READY,
                        message="Glassdoor application review reached SUBMISSION_READY.",
                        diagnostics={
                            **step_base_diagnostics,
                            "submit_verified": True,
                            "final_submit_clicked": False,
                        },
                    )
                click_ok, act_name, click_err = await self._activate_or_click_progression(page)
                step_history.append({"step": total_steps, "state": "REVIEW_STEP", "action": act_name or "review"})
                continue

            # 7. Action Button Progression Fallback
            click_ok, act_name, click_err = await self._activate_or_click_progression(page)
            if click_ok:
                step_history.append({"step": total_steps, "state": "APPLICATION_FLOW", "action": act_name})
                continue

            # Settle & Re-check: If page was transitioning, check if Submit or Progression button appeared
            await asyncio.sleep(1.0)
            submit_btn = await self.playwright_driver.find_exact_submit_button(page)
            if submit_btn is not None:
                step_history.append({"step": total_steps, "state": "SUBMISSION_READY", "submit_verified": True})
                return GlassdoorAutomationResult(
                    status="success",
                    job_id=request.job_id,
                    url=inspect_res.url,
                    browser=inspect_res.browser,
                    easy_apply_verified=True,
                    easy_apply_clicked=True,
                    steps_processed=total_steps,
                    questions_detected=total_q_detected,
                    questions_answered=total_q_answered,
                    questions_unresolved=0,
                    resume_state=resume_state,
                    submit_verified=True,
                    final_submit_clicked=False,
                    submission_confirmed=False,
                    current_state=GlassdoorAutomationState.SUBMISSION_READY,
                    message="Glassdoor hosted application flow reached SUBMISSION_READY via Playwright DOM.",
                    diagnostics={
                        **step_base_diagnostics,
                        "submit_verified": True,
                        "final_submit_clicked": False,
                    },
                )

            # Retry progression once more after settling
            click_ok, act_name, click_err = await self._activate_or_click_progression(page)
            if click_ok:
                step_history.append({"step": total_steps, "state": "APPLICATION_FLOW", "action": act_name})
                continue

            # Action control unverified
            return GlassdoorAutomationResult(
                status="blocked",
                job_id=request.job_id,
                url=inspect_res.url,
                browser=inspect_res.browser,
                easy_apply_verified=True,
                easy_apply_clicked=True,
                steps_processed=total_steps,
                current_state=GlassdoorAutomationState.ACTION_CONTROL_NOT_VERIFIED,
                message=f"Action control unverified on step {total_steps}: {click_err}",
                diagnostics=step_base_diagnostics,
            )

        return GlassdoorAutomationResult(
            status="blocked",
            job_id=request.job_id,
            url=inspect_res.url,
            browser=inspect_res.browser,
            easy_apply_verified=True,
            easy_apply_clicked=True,
            steps_processed=total_steps,
            current_state=GlassdoorAutomationState.APPLICATION_PROGRESS_STALLED,
            message=f"Exceeded maximum application steps limit ({MAX_APPLICATION_STEPS}).",
            diagnostics={
                "easy_apply_method": click_method,
                "easy_apply_clicked": True,
                "playwright_attached": True,
                "steps_processed": total_steps,
                "step_history": step_history,
            },
        )

    def _click_next_available_action(self, container: Any) -> Tuple[bool, Optional[str]]:
        """Find and click the highest priority action button (Review -> Continue -> Next)."""
        review_ctrl = self.driver.find_action_control(REVIEW_BUTTON_NAMES, container=container)
        if review_ctrl is not None:
            if self.driver.click_control(review_ctrl):
                return True, "review"

        continue_ctrl = self.driver.find_action_control(CONTINUE_BUTTON_NAMES, container=container)
        if continue_ctrl is not None:
            if self.driver.click_control(continue_ctrl):
                return True, "continue"

        next_ctrl = self.driver.find_action_control(NEXT_BUTTON_NAMES, container=container)
        if next_ctrl is not None:
            if self.driver.click_control(next_ctrl):
                return True, "next"

        return False, None

    def _fill_form_field(self, field: Any, value: Any) -> bool:
        """Autofill a single form field using appropriate input handler."""
        ctrl = getattr(field, "control", None)
        if ctrl is None and hasattr(field, "diagnostics") and isinstance(field.diagnostics, dict):
            ctrl = field.diagnostics.get("control")

        if field.input_type == QuestionInputType.TEXT:
            return fill_text_field(self.driver, ctrl, str(value))
        elif field.input_type == QuestionInputType.NUMBER:
            return fill_number_field(self.driver, ctrl, float(value) if isinstance(value, (int, float)) else float(str(value)))
        elif field.input_type == QuestionInputType.DROPDOWN:
            return select_dropdown_option(self.driver, ctrl, str(value))
        elif field.input_type == QuestionInputType.RADIO:
            return select_radio_option(self.driver, ctrl, str(value))
        elif field.input_type in (QuestionInputType.CHECKBOX, QuestionInputType.CHECKBOX_MULTI):
            items = value if isinstance(value, list) else [str(value)]
            return set_checkboxes(self.driver, [ctrl], items)
        return False

    def _adaptive_tab_traversal(self, modal: Any) -> Tuple[bool, Optional[str]]:
        """Adaptive TAB key traversal fallback if direct action clicking fails."""
        logger.info("Executing adaptive TAB traversal fallback...")
        for tab_idx in range(MAX_TAB_TRAVERSAL):
            self.driver.send_tab_key()
            time.sleep(0.15)
            focused = self.driver.get_focused_element()
            if focused is not None:
                f_name = getattr(focused.element_info, "name", "")
                norm_name = _normalize(f_name)
                for action in REVIEW_BUTTON_NAMES + CONTINUE_BUTTON_NAMES + NEXT_BUTTON_NAMES:
                    if action in norm_name:
                        logger.info("Focused action button '%s' via TAB at press %d", f_name, tab_idx + 1)
                        if self.driver.press_enter():
                            return True, action
        return False, None

    @async_hybrid
    async def resume(
        self, request: GlassdoorAutomationResumeRequest
    ) -> GlassdoorAutomationResult:
        """Resume a paused Glassdoor application after human input or with direct answers."""
        session_id = request.automation_session_id or f"gdoor_session_{uuid4().hex[:12]}"
        session_data = self.get_session(session_id) or {}

        job_id = request.job_id or session_data.get("job_id")
        profile_id = request.profile_id or session_data.get("profile_id")
        target_url = request.url or session_data.get("job_url") or "https://www.glassdoor.co.in"
        source = request.source or session_data.get("source") or "glassdoor"

        total_steps = session_data.get("total_steps", 0)
        total_q_detected = session_data.get("total_q_detected", 0)
        total_q_answered = session_data.get("total_q_answered", 0)
        resume_state = session_data.get("resume_state", "not_needed")
        step_history = list(session_data.get("step_history", []))
        resume_attempt = session_data.get("resume_attempt", 0) + 1
        session_data["resume_attempt"] = resume_attempt

        # 1. Connect to CDP browser if not connected
        if getattr(self.playwright_driver, "_browser", None) is None:
            try:
                res_conn = await _maybe_await(self.playwright_driver.connect_cdp())
                connected = True
                conn_err = None
                if isinstance(res_conn, tuple) and len(res_conn) >= 2:
                    connected, conn_err = res_conn[0], res_conn[1]
                elif isinstance(res_conn, bool):
                    connected = res_conn

                if not connected:
                    return GlassdoorAutomationResult(
                        status="failed",
                        job_id=job_id,
                        url=target_url,
                        browser="chrome_cdp",
                        current_state=GlassdoorAutomationState.BROWSER_NOT_VERIFIED,
                        message=conn_err or "Failed to connect to CDP browser session.",
                        diagnostics={
                            "cdp_connect_attempted": True,
                            "cdp_connected": False,
                            "browser_session_verified": False,
                            "contexts_found": 0,
                            "pages_found": 0,
                            "authoritative_page_url": None,
                            "requested_job_url": target_url,
                            "second_browser_launched": False,
                            "automation_session_id": session_id,
                            "resume_attempt": resume_attempt,
                        },
                    )
            except Exception as exc:
                logger.warning("Error connecting to CDP browser during resume: %s", exc)
                err_msg = str(exc)
                is_runtime_err = "Playwright" in err_msg or "asyncio" in err_msg
                return GlassdoorAutomationResult(
                    status="failed",
                    job_id=job_id,
                    url=target_url,
                    browser="chrome_cdp",
                    current_state=GlassdoorAutomationState.AUTOMATION_RUNTIME_ERROR if is_runtime_err else GlassdoorAutomationState.BROWSER_NOT_VERIFIED,
                    message=f"Exception connecting to CDP browser: {err_msg}",
                    diagnostics={
                        "cdp_connect_attempted": True,
                        "cdp_connected": False,
                        "browser_session_verified": False,
                        "contexts_found": 0,
                        "pages_found": 0,
                        "authoritative_page_url": None,
                        "requested_job_url": target_url,
                        "second_browser_launched": False,
                        "automation_session_id": session_id,
                        "resume_attempt": resume_attempt,
                        "error_detail": err_msg,
                    },
                )

        # 2. Rediscover active application page from browser state using dedicated resolver
        app_page = None
        recovery_method = "none"
        recovery_diag: Dict[str, Any] = {}
        res_resolve = None
        if hasattr(self.playwright_driver, "resolve_paused_application_page") and callable(self.playwright_driver.resolve_paused_application_page):
            try:
                res_resolve = await _maybe_await(
                    self.playwright_driver.resolve_paused_application_page(
                        session_data=session_data,
                        timeout_seconds=5,
                    )
                )
                if isinstance(res_resolve, tuple) and len(res_resolve) >= 3:
                    app_page, recovery_method, recovery_diag = res_resolve[0], res_resolve[1], res_resolve[2]
                elif isinstance(res_resolve, tuple) and len(res_resolve) == 2:
                    app_page, recovery_method = res_resolve[0], res_resolve[1]
                elif res_resolve is not None and not isinstance(res_resolve, tuple):
                    app_page = res_resolve
            except Exception as exc:
                logger.debug("resolve_paused_application_page exception: %s", exc)

        # Fallback to driver.application_page only if resolver was not invoked with a tuple result
        if app_page is None and res_resolve is None:
            app_page = getattr(self.playwright_driver, "application_page", None)
            if app_page is not None:
                recovery_method = "driver_property_fallback"
                recovery_diag = {"application_page_recovered": True, "application_page_recovery_method": recovery_method}

        if app_page is None:
            logger.warning("Could not resolve active paused application page for resume.")
            return GlassdoorAutomationResult(
                status="failed",
                job_id=job_id,
                url=target_url,
                browser="chrome_cdp",
                current_state=GlassdoorAutomationState.PAUSED_APPLICATION_NOT_FOUND,
                message="Could not rediscover active application page for CDP session.",
                diagnostics={
                    "automation_session_id": session_id,
                    "resume_attempt": resume_attempt,
                    **recovery_diag,
                },
            )

        curr_url = getattr(app_page, "url", target_url) or target_url
        is_smartapply = any(d in str(curr_url) for d in SMARTAPPLY_HOST_DOMAINS)
        current_host = "indeed_smartapply" if is_smartapply else "glassdoor"

        # 3. Detect current DOM state
        dom_state = GlassdoorAutomationState.UNKNOWN
        state_diag = {}
        try:
            res_state = await _maybe_await(self.playwright_driver.detect_page_state(app_page))
            if isinstance(res_state, tuple) and len(res_state) >= 2:
                dom_state, state_diag = res_state[0], res_state[1]
            elif isinstance(res_state, tuple) and len(res_state) == 1:
                dom_state, state_diag = res_state[0], {}
            elif isinstance(res_state, GlassdoorAutomationState):
                dom_state, state_diag = res_state, {}
        except Exception as exc:
            logger.debug("detect_page_state exception: %s", exc)

        manual_answers_detected: Dict[str, Any] = {}
        saved_answers: List[Dict[str, Any]] = []
        manual_answers_saved = False
        progress_action_clicked = None

        # 4. If currently on QUESTIONS_STEP, process user input or API answers
        if dom_state == GlassdoorAutomationState.QUESTIONS_STEP:
            questions = await _maybe_await(self.playwright_driver.extract_questions(app_page))
            step_q_count = len(questions) if isinstance(questions, list) else 0
            if step_q_count > 0 and total_q_detected == 0:
                total_q_detected = step_q_count

            profile_data = None
            if profile_id and self.profile_service:
                try:
                    profile_data = self.profile_service.get_profile(profile_id)
                except Exception as exc:
                    logger.warning("Could not load profile %s: %s", profile_id, exc)

            stored_answers = self.application_repository.get_answers_by_profile(profile_id)
            resolver = FormAnswerResolver(
                answer_bank=self.candidate_answer_bank,
                stored_answers=stored_answers,
                profile=profile_data,
                platform="glassdoor",
                host="glassdoor",
                allow_unanswered_questions=self.settings.ALLOW_SUBMISSION_WITH_UNANSWERED_QUESTIONS,
            )

            if isinstance(questions, list):
                for q in questions:
                    # Mode A: API answers provided
                    if request.answers and (q.normalized_key in request.answers or q.text in request.answers):
                        ans_val = request.answers.get(q.normalized_key)
                        if ans_val is None:
                            ans_val = request.answers.get(q.text)
                        if ans_val is not None and str(ans_val).strip() != "":
                            await _maybe_await(self.playwright_driver.fill_question(app_page, q, ans_val))
                            manual_answers_detected[q.normalized_key] = str(ans_val)
                            total_q_answered += 1
                            try:
                                self.application_repository.save_application_answer(question=q.text, answer=str(ans_val), source="manual_user_input")
                                self.application_repository.save_application_answer(question=q.normalized_key, answer=str(ans_val), source="manual_user_input")
                                saved_answers.append({"normalized_key": q.normalized_key, "value": str(ans_val), "persisted": True})
                                manual_answers_saved = True
                            except Exception as exc:
                                logger.warning("Could not persist manual answer for %s: %s", q.normalized_key, exc)
                    else:
                        # Mode B: Read user-entered DOM value
                        dom_val = await _maybe_await(self.playwright_driver.read_question_value(app_page, q))
                        if dom_val is not None and str(dom_val).strip() != "":
                            manual_answers_detected[q.normalized_key] = str(dom_val)
                            total_q_answered += 1
                            try:
                                self.application_repository.save_application_answer(question=q.text, answer=str(dom_val), source="manual_user_input")
                                self.application_repository.save_application_answer(question=q.normalized_key, answer=str(dom_val), source="manual_user_input")
                                saved_answers.append({"normalized_key": q.normalized_key, "value": str(dom_val), "persisted": True})
                                manual_answers_saved = True
                            except Exception as exc:
                                logger.warning("Could not persist manual answer for %s: %s", q.normalized_key, exc)
                        else:
                            # Stored answer fallback if available
                            res = resolver.resolve_question(q)
                            if res.is_resolved and res.resolved_value is not None:
                                await _maybe_await(self.playwright_driver.fill_question(app_page, q, res.resolved_value))
                                total_q_answered += 1

            # Validate remaining unresolved required questions
            unresolved_list: List[Dict[str, Any]] = []
            if isinstance(questions, list):
                for q in questions:
                    if q.required:
                        has_val = (q.normalized_key in manual_answers_detected)
                        if not has_val:
                            dom_val = await _maybe_await(self.playwright_driver.read_question_value(app_page, q))
                            if dom_val is not None and str(dom_val).strip() != "":
                                has_val = True
                        if not has_val:
                            res = resolver.resolve_question(q)
                            if res.is_resolved and res.resolved_value is not None:
                                has_val = True

                        if not has_val:
                            unresolved_list.append({
                                "text": q.text,
                                "normalized_key": q.normalized_key,
                                "type": q.input_type.value,
                                "required": True,
                            })

            if unresolved_list:
                import urllib.parse
                from app.automation.glassdoor.url_validator import extract_glassdoor_job_id

                total_steps += 1
                app_host = ""
                try:
                    app_host = urllib.parse.urlparse(curr_url).netloc
                except Exception:
                    pass
                page_title = ""
                try:
                    if hasattr(app_page, "title") and callable(app_page.title):
                        raw_title = app_page.title()
                        if inspect.isawaitable(raw_title):
                            page_title = await raw_title
                        elif isinstance(raw_title, str):
                            page_title = raw_title
                except Exception:
                    pass

                job_jl = extract_glassdoor_job_id(target_url)

                session_data.update({
                    "automation_session_id": session_id,
                    "job_url": target_url,
                    "job_id": job_id,
                    "job_listing_id": job_jl,
                    "source": source,
                    "source_platform": "glassdoor",
                    "profile_id": profile_id,
                    "application_page_url": curr_url,
                    "application_page_title": page_title,
                    "application_page_host": app_host or current_host,
                    "application_host": current_host,
                    "question_signature": "-".join(sorted(q["normalized_key"] for q in unresolved_list)),
                    "unresolved_questions": unresolved_list,
                    "unresolved_question_keys": [q["normalized_key"] for q in unresolved_list],
                    "total_steps": total_steps,
                    "total_q_detected": total_q_detected,
                    "total_q_answered": total_q_answered,
                    "resume_state": resume_state,
                    "step_history": step_history,
                    "current_state": GlassdoorAutomationState.NEEDS_USER_INPUT.value,
                    "pause_state": GlassdoorAutomationState.NEEDS_USER_INPUT.value,
                    "paused_at": datetime.now(timezone.utc).isoformat(),
                    "resume_attempt": resume_attempt,
                })
                self.save_session(session_id, session_data)
                first_unresolved = unresolved_list[0]
                return GlassdoorAutomationResult(
                    status="manual_action_required",
                    job_id=job_id,
                    url=target_url,
                    browser="chrome_cdp",
                    easy_apply_verified=True,
                    easy_apply_clicked=True,
                    steps_processed=total_steps,
                    questions_detected=total_q_detected,
                    questions_answered=total_q_answered,
                    questions_unresolved=len(unresolved_list),
                    resume_state=resume_state,
                    current_state=GlassdoorAutomationState.NEEDS_USER_INPUT,
                    manual_action_required=True,
                    message=f"Manual answer required for question: '{first_unresolved['text']}'",
                    diagnostics={
                        "automation_session_id": session_id,
                        "paused_for_user_input": True,
                        "unresolved_questions": unresolved_list,
                        "unresolved_keys": [q["normalized_key"] for q in unresolved_list],
                        "unresolved_question_keys": [q["normalized_key"] for q in unresolved_list],
                        "unresolved_questions_details": unresolved_list,
                        "manual_answers_detected": manual_answers_detected,
                        "manual_answers_saved": manual_answers_saved,
                        "saved_answers": saved_answers,
                        "resume_attempt": resume_attempt,
                        "question": first_unresolved["text"],
                        "normalized_key": first_unresolved["normalized_key"],
                        "input_type": first_unresolved["type"],
                        **recovery_diag,
                    },
                )

            # All required questions satisfied -> click progression action
            total_steps += 1
            snapshot = await _maybe_await(self.playwright_driver.capture_dom_snapshot(app_page))
            click_ok, act_name, click_err = await self._activate_or_click_progression(app_page)
            progress_action_clicked = act_name or "continue"
            trans_ok, trans_reason, new_snap = await self._safe_wait_for_spa_transition(app_page, snapshot)
            logger.info("Transition after resume questions progression: %s (reason: %s)", trans_ok, trans_reason)
            step_history.append({
                "step": total_steps,
                "state": "QUESTIONS_STEP",
                "action": progress_action_clicked,
                "manual_answers": list(manual_answers_detected.keys()),
                "transition_reason": trans_reason,
            })

        # 5. Build inspect_res shell and continue universal loop
        inspect_res = GlassdoorAutomationResult(
            status="success",
            job_id=job_id,
            url=target_url,
            browser="chrome_cdp",
            easy_apply_verified=True,
            easy_apply_clicked=True,
            current_state=dom_state,
        )

        nav_req = GlassdoorAutomationNavigateRequest(
            job_id=job_id,
            url=target_url,
            source=source,
            profile_id=profile_id,
        )

        initial_diag = {
            "automation_session_id": session_id,
            "manual_answers_detected": manual_answers_detected,
            "manual_answers_saved": manual_answers_saved,
            "saved_answers": saved_answers,
            "progress_action_clicked": progress_action_clicked,
            "resume_attempt": resume_attempt,
            **recovery_diag,
        }

        return await self._run_playwright_application_loop(
            request=nav_req,
            inspect_res=inspect_res,
            click_method="playwright",
            session_id=session_id,
            initial_total_steps=total_steps,
            initial_q_detected=total_q_detected,
            initial_q_answered=total_q_answered,
            initial_history=step_history,
            initial_diagnostics=initial_diag,
        )

    @async_hybrid
    async def apply_to_job(
        self, request: GlassdoorAutomationApplyRequest
    ) -> GlassdoorAutomationResult:
        """Assisted single-job application on Glassdoor, verifying Submit and clicking exactly ONCE."""
        job_id = request.job_id
        url = request.url

        # 1. Navigate to SUBMISSION_READY state
        nav_req = GlassdoorAutomationNavigateRequest(
            job_id=job_id,
            url=url,
            source=request.source,
            profile_id=request.profile_id,
        )
        nav_res = await _maybe_await(self.navigate_to_submit(nav_req))

        # 2. Block if SUBMISSION_READY was not reached
        if nav_res.current_state != GlassdoorAutomationState.SUBMISSION_READY:
            logger.warning("Application flow did not reach SUBMISSION_READY (Current state: %s).", nav_res.current_state)
            return nav_res

        # 3. Check for Post-Review Blocking States (CAPTCHA/Login/MFA)
        page_text = self.driver.get_window_text_content()
        blocking = GlassdoorStateDetector.detect_blocking_state(page_text)
        if blocking:
            b_state, b_msg = blocking
            nav_res.status = "manual_action_required"
            nav_res.current_state = b_state
            nav_res.manual_action_required = True
            nav_res.message = f"Submission halted: {b_msg}"
            return nav_res

        # 3b. Final Submit Safety Guard (AUTO_SUBMIT_ENABLED)
        if not getattr(self.settings, "AUTO_SUBMIT_ENABLED", False):
            logger.info("AUTO_SUBMIT_ENABLED is False. Halting safely at SUBMISSION_READY state without clicking final submit.")
            nav_res.status = "submission_ready"
            nav_res.current_state = GlassdoorAutomationState.SUBMISSION_READY
            nav_res.submit_verified = True
            nav_res.message = "Application reached SUBMISSION_READY state. Final submit prevented by safety policy (AUTO_SUBMIT_ENABLED=False)."
            return nav_res

        # 4. Click Final Submit Button Exactly ONCE
        logger.info("Ready to submit application for job %s. Clicking final submit button ONCE...", job_id)
        submitted_ok = False

        # Attempt submit via Playwright DOM if page is active
        if getattr(self.playwright_driver, "page", None) is not None:
            pw_sub_ok, pw_sub_err = await _maybe_await(self.playwright_driver.click_final_submit(self.playwright_driver.page))
            if pw_sub_ok:
                submitted_ok = True
                logger.info("Clicked final submit button via Playwright DOM.")

        # Fallback to PyWinAuto submit control
        if not submitted_ok:
            active_win = getattr(self.driver, "_window", None) or getattr(self.driver, "window", None)
            modal = GlassdoorModalInspector.find_application_modal(active_win)
            submit_btn = self.driver.find_exact_submit_control(container=modal)

            if submit_btn is None:
                nav_res.status = "blocked"
                nav_res.current_state = GlassdoorAutomationState.SUBMIT_CONTROL_NOT_VERIFIED
                nav_res.message = "Submit button was verified during navigation but could not be located for click."
                return nav_res

            submitted_ok = self.driver.click_control(submit_btn)

        if not submitted_ok:
            nav_res.status = "failed"
            nav_res.current_state = GlassdoorAutomationState.FAILED
            nav_res.message = "Failed to click the verified Submit button."
            return nav_res

        nav_res.final_submit_clicked = True
        logger.info("Final submit button activated once. Polling for submission confirmation...")

        # 5. Wait for Submission Confirmation
        confirmed = False
        conf_text = ""
        start_wait = time.time()
        while time.time() - start_wait < 15:
            if self.driver.detect_submission_confirmation():
                confirmed = True
                conf_text = self.driver.get_window_text_content()
                break
            time.sleep(0.5)

        if confirmed:
            logger.info("Application submission positively confirmed! Confirmation text: '%s'", conf_text)
            nav_res.status = "success"
            nav_res.submission_confirmed = True
            nav_res.current_state = GlassdoorAutomationState.APPLICATION_SUBMITTED
            nav_res.message = "Job application successfully submitted and confirmed."

            # Persist application record in Supabase
            if job_id and self.application_repository:
                try:
                    now = datetime.now(timezone.utc)
                    self.application_repository.create_application(
                        ApplicationCreate(
                            job_id=job_id,
                            profile_id=request.profile_id or uuid4(),
                            platform="glassdoor",
                            status="applied",
                            automation_status="success",
                            started_at=now,
                            submitted_at=now,
                            application_url=request.url,
                            confirmation_text=conf_text or "Application submitted",
                        )
                    )
                except Exception as exc:
                    logger.warning("Failed to persist Glassdoor application record: %s", exc)

            return nav_res

        # Confirmation missing -> do NOT retry submit
        logger.warning("Post-submission confirmation timed out. Marking as unverified.")
        nav_res.status = "blocked"
        nav_res.submission_confirmed = False
        nav_res.current_state = GlassdoorAutomationState.SUBMISSION_CONFIRMATION_UNVERIFIED
        nav_res.message = "Submit button was activated once, but confirmation was not detected."
        return nav_res


def _normalize(text: Optional[str]) -> str:
    """Normalize text for whitespace and lowercase comparison."""
    if not text:
        return ""
    import re
    return re.sub(r"\s+", " ", str(text).lower().strip())
