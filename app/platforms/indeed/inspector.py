"""Indeed Job Page & Application Flow Inspector.

Provides safe, read-only navigation, page metadata extraction, application method
detection, and form field discovery using persistent Chromium browser contexts.
"""

from datetime import datetime
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from playwright.async_api import BrowserContext, Page
from pydantic import BaseModel, Field

from app.automation.playwright_manager import PlaywrightManager
from app.automation.session_manager import SessionState
from app.config.settings import Settings, get_settings
from app.platforms.indeed.session import IndeedSessionManager
from app.platforms.policy import ActionState, AutomationAction, get_platform_policy

logger = logging.getLogger(__name__)


class FormFieldInfo(BaseModel):
    """Structured representation of an accessible form field (read-only metadata only)."""

    label: str = Field(..., description="Accessible label or placeholder of the field")
    name: Optional[str] = Field(default=None, description="HTML input name attribute")
    type: str = Field(default="text", description="HTML field type (e.g. text, file, select, checkbox)")
    required: bool = Field(default=False, description="Whether the field is marked as required")
    options: Optional[List[str]] = Field(default=None, description="Select or radio options if applicable")


class IndeedJobInspectionResult(BaseModel):
    """Structured result from inspecting an Indeed job page and application workflow."""

    platform: str = Field(default="indeed", description="Target platform")
    authenticated: bool = Field(..., description="Whether persistent session was authenticated")
    job_url: str = Field(..., description="Requested job URL")
    final_url: Optional[str] = Field(default=None, description="Final URL after navigation")
    page_title: Optional[str] = Field(default=None, description="Browser document title")
    job_title: Optional[str] = Field(default=None, description="Extracted visible job title")
    company: Optional[str] = Field(default=None, description="Extracted hiring company name")
    location: Optional[str] = Field(default=None, description="Extracted job location")
    application_method: str = Field(
        ...,
        description="Detected application mechanism: 'indeed_hosted', 'external', 'none', 'unknown'",
    )
    apply_button_found: bool = Field(default=False, description="Whether an Apply control was identified")
    apply_button_text: Optional[str] = Field(default=None, description="Visible text of the Apply control")
    external_url: Optional[str] = Field(default=None, description="External redirect URL if application is hosted externally")
    form_detected: bool = Field(default=False, description="Whether an accessible application form was detected")
    form_count: int = Field(default=0, description="Number of form elements detected")
    field_count: int = Field(default=0, description="Total count of discovered form fields")
    fields: List[FormFieldInfo] = Field(default_factory=list, description="List of discovered form fields")
    status: str = Field(
        default="review_required",
        description="Action state: 'review_required', 'login_required', 'user_action_required', 'blocked', 'error'",
    )
    page_load_ms: Optional[int] = Field(default=None, description="Page navigation latency in milliseconds")
    screenshot_saved: bool = Field(default=False, description="Whether diagnostic screenshot was saved")
    screenshot_path: Optional[str] = Field(default=None, description="Path to captured screenshot")
    trace_saved: bool = Field(default=False, description="Whether Playwright trace was recorded")
    trace_path: Optional[str] = Field(default=None, description="Path to recorded trace archive")
    reason: Optional[str] = Field(default=None, description="Explanatory commentary or policy notes")


class IndeedJobInspector:
    """Read-only inspector for Indeed job postings and application flow discovery."""

    # Selectors for Indeed-hosted applications
    INDEED_APPLY_SELECTORS = [
        "#indeedApplyButton",
        'button[data-testid="indeedApplyButton"]',
        'button:has-text("Apply now")',
        'button:has-text("Easily apply")',
        'button[aria-label*="Apply now"]',
        'button[aria-label*="Easily apply"]',
        'a:has-text("Apply now")',
        'a:has-text("Easily apply")',
    ]

    # Selectors for External company site applications
    EXTERNAL_APPLY_SELECTORS = [
        'a:has-text("Apply on company site")',
        'a:has-text("Apply on employer site")',
        'a[data-testid="apply-button"]',
        'a[href*="/apply/redirect"]',
        'button:has-text("Apply on company site")',
        'button:has-text("Apply on employer site")',
    ]

    def __init__(
        self,
        settings: Optional[Settings] = None,
        playwright_manager: Optional[PlaywrightManager] = None,
        session_manager: Optional[IndeedSessionManager] = None,
    ) -> None:
        self.settings = settings if isinstance(settings, Settings) else get_settings()
        self.playwright_manager = playwright_manager or PlaywrightManager(self.settings)
        self.session_manager = session_manager or IndeedSessionManager(
            settings=self.settings,
            playwright_manager=self.playwright_manager,
        )
        self.profile_path = self.settings.get_indeed_profile_path()
        self.screenshots_path = self.settings.get_screenshots_path() / "indeed"
        self.artifacts_path = self.settings.get_playwright_artifacts_path()

    @staticmethod
    def validate_indeed_url(url: str) -> bool:
        """Validate whether the URL belongs to a legitimate Indeed domain."""
        if not url or not isinstance(url, str):
            return False
        try:
            parsed = urlparse(url.strip())
            netloc = parsed.netloc.lower()
            # Must have http/https scheme and match indeed domain
            if parsed.scheme not in ("http", "https"):
                return False
            valid_domains = [
                "indeed.com",
                "www.indeed.com",
                "in.indeed.com",
                "uk.indeed.com",
                "ca.indeed.com",
                "au.indeed.com",
                "secure.indeed.com",
            ]
            return any(netloc == d or netloc.endswith(f".{d}") for d in valid_domains)
        except Exception:
            return False

    async def inspect_job(
        self,
        job_url: str,
        save_screenshot: bool = True,
        headless: Optional[bool] = None,
    ) -> IndeedJobInspectionResult:
        """Inspect an Indeed job posting URL and determine its application flow."""
        # 1. URL Domain Validation
        if not self.validate_indeed_url(job_url):
            logger.warning("Invalid or non-Indeed URL rejected: %s", job_url)
            return IndeedJobInspectionResult(
                platform="indeed",
                authenticated=False,
                job_url=job_url,
                application_method="unknown",
                apply_button_found=False,
                status="error",
                reason="Invalid Indeed URL. URL must belong to a valid Indeed domain (*.indeed.com).",
            )

        # 2. Platform Policy Check
        policy = get_platform_policy("indeed")
        decision = policy.evaluate_action(AutomationAction.INSPECT_JOB)
        if not decision.allowed:
            logger.warning("Indeed inspection blocked by policy: %s", decision.reason)
            return IndeedJobInspectionResult(
                platform="indeed",
                authenticated=False,
                job_url=job_url,
                application_method="unknown",
                apply_button_found=False,
                status=decision.action_state.value,
                reason=decision.reason,
            )

        # 3. Session Authentication Check
        session_status = await self.session_manager.check_session_status(headless=True)
        if not session_status.authenticated:
            logger.info("Indeed session unauthenticated. Returning LOGIN_REQUIRED.")
            return IndeedJobInspectionResult(
                platform="indeed",
                authenticated=False,
                job_url=job_url,
                application_method="unknown",
                apply_button_found=False,
                status=SessionState.LOGIN_REQUIRED.value,
                reason="Authenticated Indeed session required. Manual login required before inspecting jobs.",
            )

        # 4. Launch Persistent Browser Context
        is_headless = headless if headless is not None else self.settings.PLAYWRIGHT_HEADLESS
        context: Optional[BrowserContext] = None
        trace_path: Optional[Path] = None

        try:
            context = await self.playwright_manager.launch_persistent_context(
                user_data_dir=self.profile_path,
                headless=is_headless,
            )

            # Start Playwright tracing if configured
            if self.settings.PLAYWRIGHT_TRACE:
                self.artifacts_path.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                trace_path = self.artifacts_path / f"indeed_trace_{timestamp}.zip"
                await context.tracing.start(screenshots=True, snapshots=True)

            page = context.pages[0] if context.pages else await context.new_page()

            # 5. Navigate to Job URL & Measure Latency
            logger.info("Navigating to Indeed job URL: %s", job_url)
            start_time = time.time()
            await page.goto(job_url, wait_until="domcontentloaded", timeout=self.settings.PLAYWRIGHT_TIMEOUT_MS)
            page_load_ms = int((time.time() - start_time) * 1000)

            # Small hydration wait for dynamic React components
            await page.wait_for_timeout(1000)

            final_url = page.url
            page_title = await page.title()

            # 6. Extract Page Metadata
            job_title = await self._extract_job_title(page)
            company = await self._extract_company(page)
            location = await self._extract_location(page)

            # 7. Application Method Detection
            app_method, apply_found, apply_text, external_url = await self._detect_application_method(page)

            # 8. Read-Only Form Discovery (Inspect without filling or submitting)
            forms_info = await self._inspect_accessible_forms(page)

            # 9. Diagnostic Screenshot (Safe, non-sensitive)
            screenshot_path_str: Optional[str] = None
            screenshot_saved = False
            if save_screenshot:
                self.screenshots_path.mkdir(parents=True, exist_ok=True)
                target_screenshot = self.screenshots_path / "job_inspection.png"
                await page.screenshot(path=str(target_screenshot), full_page=False)
                screenshot_saved = target_screenshot.exists()
                screenshot_path_str = str(target_screenshot)
                logger.info("Saved job inspection screenshot: %s", target_screenshot)

            # 10. Stop Tracing if Active
            trace_saved = False
            if self.settings.PLAYWRIGHT_TRACE and trace_path:
                await context.tracing.stop(path=str(trace_path))
                trace_saved = trace_path.exists()
                logger.info("Saved Playwright trace: %s", trace_path)

            # 11. Accurate Reason Formulation based on actual findings
            if app_method == "unknown" and not apply_found:
                reason = "Application workflow could not be reliably determined (possible security challenge or unclassifiable page). Manual user review required."
            elif app_method == "none" or not apply_found:
                reason = "No active application controls detected on this job posting. Manual user review required."
            elif app_method == "indeed_hosted":
                reason = "Job page inspected successfully. Indeed-hosted application controls identified. Manual user review required."
            elif app_method == "external":
                reason = f"Job page inspected successfully. External application link identified ({external_url or 'employer site'}). Manual user review required."
            else:
                reason = "Job page inspected successfully. Manual user review required."

            return IndeedJobInspectionResult(
                platform="indeed",
                authenticated=True,
                job_url=job_url,
                final_url=final_url,
                page_title=page_title.strip() if page_title else None,
                job_title=job_title.strip() if job_title else None,
                company=company.strip() if company else None,
                location=location.strip() if location else None,
                application_method=app_method,
                apply_button_found=apply_found,
                apply_button_text=apply_text,
                external_url=external_url,
                form_detected=len(forms_info) > 0,
                form_count=1 if forms_info else 0,
                field_count=len(forms_info),
                fields=forms_info,
                status=ActionState.REVIEW_REQUIRED.value,
                page_load_ms=page_load_ms,
                screenshot_saved=screenshot_saved,
                screenshot_path=screenshot_path_str,
                trace_saved=trace_saved,
                trace_path=str(trace_path) if trace_saved and trace_path else None,
                reason=reason,
            )
        except Exception as exc:
            logger.error("Error during Indeed job inspection: %s", exc)
            return IndeedJobInspectionResult(
                platform="indeed",
                authenticated=True,
                job_url=job_url,
                application_method="unknown",
                apply_button_found=False,
                status="error",
                reason=f"Inspection failed: {str(exc)}",
            )
        finally:
            if context:
                await self.playwright_manager.close_context(context)

    async def _extract_job_title(self, page: Page) -> Optional[str]:
        """Extract visible job title using standard Indeed job-specific DOM selectors only."""
        selectors = [
            'h1[data-testid="jobsearch-JobInfoHeader-title"]',
            "h1.jobsearch-JobInfoHeader-title",
            '[data-testid="jobsearch-JobInfoHeader-title"]',
            ".jobsearch-JobInfoHeader-title",
            '[data-testid="simpler-jobTitle"]',
        ]
        for sel in selectors:
            try:
                elem = await page.query_selector(sel)
                if elem and await elem.is_visible():
                    text = await elem.inner_text()
                    if text and text.strip():
                        cleaned = text.strip()
                        # Reject generic headings
                        lower = cleaned.lower()
                        if lower not in (
                            "just a moment...",
                            "just a moment",
                            "additional verification required",
                            "welcome",
                            "sign in",
                            "log in",
                        ) and not lower.startswith("welcome,"):
                            return cleaned
            except Exception:
                continue
        return None

    async def _extract_company(self, page: Page) -> Optional[str]:
        """Extract visible hiring company name."""
        selectors = [
            '[data-testid="inlineHeader-companyName"]',
            '[data-company-name="true"]',
            ".jobsearch-CompanyInfoContainer a",
            ".jobsearch-InlineCompanyRating-companyHeader",
        ]
        for sel in selectors:
            try:
                elem = await page.query_selector(sel)
                if elem and await elem.is_visible():
                    text = await elem.inner_text()
                    if text and text.strip():
                        return text.strip()
            except Exception:
                continue
        return None

    async def _extract_location(self, page: Page) -> Optional[str]:
        """Extract visible job location."""
        selectors = [
            '[data-testid="inlineHeader-companyLocation"]',
            '[data-testid="job-location"]',
            ".jobsearch-JobInfoHeader-subtitle div",
        ]
        for sel in selectors:
            try:
                elem = await page.query_selector(sel)
                if elem and await elem.is_visible():
                    text = await elem.inner_text()
                    if text and text.strip():
                        return text.strip()
            except Exception:
                continue
        return None

    async def _detect_application_method(self, page: Page) -> tuple[str, bool, Optional[str], Optional[str]]:
        """Detect whether the job uses Indeed-hosted apply, external redirect, or has no apply button.

        Returns:
            tuple of (application_method, apply_button_found, apply_button_text, external_url)
        """
        # 1. Check for security challenge / blocked page case-insensitively
        try:
            content = await page.content()
            content_lower = content.lower()
            challenge_markers = (
                "challenge-running",
                "cf-turnstile",
                "please verify you are a human",
                "additional verification required",
                "just a moment",
            )
            if any(marker in content_lower for marker in challenge_markers):
                logger.info("Security challenge marker detected on page. Classifying as unknown application method.")
                return ("unknown", False, None, None)
        except Exception:
            pass

        # 2. Check for Indeed-Hosted Apply button
        for sel in self.INDEED_APPLY_SELECTORS:
            try:
                elem = await page.query_selector(sel)
                if elem and await elem.is_visible():
                    btn_text = (await elem.inner_text()).strip() if await elem.inner_text() else "Apply now"
                    logger.info("Detected Indeed-hosted apply button with selector '%s' (text: '%s')", sel, btn_text)
                    return ("indeed_hosted", True, btn_text, None)
            except Exception:
                continue

        # 3. Check for External Apply link / button
        for sel in self.EXTERNAL_APPLY_SELECTORS:
            try:
                elem = await page.query_selector(sel)
                if elem and await elem.is_visible():
                    btn_text = (await elem.inner_text()).strip() if await elem.inner_text() else "Apply on company site"
                    href = await elem.get_attribute("href")
                    logger.info("Detected external apply link with selector '%s' (href: '%s')", sel, href)
                    return ("external", True, btn_text, href)
            except Exception:
                continue

        # 4. No apply button detected
        return ("none", False, None, None)

    async def _inspect_accessible_forms(self, page: Page) -> List[FormFieldInfo]:
        """Inspect visible application-form fields only.

        Search/navigation forms are deliberately excluded.
        No current field values are read.
        """
        fields: List[FormFieldInfo] = []

        try:
            forms = await page.query_selector_all("form")

            for form in forms:
                try:
                    if not await form.is_visible():
                        continue

                    # Ignore site search/navigation forms
                    role = (await form.get_attribute("role") or "").lower()
                    aria_label = (await form.get_attribute("aria-label") or "").lower()
                    action = (await form.get_attribute("action") or "").lower()

                    if (
                        role == "search"
                        or "search" in aria_label
                        or "search" in action
                    ):
                        continue

                    inputs = await form.query_selector_all("input, select, textarea")

                    # Indeed's normal job-search form commonly contains q/l
                    names = set()
                    for inp in inputs:
                        name = await inp.get_attribute("name")
                        if name:
                            names.add(name.lower())

                    if names and names.issubset({"q", "l"}):
                        continue

                    # Be conservative: only treat the form as an application form if it contains application-like evidence
                    has_file_upload = await form.query_selector('input[type="file"]')

                    form_text = ""
                    try:
                        form_text = (await form.inner_text()).lower()
                    except Exception:
                        pass

                    application_terms = (
                        "application",
                        "resume",
                        "cv",
                        "cover letter",
                        "screening",
                        "phone",
                        "email",
                    )

                    looks_like_application = (
                        has_file_upload is not None
                        or any(term in form_text for term in application_terms)
                    )

                    if not looks_like_application:
                        continue

                    for inp in inputs:
                        try:
                            if not await inp.is_visible():
                                continue

                            tag_name = await (await inp.get_property("tagName")).json_value()
                            field_type = (await inp.get_attribute("type") or tag_name.lower())

                            if field_type.lower() in ("hidden", "submit", "button"):
                                continue

                            name = await inp.get_attribute("name")
                            required = (
                                await inp.get_attribute("required") is not None
                                or await inp.get_attribute("aria-required") == "true"
                            )
                            label = (
                                await inp.get_attribute("aria-label")
                                or await inp.get_attribute("placeholder")
                                or name
                                or "input_field"
                            )

                            fields.append(
                                FormFieldInfo(
                                    label=label.strip(),
                                    name=name.strip() if name else None,
                                    type=field_type.lower(),
                                    required=required,
                                )
                            )
                        except Exception:
                            continue

                except Exception:
                    continue

        except Exception as exc:
            logger.debug("Application form inspection skipped: %s", exc)

        return fields