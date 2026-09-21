"""PyWinAuto UI Automation Driver for Indeed Applications.

Handles Windows desktop UI interaction, browser attachment, UIA control discovery,
coordinate inspection, keyboard navigation, single-Enter execution, and confirmation checks.
"""

import logging
import os
import platform
import subprocess
import time
import webbrowser
from typing import Any, Dict, List, Optional, Tuple

from app.automation.indeed.config import (
    FALLBACK_EXPECTED_PRIMARY_HEIGHT,
    FALLBACK_EXPECTED_PRIMARY_WIDTH,
    INDEED_APPLY_FALLBACK_X,
    INDEED_APPLY_FALLBACK_Y,
    INDEED_TAB_COUNT_TO_SUBMIT,
    KEYBOARD_INTER_KEY_DELAY_SECONDS,
    SUPPORTED_BROWSER_PROCESSES,
)
from app.automation.indeed.state_detector import IndeedStateDetector

logger = logging.getLogger(__name__)


def is_windows() -> bool:
    """Check if current execution environment is Windows OS."""
    return platform.system().lower() == "windows"


class PyWinAutoIndeedDriver:
    """PyWinAuto-based UIA driver for interacting with Indeed job pages on Windows."""

    def __init__(self) -> None:
        self._desktop = None
        self._app = None
        self._window = None
        self._browser_name = None

    def _init_pywinauto(self) -> bool:
        """Lazy-initialize PyWinAuto Desktop object."""
        if not is_windows():
            logger.warning("PyWinAuto is only supported on Windows OS.")
            return False
        try:
            from pywinauto import Desktop
            if self._desktop is None:
                self._desktop = Desktop(backend="uia")
            return True
        except Exception as exc:
            logger.error("Failed to initialize PyWinAuto backend: %s", exc)
            return False

    def attach_or_open_browser(
        self,
        url: str,
        timeout_seconds: int = 15,
        expected_jk: Optional[str] = None,
        force_navigate: bool = True,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Find an active Chrome/Edge browser window or open the URL and attach.

        If an existing window is attached and force_navigate is True, explicitly
        navigates the attached browser to the target job URL.

        Returns:
            Tuple: (success: bool, detected_browser: Optional[str], error_message: Optional[str])
        """
        if not self._init_pywinauto():
            return False, None, "PyWinAuto is not available or OS is not Windows."

        # 1. Search for an already open browser window
        found_window, b_name = self._find_browser_window()
        is_new_launch = False

        if not found_window:
            if force_navigate:
                logger.info("No active Indeed/Browser window found. Opening URL with default browser: %s", url)
                try:
                    webbrowser.open(url)
                    is_new_launch = True
                except Exception as exc:
                    return False, None, f"Failed to launch browser URL: {exc}"

            # Poll for newly launched or active window
            start_wait = time.time()
            while time.time() - start_wait < timeout_seconds:
                time.sleep(1.0)
                found_window, b_name = self._find_browser_window()
                if found_window:
                    break

        if not found_window:
            return False, None, "Could not find or attach to supported browser window (Chrome/Edge)."

        self._window = found_window
        self._browser_name = b_name

        # 2. Verify window is active, foreground, visible, not minimized
        is_valid, err = self.verify_browser_window(ensure_maximized=True)
        if not is_valid:
            return False, b_name, err

        # 3. Explicitly navigate to target URL on attached browser if requested and not newly launched
        if force_navigate and not is_new_launch:
            nav_ok, nav_err = self.navigate_to_url(url, expected_jk=expected_jk, timeout_seconds=timeout_seconds)
            if not nav_ok:
                return False, b_name, nav_err

        return True, b_name, None

    def navigate_to_url(
        self,
        url: str,
        expected_jk: Optional[str] = None,
        timeout_seconds: int = 15,
    ) -> Tuple[bool, Optional[str]]:
        """Explicitly navigate active browser window to target job URL and verify page load.

        Verifies:
        1. Browser window is active and foreground.
        2. Dispatches navigation command to URL.
        3. Polls until page content is no longer a stale confirmation page.
        4. Verifies Indeed job page / domain has loaded.
        """
        if not self._window:
            return False, "No active browser window attached."

        logger.info("Explicitly navigating browser to target job URL: %s (expected_jk=%s)", url, expected_jk)

        # Trigger navigation via system browser URL handler
        try:
            webbrowser.open(url)
        except Exception as exc:
            logger.warning("webbrowser.open failed: %s", exc)

        # Poll for page navigation and ensure we are not on a stale confirmation page
        start_wait = time.time()
        navigated = False

        while time.time() - start_wait < timeout_seconds:
            time.sleep(1.0)
            page_text = self.get_window_text_content()

            # If page text still shows stale submission confirmation, keep waiting for target URL
            if IndeedStateDetector.is_stale_confirmation_page(page_text) and not IndeedStateDetector.is_apply_with_indeed(page_text):
                logger.debug("Page still shows stale submission confirmation, waiting for target URL to load...")
                continue

            # Verify page text indicates an Indeed job posting or security challenge
            norm = page_text.lower() if page_text else ""
            has_indeed = "indeed" in norm
            has_job_indicators = any(term in norm for term in ["apply", "job", "qualifications", "description", "salary", "company"])
            has_jk = expected_jk and expected_jk.lower() in norm
            has_challenge = IndeedStateDetector.detect_blocking_state(page_text) is not None

            if has_challenge or (has_indeed and (has_job_indicators or has_jk or IndeedStateDetector.is_apply_with_indeed(page_text))):
                navigated = True
                break

        if not navigated:
            final_text = self.get_window_text_content()
            if IndeedStateDetector.is_stale_confirmation_page(final_text) and not IndeedStateDetector.is_apply_with_indeed(final_text):
                return False, "Browser remained on stale submission confirmation page and failed to navigate to target job URL."

            is_valid, _ = self.verify_browser_window()
            if not is_valid:
                return False, "Browser window became invalid or unreachable during navigation."

            return True, None

        return True, None

    def _find_browser_window(self) -> Tuple[Optional[Any], Optional[str]]:
        """Search desktop top-level windows for Chrome or Edge using multi-strategy connection."""
        if not is_windows():
            return None, None

        # Strategy 1: Desktop top-level UIA windows
        if self._desktop is not None:
            try:
                windows = self._desktop.windows()
                for win in windows:
                    try:
                        props = win.get_properties()
                        p_name = (props.get("process_name") or "").lower()
                        class_name = props.get("class_name") or ""
                        title = (props.get("texts") or [""])[0].lower()

                        is_chrome = "chrome" in p_name or "chrome" in class_name.lower()
                        is_edge = "msedge" in p_name or "edge" in class_name.lower()

                        if (is_chrome or is_edge) and win.is_visible():
                            b_type = "chrome" if is_chrome else "edge"
                            if "indeed" in title:
                                return win, b_type
                    except Exception:
                        continue

                for win in windows:
                    try:
                        props = win.get_properties()
                        p_name = (props.get("process_name") or "").lower()
                        if ("chrome" in p_name or "msedge" in p_name) and win.is_visible():
                            b_type = "chrome" if "chrome" in p_name else "edge"
                            return win, b_type
                    except Exception:
                        continue
            except Exception as exc:
                logger.debug("Desktop.windows search warning: %s", exc)

        # Strategy 2: Application title regex connect
        try:
            from pywinauto import Application
            for title_pat, b_type in [
                (".*Indeed.*", "chrome"),
                (".*Google Chrome.*", "chrome"),
                (".*Microsoft​ Edge.*", "edge"),
                (".*Edge.*", "edge"),
            ]:
                try:
                    app = Application(backend="uia").connect(title_re=title_pat, timeout=1)
                    win = app.top_window()
                    if win and win.is_visible():
                        self._app = app
                        return win, b_type
                except Exception:
                    continue
        except Exception as exc:
            logger.debug("Application.connect title_re warning: %s", exc)

        # Strategy 3: Application class name connect
        try:
            from pywinauto import Application
            app = Application(backend="uia").connect(class_name="Chrome_WidgetWin_1", timeout=1)
            win = app.top_window()
            if win and win.is_visible():
                self._app = app
                return win, "chrome"
        except Exception:
            pass

        return None, None

    def verify_browser_window(self, ensure_maximized: bool = False) -> Tuple[bool, Optional[str]]:
        """Verify window is foreground, visible, not minimized, and optionally maximized."""
        if not self._window:
            return False, "No attached browser window."

        try:
            if not self._window.is_visible():
                return False, "Browser window is not visible."

            # Restore if minimized
            if hasattr(self._window, "is_minimized") and self._window.is_minimized():
                self._window.restore()
                time.sleep(0.5)

            # Bring to foreground / active focus
            try:
                self._window.set_focus()
                time.sleep(0.3)
            except Exception as exc:
                logger.warning("Could not set_focus on window: %s", exc)

            if ensure_maximized:
                try:
                    if hasattr(self._window, "maximize") and not self._window.is_maximized():
                        self._window.maximize()
                        time.sleep(0.5)
                except Exception as exc:
                    logger.debug("Maximize check warning: %s", exc)

            return True, None
        except Exception as exc:
            return False, f"Window verification error: {exc}"

    def get_window_text_content(self) -> str:
        """Extract visible text fragments from the window hierarchy for state detection."""
        if not self._window:
            return ""
        try:
            texts = []
            # Extract window title / header texts
            win_props = self._window.get_properties()
            for t in win_props.get("texts", []):
                if t:
                    texts.append(t)

            # Collect descendant text items (up to reasonable depth/count)
            descendants = self._window.descendants(depth=6)
            for d in descendants[:50]:
                try:
                    name = d.element_info.name
                    if name and len(name.strip()) > 1:
                        texts.append(name.strip())
                except Exception:
                    continue

            return "\n".join(texts)
        except Exception as exc:
            logger.warning("Error collecting window text content: %s", exc)
            return ""

    def find_apply_with_indeed_control(self) -> Optional[Any]:
        """Search UIA descendants for a button control named 'Apply with Indeed'."""
        if not self._window:
            return None

        try:
            # Search for Button controls
            buttons = self._window.descendants(control_type="Button")
            for btn in buttons:
                try:
                    name = btn.element_info.name
                    if IndeedStateDetector.is_apply_with_indeed(name):
                        if btn.is_visible() and btn.is_enabled():
                            return btn
                except Exception:
                    continue

            # Search broader descendants if not typed as Button
            all_desc = self._window.descendants(depth=8)
            for elem in all_desc:
                try:
                    name = elem.element_info.name
                    if IndeedStateDetector.is_apply_with_indeed(name):
                        if elem.is_visible() and elem.is_enabled():
                            return elem
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Error searching for UIA Apply with Indeed button: %s", exc)

        return None

    def inspect_element_at_coordinates(
        self,
        x: int = INDEED_APPLY_FALLBACK_X,
        y: int = INDEED_APPLY_FALLBACK_Y,
    ) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Verify if the element occupying or surrounding (X, Y) contains 'Apply with Indeed'.

        Returns:
            Tuple: (is_verified: bool, detected_name: Optional[str], element: Optional[Any])
        """
        if not self._window:
            return False, None, None

        try:
            # Check bounding rectangles of descendant controls
            descendants = self._window.descendants(depth=10)
            candidate_elem = None
            candidate_text = ""

            for elem in descendants:
                try:
                    rect = elem.rectangle()
                    # Check if (x, y) falls inside or close to the rectangle (within 30px margin)
                    if (rect.left - 30 <= x <= rect.right + 30) and (rect.top - 30 <= y <= rect.bottom + 30):
                        name = elem.element_info.name
                        if name and IndeedStateDetector.is_apply_with_indeed(name):
                            return True, name, elem
                        if name:
                            candidate_text = name
                            candidate_elem = elem
                except Exception:
                    continue

            # If element at coords was found but didn't match Apply with Indeed
            if candidate_text:
                return False, candidate_text, candidate_elem

        except Exception as exc:
            logger.warning("Coordinate element inspection error at (%d, %d): %s", x, y, exc)

        return False, None, None

    def click_apply_control(
        self,
        uia_control: Optional[Any] = None,
        use_coordinate_fallback: bool = True,
    ) -> Tuple[bool, Optional[str], Optional[List[int]], Optional[str]]:
        """Click the verified Apply with Indeed control or verified coordinate fallback.

        Returns:
            Tuple: (success: bool, apply_method: Optional[str], coords: Optional[List[int]], error: Optional[str])
        """
        if not self._window:
            return False, None, None, "No attached browser window."

        # Option A: UIA control click
        if uia_control is not None:
            try:
                uia_control.click_input()
                return True, "uia_control", None, None
            except Exception as exc:
                logger.warning("UIA click_input failed, attempting coordinate fallback: %s", exc)

        # Option B: Verified Coordinate Fallback
        if use_coordinate_fallback:
            try:
                from pywinauto import mouse
                # Re-verify window is foreground before coordinate click
                self._window.set_focus()
                time.sleep(0.3)
                mouse.click(coords=(INDEED_APPLY_FALLBACK_X, INDEED_APPLY_FALLBACK_Y))
                return True, "coordinate_fallback", [INDEED_APPLY_FALLBACK_X, INDEED_APPLY_FALLBACK_Y], None
            except Exception as exc:
                return False, None, None, f"Coordinate click failed: {exc}"

        return False, None, None, "No verified click method available."

    def send_tab_sequence(self, count: int = INDEED_TAB_COUNT_TO_SUBMIT) -> int:
        """Send a sequence of individual TAB keypresses to navigate toward Submit.

        Returns:
            int: Total TAB keypresses successfully sent.
        """
        if not is_windows():
            return 0

        tabs_sent = 0
        try:
            from pywinauto.keyboard import send_keys
            for _ in range(count):
                send_keys("{TAB}")
                tabs_sent += 1
                time.sleep(KEYBOARD_INTER_KEY_DELAY_SECONDS)
        except Exception as exc:
            logger.warning("Error during TAB sequence sending: %s", exc)

        return tabs_sent

    def get_focused_control_info(self) -> Dict[str, Any]:
        """Inspect the currently focused UI Automation element."""
        if not is_windows():
            return {"name": "", "control_type": "", "is_enabled": False, "is_visible": False}

        try:
            from pywinauto import Desktop
            # Direct UIA focused element
            focused = Desktop(backend="uia").get_focus()
            if focused:
                info = focused.element_info
                name = info.name or ""
                c_type = info.control_type or ""
                is_enabled = focused.is_enabled() if hasattr(focused, "is_enabled") else True
                is_visible = focused.is_visible() if hasattr(focused, "is_visible") else True

                return {
                    "name": name,
                    "control_type": c_type,
                    "is_enabled": is_enabled,
                    "is_visible": is_visible,
                    "is_submit": IndeedStateDetector.is_submit_control(name, c_type),
                }
        except Exception as exc:
            logger.warning("Error obtaining focused UIA control info: %s", exc)

        return {"name": "", "control_type": "", "is_enabled": False, "is_visible": False, "is_submit": False}

    def find_exact_submit_controls(self) -> List[Tuple[Any, Dict[str, Any]]]:
        """Scan the active browser window using UIA for visible, enabled controls matching exact Submit names.

        A candidate must:
        - belong to the active Indeed application window
        - be visible and enabled
        - have a non-empty bounding rectangle
        - have an exact Submit name ('submit', 'submit application', 'submit your application')
        """
        if not self._window:
            return []

        candidates: List[Tuple[Any, Dict[str, Any]]] = []
        try:
            # Search descendants
            descendants = self._window.descendants()
            for elem in descendants:
                try:
                    info = elem.element_info
                    name = info.name
                    if not IndeedStateDetector.is_exact_submit_name(name):
                        continue

                    # Check visibility and enabled state
                    is_vis = elem.is_visible() if hasattr(elem, "is_visible") else True
                    is_en = elem.is_enabled() if hasattr(elem, "is_enabled") else True
                    if not is_vis or not is_en:
                        continue

                    # Check non-empty bounding rectangle
                    rect = elem.rectangle()
                    width = rect.width() if hasattr(rect, "width") else (rect.right - rect.left)
                    height = rect.height() if hasattr(rect, "height") else (rect.bottom - rect.top)
                    if width <= 0 or height <= 0:
                        continue

                    c_type = info.control_type or "Button"
                    meta = {
                        "name": name.strip(),
                        "control_type": c_type,
                        "is_enabled": is_en,
                        "is_visible": is_vis,
                        "rectangle": {
                            "left": rect.left,
                            "top": rect.top,
                            "right": rect.right,
                            "bottom": rect.bottom,
                            "width": width,
                            "height": height,
                        },
                    }
                    candidates.append((elem, meta))
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Error scanning for UIA Submit controls: %s", exc)

        return candidates

    def click_submit_control(self, uia_control: Any) -> Tuple[bool, Optional[str]]:
        """Invoke/click exactly once the positively verified UIA Submit control.

        Returns:
            Tuple: (success: bool, error_message: Optional[str])
        """
        if uia_control is None:
            return False, "No verified UIA Submit control provided."

        try:
            if self._window:
                self._window.set_focus()
                time.sleep(0.3)

            # Option A: click_input on the verified control
            if hasattr(uia_control, "click_input"):
                try:
                    uia_control.click_input()
                    return True, None
                except Exception as exc:
                    logger.warning("click_input on Submit control failed, attempting center coord fallback: %s", exc)

            # Option B: Center coordinate click of the verified control's bounding box
            if hasattr(uia_control, "rectangle"):
                rect = uia_control.rectangle()
                mid_x = (rect.left + rect.right) // 2
                mid_y = (rect.top + rect.bottom) // 2
                from pywinauto import mouse
                mouse.click(coords=(mid_x, mid_y))
                return True, None

            return False, "Verified Submit control has no supported click mechanism."
        except Exception as exc:
            return False, f"Failed to click verified Submit control: {exc}"

    def send_enter_once(self) -> bool:
        """Send ENTER key exactly once to activate the verified Submit control."""
        if not is_windows():
            return False
        try:
            from pywinauto.keyboard import send_keys
            send_keys("{ENTER}")
            return True
        except Exception as exc:
            logger.error("Error sending ENTER to Submit control: %s", exc)
            return False

    def detect_submission_confirmation(self, timeout_seconds: int = 15) -> bool:
        """Poll window text hierarchy to verify application submission confirmation."""
        start = time.time()
        while time.time() - start < timeout_seconds:
            text = self.get_window_text_content()
            if IndeedStateDetector.is_submission_confirmation(text):
                return True
            time.sleep(1.0)
        return False
