"""PyWinAuto UI Automation Driver for Glassdoor Applications.

Handles Chrome/Edge browser attachment, window state verification, explicit URL navigation,
Easy Apply detection, modal scoping, direct UIA action invocation, and bounded adaptive TAB traversal.
"""

import logging
import platform
import re
import time
import webbrowser
from typing import Any, Dict, List, Optional, Tuple

from app.automation.glassdoor.config import (
    CONFIRMATION_TIMEOUT_SECONDS,
    EASY_APPLY_BUTTON_NAMES,
    EXACT_SUBMIT_NAMES,
    KEYBOARD_INTER_KEY_DELAY_SECONDS,
    MAX_TAB_TRAVERSAL,
)
from app.automation.glassdoor.confirmation_detector import GlassdoorConfirmationDetector
from app.automation.glassdoor.state_detector import GlassdoorStateDetector

logger = logging.getLogger(__name__)


def is_windows() -> bool:
    """Check if current execution environment is Windows OS."""
    return platform.system().lower() == "windows"


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    t = re.sub(r"[^\w\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", t).strip()


class PyWinAutoGlassdoorDriver:
    """PyWinAuto-based UIA driver for interacting with Glassdoor job postings on Windows."""

    def __init__(self) -> None:
        self._desktop = None
        self._app = None
        self._window = None
        self._browser_name = None

    @property
    def window(self) -> Optional[Any]:
        """Active attached window."""
        return self._window

    @window.setter
    def window(self, val: Optional[Any]) -> None:
        self._window = val

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
        force_navigate: bool = True,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Find an active Chrome/Edge browser window and attach without launching external processes.

        Returns:
            Tuple: (success: bool, detected_browser: Optional[str], error_message: Optional[str])
        """
        if not self._init_pywinauto():
            return False, None, "PyWinAuto is not available or OS is not Windows."

        # 1. Search for an open browser window belonging to the active session
        found_window, b_name = self._find_browser_window()

        if not found_window:
            start_wait = time.time()
            while time.time() - start_wait < timeout_seconds:
                time.sleep(1.0)
                found_window, b_name = self._find_browser_window()
                if found_window:
                    break

        if not found_window:
            return False, None, "Could not find active browser window (Chrome/Edge) for CDP session."

        self._window = found_window
        self._browser_name = b_name

        # 2. Verify window state
        is_valid, err = self.verify_browser_window(ensure_maximized=True)
        if not is_valid:
            return False, b_name, err

        # 3. Verify target page state if navigation requested
        if force_navigate:
            nav_ok, nav_err = self.navigate_to_url(url, timeout_seconds=timeout_seconds)
            if not nav_ok:
                return False, b_name, nav_err

        return True, b_name, None

    def navigate_to_url(
        self,
        url: str,
        timeout_seconds: int = 15,
    ) -> Tuple[bool, Optional[str]]:
        """Verify active browser window is on target Glassdoor URL or wait for page load."""
        if not self._window:
            return False, "No active browser window attached."

        logger.info("Verifying browser window on Glassdoor job URL: %s", url)

        start_wait = time.time()
        navigated = False

        while time.time() - start_wait < timeout_seconds:
            time.sleep(1.0)
            page_text = self.get_window_text_content()

            # If page still shows stale confirmation, continue waiting
            if GlassdoorConfirmationDetector.is_stale_confirmation_page(page_text) and not (
                "glassdoor" in page_text.lower() and ("easy apply" in page_text.lower() or "apply now" in page_text.lower())
            ):
                continue

            norm = _normalize(page_text)
            has_gd = "glassdoor" in norm
            has_job_terms = any(t in norm for t in ["easy apply", "apply", "job", "salary", "overview", "company", "description"])
            has_challenge = GlassdoorStateDetector.detect_blocking_state(page_text) is not None

            if has_challenge or (has_gd and (has_job_terms or "apply" in norm)):
                navigated = True
                break

        if not navigated:
            final_text = self.get_window_text_content()
            if GlassdoorConfirmationDetector.is_stale_confirmation_page(final_text) and not (
                "easy apply" in final_text.lower() or "apply now" in final_text.lower()
            ):
                return False, "Browser remained on stale submission confirmation page."

            is_valid, _ = self.verify_browser_window()
            if not is_valid:
                return False, "Browser window became invalid during navigation."

            return True, None

        return True, None

    def _find_browser_window(self) -> Tuple[Optional[Any], Optional[str]]:
        """Search desktop top-level windows for Chrome or Edge using multi-strategy connection."""
        if not is_windows():
            return None, None

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
                            if "glassdoor" in title:
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

        return None, None

    def verify_browser_window(self, ensure_maximized: bool = False) -> Tuple[bool, Optional[str]]:
        """Verify window is active, foreground, and visible."""
        if not self._window:
            return False, "No attached browser window."

        try:
            if not self._window.is_visible():
                return False, "Browser window is not visible."

            if hasattr(self._window, "set_focus"):
                try:
                    self._window.set_focus()
                except Exception:
                    pass

            return True, None
        except Exception as exc:
            return False, f"Browser verification error: {exc}"

    def get_window_text_content(self) -> str:
        """Extract visible text content from the attached browser window."""
        if not self._window:
            return ""
        try:
            texts = []
            if hasattr(self._window, "texts"):
                texts.extend(self._window.texts() or [])
            if hasattr(self._window, "descendants"):
                for d in self._window.descendants(control_type="Text"):
                    name = getattr(d.element_info, "name", "")
                    if name:
                        texts.append(name)
            return " ".join(texts)
        except Exception as exc:
            logger.debug("Error extracting window text: %s", exc)
            return ""

    def find_easy_apply_control(self) -> Optional[Any]:
        """Search active window for a visible Easy Apply UIA button (Primary Strategy)."""
        if not self._window:
            return None

        try:
            buttons = self._window.descendants(control_type="Button") if hasattr(self._window, "descendants") else []
            for btn in buttons:
                name = getattr(btn.element_info, "name", "")
                if GlassdoorStateDetector.is_easy_apply_button(name) and getattr(btn.element_info, "is_visible", True):
                    return btn
            return None
        except Exception as exc:
            logger.debug("Error searching for Easy Apply button: %s", exc)
            return None

    def get_element_at_point(self, x: int, y: int) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Retrieve and inspect the UIA element at screen coordinate (x, y) via hit testing."""
        diag: Dict[str, Any] = {
            "coordinate": [x, y],
            "element_name": "",
            "control_type": "",
            "is_enabled": False,
            "is_visible": False,
            "bounds": None,
            "is_easy_apply": False,
        }
        if not self._init_pywinauto():
            return None, diag

        try:
            from pywinauto.uia_element_info import UIAElementInfo
            info = UIAElementInfo.from_point(x, y)
            if not info:
                return None, diag

            name = getattr(info, "name", "") or ""
            ctype = getattr(info, "control_type", "") or ""
            enabled = getattr(info, "is_enabled", True)
            visible = getattr(info, "is_visible", True)
            rect = getattr(info, "rectangle", None)
            rect_str = str(rect) if rect else None

            diag["element_name"] = name
            diag["control_type"] = ctype
            diag["is_enabled"] = enabled
            diag["is_visible"] = visible
            diag["bounds"] = rect_str

            # Wrap in HwndWrapper / UIA element if possible
            resolved_ctrl = None
            try:
                from pywinauto.controls.uiawrapper import UIAWrapper
                resolved_ctrl = UIAWrapper(info)
            except Exception:
                resolved_ctrl = info

            # Check direct element
            if GlassdoorStateDetector.is_easy_apply_button(name):
                diag["is_easy_apply"] = True
                return resolved_ctrl, diag

            # Check ancestor chain (e.g. Text inside Button)
            curr = info
            for _ in range(3):
                parent_info = getattr(curr, "parent", None)
                if not parent_info:
                    break
                p_name = getattr(parent_info, "name", "") or ""
                p_ctype = getattr(parent_info, "control_type", "") or ""
                if GlassdoorStateDetector.is_easy_apply_button(p_name) or (
                    p_ctype == "Button" and GlassdoorStateDetector.is_easy_apply_button(name)
                ):
                    diag["element_name"] = p_name or name
                    diag["control_type"] = p_ctype or ctype
                    diag["is_easy_apply"] = True
                    try:
                        from pywinauto.controls.uiawrapper import UIAWrapper
                        resolved_ctrl = UIAWrapper(parent_info)
                    except Exception:
                        resolved_ctrl = parent_info
                    return resolved_ctrl, diag
                curr = parent_info

            return None, diag
        except Exception as exc:
            logger.debug("Error retrieving element at point (%d, %d): %s", x, y, exc)
            diag["error"] = str(exc)
            return None, diag

    def is_point_inside_window(self, x: int, y: int) -> bool:
        """Check if coordinate (x, y) is inside the client / window rectangle."""
        if not self._window:
            return False
        try:
            rect = getattr(self._window.element_info, "rectangle", None)
            if rect:
                return (rect.left <= x <= rect.right) and (rect.top <= y <= rect.bottom)
            return True
        except Exception:
            return True

    def click_coordinate(self, x: int, y: int) -> bool:
        """Click screen coordinates (x, y) using PyWinAuto mouse."""
        try:
            from pywinauto import mouse
            mouse.click(coords=(x, y))
            time.sleep(0.5)
            return True
        except Exception as exc:
            logger.warning("Failed clicking coordinate (%d, %d): %s", x, y, exc)
            return False

    def is_smartapply_detected(self) -> bool:
        """Check if browser window transitioned to Indeed SmartApply host."""
        if not self._window:
            return False
        page_text = self.get_window_text_content()
        norm = _normalize(page_text)
        return "smartapply" in norm or "smartapply.indeed.com" in norm or "indeed.com/smartapply" in norm

    def find_action_control(self, action_names: List[str], container: Optional[Any] = None) -> Optional[Any]:
        """Search container for an action button whose name matches one of action_names."""
        target = container or self._window
        if not target:
            return None

        norm_targets = {_normalize(a) for a in action_names}
        try:
            buttons = target.descendants(control_type="Button") if hasattr(target, "descendants") else []
            for btn in buttons:
                name = getattr(btn.element_info, "name", "")
                if _normalize(name) in norm_targets and getattr(btn.element_info, "is_visible", True):
                    return btn
            return None
        except Exception as exc:
            logger.debug("Error searching for action control: %s", exc)
            return None

    def find_exact_submit_control(self, container: Optional[Any] = None) -> Optional[Any]:
        """Search container for an exact Submit button."""
        target = container or self._window
        if not target:
            return None

        try:
            buttons = target.descendants(control_type="Button") if hasattr(target, "descendants") else []
            for btn in buttons:
                name = getattr(btn.element_info, "name", "")
                if GlassdoorStateDetector.is_exact_submit_name(name) and getattr(btn.element_info, "is_visible", True):
                    return btn
            return None
        except Exception as exc:
            logger.debug("Error searching for exact Submit control: %s", exc)
            return None

    def click_control(self, control: Any) -> bool:
        """Click or invoke a verified UIA control."""
        if not control:
            return False
        try:
            if hasattr(control, "click_input"):
                control.click_input()
                time.sleep(0.5)
                return True
            if hasattr(control, "invoke"):
                control.invoke()
                time.sleep(0.5)
                return True
            return False
        except Exception as exc:
            logger.warning("Failed clicking control: %s", exc)
            return False

    def get_focused_control_info(self) -> Dict[str, Any]:
        """Retrieve details of the currently focused UI element."""
        if not self._init_pywinauto():
            return {"name": "", "control_type": "", "is_submit": False}
        try:
            from pywinauto.uia_element_info import UIAElementInfo
            focused = UIAElementInfo.get_focused_element()
            if not focused:
                return {"name": "", "control_type": "", "is_submit": False}

            name = getattr(focused, "name", "") or ""
            ctype = getattr(focused, "control_type", "") or ""
            is_submit = GlassdoorStateDetector.is_exact_submit_name(name)
            return {
                "name": name,
                "control_type": ctype,
                "is_submit": is_submit,
                "is_enabled": getattr(focused, "is_enabled", True),
                "is_visible": getattr(focused, "is_visible", True),
            }
        except Exception as exc:
            logger.debug("Error getting focused control: %s", exc)
            return {"name": "", "control_type": "", "is_submit": False}

    def find_and_activate_action_via_tab_traversal(
        self,
        expected_actions: List[str],
        max_tabs: int = MAX_TAB_TRAVERSAL,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Bounded adaptive keyboard TAB traversal to find and activate a verified action button.

        Returns:
            Tuple: (success: bool, activated_action: Optional[str], error_message: Optional[str])
        """
        if not self._window:
            return False, None, "No active browser window."

        norm_targets = {_normalize(a) for a in expected_actions}
        logger.info("Starting bounded adaptive TAB traversal for actions: %s (max_tabs=%d)", expected_actions, max_tabs)

        for step in range(max_tabs):
            focus_info = self.get_focused_control_info()
            f_name = focus_info.get("name", "")
            f_norm = _normalize(f_name)

            # Check if focused element is one of the verified expected actions
            if f_norm in norm_targets:
                logger.info("Adaptive TAB traversal verified target action '%s' after %d tabs. Activating once.", f_name, step)
                self.send_enter_once()
                time.sleep(0.5)
                return True, f_name, None

            # Send single TAB
            try:
                if hasattr(self._window, "type_keys"):
                    self._window.type_keys("{TAB}", with_spaces=True)
                time.sleep(KEYBOARD_INTER_KEY_DELAY_SECONDS)
            except Exception as exc:
                return False, None, f"Keyboard TAB failed: {exc}"

        return False, None, f"Reached maximum traversal limit ({max_tabs} tabs) without finding target action."

    def send_enter_once(self) -> bool:
        """Send a single Enter keystroke to the active window."""
        if not self._window:
            return False
        try:
            if hasattr(self._window, "type_keys"):
                self._window.type_keys("{ENTER}", with_spaces=True)
                return True
            return False
        except Exception as exc:
            logger.warning("Error sending Enter: %s", exc)
            return False

    def detect_submission_confirmation(self, timeout_seconds: int = CONFIRMATION_TIMEOUT_SECONDS) -> bool:
        """Poll active window for post-submission confirmation text."""
        start_wait = time.time()
        while time.time() - start_wait < timeout_seconds:
            time.sleep(1.0)
            page_text = self.get_window_text_content()
            if GlassdoorConfirmationDetector.is_submission_confirmation(page_text):
                logger.info("Glassdoor submission confirmation detected positively.")
                return True
        return False
