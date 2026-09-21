"""Live Browser Testing Suite for Glassdoor Automation (Phase 5.6).

These tests run strictly against a live dedicated Chrome CDP instance (port 9222).
They are excluded from default unit test runs and must be invoked explicitly with:
    pytest tests/live/test_glassdoor_live.py -v -m live_browser
"""

import os
import pytest
from app.automation.glassdoor.config import ALLOW_LIVE_CDP, CDP_HOST, CDP_REMOTE_DEBUGGING_PORT
from app.automation.glassdoor.playwright_driver import GlassdoorPlaywrightDriver
from app.automation.glassdoor.apply_service import GlassdoorApplyService
from app.automation.glassdoor.models import GlassdoorAutomationInspectRequest


@pytest.mark.live_browser
def test_live_cdp_connection_status():
    """Verify live CDP connection when explicitly enabled."""
    if not ALLOW_LIVE_CDP and os.getenv("ALLOW_LIVE_CDP", "false").lower() not in ("true", "1", "yes"):
        pytest.skip("ALLOW_LIVE_CDP is false. Set ALLOW_LIVE_CDP=true to run live browser tests.")

    driver = GlassdoorPlaywrightDriver(host=CDP_HOST, port=CDP_REMOTE_DEBUGGING_PORT)
    ok, err = driver.connect_cdp(timeout_seconds=5)
    assert ok is True, f"Live CDP connection failed: {err}"
    assert driver.page is not None
    driver.close()
