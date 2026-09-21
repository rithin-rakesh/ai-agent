"""Global pytest fixtures and isolation configuration."""

import os
import pytest


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch):
    """Ensure all test execution runs in strict isolation with live CDP and live Apify disabled by default."""
    from app.config.settings import get_settings
    monkeypatch.setenv("ALLOW_LIVE_CDP", "false")
    monkeypatch.setenv("APIFY_GLASSDOOR_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


