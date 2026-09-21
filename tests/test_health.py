"""Tests for Health Check, Root Endpoint, and Configuration."""

import os
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.main import app
from app.config.settings import Settings, get_settings
from app.database.supabase import (
    get_supabase_client,
    get_supabase_service_client,
    reset_clients,
)


@pytest.fixture(autouse=True)
def clean_database_clients():
    """Ensure database singleton clients are reset before and after each test."""
    reset_clients()
    yield
    reset_clients()


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_health_check_success(client: TestClient):
    """Test that GET /health returns 200 OK and expected structure."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "environment" in data
    assert isinstance(data["environment"], str)


def test_root_endpoint_success(client: TestClient):
    """Test that GET / returns 200 OK and application identifier message."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "message" in data
    assert "AI Job Application Agent" in data["message"]
    assert "phase" in data
    assert "Phase" in data["phase"]


def test_settings_secret_masking():
    """Test that secret keys are masked in string representations and logs."""
    test_settings = Settings(
        SUPABASE_URL="https://example.supabase.co",
        SUPABASE_ANON_KEY="anon-public-key",
        SUPABASE_SERVICE_ROLE_KEY=SecretStr("super-secret-service-role-key"),
        APP_ENV="test",
    )

    settings_repr = repr(test_settings)
    assert "super-secret-service-role-key" not in settings_repr
    assert "**********" in settings_repr or "SecretStr" in settings_repr

    # Verify secret value can be accessed safely via method
    assert test_settings.get_service_role_key() == "super-secret-service-role-key"


def test_supabase_client_requires_credentials():
    """Test that Supabase client factory raises ValueError when credentials are missing."""
    empty_settings = Settings(
        SUPABASE_URL="",
        SUPABASE_ANON_KEY="",
        SUPABASE_SERVICE_ROLE_KEY=None,
        APP_ENV="test",
    )

    with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_ANON_KEY must be set"):
        get_supabase_client(settings=empty_settings)

    with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set"):
        get_supabase_service_client(settings=empty_settings)
