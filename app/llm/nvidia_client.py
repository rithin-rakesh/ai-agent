"""NVIDIA NIM API Client Layer.

Handles secure communication with the NVIDIA API endpoints (embeddings, chat completions)
with timeout controls, bounded retries, authentication validation, and secret protection.
"""

import logging
import time
from typing import Any, Dict, Optional
import httpx

from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class NVIDIAClientError(Exception):
    """Base exception for NVIDIA API client errors."""

    pass


class NVIDIAAuthError(NVIDIAClientError):
    """Authentication or authorization failure (HTTP 401/403)."""

    pass


class NVIDIAModelNotFoundError(NVIDIAClientError):
    """Specified model identifier was not found on NVIDIA NIM (HTTP 404)."""

    pass


class NVIDIAClient:
    """HTTP Client for interacting with NVIDIA NIM API."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.NVIDIA_BASE_URL.rstrip("/")
        self.timeout = self.settings.NVIDIA_REQUEST_TIMEOUT_SECONDS

    @property
    def is_configured(self) -> bool:
        """Check if NVIDIA API key is configured."""
        key = self.settings.get_nvidia_api_key()
        return bool(key and key.strip())

    def _get_headers(self) -> Dict[str, str]:
        """Construct HTTP headers without exposing secrets."""
        key = self.settings.get_nvidia_api_key()
        if not key:
            raise NVIDIAAuthError("NVIDIA_API_KEY is not configured in settings or environment.")
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AI-Job-Application-Agent/0.4.0",
        }

    def post(
        self,
        path: str,
        payload: Dict[str, Any],
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """Execute a POST request against NVIDIA NIM API with bounded retries on transient errors.

        Args:
            path: Relative path endpoint (e.g. "/embeddings" or "/chat/completions")
            payload: JSON request body dictionary
            max_retries: Number of retry attempts on 5xx or connection errors

        Returns:
            Dict[str, Any]: Parsed JSON response dictionary

        Raises:
            NVIDIAAuthError: If authentication fails (401/403)
            NVIDIAModelNotFoundError: If model is not found (404)
            NVIDIAClientError: For all other unrecoverable errors
        """
        endpoint = f"{self.base_url}/{path.lstrip('/')}"
        headers = self._get_headers()

        attempts = 0
        last_error = None

        while attempts <= max_retries:
            attempts += 1
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(endpoint, json=payload, headers=headers)

                if response.status_code in (401, 403):
                    error_msg = f"NVIDIA API Authentication failed ({response.status_code}). Please verify your NVIDIA_API_KEY."
                    logger.error("NVIDIA authentication failed: %s", response.status_code)
                    raise NVIDIAAuthError(error_msg)

                if response.status_code == 404:
                    error_body = response.text[:200]
                    error_msg = f"NVIDIA Model/Endpoint not found (404): {error_body}"
                    logger.error("NVIDIA model not found: %s", error_msg)
                    raise NVIDIAModelNotFoundError(error_msg)

                if response.status_code >= 500:
                    logger.warning(
                        "NVIDIA API returned server error (%d). Attempt %d/%d...",
                        response.status_code,
                        attempts,
                        max_retries + 1,
                    )
                    if attempts <= max_retries:
                        time.sleep(1.0 * attempts)
                        continue
                    response.raise_for_status()

                response.raise_for_status()
                return response.json()

            except (NVIDIAAuthError, NVIDIAModelNotFoundError):
                raise
            except httpx.RequestError as exc:
                last_error = exc
                logger.warning(
                    "Network error connecting to NVIDIA API: %s. Attempt %d/%d...",
                    type(exc).__name__,
                    attempts,
                    max_retries + 1,
                )
                if attempts <= max_retries:
                    time.sleep(1.0 * attempts)
                    continue
                raise NVIDIAClientError(f"Failed to communicate with NVIDIA API: {exc}") from exc
            except httpx.HTTPStatusError as exc:
                last_error = exc
                logger.error("NVIDIA API HTTP error: %d - %s", exc.response.status_code, exc.response.text[:200])
                raise NVIDIAClientError(f"NVIDIA API error ({exc.response.status_code}): {exc.response.text[:200]}") from exc

        raise NVIDIAClientError(f"NVIDIA API request failed after {max_retries + 1} attempts: {last_error}")
