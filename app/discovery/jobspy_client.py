"""JobSpy MCP Client module.

Handles isolated communication with the standalone JobSpy MCP Server,
dispatching independent queries per source and capturing detailed diagnostics.
"""

import logging
import time
from typing import Any, Dict, List, Literal, Optional, Set
import httpx
from pydantic import BaseModel, Field

from app.config.settings import Settings, get_settings
from app.discovery.location import normalize_glassdoor_location

logger = logging.getLogger(__name__)

# Supported job boards in Phase 2.2
SUPPORTED_JOB_SOURCES: Set[str] = {
    "linkedin",
    "indeed",
    "naukri",
    "glassdoor",
}


class SourceResult(BaseModel):
    """Result of querying a single job source."""

    provider: str = Field(default="jobspy", description="Provider name")
    source: str = Field(..., description="Target source name (e.g. linkedin, indeed, glassdoor)")
    status: Literal["success", "failed"] = Field(..., description="'success' or 'failed'")
    provider_status: Optional[str] = Field(
        default=None,
        description="Granular provider status e.g. SUCCESS, UPSTREAM_BLOCKED, LOCATION_PARSE_ERROR, BRIDGE_UNAVAILABLE, NO_RESULTS",
    )
    http_status: Optional[int] = Field(default=None, description="HTTP status code from bridge or upstream")
    retryable: bool = Field(default=False, description="Whether this query failure is transient and retryable")
    jobs_returned: int = Field(default=0, description="Count of raw jobs returned")
    error: Optional[str] = Field(default=None, description="Diagnostic error message if failed")
    duration_ms: int = Field(default=0, description="Time taken in milliseconds")
    raw_jobs: List[Dict[str, Any]] = Field(default_factory=list, description="Raw job dictionaries from JobSpy")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Outbound request and execution diagnostics")


class JobSpyClient:
    """Client for communicating with the JobSpy MCP Server."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._http_client = http_client

    async def search_single_source(
        self,
        source: str,
        search_term: str,
        location: str,
        is_remote: bool = False,
        results_wanted: int = 20,
        offset: int = 0,
        hours_old: int = 72,
        country_indeed: str = "India",
        linkedin_fetch_description: bool = False,
    ) -> SourceResult:
        """Query a single job source independently and capture error details.

        Args:
            source: Name of the job site (e.g. 'glassdoor', 'indeed')
            search_term: Job search query or title
            location: Target location
            is_remote: Whether to filter for remote jobs only
            results_wanted: Number of results wanted
            offset: Pagination offset
            hours_old: Maximum age of jobs in hours
            country_indeed: Country code/name for queries
            linkedin_fetch_description: Whether to fetch full LinkedIn descriptions

        Returns:
            SourceResult with status, job counts, raw data, and structured error diagnostics.
        """
        start_time = time.perf_counter()

        if not self.settings.JOBSPY_ENABLED:
            return SourceResult(
                source=source,
                status="failed",
                provider_status="PROVIDER_DISABLED",
                error="JobSpy integration is disabled in settings (JOBSPY_ENABLED=false)",
                duration_ms=0,
            )

        # Normalize location and separate remote queries
        effective_is_remote = is_remote
        effective_location = location

        if source.lower() == "glassdoor":
            norm_loc, is_rem_detected = normalize_glassdoor_location(location)
            if is_remote or is_rem_detected:
                effective_is_remote = True
                effective_location = "remote"
            else:
                effective_is_remote = False
                effective_location = norm_loc

        api_url = self.settings.jobspy_api_url
        timeout_ms = int(self.settings.JOBSPY_TIMEOUT_SECONDS * 1000)

        # Build schema-compliant payload
        payload = {
            "siteNames": [source] if isinstance(source, str) else source,
            "searchTerm": search_term,
            "location": effective_location,
            "isRemote": effective_is_remote,
            "resultsWanted": results_wanted,
            "offset": offset,
            "hoursOld": hours_old,
            "countryIndeed": country_indeed,
            "linkedinFetchDescription": linkedin_fetch_description,
            "format": "json",
            "timeout": timeout_ms,
        }

        # Safe diagnostic metadata (no secrets)
        outbound_diag = {
            "provider": "jobspy",
            "site": source,
            "search_term": search_term,
            "location": effective_location,
            "country": country_indeed,
            "is_remote": effective_is_remote,
            "results_wanted": results_wanted,
            "offset": offset,
            "bridge_url": api_url,
            "timeout": timeout_ms,
            "request_payload": payload,
        }

        logger.info(
            "JobSpy outbound request: provider=jobspy site=%s search_term='%s' location='%s' country='%s' is_remote=%s results_wanted=%d offset=%d bridge_url=%s timeout=%d",
            source,
            search_term,
            effective_location,
            country_indeed,
            effective_is_remote,
            results_wanted,
            offset,
            api_url,
            timeout_ms,
        )

        try:
            timeout_cfg = httpx.Timeout(float(self.settings.JOBSPY_TIMEOUT_SECONDS) + 5.0)

            if self._http_client:
                response = await self._http_client.post(api_url, json=payload, timeout=timeout_cfg)
            else:
                async with httpx.AsyncClient(timeout=timeout_cfg) as client:
                    response = await client.post(api_url, json=payload)

            duration_ms = int((time.perf_counter() - start_time) * 1000)

            if response.status_code == 200:
                data = response.json()
                raw_jobs = data.get("jobs", [])

                if raw_jobs:
                    logger.info(
                        "Source '%s' search succeeded with %d jobs in %d ms",
                        source,
                        len(raw_jobs),
                        duration_ms,
                    )
                    return SourceResult(
                        source=source,
                        status="success",
                        provider_status="SUCCESS",
                        http_status=200,
                        retryable=False,
                        jobs_returned=len(raw_jobs),
                        error=None,
                        duration_ms=duration_ms,
                        raw_jobs=raw_jobs,
                        diagnostics=outbound_diag,
                    )
                else:
                    # Check if body or logs indicate an upstream error/block
                    msg_str = (str(data.get("message") or "") + " " + str(data.get("error") or "")).lower()
                    if "403" in msg_str or "bad response status code: 403" in msg_str or "forbidden" in msg_str or "blocked" in msg_str:
                        logger.warning("Source '%s' encountered upstream HTTP 403 block", source)
                        return SourceResult(
                            source=source,
                            status="failed",
                            provider_status="UPSTREAM_BLOCKED",
                            http_status=403,
                            retryable=False,
                            jobs_returned=0,
                            error="Upstream Glassdoor request blocked by provider (HTTP 403 Security Check).",
                            duration_ms=duration_ms,
                            raw_jobs=[],
                            diagnostics=outbound_diag,
                        )
                    elif "location not parsed" in msg_str:
                        logger.warning("Source '%s' encountered location parse failure", source)
                        return SourceResult(
                            source=source,
                            status="failed",
                            provider_status="LOCATION_PARSE_ERROR",
                            http_status=400,
                            retryable=False,
                            jobs_returned=0,
                            error="Glassdoor location could not be parsed by upstream provider.",
                            duration_ms=duration_ms,
                            raw_jobs=[],
                            diagnostics=outbound_diag,
                        )
                    else:
                        return SourceResult(
                            source=source,
                            status="success",
                            provider_status="NO_RESULTS",
                            http_status=200,
                            retryable=False,
                            jobs_returned=0,
                            error=None,
                            duration_ms=duration_ms,
                            raw_jobs=[],
                            diagnostics=outbound_diag,
                        )

            elif response.status_code == 403:
                err_text = response.text
                logger.warning("Source '%s' returned HTTP 403: %s", source, err_text[:200])
                return SourceResult(
                    source=source,
                    status="failed",
                    provider_status="UPSTREAM_BLOCKED",
                    http_status=403,
                    retryable=False,
                    jobs_returned=0,
                    error=f"Upstream provider blocked request: {err_text[:200]}",
                    duration_ms=duration_ms,
                    raw_jobs=[],
                    diagnostics=outbound_diag,
                )

            elif response.status_code == 400:
                err_text = response.text
                prov_st = "LOCATION_PARSE_ERROR" if "location" in err_text.lower() else "REQUEST_SCHEMA_ERROR"
                logger.warning("Source '%s' returned HTTP 400 (%s): %s", source, prov_st, err_text[:200])
                return SourceResult(
                    source=source,
                    status="failed",
                    provider_status=prov_st,
                    http_status=400,
                    retryable=False,
                    jobs_returned=0,
                    error=f"Bridge request validation error: {err_text[:200]}",
                    duration_ms=duration_ms,
                    raw_jobs=[],
                    diagnostics=outbound_diag,
                )

            elif response.status_code == 429:
                err_text = response.text
                logger.warning("Source '%s' rate limited (429): %s", source, err_text[:200])
                return SourceResult(
                    source=source,
                    status="failed",
                    provider_status="UPSTREAM_BLOCKED",
                    http_status=429,
                    retryable=True,
                    jobs_returned=0,
                    error=f"Upstream provider rate limited: {err_text[:200]}",
                    duration_ms=duration_ms,
                    raw_jobs=[],
                    diagnostics=outbound_diag,
                )

            else:
                err_text = response.text
                logger.warning("Source '%s' returned HTTP %d: %s", source, response.status_code, err_text[:200])
                return SourceResult(
                    source=source,
                    status="failed",
                    provider_status="UPSTREAM_REQUEST_FAILED",
                    http_status=response.status_code,
                    retryable=False,
                    jobs_returned=0,
                    error=f"HTTP {response.status_code}: {err_text[:200]}",
                    duration_ms=duration_ms,
                    raw_jobs=[],
                    diagnostics=outbound_diag,
                )

        except httpx.ConnectError as exc:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            err_msg = f"Cannot connect to JobSpy MCP Server at {api_url}: {str(exc)}"
            logger.error("Source '%s' connection error: %s", source, err_msg)
            return SourceResult(
                source=source,
                status="failed",
                provider_status="BRIDGE_UNAVAILABLE",
                http_status=503,
                retryable=True,
                jobs_returned=0,
                error=err_msg,
                duration_ms=duration_ms,
                raw_jobs=[],
                diagnostics=outbound_diag,
            )

        except httpx.TimeoutException:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            err_msg = f"JobSpy request timed out after {self.settings.JOBSPY_TIMEOUT_SECONDS}s"
            logger.warning("Source '%s' timed out in %d ms", source, duration_ms)
            return SourceResult(
                source=source,
                status="failed",
                provider_status="UPSTREAM_TIMEOUT",
                http_status=504,
                retryable=True,
                jobs_returned=0,
                error=err_msg,
                duration_ms=duration_ms,
                raw_jobs=[],
                diagnostics=outbound_diag,
            )

        except Exception as exc:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            err_msg = f"{type(exc).__name__}: {str(exc)}"
            logger.error("Source '%s' unexpected error: %s", source, err_msg)
            return SourceResult(
                source=source,
                status="failed",
                provider_status="UNKNOWN_ERROR",
                http_status=500,
                retryable=False,
                jobs_returned=0,
                error=err_msg,
                duration_ms=duration_ms,
                raw_jobs=[],
                diagnostics=outbound_diag,
            )

    async def search_jobs(
        self,
        site_names: List[str],
        search_term: str,
        location: str,
        is_remote: bool = False,
        results_wanted: int = 20,
        offset: int = 0,
        hours_old: int = 72,
        country_indeed: str = "India",
        linkedin_fetch_description: bool = False,
    ) -> List[SourceResult]:
        """Query multiple job sources sequentially, ensuring complete error isolation and dedicated timeout budgets."""
        results: List[SourceResult] = []
        for site in site_names:
            sr = await self.search_single_source(
                source=site,
                search_term=search_term,
                location=location,
                is_remote=is_remote,
                results_wanted=results_wanted,
                offset=offset,
                hours_old=hours_old,
                country_indeed=country_indeed,
                linkedin_fetch_description=linkedin_fetch_description,
            )
            results.append(sr)

        return results

    async def health_check(self) -> Dict[str, Any]:
        """Check connection to the JobSpy bridge server."""
        health_url = f"http://{self.settings.JOBSPY_HOST}:{self.settings.JOBSPY_PORT}/health"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(health_url)
                if res.status_code == 200:
                    return {
                        "status": "healthy",
                        "provider_status": "SUCCESS",
                        "bridge_url": health_url,
                        "http_status": 200,
                    }
                return {
                    "status": "unhealthy",
                    "provider_status": "BRIDGE_ERROR",
                    "bridge_url": health_url,
                    "http_status": res.status_code,
                }
        except httpx.ConnectError:
            return {
                "status": "unhealthy",
                "provider_status": "BRIDGE_UNAVAILABLE",
                "bridge_url": health_url,
                "http_status": 503,
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "provider_status": "UNKNOWN_ERROR",
                "bridge_url": health_url,
                "error": str(exc),
            }
