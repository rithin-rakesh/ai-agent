"""JobSpy implementation of JobDiscoveryProvider."""

from typing import Any, Dict, Optional
from app.config.settings import Settings, get_settings
from app.discovery.jobspy_client import JobSpyClient, SourceResult
from app.discovery.providers.base import JobDiscoveryProvider


class JobSpyDiscoveryProvider(JobDiscoveryProvider):
    """Discovery provider backed by the local JobSpy bridge."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[JobSpyClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or JobSpyClient(settings=self.settings)

    async def search(
        self,
        source: str,
        search_term: str,
        location: Optional[str] = None,
        is_remote: bool = False,
        results_wanted: int = 20,
        hours_old: int = 72,
        country: str = "India",
        **kwargs,
    ) -> SourceResult:
        """Query JobSpy for a single job source."""
        return await self.client.search_single_source(
            source=source,
            search_term=search_term,
            location=location or ("remote" if is_remote else "India"),
            is_remote=is_remote,
            results_wanted=results_wanted,
            hours_old=hours_old,
            country_indeed=country,
            **kwargs,
        )

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the JobSpy bridge."""
        return await self.client.health_check()
