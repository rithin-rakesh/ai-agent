"""Provider abstraction layer for multi-source job discovery."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from app.discovery.jobspy_client import SourceResult


class JobDiscoveryProvider(ABC):
    """Abstract base class for job discovery providers."""

    @abstractmethod
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
        """Execute a job search against the provider.

        Args:
            source: Job board identifier (e.g. 'glassdoor', 'indeed')
            search_term: Role or keyword query
            location: Normalized physical location
            is_remote: Whether to search for remote jobs
            results_wanted: Number of items requested
            hours_old: Recency window in hours
            country: Target country

        Returns:
            SourceResult with status, job counts, raw items, and error details.
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Verify provider availability and connection status."""
        pass
