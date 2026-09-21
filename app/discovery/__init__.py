"""Job Discovery package.

Provides resilient integration with JobSpy MCP, profile-driven search planning,
normalization, cross-query deduplication, database persistence, and matched URL delivery.
"""

from app.discovery.jobspy_client import SUPPORTED_JOB_SOURCES, JobSpyClient, SourceResult
from app.discovery.normalizer import JobNormalizer, normalize_text_key, normalize_url
from app.discovery.profile_discovery_service import ProfileDiscoveryService
from app.discovery.search_planner import ProfileSearchPlanner
from app.discovery.service import JobDiscoveryService

__all__ = [
    "SUPPORTED_JOB_SOURCES",
    "JobSpyClient",
    "SourceResult",
    "JobNormalizer",
    "normalize_url",
    "normalize_text_key",
    "JobDiscoveryService",
    "ProfileSearchPlanner",
    "ProfileDiscoveryService",
]
