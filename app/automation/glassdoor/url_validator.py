"""URL and Source Validator for Glassdoor Automation.

Enforces platform boundary: Glassdoor ONLY.
Validates job source metadata and HTTP(S) URL hostnames.
"""

import re
from typing import Optional, Tuple
from urllib.parse import urlparse
from app.automation.glassdoor.config import SUPPORTED_GLASSDOOR_DOMAINS


def is_glassdoor_domain(hostname: Optional[str]) -> bool:
    """Check if the provided hostname belongs to a legitimate Glassdoor domain."""
    if not hostname:
        return False
    host = hostname.lower().strip()
    if host in SUPPORTED_GLASSDOOR_DOMAINS:
        return True
    pattern = r"^([a-z0-9\-]+\.)*glassdoor\.(com|co\.in|ca|co\.uk|com\.au|de|fr|nl|ie|es|ch|at|be|in)$"
    return bool(re.match(pattern, host))


def validate_glassdoor_request(
    url: str, source: Optional[str] = "glassdoor"
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate that the request target is a Glassdoor job on a legitimate Glassdoor domain.

    Returns:
        Tuple: (is_valid: bool, normalized_url: Optional[str], error_code: Optional[str])
    """
    if not url or not isinstance(url, str) or not url.strip():
        return False, None, "INVALID_JOB_URL"

    # 1. Verify Job Source Metadata
    if source is not None and str(source).lower().strip() != "glassdoor":
        return False, None, "UNSUPPORTED_AUTOMATION_SOURCE"

    # 2. Parse and Validate URL Scheme and Hostname
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        return False, None, "INVALID_JOB_URL"

    hostname = parsed.hostname
    if not is_glassdoor_domain(hostname):
        return False, None, "INVALID_JOB_URL"

    return True, url.strip(), None


def extract_glassdoor_job_id(url: Optional[str]) -> Optional[str]:
    """Extract Glassdoor job listing ID or jl parameter from URL."""
    if not url or not isinstance(url, str):
        return None
    match_jl = re.search(r"[-_?&]jl=([0-9]+)", url)
    if match_jl:
        return match_jl.group(1)
    match_job = re.search(r"jobListingId=([0-9]+)", url)
    if match_job:
        return match_job.group(1)
    match_path = re.search(r"/job-listing/[^?]*[?&]jl=([0-9]+)", url)
    if match_path:
        return match_path.group(1)
    return None
