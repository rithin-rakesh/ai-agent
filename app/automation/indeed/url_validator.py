"""URL and Source Validator for Indeed Automation.

Enforces the strict platform boundary: Indeed ONLY.
Validates both job source metadata and HTTP(S) URL hostnames.
"""

import re
from typing import Optional, Tuple
from urllib.parse import urlparse

from app.automation.indeed.config import SUPPORTED_INDEED_DOMAINS


def is_indeed_domain(hostname: Optional[str]) -> bool:
    """Check if the provided hostname belongs to a legitimate Indeed domain."""
    if not hostname:
        return False
    host = hostname.lower().strip()
    if host in SUPPORTED_INDEED_DOMAINS:
        return True
    # Match standard patterns: indeed.com, *.indeed.com
    pattern = r"^([a-z0-9\-]+\.)*indeed\.com$"
    return bool(re.match(pattern, host))


def validate_indeed_request(url: str, source: Optional[str] = "indeed") -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate that the request target is an Indeed job on a legitimate Indeed domain.

    Args:
        url: Direct job posting URL
        source: Job board platform metadata string (default "indeed")

    Returns:
        Tuple: (is_valid: bool, normalized_url: Optional[str], error_code: Optional[str])
    """
    if not url or not isinstance(url, str) or not url.strip():
        return False, None, "INVALID_JOB_URL"

    # 1. Verify Job Source Metadata
    if source is not None and str(source).lower().strip() != "indeed":
        return False, None, "UNSUPPORTED_AUTOMATION_SOURCE"

    # 2. Parse and Validate URL Scheme and Hostname
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        return False, None, "INVALID_JOB_URL"

    hostname = parsed.hostname
    if not is_indeed_domain(hostname):
        return False, None, "INVALID_JOB_URL"

    return True, url.strip(), None


def extract_indeed_jk(url: Optional[str]) -> Optional[str]:
    """Extract Indeed job key (jk) parameter or path segment from URL."""
    if not url:
        return None
    match = re.search(r"[?&]jk=([a-zA-Z0-9]+)", url)
    if match:
        return match.group(1)
    match_vjk = re.search(r"[?&]vjk=([a-zA-Z0-9]+)", url)
    if match_vjk:
        return match_vjk.group(1)
    match_view = re.search(r"/viewjob\?.*jk=([a-zA-Z0-9]+)", url)
    if match_view:
        return match_view.group(1)
    match_rc = re.search(r"/rc/clk\?.*jk=([a-zA-Z0-9]+)", url)
    if match_rc:
        return match_rc.group(1)
    match_slug = re.search(r"/(?:job|jobs|viewjob)/[^\?]*?([a-zA-Z0-9]{16})", url)
    if match_slug:
        return match_slug.group(1)
    return None
