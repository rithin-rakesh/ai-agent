"""Job Normalization and Canonicalization module.

Transforms raw JobSpy results into standardized Pydantic Job models and provides
URL / signature normalization for deduplication.
"""

import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from app.models.job import JobCreate

logger = logging.getLogger(__name__)

# Query parameters commonly used for tracking that should be stripped for URL deduplication
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "refId",
    "trackingId",
    "trk",
    "position",
    "pageNum",
    "currentJobId",
    "ref",
    "fbclid",
    "gclid",
}


def normalize_url(url: Optional[str]) -> Optional[str]:
    """Clean and canonicalize a job URL by removing tracking query parameters and trailing slashes.

    Args:
        url: Raw URL string

    Returns:
        Cleaned canonical URL or None
    """
    if not url or not isinstance(url, str) or not url.strip():
        return None

    try:
        parsed = urlparse(url.strip())
        query_dict = parse_qs(parsed.query, keep_blank_values=False)

        # Remove tracking parameters
        filtered_query = {
            k: v for k, v in query_dict.items() if k not in TRACKING_PARAMS and not k.startswith("utm_")
        }

        # Re-encode query parameters deterministically
        clean_query = urlencode(filtered_query, doseq=True)

        # Rebuild URL with clean path (remove trailing slash unless root)
        path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
        canonical = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.params,
            clean_query,
            "",  # Remove fragment
        ))
        return canonical
    except Exception as exc:
        logger.debug("Failed to normalize URL '%s': %s", url, exc)
        return url.strip()


def normalize_text_key(text: Optional[str]) -> str:
    """Normalize a text string for fuzzy deduplication signature matching."""
    if not text:
        return ""
    # Lowercase, replace non-alphanumeric characters with single space, strip
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", str(text).lower()).strip()
    return cleaned


def parse_datetime_safe(val: Any) -> Optional[datetime]:
    """Safely parse various datetime representations into a timezone-aware or standard datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val

    # Numeric timestamp (seconds or milliseconds)
    if isinstance(val, (int, float)):
        try:
            # If greater than 10^11, assume milliseconds
            ts = val / 1000.0 if val > 1e11 else float(val)
            return datetime.fromtimestamp(ts)
        except Exception:
            return None

    # String timestamp (ISO 8601 or similar)
    if isinstance(val, str) and val.strip():
        cleaned = val.strip()
        # Handle trailing Z
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(cleaned)
        except Exception:
            pass

    return None


class JobNormalizer:
    """Normalizes raw dictionary payloads from JobSpy into Pydantic JobCreate entities."""

    @staticmethod
    def normalize(raw: Dict[str, Any], default_source: str = "unknown") -> Tuple[JobCreate, Dict[str, str]]:
        """Normalize a raw JobSpy dictionary into a JobCreate model and compute dedup keys.

        Args:
            raw: Raw dictionary from JobSpy
            default_source: Fallback source name if missing in payload

        Returns:
            Tuple of (JobCreate instance, dict of deduplication keys)
        """
        # Determine source platform
        source = (
            raw.get("site")
            or raw.get("source")
            or raw.get("posted_via")
            or default_source
        )
        source = str(source).lower().strip()
        raw_url_hint = str(raw.get("url") or raw.get("jobUrl") or raw.get("seoUrl") or "").lower()
        if source in ("glassdoor", "gd") or "glassdoor" in raw_url_hint:
            source = "glassdoor"

        # Determine URL
        raw_url = (
            raw.get("url")
            or raw.get("jobUrl")
            or raw.get("job_url")
            or raw.get("seoUrl")
            or raw.get("applyUrl")
            or raw.get("URL")
            or raw.get("jobUrlDirect")
            or raw.get("job_url_direct")
        )
        clean_url = normalize_url(raw_url)

        # Determine external ID
        external_id = (
            raw.get("id")
            or raw.get("key")
            or raw.get("jobId")
            or raw.get("externalId")
            or raw.get("external_id")
        )

        if source == "glassdoor":
            from app.automation.glassdoor.url_validator import extract_glassdoor_job_id
            url_jl = (
                extract_glassdoor_job_id(clean_url)
                or extract_glassdoor_job_id(raw_url)
                or extract_glassdoor_job_id(raw.get("seoUrl"))
            )
            if url_jl:
                external_id = url_jl
            elif external_id:
                ext_str = str(external_id).strip()
                if ext_str.lower().startswith("gd-"):
                    external_id = ext_str[3:]
                else:
                    external_id = ext_str
            # Ensure clean Glassdoor URL if clean_url is missing
            if external_id and not clean_url:
                clean_url = f"https://www.glassdoor.com/job-listing/j?jl={external_id}"

        # Title & Company (supporting nested employer/company objects)
        title = str(raw.get("title") or raw.get("job_title") or raw.get("jobTitle") or "Untitled Position").strip()
        
        company_raw = raw.get("company") or raw.get("company_name") or raw.get("companyName")
        if isinstance(raw.get("employer"), dict):
            company_raw = raw["employer"].get("name") or company_raw
        elif isinstance(company_raw, dict):
            company_raw = company_raw.get("companyName") or company_raw.get("name")
        company = str(company_raw or "Unknown Company").strip()

        # Location (supporting nested location objects)
        location_raw = raw.get("location") or raw.get("location_city") or raw.get("job_location")
        if isinstance(location_raw, dict):
            location_raw = location_raw.get("name") or location_raw.get("city")
        location = str(location_raw).strip() if location_raw is not None else None

        # Fallback external_id if missing
        if not external_id:
            # Compute deterministic hash from clean_url or title+company
            hash_basis = clean_url or f"{source}:{title}:{company}:{location}"
            external_id = f"{source[:2]}-{hashlib.sha256(hash_basis.encode()).hexdigest()[:16]}"
        else:
            external_id = str(external_id).strip()

        # Description
        description = raw.get("description")
        if description is not None:
            description = str(description).strip() or None

        # Salary fields (supporting nested pay objects from valig)
        def to_float(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                f = float(v)
                return f if f >= 0.0 else None
            except (ValueError, TypeError):
                return None

        if isinstance(raw.get("pay"), dict):
            salary_min = to_float(raw["pay"].get("min"))
            salary_max = to_float(raw["pay"].get("max"))
        elif raw.get("baseSalary_min") is not None or raw.get("baseSalary_max") is not None:
            salary_min = to_float(raw.get("baseSalary_min"))
            salary_max = to_float(raw.get("baseSalary_max"))
        else:
            salary_min = to_float(raw.get("minAmount") or raw.get("min_amount") or raw.get("salary_min"))
            salary_max = to_float(raw.get("maxAmount") or raw.get("max_amount") or raw.get("salary_max"))

        # Fallback to parse salary text range if present (e.g. "$120,000 - $150,000")
        if salary_min is None and salary_max is None and raw.get("salary"):
            sal_str = str(raw["salary"]).replace(",", "")
            amounts = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", sal_str)]
            if len(amounts) >= 2:
                salary_min = min(amounts[0], amounts[1])
                salary_max = max(amounts[0], amounts[1])
            elif len(amounts) == 1:
                salary_min = amounts[0]

        # Job Type & Remote
        job_type = raw.get("jobType") or raw.get("job_type")
        if job_type is not None:
            job_type = str(job_type).strip() or None

        is_remote_raw = raw.get("isRemote") or raw.get("is_remote") or raw.get("remote")
        if is_remote_raw is not None:
            remote = bool(is_remote_raw)
        else:
            remote = "remote" in str(location).lower() if location else False

        # Posted timestamp
        posted_at = parse_datetime_safe(
            raw.get("datePosted") or raw.get("date_posted") or raw.get("date") or raw.get("posted_at")
        )

        # Easy Apply
        easy_apply_raw = raw.get("easyApply") or raw.get("easy_apply") or raw.get("isEasyApply")
        easy_apply = bool(easy_apply_raw) if easy_apply_raw is not None else None

        # Experience text
        experience_text = raw.get("experienceRange") or raw.get("experience_range") or raw.get("experience_text")
        if experience_text is not None:
            experience_text = str(experience_text).strip() or None

        job_create = JobCreate(
            source=source,
            external_id=external_id,
            title=title,
            company=company,
            location=location,
            description=description,
            url=clean_url,
            salary_min=salary_min,
            salary_max=salary_max,
            experience_text=experience_text,
            job_type=job_type,
            remote=remote,
            posted_at=posted_at,
            easy_apply=easy_apply,
            raw_data=raw,
        )

        # Compute 3-tier deduplication keys
        dedup_keys = {
            "source_external_id": f"{source}:{external_id}",
            "canonical_url": clean_url or "",
            "signature": f"{normalize_text_key(title)}|{normalize_text_key(company)}|{normalize_text_key(location)}",
        }

        return job_create, dedup_keys
