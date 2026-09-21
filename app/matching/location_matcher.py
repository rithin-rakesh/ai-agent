"""Deterministic Location and Remote Matcher.

Evaluates geographical proximity, preferred locations, state/region overlap,
and remote work preferences between candidate and job posting.
"""

import re
from typing import List, Optional, Tuple


def _clean_location(text: Optional[str]) -> str:
    """Normalize location string by removing extra whitespace and punctuation."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_location_tokens(text: Optional[str]) -> List[str]:
    """Extract location components (e.g. ['kochi', 'kerala', 'india'])."""
    if not text:
        return []
    parts = re.split(r"[,/|•\-\n]+", text.lower())
    return [p.strip() for p in parts if p.strip()]


class LocationMatcher:
    """Evaluates location and remote alignment."""

    def match(
        self,
        job_location: Optional[str],
        job_is_remote: bool,
        candidate_location: Optional[str],
        preferred_locations: List[str],
        remote_preference: str = "any",
    ) -> Tuple[float, str]:
        """Compute location alignment score (0-100).

        Returns:
            Tuple: (score_0_to_100, status_string)
        """
        clean_job_loc = _clean_location(job_location)
        clean_cand_loc = _clean_location(candidate_location)
        remote_pref = remote_preference.lower()

        # Case 1: Job is explicitly Remote
        if job_is_remote or "remote" in clean_job_loc:
            if remote_pref in ("any", "remote_only", "hybrid"):
                return 100.0, "remote_matched"
            return 80.0, "remote_acceptable"

        # Case 2: Candidate strictly wants Remote only, but job is on-site
        if remote_pref == "remote_only" and not job_is_remote:
            return 30.0, "remote_required_but_onsite"

        # Case 3: Unspecified job location
        if not clean_job_loc:
            return 75.0, "location_unspecified"

        job_tokens = _extract_location_tokens(clean_job_loc)

        # Case 4: Exact match with candidate current location
        if clean_cand_loc and (clean_cand_loc in clean_job_loc or clean_job_loc in clean_cand_loc):
            return 100.0, "exact_current_location"

        # Case 5: Match with any candidate preferred location
        for pref in preferred_locations:
            clean_pref = _clean_location(pref)
            if not clean_pref:
                continue

            if clean_pref in ("remote", "anywhere") and job_is_remote:
                return 100.0, "preferred_remote_matched"

            if clean_pref in clean_job_loc or clean_job_loc in clean_pref:
                return 100.0, f"preferred_location_exact_{pref}"

            # Check token/city/state overlap
            pref_tokens = _extract_location_tokens(clean_pref)
            common = set(pref_tokens).intersection(set(job_tokens))
            if common:
                # E.g., both contain 'kerala' or 'india'
                if any(t in ("india", "us", "usa", "worldwide") for t in common) and len(common) == 1:
                    continue  # Country-only match isn't enough for high score
                return 85.0, f"preferred_region_matched_{next(iter(common))}"

        # Case 6: Candidate current location shares state/region with job
        if clean_cand_loc:
            cand_tokens = _extract_location_tokens(clean_cand_loc)
            common = set(cand_tokens).intersection(set(job_tokens))
            # Remove country broad tokens
            specific_common = {t for t in common if t not in ("india", "us", "usa", "worldwide")}
            if specific_common:
                return 80.0, f"same_state_region_{next(iter(specific_common))}"

        return 35.0, "location_mismatch"
