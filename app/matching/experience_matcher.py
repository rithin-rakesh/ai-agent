"""Deterministic Experience Matcher.

Extracts required experience years from job descriptions using regex patterns,
compares against candidate profile experience, and computes alignment score (0-100).
"""

import re
from typing import Optional, Tuple

# Patterns for extracting experience requirements
EXP_RANGE_PATTERN = re.compile(
    r"\b(\d+)\s*(?:-|to)\s*(\d+)\s*(?:years?|yrs?|yr)\b",
    re.IGNORECASE,
)
EXP_PLUS_PATTERN = re.compile(
    r"\b(\d+)\s*\+\s*(?:years?|yrs?|yr)\b",
    re.IGNORECASE,
)
EXP_MIN_PATTERN = re.compile(
    r"\b(?:minimum|min|at\s+least|requires?|requiring|with)\s+(?:of\s+)?(\d+)\s*\+?\s*(?:years?|yrs?|yr)\b",
    re.IGNORECASE,
)
EXP_PHRASE_PATTERN = re.compile(
    r"\b(\d+)\s*\+?\s*(?:years?|yrs?|yr)\s+(?:of\s+)?(?:hands-on\s+|relevant\s+|professional\s+|industry\s+|work\s+)?(?:experience|exp)\b",
    re.IGNORECASE,
)
EXP_FRESHER_PATTERN = re.compile(
    r"\b(?:fresher|freshers|entry\s*level|graduate\s*trainee|0\s*years?)\b",
    re.IGNORECASE,
)


def parse_job_experience(text: str) -> Tuple[Optional[float], Optional[float], bool]:
    """Parse text to extract minimum and maximum required years of experience."""
    if not text or not text.strip():
        return None, None, False

    # Check for freshers
    if EXP_FRESHER_PATTERN.search(text):
        return 0.0, 1.0, True

    # 1. Range: e.g. "3-5 years" or "2 to 4 yrs"
    range_match = EXP_RANGE_PATTERN.search(text)
    if range_match:
        try:
            min_v = float(range_match.group(1))
            max_v = float(range_match.group(2))
            if min_v <= max_v and min_v <= 40.0:
                return min_v, max_v, False
        except (ValueError, IndexError):
            pass

    extracted_mins = []

    # 2. Plus pattern: e.g. "15+ years"
    for m in EXP_PLUS_PATTERN.finditer(text):
        try:
            v = float(m.group(1))
            if v <= 40.0:
                extracted_mins.append(v)
        except (ValueError, IndexError):
            pass

    # 3. Minimum pattern: e.g. "minimum 2 years", "at least 4 years"
    for m in EXP_MIN_PATTERN.finditer(text):
        try:
            v = float(m.group(1))
            if v <= 40.0:
                extracted_mins.append(v)
        except (ValueError, IndexError):
            pass

    # 4. Phrase pattern: e.g. "15+ years hands-on custom layout experience", "5 years of experience"
    for m in EXP_PHRASE_PATTERN.finditer(text):
        start_idx = m.start()
        preceding = text[max(0, start_idx - 5) : start_idx]
        if not re.search(r"\d+\s*[-to]\s*$", preceding, re.IGNORECASE):
            try:
                v = float(m.group(1))
                if v <= 40.0:
                    extracted_mins.append(v)
            except (ValueError, IndexError):
                pass

    if extracted_mins:
        min_y = max(extracted_mins)
        return min_y, None, False

    return None, None, False


class ExperienceMatcher:
    """Evaluates candidate years of experience against job requirements."""

    def match(
        self,
        candidate_years: float,
        job_experience_text: Optional[str],
        job_description: Optional[str],
    ) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
        """Compute experience compatibility score (0-100) or None if unspecified.

        Returns:
            Tuple: (score_0_to_100_or_None, detected_min_years, detected_max_years, status_string)
        """
        combined = f"{job_experience_text or ''}\n{job_description or ''}"
        min_y, max_y, is_fresher = parse_job_experience(combined)

        # Case 1: Unspecified experience requirement
        if min_y is None and max_y is None and not is_fresher:
            return None, None, None, "unspecified"

        # Case 2: Fresher position
        if is_fresher:
            if candidate_years <= 2.0:
                return 100.0, 0.0, 1.0, "fresher_match"
            return 80.0, 0.0, 1.0, "experienced_for_fresher"

        # Case 3: Specific experience bounds
        min_req = min_y or 0.0
        max_req = max_y or 99.0

        if min_req <= candidate_years <= max_req:
            # Fully in target range
            return 100.0, min_y, max_y, "exact_match"

        if candidate_years < min_req:
            deficit = min_req - candidate_years
            if deficit <= 1.0:
                score = 80.0  # reasonable stretch
                status = "slight_deficit"
            elif deficit <= 2.0:
                score = 60.0  # stretch / underqualified
                status = "underqualified"
            elif deficit <= 4.0:
                score = 25.0  # significant deficit
                status = "underqualified"
            else:
                score = 0.0   # extreme mismatch (e.g. 15 yrs vs 3 yrs)
                status = "extreme_mismatch"
            return round(score, 2), min_y, max_y, status

        if candidate_years > max_req:
            surplus = candidate_years - max_req
            score = max(70.0, 95.0 - (surplus * 5.0))
            return round(score, 2), min_y, max_y, "overqualified"

        return 80.0, min_y, max_y, "compatible"
