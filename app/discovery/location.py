"""Location normalization helper for Glassdoor discovery.

Ensures candidate location strings conform to Glassdoor's expected formats
and separates remote search from physical location search.
"""

import re
from typing import Optional, Tuple

# Common city mappings to canonical "City, Country" or "City, State, Country"
COMMON_LOCATION_MAPPINGS = {
    "bangalore": "Bangalore, India",
    "bengaluru": "Bengaluru, India",
    "chennai": "Chennai, India",
    "kochi": "Kochi, Kerala, India",
    "cochin": "Kochi, Kerala, India",
    "kerala": "Kerala, India",
    "hyderabad": "Hyderabad, India",
    "mumbai": "Mumbai, India",
    "pune": "Pune, India",
    "delhi": "Delhi, India",
    "new delhi": "New Delhi, India",
    "noida": "Noida, India",
    "gurgaon": "Gurgaon, India",
    "gurugram": "Gurugram, India",
    "kolkata": "Kolkata, India",
    "calcutta": "Kolkata, India",
    "ahmedabad": "Ahmedabad, India",
    "trivandrum": "Thiruvananthapuram, Kerala, India",
    "thiruvananthapuram": "Thiruvananthapuram, Kerala, India",
    "india": "India",
}

REMOTE_INDICATORS = {
    "remote",
    "wfh",
    "work from home",
    "anywhere",
    "virtual",
    "telecommute",
}


def normalize_glassdoor_location(location: Optional[str]) -> Tuple[str, bool]:
    """Normalize a raw location string for Glassdoor discovery.

    Args:
        location: Raw input location string (e.g. "Bangalore", "remote", "Chennai, India")

    Returns:
        Tuple of (normalized_location_str, is_remote_bool)
        If remote is detected, is_remote is True and location is formatted for remote queries.
    """
    if not location or not isinstance(location, str):
        return "remote", True

    cleaned = location.strip()
    lower_loc = cleaned.lower()

    # Check for remote indicators
    if lower_loc in REMOTE_INDICATORS or any(ind in lower_loc for ind in ["work from home", "remote only"]):
        return "remote", True

    # Exact known alias lookup
    if lower_loc in COMMON_LOCATION_MAPPINGS:
        return COMMON_LOCATION_MAPPINGS[lower_loc], False

    # Check if already in "City, Country" or "City, State, Country" format
    if "," in cleaned:
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if parts:
            return ", ".join(parts), False

    # Check if a single word matches one of our known cities
    words = re.findall(r"\b\w+\b", lower_loc)
    for word in words:
        if word in COMMON_LOCATION_MAPPINGS:
            return COMMON_LOCATION_MAPPINGS[word], False

    # Fallback: title-case the cleaned string
    return cleaned.title(), False
