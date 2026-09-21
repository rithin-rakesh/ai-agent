"""India Location Validator module.

Enforces strict India-only filtering on all discovered job postings:
- Validates location against known Indian cities, states, territories, and 'Remote - India'
- Normalizes Indian locations to clean canonical strings
- Rejects foreign locations
- Handles unknown locations with safe fail-closed policy
- Returns structured filter diagnostics
"""

import re
from typing import Optional, Tuple

INDIAN_STATES_AND_UT = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
    "delhi", "ncr", "chandigarh", "puducherry", "jammu and kashmir", "ladakh",
}

INDIAN_CITIES_AND_HUBS = {
    # South
    "bangalore": "Bangalore, India",
    "bengaluru": "Bengaluru, India",
    "chennai": "Chennai, India",
    "madras": "Chennai, India",
    "hyderabad": "Hyderabad, India",
    "secunderabad": "Secunderabad, India",
    "kochi": "Kochi, Kerala, India",
    "cochin": "Kochi, Kerala, India",
    "ernakulam": "Kochi, Kerala, India",
    "trivandrum": "Thiruvananthapuram, Kerala, India",
    "thiruvananthapuram": "Thiruvananthapuram, Kerala, India",
    "calicut": "Kozhikode, Kerala, India",
    "kozhikode": "Kozhikode, Kerala, India",
    "coimbatore": "Coimbatore, Tamil Nadu, India",
    "mysore": "Mysuru, Karnataka, India",
    "mysuru": "Mysuru, Karnataka, India",
    "visakhapatnam": "Visakhapatnam, Andhra Pradesh, India",
    "vizag": "Visakhapatnam, Andhra Pradesh, India",
    "vijayawada": "Vijayawada, Andhra Pradesh, India",
    # West
    "mumbai": "Mumbai, India",
    "bombay": "Mumbai, India",
    "pune": "Pune, India",
    "navi mumbai": "Navi Mumbai, India",
    "thane": "Thane, India",
    "nagpur": "Nagpur, Maharashtra, India",
    "ahmedabad": "Ahmedabad, India",
    "surat": "Surat, Gujarat, India",
    "vadodara": "Vadodara, Gujarat, India",
    # North
    "delhi": "Delhi, India",
    "new delhi": "New Delhi, India",
    "noida": "Noida, India",
    "greater noida": "Greater Noida, India",
    "gurgaon": "Gurgaon, India",
    "gurugram": "Gurugram, India",
    "faridabad": "Faridabad, India",
    "ghaziabad": "Ghaziabad, India",
    "chandigarh": "Chandigarh, India",
    "mohali": "Mohali, Punjab, India",
    "jaipur": "Jaipur, Rajasthan, India",
    "lucknow": "Lucknow, Uttar Pradesh, India",
    "kanpur": "Kanpur, Uttar Pradesh, India",
    "indore": "Indore, Madhya Pradesh, India",
    "bhopal": "Bhopal, Madhya Pradesh, India",
    # East
    "kolkata": "Kolkata, India",
    "calcutta": "Kolkata, India",
    "bhubaneswar": "Bhubaneswar, Odisha, India",
    "patna": "Patna, Bihar, India",
    "guwahati": "Guwahati, Assam, India",
}

FOREIGN_INDICATORS = {
    "united states", "usa", "u.s.", "california", "new york", "texas", "washington",
    "florida", "seattle", "austin", "san francisco", "chicago", "boston",
    "united kingdom", "uk", "u.k.", "london", "manchester", "birmingham",
    "canada", "toronto", "vancouver", "montreal", "ottawa",
    "germany", "berlin", "munich", "frankfurt",
    "france", "paris", "netherlands", "amsterdam",
    "australia", "sydney", "melbourne", "brisbane",
    "singapore", "dubai", "uae", "u.a.e.", "japan", "tokyo", "ireland", "dublin",
}


def validate_india_location(
    location: Optional[str],
    allow_unknown: bool = False,
) -> Tuple[bool, str, str]:
    """Validate and normalize whether a job location belongs to India.

    Args:
        location: Raw location string (e.g. 'Bangalore', 'New York, NY', 'Remote - India')
        allow_unknown: If True, unknown locations pass filter; if False, fail-closed

    Returns:
        Tuple of:
            - is_india (bool): True if positively validated as Indian location
            - normalized_location (str): Standardized canonical location string
            - filter_status (str): 'india_filter_passed', 'india_filter_rejected', 'india_location_unknown'
    """
    if not location or not isinstance(location, str) or not location.strip():
        return (allow_unknown, "Unknown", "india_location_unknown")

    cleaned = location.strip()
    lower = cleaned.lower()

    # 1. Check explicit foreign indicators first (Fail fast)
    for foreign in FOREIGN_INDICATORS:
        pattern = r"\b" + re.escape(foreign) + r"\b"
        if re.search(pattern, lower):
            # Exception: if it says something like "London, UK (Wait, no: but check if India is also there)"
            if "india" not in lower:
                return (False, cleaned, "india_filter_rejected")

    # 2. Check Remote India patterns
    if "remote" in lower:
        if any(ind in lower for ind in ["india", "in", "bangalore", "delhi", "mumbai", "hyderabad", "chennai", "pune"]):
            return (True, "Remote, India", "india_filter_passed")
        else:
            # Ambiguous "Remote" without country context
            status = "india_location_unknown"
            return (allow_unknown, cleaned, status)

    # 3. Check explicit "India" or ", in"
    if "india" in lower or lower.endswith(", in") or " bharat" in lower or lower.startswith("in-"):
        matched_keys = [k for k in INDIAN_CITIES_AND_HUBS if k in lower]
        if matched_keys:
            best_city = max(matched_keys, key=len)
            return (True, INDIAN_CITIES_AND_HUBS[best_city], "india_filter_passed")
        return (True, cleaned.title() if not cleaned.endswith(", India") else cleaned, "india_filter_passed")

    # 4. Check known Indian tech hubs and cities
    for city_key, canonical in INDIAN_CITIES_AND_HUBS.items():
        pattern = r"\b" + re.escape(city_key) + r"\b"
        if re.search(pattern, lower):
            return (True, canonical, "india_filter_passed")

    # 5. Check Indian States / Territories
    for state in INDIAN_STATES_AND_UT:
        pattern = r"\b" + re.escape(state) + r"\b"
        if re.search(pattern, lower):
            return (True, f"{cleaned}, India" if not cleaned.lower().endswith("india") else cleaned, "india_filter_passed")

    # 6. Location could not be determined
    return (allow_unknown, cleaned, "india_location_unknown")


QUERY_LOCATION_MAPPINGS = {
    "bangalore": "Bangalore, India",
    "bengaluru": "Bangalore, India",
    "chennai": "Chennai, India",
    "madras": "Chennai, India",
    "kochi": "Kochi, Kerala, India",
    "cochin": "Kochi, Kerala, India",
    "ernakulam": "Kochi, Kerala, India",
    "trivandrum": "Trivandrum, Kerala, India",
    "thiruvananthapuram": "Trivandrum, Kerala, India",
    "kozhikode": "Kozhikode, Kerala, India",
    "calicut": "Kozhikode, Kerala, India",
    "kerala": "Kerala, India",
    "mumbai": "Mumbai, India",
    "pune": "Pune, India",
    "hyderabad": "Hyderabad, India",
    "delhi": "Delhi, India",
    "noida": "Noida, India",
    "gurgaon": "Gurgaon, India",
    "gurugram": "Gurugram, India",
}


def normalize_query_location(
    location: Optional[str],
    provider: str = "indeed",
) -> Tuple[str, bool]:
    """Normalize a search query location for provider dispatch and separate remote intent.

    Args:
        location: Raw planned query location string (e.g. 'Bangalore, Karnataka', 'Remote')
        provider: Target provider ('indeed' or 'glassdoor')

    Returns:
        Tuple of (normalized_location: str, is_remote: bool)
    """
    if not location or not str(location).strip():
        return ("", True) if provider == "indeed" else ("India", True)

    cleaned = str(location).strip()
    lower = cleaned.lower()

    # 1. Detect Remote
    if lower in ("remote", "remote - india", "remote, india", "any", "wfh", "work from home") or lower.startswith("remote"):
        if provider == "indeed":
            return ("", True)
        return ("India", True)

    # 2. Check canonical mappings
    for key, canonical in QUERY_LOCATION_MAPPINGS.items():
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, lower):
            return (canonical, False)

    # 3. Fallback: if it's already an Indian location e.g. "Tamil Nadu, India"
    if "india" in lower:
        return (cleaned, False)

    # 4. Check known states
    for state in INDIAN_STATES_AND_UT:
        pattern = r"\b" + re.escape(state) + r"\b"
        if re.search(pattern, lower):
            return (f"{cleaned}, India" if not cleaned.lower().endswith("india") else cleaned, False)

    return (cleaned, False)

