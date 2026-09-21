"""Deterministic Education Matcher.

Extracts educational credentials from job descriptions and compares against candidate degrees.
"""

import re
from typing import List, Optional, Tuple

DEGREE_HIERARCHY = {
    "phd": 4,
    "doctorate": 4,
    "master": 3,
    "masters": 3,
    "m.tech": 3,
    "mtech": 3,
    "ms": 3,
    "msc": 3,
    "mca": 3,
    "bachelor": 2,
    "bachelors": 2,
    "b.tech": 2,
    "btech": 2,
    "bs": 2,
    "bsc": 2,
    "bca": 2,
    "diploma": 1,
}


class EducationMatcher:
    """Evaluates candidate educational qualifications against job requirements."""

    def match(
        self,
        job_text: str,
        candidate_degree: Optional[str],
        candidate_field: Optional[str],
    ) -> Tuple[float, str]:
        """Compute education match score (0-100).

        Returns:
            Tuple: (score_0_to_100, status_string)
        """
        if not job_text:
            return None, "education_unspecified"

        lower_text = job_text.lower()

        # Check what degree is required
        detected_level = 0
        detected_deg_name = ""
        for deg, level in DEGREE_HIERARCHY.items():
            pattern = r"(?:\b|\W)" + re.escape(deg) + r"(?:\b|\W)"
            if re.search(pattern, lower_text):
                if level > detected_level:
                    detected_level = level
                    detected_deg_name = deg

        # If no degree requirement is mentioned in the job text, mark unavailable (None)
        if detected_level == 0:
            return None, "no_degree_requirement_detected"

        # Determine candidate's degree level
        cand_level = 2  # default bachelor if unspecified
        if candidate_degree:
            c_deg = candidate_degree.lower()
            for deg, level in DEGREE_HIERARCHY.items():
                if deg in c_deg:
                    if level > cand_level:
                        cand_level = level

        if cand_level >= detected_level:
            return 100.0, f"meets_or_exceeds_{detected_deg_name}"
        elif cand_level == detected_level - 1:
            return 75.0, f"one_level_below_{detected_deg_name}"
        else:
            return 50.0, f"below_required_{detected_deg_name}"
