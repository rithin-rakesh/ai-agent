"""Deterministic Salary Matcher.

Evaluates compensation alignment between job posting ranges and candidate salary expectations.
"""

from typing import Optional, Tuple


class SalaryMatcher:
    """Evaluates compensation compatibility."""

    def match(
        self,
        job_salary_min: Optional[float],
        job_salary_max: Optional[float],
        candidate_salary_min: Optional[float],
        candidate_salary_max: Optional[float],
    ) -> Tuple[float, bool, str]:
        """Compute salary compatibility score (0-100).

        Returns:
            Tuple: (score_0_to_100, is_compatible_bool, status_string)
        """
        # If job does not disclose salary, mark unavailable
        if job_salary_min is None and job_salary_max is None:
            return None, True, "salary_unknown"

        # If candidate has not set salary preferences, mark unavailable
        if candidate_salary_min is None and candidate_salary_max is None:
            return None, True, "candidate_preference_open"

        cand_min = candidate_salary_min or 0.0
        cand_max = candidate_salary_max or float("inf")

        j_min = job_salary_min or job_salary_max or 0.0
        j_max = job_salary_max or job_salary_min or float("inf")

        # Overlap condition
        overlap_start = max(cand_min, j_min)
        overlap_end = min(cand_max, j_max)

        if overlap_start <= overlap_end:
            # Overlapping intervals
            return 100.0, True, "salary_within_range"

        if j_max < cand_min:
            # Job pays less than candidate minimum
            deficit_pct = (cand_min - j_max) / max(1.0, cand_min)
            if deficit_pct <= 0.15:
                return 75.0, False, "slightly_below_minimum"
            elif deficit_pct <= 0.30:
                return 50.0, False, "below_minimum"
            else:
                return 25.0, False, "significantly_below_minimum"

        if j_min > cand_max:
            # Job pays more than candidate expected maximum
            return 95.0, True, "above_expected_maximum"

        return 80.0, True, "compatible"
