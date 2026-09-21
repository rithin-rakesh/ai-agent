"""Deterministic Job Type Matcher.

Evaluates employment type alignment (full-time, contract, internship, part-time).
"""

from typing import Optional, Tuple


class JobTypeMatcher:
    """Evaluates employment type compatibility."""

    def match(
        self,
        job_type: Optional[str],
        candidate_job_type: Optional[str],
    ) -> Tuple[float, str]:
        """Compute job type alignment score (0-100).

        Returns:
            Tuple: (score_0_to_100, status_string)
        """
        if not job_type or not job_type.strip():
            return None, "job_type_unspecified"

        if not candidate_job_type or not candidate_job_type.strip():
            return None, "candidate_preference_open"

        clean_job_type = job_type.strip().lower()
        clean_cand_type = candidate_job_type.strip().lower()

        if clean_cand_type in clean_job_type or clean_job_type in clean_cand_type:
            return 100.0, "exact_job_type_match"

        # Special cases: full-time vs contract/internship
        if "full" in clean_cand_type and "contract" in clean_job_type:
            return 50.0, "candidate_fulltime_job_contract"

        if "intern" in clean_job_type and "full" in clean_cand_type:
            return 40.0, "candidate_fulltime_job_internship"

        return 60.0, "job_type_mismatch"
