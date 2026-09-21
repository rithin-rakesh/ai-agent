"""Explanation and Rationale Generator for Deterministic Match Engine.

Translates sub-matcher scores, matched/missing skills, experience alignments,
and location evaluations into clear, human-readable explanations and warnings.
"""

from typing import List, Optional, Tuple
from app.models.match import MatchBreakdown


class ExplanationGenerator:
    """Builds human-readable explanations and warnings for job match results."""

    def generate(
        self,
        final_score: float,
        decision: str,
        breakdown: MatchBreakdown,
        matched_skills: List[str],
        missing_skills: List[str],
        job_title: str,
        matched_role: str,
        exp_status: str,
        min_years: Optional[float],
        candidate_years: float,
        loc_status: str,
        salary_status: str,
    ) -> Tuple[str, List[str], List[str]]:
        """Construct summary reason text, bulleted positive reasons, and warnings.

        Returns:
            Tuple: (summary_reason_string, reasons_list, warnings_list)
        """
        reasons: List[str] = []
        warnings: List[str] = []

        # 1. Skills rationale
        if matched_skills:
            top_skills = ", ".join(matched_skills[:5])
            reasons.append(
                f"Matched {len(matched_skills)} core skills ({top_skills})"
                + (f" and {len(matched_skills) - 5} more" if len(matched_skills) > 5 else "")
                + f" (Skills Score: {breakdown.skills}/100)"
            )
        if missing_skills:
            missing_text = ", ".join(missing_skills[:4])
            warnings.append(
                f"Missing {len(missing_skills)} requested skill(s): {missing_text}"
            )

        # 2. Title rationale
        if breakdown.title >= 90.0:
            reasons.append(f"Strong role match between '{job_title}' and preferred role '{matched_role}' ({breakdown.title}/100)")
        elif breakdown.title >= 70.0:
            reasons.append(f"Relevant role overlap with preferred role '{matched_role}' ({breakdown.title}/100)")
        else:
            warnings.append(f"Low title similarity with target roles ({breakdown.title}/100)")

        # 3. Experience rationale
        if exp_status == "exact_match":
            reasons.append(f"Experience requirement ({min_years or 0}+ yrs) matches candidate profile ({candidate_years} yrs)")
        elif exp_status == "extreme_mismatch":
            warnings.append(f"Severe experience deficit: candidate has {candidate_years} yrs; job requires {min_years or 0}+ yrs")
        elif exp_status == "underqualified":
            warnings.append(f"Candidate has {candidate_years} yrs; job suggests {min_years or 0}+ yrs requirement")
        elif exp_status == "overqualified":
            reasons.append(f"Candidate has ample experience ({candidate_years} yrs)")
        elif exp_status == "fresher_match":
            reasons.append("Entry-level / Fresher position matches profile")

        # 4. Location rationale
        if "remote" in loc_status:
            reasons.append("Remote job matches candidate remote preference")
        elif "exact" in loc_status or "preferred" in loc_status or "same_state" in loc_status:
            score_txt = f"{breakdown.location}/100" if breakdown.location is not None else "Aligned"
            reasons.append(f"Location aligns with candidate preferred regions (Score: {score_txt})")
        elif breakdown.location is not None and breakdown.location < 50.0:
            warnings.append("Job location is outside preferred geographic areas")

        # 5. Salary rationale
        if salary_status == "salary_within_range":
            reasons.append("Compensation range is within candidate salary expectations")
        elif "below" in salary_status:
            warnings.append("Disclosed compensation is below candidate target minimum")

        # Build summary paragraph for Supabase 'reason' column
        summary_parts = []
        summary_parts.append(f"Match Score: {final_score}/100 ({decision.upper()}).")
        if reasons:
            summary_parts.append(" Strengths: " + "; ".join(reasons[:2]) + ".")
        if warnings:
            summary_parts.append(" Considerations: " + "; ".join(warnings[:2]) + ".")

        summary_reason = "".join(summary_parts)
        return summary_reason, reasons, warnings
