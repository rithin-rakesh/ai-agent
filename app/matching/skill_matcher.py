"""Deterministic Skill Matcher.

Extracts required and preferred technical skills from job descriptions, normalizes
them using the alias catalog, compares against candidate profile skills, and computes
a weighted skill match score (0-100).
"""

import re
from typing import Dict, List, Set, Tuple

from app.models.job import Job
from app.models.profile import CandidateProfileData, SkillItem
from app.profile.profile_loader import SKILL_ALIASES, normalize_skill_name

# Skill importance multipliers
IMPORTANCE_MULTIPLIERS = {
    "critical": 1.5,
    "high": 1.2,
    "medium": 1.0,
    "low": 0.7,
}


def _extract_keywords_from_text(text: str, candidate_skill_names: Set[str]) -> Set[str]:
    """Scan job description and title for mentions of technical skills."""
    if not text:
        return set()

    found_skills: Set[str] = set()
    lower_text = " " + text.lower() + " "

    # Check against known aliases
    for alias, canonical in SKILL_ALIASES.items():
        pattern = r"(?:\b|\W)" + re.escape(alias) + r"(?:\b|\W)"
        if re.search(pattern, lower_text):
            found_skills.add(canonical)

    # Check against candidate's specific skill inventory
    for c_skill in candidate_skill_names:
        c_lower = c_skill.lower()
        pattern = r"(?:\b|\W)" + re.escape(c_lower) + r"(?:\b|\W)"
        if re.search(pattern, lower_text):
            found_skills.add(c_skill)

    return found_skills


class SkillMatcher:
    """Evaluates alignment between candidate skills and job requirements."""

    def match(
        self,
        job: Job,
        profile_data: CandidateProfileData,
    ) -> Tuple[float, List[str], List[str], Dict[str, any]]:
        """Compute deterministic skill match score (0-100).

        Returns:
            Tuple: (score_0_to_100, matched_skills_list, missing_skills_list, diagnostics_dict)
        """
        candidate_skills: Dict[str, SkillItem] = {
            s.skill.lower(): s for s in profile_data.skills
        }
        candidate_skill_names = {s.skill for s in profile_data.skills}

        # Combine job title and description for extraction
        combined_text = f"{job.title}\n{job.description or ''}"
        job_extracted_skills = _extract_keywords_from_text(combined_text, candidate_skill_names)

        # If job text has very few detected skills, evaluate against title-based core skills
        if not job_extracted_skills:
            job_extracted_skills = _extract_keywords_from_text(job.title, candidate_skill_names)

        # If still no skills detectable from the posting, award high neutral score (80.0)
        if not job_extracted_skills:
            return 80.0, [], [], {"status": "no_skills_detected", "coverage_ratio": 1.0}

        matched_skills: List[str] = []
        missing_skills: List[str] = []
        total_required_weight = 0.0
        earned_weight = 0.0

        for required_skill in job_extracted_skills:
            req_lower = required_skill.lower()
            if req_lower in candidate_skills:
                item = candidate_skills[req_lower]
                multiplier = IMPORTANCE_MULTIPLIERS.get(item.importance.lower(), 1.0)
                if item.weight:
                    multiplier = item.weight

                weight = 1.0 * multiplier
                total_required_weight += weight
                earned_weight += weight
                matched_skills.append(item.skill)
            else:
                # Skill required by job but missing from candidate
                weight = 1.0
                total_required_weight += weight
                missing_skills.append(required_skill)

        if total_required_weight > 0:
            coverage = earned_weight / total_required_weight
            score = round(min(100.0, coverage * 100.0), 2)
        else:
            score = 100.0

        # Sort lists alphabetically for deterministic consistency
        matched_skills = sorted(list(set(matched_skills)))
        missing_skills = sorted(list(set(missing_skills)))

        diagnostics = {
            "total_detected_in_job": len(job_extracted_skills),
            "matched_count": len(matched_skills),
            "missing_count": len(missing_skills),
            "coverage_ratio": round(len(matched_skills) / max(1, len(job_extracted_skills)), 2),
        }

        return score, matched_skills, missing_skills, diagnostics
