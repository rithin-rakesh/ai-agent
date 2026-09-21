"""Field Mapper for Indeed Application Forms.

Separates candidate profile data mapping from UI interaction logic,
and identifies unresolved screening questions without fabricating answers.
"""

from typing import Any, Dict, List, Optional
from app.models.profile import CandidateProfileData


class IndeedFieldMapper:
    """Safely extracts candidate personal and career attributes for application form inputs."""

    @staticmethod
    def map_candidate_data(profile_data: Optional[CandidateProfileData]) -> Dict[str, Any]:
        """Extract standardized candidate fields from CandidateProfileData."""
        if not profile_data:
            return {}

        personal = profile_data.personal
        career = profile_data.career

        return {
            "full_name": personal.name if personal else None,
            "email": personal.email if personal else None,
            "phone": personal.phone if personal else None,
            "location": personal.location if personal else None,
            "years_experience": career.experience_years if career else None,
            "education_degree": career.education_degree if career else None,
            "education_field": career.education_field if career else None,
        }

    @staticmethod
    def identify_screening_questions(form_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify custom screening prompts that require human input."""
        unresolved = []
        known_standard_fields = {"name", "email", "phone", "location", "resume", "cv"}

        for elem in form_elements:
            label = (elem.get("label") or "").lower()
            if not any(k in label for k in known_standard_fields):
                unresolved.append(elem)

        return unresolved
