"""Supabase Database Repository for Profile and Skill entities.

Handles all persistence, retrieval, updating, and skill replacement operations
using the Supabase client.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID
from supabase import Client

from app.database.supabase import get_supabase_client, get_supabase_service_client
from app.models.profile import (
    CandidateProfileData,
    Profile,
    ProfileCreate,
    ProfileUpdate,
    Skill,
    SkillCreate,
    SkillItem,
)

logger = logging.getLogger(__name__)


def _serialize_profile_dict(profile_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure JSON fields like preferred_locations and preferred_roles are correctly structured."""
    payload = dict(profile_dict)
    if "preferred_locations" in payload and payload["preferred_locations"] is None:
        payload["preferred_locations"] = []
    if "preferred_roles" in payload and payload["preferred_roles"] is None:
        payload["preferred_roles"] = []
    return payload


class ProfileRepository:
    """Repository handling CRUD and synchronization for Profile and Skill records in Supabase."""

    def __init__(self, client: Optional[Client] = None) -> None:
        self._client = client

    @property
    def client(self) -> Client:
        """Retrieve active Supabase client. Prefers service client for DB operations."""
        if self._client is not None:
            return self._client
        try:
            return get_supabase_service_client()
        except ValueError:
            return get_supabase_client()

    def get_profile_by_id(self, profile_id: UUID) -> Optional[Profile]:
        """Fetch a single profile by UUID along with its associated skills."""
        try:
            response = (
                self.client.table("profiles")
                .select("*")
                .eq("id", str(profile_id))
                .limit(1)
                .execute()
            )
            if not response.data:
                return None

            profile_dict = response.data[0]
            skills = self.get_skills(profile_id)
            profile_dict["skills"] = [s.model_dump() for s in skills]
            return Profile.model_validate(profile_dict)
        except Exception as exc:
            logger.error("Error fetching profile by id (%s): %s", profile_id, exc)
            return None

    def get_profile_by_email(self, email: str) -> Optional[Profile]:
        """Fetch a single profile by email address along with its associated skills."""
        try:
            response = (
                self.client.table("profiles")
                .select("*")
                .eq("email", email)
                .limit(1)
                .execute()
            )
            if not response.data:
                return None

            profile_dict = response.data[0]
            profile_id = UUID(profile_dict["id"])
            skills = self.get_skills(profile_id)
            profile_dict["skills"] = [s.model_dump() for s in skills]
            return Profile.model_validate(profile_dict)
        except Exception as exc:
            logger.error("Error fetching profile by email (%s): %s", email, exc)
            return None

    def get_default_profile(self) -> Optional[Profile]:
        """Retrieve the primary/most recently created candidate profile."""
        try:
            response = (
                self.client.table("profiles")
                .select("*")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if not response.data:
                return None

            profile_dict = response.data[0]
            profile_id = UUID(profile_dict["id"])
            skills = self.get_skills(profile_id)
            profile_dict["skills"] = [s.model_dump() for s in skills]
            return Profile.model_validate(profile_dict)
        except Exception as exc:
            logger.error("Error fetching default profile: %s", exc)
            return None

    def create_profile(self, profile_create: ProfileCreate) -> Optional[Profile]:
        """Insert a new profile into Supabase."""
        try:
            payload = _serialize_profile_dict(profile_create.model_dump(mode="python"))
            response = self.client.table("profiles").insert(payload).execute()
            if response.data:
                created = response.data[0]
                created["skills"] = []
                logger.info("Successfully created profile: %s (%s)", created.get("name"), created.get("id"))
                return Profile.model_validate(created)
            return None
        except Exception as exc:
            logger.error("Failed to create profile: %s", exc)
            return None

    def update_profile(self, profile_id: UUID, update_data: ProfileUpdate) -> Optional[Profile]:
        """Partially update an existing profile."""
        try:
            payload = update_data.model_dump(mode="python", exclude_unset=True)
            if not payload:
                return self.get_profile_by_id(profile_id)

            payload = _serialize_profile_dict(payload)
            response = (
                self.client.table("profiles")
                .update(payload)
                .eq("id", str(profile_id))
                .execute()
            )
            if response.data:
                return self.get_profile_by_id(profile_id)
            return None
        except Exception as exc:
            logger.error("Failed to update profile (%s): %s", profile_id, exc)
            return None

    def get_skills(self, profile_id: UUID) -> List[Skill]:
        """Fetch all skills associated with a profile."""
        try:
            response = (
                self.client.table("skills")
                .select("*")
                .eq("profile_id", str(profile_id))
                .order("created_at", desc=False)
                .execute()
            )
            if not response.data:
                return []
            return [Skill.model_validate(item) for item in response.data]
        except Exception as exc:
            logger.error("Failed to get skills for profile (%s): %s", profile_id, exc)
            return []

    def replace_skills(self, profile_id: UUID, skills: List[SkillCreate]) -> List[Skill]:
        """Atomically replace all skills for a candidate profile."""
        try:
            # Delete existing skills
            self.client.table("skills").delete().eq("profile_id", str(profile_id)).execute()

            if not skills:
                return []

            payloads = [s.model_dump(mode="python") for s in skills]
            # Convert UUIDs to strings in payloads
            for p in payloads:
                p["profile_id"] = str(p["profile_id"])

            response = self.client.table("skills").insert(payloads).execute()
            if response.data:
                logger.info("Replaced %d skills for profile %s", len(response.data), profile_id)
                return [Skill.model_validate(item) for item in response.data]
            return []
        except Exception as exc:
            logger.error("Failed to replace skills for profile (%s): %s", profile_id, exc)
            return []

    def sync_profile_from_data(self, profile_data: CandidateProfileData) -> Optional[Profile]:
        """Create or update canonical profile and replace skills from structured CandidateProfileData."""
        email = profile_data.personal.email
        if not email:
            logger.error("Cannot sync profile: email is required in personal details")
            return None

        existing = self.get_profile_by_email(email)
        profile_fields = {
            "name": profile_data.personal.name or "Candidate",
            "email": email,
            "phone": profile_data.personal.phone,
            "location": profile_data.personal.location,
            "experience_years": profile_data.career.experience_years,
            "preferred_locations": profile_data.career.preferred_locations,
            "preferred_roles": profile_data.career.preferred_roles,
            "salary_min": profile_data.career.salary_min,
            "salary_max": profile_data.career.salary_max,
        }

        if existing:
            profile_id = existing.id
            update_model = ProfileUpdate.model_validate(profile_fields)
            profile = self.update_profile(profile_id, update_model)
        else:
            create_model = ProfileCreate.model_validate(profile_fields)
            profile = self.create_profile(create_model)
            if not profile:
                return None
            profile_id = profile.id

        # Sync skills
        skill_creates: List[SkillCreate] = []
        for s in profile_data.skills:
            skill_creates.append(
                SkillCreate(
                    profile_id=profile_id,
                    skill=s.skill,
                    importance=s.importance,
                    years_experience=s.years_experience,
                )
            )

        updated_skills = self.replace_skills(profile_id, skill_creates)
        if profile:
            profile.skills = updated_skills
        return profile
