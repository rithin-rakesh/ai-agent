"""Candidate Profile Service.

Orchestrates loading structured local profile data, syncing to Supabase,
and providing canonical candidate profile access for the matching engine and API.
"""

import logging
from typing import List, Optional
from uuid import UUID

from app.models.profile import (
    CandidateProfileData,
    Profile,
    ProfileCreate,
    ProfileUpdate,
    Skill,
    SkillCreate,
)
from app.profile.profile_loader import ProfileLoader
from app.profile.repository import ProfileRepository

logger = logging.getLogger(__name__)


class ProfileService:
    """Service layer for Candidate Profile operations."""

    def __init__(
        self,
        repository: Optional[ProfileRepository] = None,
        loader: Optional[ProfileLoader] = None,
    ) -> None:
        self.repository = repository or ProfileRepository()
        self.loader = loader or ProfileLoader()

    def load_local_profile_data(self) -> CandidateProfileData:
        """Load and normalize candidate profile data from the local JSON file."""
        return self.loader.load_and_normalize()

    def sync_local_profile_to_database(self) -> Optional[Profile]:
        """Read local JSON profile and synchronize it into the Supabase profiles and skills tables."""
        profile_data = self.load_local_profile_data()
        logger.info("Synchronizing local profile (%s) to Supabase...", profile_data.personal.email)
        synced_profile = self.repository.sync_profile_from_data(profile_data)
        if synced_profile:
            logger.info("Profile successfully synced with ID: %s", synced_profile.id)
        return synced_profile

    def get_profile(self, profile_id: Optional[UUID] = None) -> Optional[Profile]:
        """Retrieve candidate profile by ID, or fall back to default profile from Supabase."""
        if profile_id:
            profile = self.repository.get_job_by_id(profile_id) if hasattr(self.repository, "get_job_by_id") else self.repository.get_profile_by_id(profile_id)
            if profile:
                return profile

        # Retrieve default profile from database
        profile = self.repository.get_default_profile()
        if profile:
            return profile

        # If not yet stored in DB, sync from local file automatically
        logger.info("No profile found in database. Initializing from local profile file...")
        return self.sync_local_profile_to_database()

    def get_active_profile_data(self, profile_id: Optional[UUID] = None) -> CandidateProfileData:
        """Retrieve comprehensive CandidateProfileData combining local data and canonical DB profile."""
        local_data = self.load_local_profile_data()
        db_profile = self.get_profile(profile_id)

        if not db_profile:
            return local_data

        # Merge DB updates into the rich CandidateProfileData
        local_data.personal.name = db_profile.name
        local_data.personal.email = db_profile.email
        local_data.personal.phone = db_profile.phone
        local_data.personal.location = db_profile.location

        if db_profile.experience_years is not None:
            local_data.career.experience_years = float(db_profile.experience_years)
        if db_profile.preferred_roles:
            local_data.career.preferred_roles = db_profile.preferred_roles
        if db_profile.preferred_locations:
            local_data.career.preferred_locations = db_profile.preferred_locations
        if db_profile.salary_min is not None:
            local_data.career.salary_min = float(db_profile.salary_min)
        if db_profile.salary_max is not None:
            local_data.career.salary_max = float(db_profile.salary_max)

        return local_data

    def update_profile(self, profile_id: UUID, update: ProfileUpdate) -> Optional[Profile]:
        """Partially update an existing profile in Supabase."""
        return self.repository.update_profile(profile_id, update)

    def get_skills(self, profile_id: UUID) -> List[Skill]:
        """Retrieve skills for a specific profile ID."""
        return self.repository.get_skills(profile_id)

    def replace_skills(self, profile_id: UUID, skills: List[SkillCreate]) -> List[Skill]:
        """Replace all skills for a specific profile ID."""
        return self.repository.replace_skills(profile_id, skills)
