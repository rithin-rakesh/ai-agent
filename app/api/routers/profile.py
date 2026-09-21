"""FastAPI Router for Candidate Profile and Skills endpoints."""

import logging
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.models.profile import (
    CandidateProfileData,
    Profile,
    ProfileCreate,
    ProfileUpdate,
    Skill,
    SkillCreate,
)
from app.profile.service import ProfileService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/profile", tags=["Profile"])


def get_profile_service() -> ProfileService:
    """Dependency injection for ProfileService."""
    return ProfileService()


@router.get(
    "",
    response_model=Profile,
    summary="Get active candidate profile",
    description="Retrieve the active candidate profile with associated skills.",
)
def get_profile(
    profile_id: Optional[UUID] = Query(default=None, description="Optional profile UUID"),
    service: ProfileService = Depends(get_profile_service),
) -> Profile:
    profile = service.get_profile(profile_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate profile could not be found or loaded.",
        )
    return profile


@router.post(
    "",
    response_model=Profile,
    status_code=status.HTTP_201_CREATED,
    summary="Create or sync candidate profile from local storage",
    description="Synchronizes the local candidate_profile.json file into the Supabase database.",
)
def create_or_sync_profile(
    service: ProfileService = Depends(get_profile_service),
) -> Profile:
    profile = service.sync_local_profile_to_database()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to synchronize local profile to Supabase database.",
        )
    return profile


@router.put(
    "",
    response_model=Profile,
    summary="Update candidate profile",
    description="Partially update candidate profile preferences, roles, or locations.",
)
def update_profile(
    update_data: ProfileUpdate,
    profile_id: Optional[UUID] = Query(default=None, description="Optional profile UUID"),
    service: ProfileService = Depends(get_profile_service),
) -> Profile:
    current = service.get_profile(profile_id)
    if not current:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found to update.",
        )

    updated = service.update_profile(current.id, update_data)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update candidate profile in Supabase.",
        )
    return updated


@router.get(
    "/skills",
    response_model=List[Skill],
    summary="Get candidate skills",
    description="Retrieve the complete list of candidate technical skills.",
)
def get_skills(
    profile_id: Optional[UUID] = Query(default=None, description="Optional profile UUID"),
    service: ProfileService = Depends(get_profile_service),
) -> List[Skill]:
    current = service.get_profile(profile_id)
    if not current:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate profile not found.",
        )
    return service.get_skills(current.id)


@router.put(
    "/skills",
    response_model=List[Skill],
    summary="Replace candidate skills",
    description="Atomically replaces the candidate skills list in Supabase.",
)
def replace_skills(
    skills: List[SkillCreate],
    profile_id: Optional[UUID] = Query(default=None, description="Optional profile UUID"),
    service: ProfileService = Depends(get_profile_service),
) -> List[Skill]:
    current = service.get_profile(profile_id)
    if not current:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate profile not found.",
        )
    return service.replace_skills(current.id, skills)
