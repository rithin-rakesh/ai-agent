"""Pydantic models for Profile and Skill entities."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Structured Candidate Profile Schema (Used for local JSON & Rich Profile Data)
# ---------------------------------------------------------------------------


class PersonalDetails(BaseModel):
    """Personal contact information."""

    name: str = Field(default="", description="Full legal name")
    email: str = Field(default="", description="Contact email address")
    phone: Optional[str] = Field(default=None, description="Contact phone number")
    location: Optional[str] = Field(default=None, description="Current city, state, country")


class CareerPreferences(BaseModel):
    """Career parameters and role preferences."""

    experience_years: float = Field(default=0.0, ge=0.0, description="Total professional experience in years")
    preferred_roles: List[str] = Field(default_factory=list, description="Target job titles / roles")
    preferred_locations: List[str] = Field(default_factory=list, description="Target locations or Remote")
    remote_preference: str = Field(default="any", description="Remote preference: any, remote_only, hybrid, on_site")
    job_type: str = Field(default="full-time", description="Preferred job type: full-time, part-time, contract, internship")
    salary_min: Optional[float] = Field(default=None, ge=0.0, description="Minimum expected salary")
    salary_max: Optional[float] = Field(default=None, ge=0.0, description="Maximum expected salary")
    education_degree: Optional[str] = Field(default=None, description="Highest degree attained")
    education_field: Optional[str] = Field(default=None, description="Field of study")


class SkillItem(BaseModel):
    """Detailed skill entry."""

    skill: str = Field(..., min_length=1, description="Skill name or keyword")
    category: str = Field(default="General", description="Skill category: Programming, AI/ML, Backend, Database, DevOps, etc.")
    proficiency: str = Field(default="intermediate", description="Proficiency: beginner, intermediate, advanced, expert")
    years_experience: Optional[float] = Field(default=None, ge=0.0, description="Years of experience with this skill")
    importance: str = Field(default="medium", description="Importance rating: low, medium, high, critical")
    weight: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Optional custom multiplier weight")


class ExperienceItem(BaseModel):
    """Structured work experience record."""

    company: str = Field(..., min_length=1, description="Company name")
    title: str = Field(..., min_length=1, description="Job title")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD or YYYY-MM)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD, YYYY-MM, or null for Present)")
    responsibilities: Optional[str] = Field(default=None, description="Summary of key responsibilities and impact")
    skills_used: List[str] = Field(default_factory=list, description="Skills used in this role")


class EducationItem(BaseModel):
    """Structured education record."""

    degree: str = Field(..., min_length=1, description="Degree or certificate name (e.g. B.Tech, MS)")
    field: Optional[str] = Field(default=None, description="Field of study / major")
    institution: Optional[str] = Field(default=None, description="University or institution name")
    start_date: Optional[str] = Field(default=None, description="Start date")
    end_date: Optional[str] = Field(default=None, description="Graduation date")


class CandidateProfileData(BaseModel):
    """Complete candidate profile document loaded from local storage."""

    personal: PersonalDetails = Field(default_factory=PersonalDetails)
    career: CareerPreferences = Field(default_factory=CareerPreferences)
    skills: List[SkillItem] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Database Models (Supabase 'profiles' and 'skills' tables)
# ---------------------------------------------------------------------------


class SkillBase(BaseModel):
    """Base schema for user skill in database."""

    skill: str = Field(..., min_length=1, description="Skill name or keyword")
    importance: str = Field(default="medium", description="Importance rating: low, medium, high, critical")
    years_experience: Optional[float] = Field(default=None, ge=0.0, description="Years of experience with this skill")


class SkillCreate(SkillBase):
    """Schema for creating a new skill."""

    profile_id: UUID


class Skill(SkillBase):
    """Complete Skill model with database identifiers and timestamps."""

    id: UUID
    profile_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProfileBase(BaseModel):
    """Base schema for user profile in database."""

    name: str = Field(..., min_length=1, description="Full name")
    email: str = Field(..., min_length=3, description="Email address")
    phone: Optional[str] = Field(default=None, description="Contact phone number")
    location: Optional[str] = Field(default=None, description="Current city/country")
    experience_years: Optional[float] = Field(default=None, ge=0.0, description="Total professional experience in years")
    preferred_locations: List[str] = Field(default_factory=list, description="Target locations or Remote")
    preferred_roles: List[str] = Field(default_factory=list, description="Target job titles / roles")
    salary_min: Optional[float] = Field(default=None, ge=0.0, description="Minimum expected salary")
    salary_max: Optional[float] = Field(default=None, ge=0.0, description="Maximum expected salary")
    resume_path: Optional[str] = Field(default=None, description="Path to the primary resume file")


class ProfileCreate(ProfileBase):
    """Schema for creating a new profile."""

    pass


class ProfileUpdate(BaseModel):
    """Schema for partially updating an existing profile."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    experience_years: Optional[float] = None
    preferred_locations: Optional[List[str]] = None
    preferred_roles: Optional[List[str]] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    resume_path: Optional[str] = None


class Profile(ProfileBase):
    """Complete Profile model with database identifiers and timestamps."""

    id: UUID
    created_at: datetime
    updated_at: datetime
    skills: List[Skill] = Field(default_factory=list, description="Associated skills")

    model_config = ConfigDict(from_attributes=True)
