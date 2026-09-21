"""Candidate Profile File Loader and Normalizer.

Loads structured candidate profile JSON from disk and normalizes skills, casing,
whitespace, and common aliases.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from app.config.settings import get_settings
from app.models.profile import (
    CandidateProfileData,
    CareerPreferences,
    EducationItem,
    ExperienceItem,
    PersonalDetails,
    SkillItem,
)

logger = logging.getLogger(__name__)

# Controlled skill alias mapping (normalized lowercase -> Canonical display name)
SKILL_ALIASES: Dict[str, str] = {
    "python": "Python",
    "py": "Python",
    "python3": "Python",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "machine-learning": "Machine Learning",
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "artificial-intelligence": "Artificial Intelligence",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "cv": "Computer Vision",
    "computer vision": "Computer Vision",
    "llm": "Large Language Models",
    "llms": "Large Language Models",
    "generative ai": "Generative AI",
    "genai": "Generative AI",
    "gen ai": "Generative AI",
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "tf": "TensorFlow",
    "keras": "Keras",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "psql": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "mongo": "MongoDB",
    "redis": "Redis",
    "docker": "Docker",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "aws": "Amazon Web Services",
    "amazon web services": "Amazon Web Services",
    "gcp": "Google Cloud Platform",
    "google cloud": "Google Cloud Platform",
    "azure": "Microsoft Azure",
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "linux": "Linux",
    "rest": "REST API",
    "rest api": "REST API",
    "restful": "REST API",
    "graphql": "GraphQL",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "react": "React",
    "reactjs": "React",
    "react.js": "React",
    "vue": "Vue.js",
    "vuejs": "Vue.js",
    "angular": "Angular",
    "c++": "C++",
    "cpp": "C++",
    "golang": "Go",
    "go": "Go",
    "rust": "Rust",
    "java": "Java",
    "c#": "C#",
    "csharp": "C#",
    ".net": ".NET",
    "dotnet": ".NET",
}


def normalize_skill_name(raw_skill: str) -> str:
    """Normalize a skill name using whitespace cleanup and controlled alias mapping."""
    if not raw_skill or not isinstance(raw_skill, str):
        return ""

    cleaned = re.sub(r"\s+", " ", raw_skill.strip())
    lower_key = cleaned.lower()

    if lower_key in SKILL_ALIASES:
        return SKILL_ALIASES[lower_key]

    # Preserve casing if title-cased or already formatted, else capitalize words
    if any(c.isupper() for c in cleaned):
        return cleaned
    return cleaned.title()


def normalize_string(text: Optional[str]) -> str:
    """Normalize whitespace and strip leading/trailing spaces."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


class ProfileLoader:
    """Loads and validates candidate profile JSON from local disk."""

    def __init__(self, file_path: Optional[str] = None) -> None:
        self.file_path = file_path or get_settings().CANDIDATE_PROFILE_PATH

    def load_raw_json(self) -> dict:
        """Read the JSON file from the configured path."""
        path = Path(self.file_path)
        if not path.is_absolute():
            # If relative, resolve relative to current working directory
            path = Path.cwd() / path

        if not path.exists():
            error_msg = f"Candidate profile JSON file not found at: {path.resolve()}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_and_normalize(self) -> CandidateProfileData:
        """Load JSON, normalize fields and skills, and return validated Pydantic model."""
        raw = self.load_raw_json()

        personal_raw = raw.get("personal", {})
        personal = PersonalDetails(
            name=normalize_string(personal_raw.get("name")),
            email=normalize_string(personal_raw.get("email")),
            phone=normalize_string(personal_raw.get("phone")) or None,
            location=normalize_string(personal_raw.get("location")) or None,
        )

        career_raw = raw.get("career", {})
        career = CareerPreferences(
            experience_years=float(career_raw.get("experience_years", 0.0) or 0.0),
            preferred_roles=[normalize_string(r) for r in career_raw.get("preferred_roles", []) if r],
            preferred_locations=[normalize_string(loc) for loc in career_raw.get("preferred_locations", []) if loc],
            remote_preference=normalize_string(career_raw.get("remote_preference", "any")) or "any",
            job_type=normalize_string(career_raw.get("job_type", "full-time")) or "full-time",
            salary_min=career_raw.get("salary_min"),
            salary_max=career_raw.get("salary_max"),
            education_degree=normalize_string(career_raw.get("education_degree")) or None,
            education_field=normalize_string(career_raw.get("education_field")) or None,
        )

        # Normalize skills and deduplicate by canonical name
        skills_raw = raw.get("skills", [])
        normalized_skills: List[SkillItem] = []
        seen_skills = set()

        for s in skills_raw:
            if isinstance(s, str):
                s_name = normalize_skill_name(s)
                if s_name and s_name.lower() not in seen_skills:
                    seen_skills.add(s_name.lower())
                    normalized_skills.append(SkillItem(skill=s_name))
            elif isinstance(s, dict):
                raw_name = s.get("skill", "")
                s_name = normalize_skill_name(raw_name)
                if s_name and s_name.lower() not in seen_skills:
                    seen_skills.add(s_name.lower())
                    normalized_skills.append(
                        SkillItem(
                            skill=s_name,
                            category=normalize_string(s.get("category", "General")) or "General",
                            proficiency=normalize_string(s.get("proficiency", "intermediate")) or "intermediate",
                            years_experience=s.get("years_experience"),
                            importance=normalize_string(s.get("importance", "medium")) or "medium",
                            weight=s.get("weight"),
                        )
                    )

        # Normalize experience entries
        exp_raw = raw.get("experience", [])
        experience: List[ExperienceItem] = []
        for e in exp_raw:
            if isinstance(e, dict) and e.get("company") and e.get("title"):
                skills_used = [normalize_skill_name(sk) for sk in e.get("skills_used", []) if sk]
                experience.append(
                    ExperienceItem(
                        company=normalize_string(e["company"]),
                        title=normalize_string(e["title"]),
                        start_date=e.get("start_date"),
                        end_date=e.get("end_date"),
                        responsibilities=normalize_string(e.get("responsibilities")) or None,
                        skills_used=skills_used,
                    )
                )

        # Normalize education entries
        edu_raw = raw.get("education", [])
        education: List[EducationItem] = []
        for ed in edu_raw:
            if isinstance(ed, dict) and ed.get("degree"):
                education.append(
                    EducationItem(
                        degree=normalize_string(ed["degree"]),
                        field=normalize_string(ed.get("field")) or None,
                        institution=normalize_string(ed.get("institution")) or None,
                        start_date=ed.get("start_date"),
                        end_date=ed.get("end_date"),
                    )
                )

        profile_data = CandidateProfileData(
            personal=personal,
            career=career,
            skills=normalized_skills,
            experience=experience,
            education=education,
        )

        logger.info(
            "Successfully loaded and normalized candidate profile for '%s' (%d skills, %d experience records)",
            profile_data.personal.name or "Unnamed Candidate",
            len(profile_data.skills),
            len(profile_data.experience),
        )
        return profile_data
