"""NVIDIA Embedding Provider.

Generates dense semantic embedding vectors using NVIDIA NIM (nvidia/nemotron-3-embed-1b)
with proper query/passage input_type separation, caching, and deterministic formatting.
"""

import logging
from typing import Any, Dict, List, Optional

from app.config.settings import Settings, get_settings
from app.llm.cache import SemanticCache
from app.llm.nvidia_client import NVIDIAClient
from app.models.job import Job
from app.models.profile import CandidateProfileData

logger = logging.getLogger(__name__)


def build_candidate_embedding_text(profile_data: CandidateProfileData) -> str:
    """Build a deterministic, privacy-safe textual representation of a candidate profile.

    STRICTLY EXCLUDES:
    - Phone number
    - Email address
    - Physical address / home address
    - Passwords, keys, credentials
    """
    roles = ", ".join(profile_data.career.preferred_roles) if profile_data.career.preferred_roles else "Software Engineer"
    skills = ", ".join([s.skill for s in profile_data.skills]) if profile_data.skills else "Software Development"
    exp_years = profile_data.career.experience_years

    # Aggregate experience summaries
    exp_bullets = []
    for exp in profile_data.experience:
        role_desc = f"{exp.title} at {exp.company}"
        if exp.skills_used:
            role_desc += f" (Skills: {', '.join(exp.skills_used)})"
        if exp.responsibilities:
            role_desc += f": {exp.responsibilities[:200]}"
        exp_bullets.append(role_desc)
    exp_str = "; ".join(exp_bullets) if exp_bullets else "Relevant industry experience"

    # Education
    edu_bullets = []
    for edu in profile_data.education:
        edu_desc = edu.degree
        if edu.field:
            edu_desc += f" in {edu.field}"
        edu_bullets.append(edu_desc)
    edu_str = "; ".join(edu_bullets) if edu_bullets else "Higher Education"

    pref_locs = ", ".join(profile_data.career.preferred_locations) if profile_data.career.preferred_locations else "Flexible"
    remote = profile_data.career.remote_preference

    return (
        f"Candidate Professional Profile:\n"
        f"Target Roles: {roles}\n"
        f"Experience Level: {exp_years} years\n"
        f"Core Technical Skills: {skills}\n"
        f"Professional Background: {exp_str}\n"
        f"Education: {edu_str}\n"
        f"Work Preferences: {remote}, Preferred Locations: {pref_locs}"
    )


def build_job_embedding_text(job: Job) -> str:
    """Build a deterministic textual representation of a job posting for embedding."""
    title = job.title or "Job Opportunity"
    company = job.company or "Company"
    location = job.location or "Location Unspecified"
    remote = "Remote" if getattr(job, "remote", False) else "On-site/Hybrid"
    job_type = job.job_type or "Full-time"
    description = (job.description or "").strip()
    if len(description) > 3000:
        description = description[:3000] + "..."

    return (
        f"Job Opportunity Posting:\n"
        f"Title: {title}\n"
        f"Company: {company}\n"
        f"Location: {location} ({remote})\n"
        f"Employment Type: {job_type}\n"
        f"Job Details and Requirements:\n{description}"
    )


class NVIDIAEmbeddingProvider:
    """Provider for generating and caching text embeddings via NVIDIA NIM API."""

    def __init__(
        self,
        client: Optional[NVIDIAClient] = None,
        cache: Optional[SemanticCache] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or NVIDIAClient(self.settings)
        self.cache = cache or SemanticCache()
        self.model = self.settings.NVIDIA_EMBEDDING_MODEL

    def get_embedding(
        self,
        text: str,
        input_type: str = "query",
    ) -> List[float]:
        """Generate or retrieve a single embedding vector.

        Args:
            text: Input string to embed
            input_type: Must be 'query' (for candidate search query) or 'passage' (for job description)

        Returns:
            List[float]: 1024-dimensional float embedding vector

        Raises:
            ValueError: If input_type is invalid
            NVIDIAClientError: If the API call fails
        """
        if input_type not in ("query", "passage"):
            raise ValueError(f"Invalid input_type '{input_type}'. Must be 'query' or 'passage'.")

        cached = self.cache.get_embedding(text, self.model, input_type)
        if cached is not None:
            return cached

        payload = {
            "model": self.model,
            "input": [text],
            "input_type": input_type,
        }

        response = self.client.post("/embeddings", payload)

        data = response.get("data", [])
        if not data or not isinstance(data, list):
            raise ValueError("Malformed response from NVIDIA embeddings API: missing data array.")

        embedding = data[0].get("embedding")
        if not embedding or not isinstance(embedding, list):
            raise ValueError("Malformed response from NVIDIA embeddings API: missing embedding vector.")

        # Cache the result
        self.cache.set_embedding(text, self.model, input_type, embedding)
        return embedding

    def get_embeddings_batch(
        self,
        texts: List[str],
        input_type: str = "passage",
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in a single batch request where uncached."""
        if not texts:
            return []

        if input_type not in ("query", "passage"):
            raise ValueError(f"Invalid input_type '{input_type}'. Must be 'query' or 'passage'.")

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for idx, text in enumerate(texts):
            cached = self.cache.get_embedding(text, self.model, input_type)
            if cached is not None:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            payload = {
                "model": self.model,
                "input": uncached_texts,
                "input_type": input_type,
            }
            response = self.client.post("/embeddings", payload)
            data = response.get("data", [])

            for pos, item in enumerate(data):
                embedding = item.get("embedding")
                original_idx = uncached_indices[pos]
                results[original_idx] = embedding
                self.cache.set_embedding(uncached_texts[pos], self.model, input_type, embedding)

        return [r for r in results if r is not None]
