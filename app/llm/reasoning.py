"""NVIDIA Reasoning Model Provider.

Executes qualitative semantic evaluation using the configured NVIDIA LLM
with structured JSON prompt engineering, schema validation, and error recovery.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.config.settings import Settings, get_settings
from app.llm.cache import SemanticCache
from app.llm.nvidia_client import NVIDIAClient
from app.models.job import Job
from app.models.llm import ReasoningOutput
from app.models.match import MatchResult
from app.models.profile import CandidateProfileData

logger = logging.getLogger(__name__)

# Prompt paths relative to current file
_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_SYSTEM_PROMPT_PATH = os.path.join(_PROMPT_DIR, "semantic_match_system.txt")
_USER_PROMPT_PATH = os.path.join(_PROMPT_DIR, "semantic_match_user.txt")


def _load_prompt_template(file_path: str) -> str:
    """Read a prompt template from disk."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _clean_json_response(raw_text: str) -> str:
    """Strip markdown code fences and extraneous leading/trailing text from JSON response."""
    text = raw_text.strip()

    # 1. Match code fence ```json { ... } ``` or ``` { ... } ```
    matches = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL)
    for m in reversed(matches):
        if "semantic_score" in m:
            return m.strip()

    # 2. Match a JSON block containing "semantic_score": <number>
    block_match = re.findall(r"(\{[^{}]*?\"semantic_score\"\s*:\s*[\d.]+[\s\S]*?\})", text)
    if block_match:
        return block_match[-1].strip()

    # 3. Find the last outermost balanced braces { ... } containing semantic_score
    last_brace = text.rfind("}")
    if last_brace != -1:
        depth = 0
        for idx in range(last_brace, -1, -1):
            if text[idx] == "}":
                depth += 1
            elif text[idx] == "{":
                depth -= 1
                if depth == 0:
                    candidate = text[idx : last_brace + 1].strip()
                    if "semantic_score" in candidate:
                        return candidate

    return text


class NVIDIAReasoningProvider:
    """Evaluates qualitative candidate-job fit using NVIDIA reasoning model."""

    def __init__(
        self,
        client: Optional[NVIDIAClient] = None,
        cache: Optional[SemanticCache] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or NVIDIAClient(self.settings)
        self.cache = cache or SemanticCache()
        self.model = self.settings.NVIDIA_REASONING_MODEL
        self._system_prompt = _load_prompt_template(_SYSTEM_PROMPT_PATH)
        self._user_prompt_template = _load_prompt_template(_USER_PROMPT_PATH)

    def _build_user_prompt(
        self,
        job: Job,
        profile_data: CandidateProfileData,
        deterministic_result: MatchResult,
    ) -> str:
        """Populate the user prompt template with professional candidate and job data."""
        roles = ", ".join(profile_data.career.preferred_roles) if profile_data.career.preferred_roles else "Software Engineer"
        skills = ", ".join([f"{s.skill} ({s.importance})" for s in profile_data.skills]) if profile_data.skills else "Software Engineering"
        exp_years = str(profile_data.career.experience_years)

        exp_summaries = []
        for exp in profile_data.experience:
            summary = f"{exp.title} at {exp.company}"
            if exp.responsibilities:
                summary += f": {exp.responsibilities[:250]}"
            exp_summaries.append(summary)
        exp_text = " | ".join(exp_summaries) if exp_summaries else "Professional industry experience"

        edu_list = [f"{e.degree} ({e.field or 'General'})" for e in profile_data.education]
        edu_text = ", ".join(edu_list) if edu_list else "Higher Education Degree"

        work_pref = f"{profile_data.career.remote_preference}, {profile_data.career.job_type}"

        # Job details
        job_desc = (job.description or "").strip()
        if len(job_desc) > 3500:
            job_desc = job_desc[:3500] + "\n[... truncated ...]"

        cat_breakdown = (
            f"Skills: {deterministic_result.skill_score}/100, "
            f"Title: {deterministic_result.title_score}/100, "
            f"Experience: {deterministic_result.experience_score}/100, "
            f"Location: {deterministic_result.location_score}/100"
        )

        return self._user_prompt_template.format(
            preferred_roles=roles,
            experience_years=exp_years,
            skills=skills,
            experience_summary=exp_text,
            education=edu_text,
            work_preferences=work_pref,
            job_title=job.title or "Unknown Role",
            job_company=job.company or "Unknown Company",
            job_location=job.location or "Location Unspecified",
            job_type=job.job_type or "Full-time",
            job_experience=getattr(job, "experience_text", None) or "Not explicitly stated",
            job_description=job_desc,
            deterministic_score=round(deterministic_result.final_score, 1),
            matched_skills=", ".join(deterministic_result.matched_skills) if deterministic_result.matched_skills else "None",
            missing_skills=", ".join(deterministic_result.missing_skills) if deterministic_result.missing_skills else "None",
            category_breakdown=cat_breakdown,
        )

    def evaluate_match(
        self,
        job: Job,
        profile_data: CandidateProfileData,
        deterministic_result: MatchResult,
    ) -> ReasoningOutput:
        """Send job & candidate context to NVIDIA reasoning model and parse structured evaluation.

        Returns:
            ReasoningOutput: Validated structured evaluation containing score, relevance, strengths, and concerns.

        Raises:
            NVIDIAClientError: On network or authentication failure
            ValueError: On malformed or unparseable LLM output
        """
        user_prompt = self._build_user_prompt(job, profile_data, deterministic_result)

        # Check reasoning cache
        cache_key = f"{job.id}:{profile_data.personal.name}:{deterministic_result.final_score}"
        cached = self.cache.get_reasoning(cache_key, self.model)
        if cached is not None:
            return ReasoningOutput.model_validate(cached)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 3000,
        }

        response = self.client.post("/chat/completions", payload)
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("Empty choices list in NVIDIA chat completion response.")

        message = choices[0].get("message", {})
        content = message.get("content", "").strip()
        if not content:
            raise ValueError("Empty content in NVIDIA chat completion response.")

        cleaned_json = _clean_json_response(content)

        try:
            parsed_data = json.loads(cleaned_json)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse JSON from reasoning model response: %s\nRaw: %s", exc, content[:300])
            raise ValueError(f"Reasoning model did not return valid JSON: {exc}") from exc

        # Clamp and normalize score if necessary
        if "semantic_score" in parsed_data:
            parsed_data["semantic_score"] = float(min(100.0, max(0.0, parsed_data["semantic_score"])))

        reasoning_output = ReasoningOutput.model_validate(parsed_data)

        # Store in cache
        self.cache.set_reasoning(cache_key, self.model, reasoning_output.model_dump())
        return reasoning_output
