"""Profile-Driven Job Search Planner.

Generates a broad, bounded, and prioritized discovery search plan from candidate career
profiles, role families, skill domain clusters, and optional privacy-safe NVIDIA expansion.
"""

import json
import logging
import re
from typing import List, Optional, Set, Tuple
from pydantic import BaseModel

from app.config.settings import Settings, get_settings
from app.llm.nvidia_client import NVIDIAClient
from app.models.job import SearchPlan, SearchQuery
from app.models.profile import CandidateProfileData

logger = logging.getLogger(__name__)

# Single tools/languages to avoid querying standalone without role context
_STANDALONE_SKILL_STOPWORDS = {
    "git",
    "github",
    "gitlab",
    "linux",
    "docker",
    "kubernetes",
    "sql",
    "jira",
    "confluence",
    "bash",
    "vscode",
    "postman",
}


def _normalize_search_term(term: str) -> str:
    """Normalize search phrase for deduplication (lowercase, collapse punctuation/spaces/common variations)."""
    t = term.lower().strip()
    t = re.sub(r"[/\\_-]+", " ", t)
    t = re.sub(r"\bcyber\s+security\b", "cybersecurity", t)
    t = re.sub(r"\binformation\s+security\b", "infosec", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


class ProfileSearchPlanner:
    """Generates bounded multi-query search plans from candidate profile data."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        nvidia_client: Optional[NVIDIAClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._nvidia_client = nvidia_client

    @property
    def nvidia_client(self) -> NVIDIAClient:
        """Lazy initialization of NVIDIA client."""
        if self._nvidia_client is None:
            self._nvidia_client = NVIDIAClient(self.settings)
        return self._nvidia_client

    def create_search_plan(
        self,
        profile_data: CandidateProfileData,
        max_queries: int = 15,
        site: str = "indeed",
        use_nvidia: bool = False,
    ) -> SearchPlan:
        """Create a prioritized and deduplicated search plan from candidate profile data.

        Args:
            profile_data: Canonical CandidateProfileData
            max_queries: Upper bound limit on total queries generated
            site: Target job board (e.g. 'indeed')
            use_nvidia: Whether to optionally include NVIDIA search phrase expansion

        Returns:
            SearchPlan containing bounded list of SearchQuery objects.
        """
        planned_queries: List[SearchQuery] = []
        seen_query_keys: Set[Tuple[str, str]] = set()

        # 1. Determine Target Locations
        raw_locations = self._resolve_target_locations(profile_data)
        # Avoid searching generic "India" as a location query if specific city hubs exist
        target_locations = [
            loc for loc in raw_locations
            if not (loc.strip().lower() == "india" and len(raw_locations) > 1)
        ]
        if not target_locations:
            target_locations = ["Remote"]

        primary_loc = target_locations[0]

        # 2. Priority 1: Exact Preferred Roles from Profile
        preferred_roles = profile_data.career.preferred_roles or ["Software Engineer"]
        clean_roles = [r.strip() for r in preferred_roles if r and r.strip()]

        # Pass 1: Query every preferred role in the primary location first
        # This guarantees roles like Data Scientist, AI ML Engineer, Machine Learning Engineer
        # are searched rather than starving them on a single role across multiple cities.
        for role_clean in clean_roles:
            if len(planned_queries) >= max_queries:
                break
            self._add_query_if_unique(
                planned_queries=planned_queries,
                seen_keys=seen_query_keys,
                term=role_clean,
                location=primary_loc,
                source=site,
                reason=f"Candidate preferred role: {role_clean}",
                priority=1,
            )

        # Pass 2: Distribute remaining query budget across secondary locations for candidate roles
        if len(target_locations) > 1 and len(planned_queries) < max_queries:
            for loc_idx, loc in enumerate(target_locations[1:], start=2):
                if len(planned_queries) >= max_queries:
                    break
                for role_clean in clean_roles:
                    if len(planned_queries) >= max_queries:
                        break
                    self._add_query_if_unique(
                        planned_queries=planned_queries,
                        seen_keys=seen_query_keys,
                        term=role_clean,
                        location=loc,
                        source=site,
                        reason=f"Candidate preferred role in {loc}: {role_clean}",
                        priority=loc_idx,
                    )

        # 3. Priority 2: Broad Domain / Role-Family Expansion
        domain_terms = self._generate_domain_expansions(profile_data)
        for term, reason in domain_terms:
            if len(planned_queries) >= max_queries:
                break
            self._add_query_if_unique(
                planned_queries=planned_queries,
                seen_keys=seen_query_keys,
                term=term,
                location=primary_loc,
                source=site,
                reason=reason,
                priority=3,
            )

        # 4. Optional Priority 3: Privacy-Safe NVIDIA Expansion
        if use_nvidia and len(planned_queries) < max_queries:
            try:
                ai_terms = self._expand_queries_via_nvidia(profile_data)
                for ai_term in ai_terms:
                    if len(planned_queries) >= max_queries:
                        break
                    self._add_query_if_unique(
                        planned_queries=planned_queries,
                        seen_keys=seen_query_keys,
                        term=ai_term,
                        location=primary_loc,
                        source=site,
                        reason=f"NVIDIA semantic query expansion: {ai_term}",
                        priority=4,
                    )
            except Exception as exc:
                logger.warning("NVIDIA query expansion skipped: %s", exc)

        # Enforce strict upper bound
        bounded_queries = planned_queries[:max_queries]
        logger.info(
            "SearchPlan generated: %d queries (max_queries=%d)",
            len(bounded_queries),
            max_queries,
        )

        return SearchPlan(
            queries=bounded_queries,
            generated_from="candidate_profile",
            query_count=len(bounded_queries),
        )

    def _resolve_target_locations(self, profile_data: CandidateProfileData) -> List[str]:
        """Extract prioritized locations from candidate career preferences."""
        locations: List[str] = []
        preferred = profile_data.career.preferred_locations or []

        for loc in preferred:
            clean_loc = loc.strip()
            if clean_loc and clean_loc not in locations:
                locations.append(clean_loc)

        # Handle remote preference
        remote_pref = (profile_data.career.remote_preference or "any").lower()
        if remote_pref in ("remote_only", "any", "remote") and "Remote" not in locations:
            locations.append("Remote")

        # Fallback if no locations specified
        if not locations:
            cand_loc = (profile_data.personal.location or "").strip()
            locations.append(cand_loc if cand_loc else "Remote")

        return locations

    def _generate_domain_expansions(self, profile_data: CandidateProfileData) -> List[Tuple[str, str]]:
        """Deterministically derive role family and domain expansion terms from candidate profile."""
        expansions: List[Tuple[str, str]] = []
        preferred_roles_lower = [r.lower() for r in profile_data.career.preferred_roles]
        skills_lower = {s.skill.lower() for s in profile_data.skills}

        # Domain Cluster: AI / Machine Learning
        is_ai_ml = any(any(k in r for k in ["ai", "machine learning", "ml", "data scientist"]) for r in preferred_roles_lower) or any(
            k in skills_lower for k in ["machine learning", "deep learning", "pytorch", "tensorflow", "nlp"]
        )
        if is_ai_ml:
            ai_terms = [
                ("Machine Learning Engineer", "AI/ML domain role expansion"),
                ("AI Engineer", "AI/ML domain role expansion"),
                ("Python Developer", "Core programming language for ML workflows"),
                ("Data Scientist", "Adjacent quantitative analytics and ML role"),
                ("Deep Learning Engineer", "Specialized neural network and AI domain"),
                ("MLOps Engineer", "Machine Learning operations & model deployment"),
            ]
            for term, reason in ai_terms:
                expansions.append((term, reason))

        # Domain Cluster: Cybersecurity / InfoSec
        is_security = any(any(k in r for k in ["cyber", "security", "soc", "infosec"]) for r in preferred_roles_lower) or any(
            k in skills_lower for k in ["cybersecurity", "penetration testing", "soc", "vulnerability assessment", "siem"]
        )
        if is_security:
            sec_terms = [
                ("Cybersecurity Analyst", "Cybersecurity domain role expansion"),
                ("Information Security Specialist", "Core security operations role"),
                ("SOC Analyst", "Security operations center monitoring role"),
                ("Vulnerability Assessment", "Proactive security and risk analysis"),
                ("Cloud Security Engineer", "Infrastructure and cloud defense role"),
                ("Network Security Engineer", "Network protection and threat analysis"),
            ]
            for term, reason in sec_terms:
                expansions.append((term, reason))

        # Domain Cluster: Backend / Software Engineering
        is_backend = any(any(k in r for k in ["backend", "software", "full stack", "python developer"]) for r in preferred_roles_lower)
        if is_backend and not is_ai_ml and not is_security:
            backend_terms = [
                ("Software Engineer", "Standard software development role"),
                ("Backend Developer", "Server-side and API engineering role"),
                ("Full Stack Developer", "Cross-functional web engineering role"),
                ("Python Software Engineer", "Python-focused application development"),
            ]
            for term, reason in backend_terms:
                expansions.append((term, reason))

        # Skill-Derived Domain Roles (only for critical/high skills with role context)
        for skill in profile_data.skills:
            if skill.importance in ("critical", "high"):
                sk_name = skill.skill.strip()
                sk_lower = sk_name.lower()
                if sk_lower in _STANDALONE_SKILL_STOPWORDS:
                    continue
                # Avoid single generic skill words without role context
                if len(sk_name.split()) == 1 and not is_ai_ml:
                    compound_term = f"{sk_name} Developer"
                    expansions.append((compound_term, f"Skill-based role expansion for {sk_name}"))
                elif len(sk_name.split()) > 1:
                    expansions.append((sk_name, f"Domain skill expansion: {sk_name}"))

        return expansions

    def _expand_queries_via_nvidia(self, profile_data: CandidateProfileData) -> List[str]:
        """Query NVIDIA LLM for adjacent job search phrases using privacy-safe professional summary only."""
        if not self.settings.NVIDIA_API_KEY:
            return []

        roles_str = ", ".join(profile_data.career.preferred_roles) if profile_data.career.preferred_roles else "Software Engineer"
        top_skills = ", ".join([s.skill for s in profile_data.skills if s.importance in ("critical", "high")][:6])

        system_msg = (
            "You are a recruitment search assistant. Generate adjacent, industry-standard job search titles "
            "based strictly on the candidate's target roles and skill areas. "
            "Do not invent candidate skills, experience, or history. "
            "Return ONLY a JSON list of 3 to 5 job title strings, e.g. [\"Role 1\", \"Role 2\"]."
        )
        user_msg = (
            f"Target Roles: {roles_str}\n"
            f"Key Skill Domains: {top_skills}\n"
            f"Experience Level: {profile_data.career.experience_years} years.\n"
            "Provide 3-5 adjacent job search phrases."
        )

        payload = {
            "model": self.settings.NVIDIA_REASONING_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": 500,
        }

        response = self.nvidia_client.post("/chat/completions", payload)
        choices = response.get("choices", [])
        if not choices:
            return []

        content = choices[0].get("message", {}).get("content", "").strip()
        # Parse JSON list
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if isinstance(item, str) and item.strip()]
        return []

    def _add_query_if_unique(
        self,
        planned_queries: List[SearchQuery],
        seen_keys: Set[Tuple[str, str]],
        term: str,
        location: str,
        source: str,
        reason: str,
        priority: int,
    ) -> bool:
        """Add a search query if normalized term + location pair has not been added yet."""
        norm_term = _normalize_search_term(term)
        norm_loc = location.lower().strip()

        # Reject empty or purely stopword terms
        if not norm_term or norm_term in _STANDALONE_SKILL_STOPWORDS:
            return False

        key = (norm_term, norm_loc)
        if key in seen_keys:
            return False

        seen_keys.add(key)
        planned_queries.append(
            SearchQuery(
                search_term=term.strip(),
                location=location.strip(),
                source=source,
                reason=reason,
                priority=priority,
            )
        )
        return True
