"""LinkedIn Lead Relevance Scoring Engine (Phase 6.2).

Computes multi-factor LinkedInLeadRelevanceScore (0-100) evaluating:
- Role similarity: Alignment with candidate target roles (30%)
- Skill overlap: Explicit mentions of candidate profile skills (25%)
- Hiring intent intensity: Strength of hiring signals in post (20%)
- Experience & level fit: Entry-level / junior / intern fit (10%, strictly no inferred unverified years)
- Location & remote fit: Target regions in India / Remote (10%)
- Recruiter / Authority signal: Recruiter or hiring manager author (5%)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from app.models.profile import CandidateProfileData, Profile
from app.outreach.linkedin.models import LeadType, LinkedInLeadBase
from app.profile.profile_loader import ProfileLoader

logger = logging.getLogger(__name__)

# Weight distribution (total = 1.00)
WEIGHT_ROLE = 0.30
WEIGHT_SKILL = 0.25
WEIGHT_INTENT = 0.20
WEIGHT_EXPERIENCE = 0.10
WEIGHT_LOCATION = 0.10
WEIGHT_AUTHORITY = 0.05

SENIOR_KEYWORDS = {"senior", "sr.", "lead", "staff", "principal", "architect", "director", "head of", "vp"}
JUNIOR_KEYWORDS = {"junior", "jr.", "entry", "fresher", "intern", "internship", "graduate", "trainee", "associate"}


class LinkedInLeadRelevanceEngine:
    """Evaluates LinkedIn opportunity posts against the candidate profile."""

    def __init__(self, candidate_profile: Optional[Any] = None) -> None:
        if candidate_profile is not None:
            self.profile = candidate_profile
        else:
            try:
                self.profile = ProfileLoader().load_and_normalize()
            except Exception:
                self.profile = None

        self._target_roles: Set[str] = set()
        self._skills: Set[str] = set()
        self._locations: Set[str] = set()
        self._initialize_profile_indexes()

    def _initialize_profile_indexes(self) -> None:
        """Cache lowercase sets from the loaded candidate profile."""
        if not self.profile:
            return

        roles: List[str] = []
        if isinstance(self.profile, CandidateProfileData) and self.profile.career:
            roles = self.profile.career.preferred_roles or []
        elif hasattr(self.profile, "preferred_roles") and self.profile.preferred_roles:
            roles = self.profile.preferred_roles
        elif hasattr(self.profile, "target_roles") and self.profile.target_roles:
            roles = self.profile.target_roles

        self._target_roles = {str(r).lower().strip() for r in roles if str(r).strip()}

        raw_skills = getattr(self.profile, "skills", []) or []
        skill_names: List[str] = []
        for s in raw_skills:
            if hasattr(s, "skill"):
                skill_names.append(s.skill)
            elif isinstance(s, dict) and "skill" in s:
                skill_names.append(s["skill"])
            elif isinstance(s, str):
                skill_names.append(s)

        self._skills = {str(s).lower().strip() for s in skill_names if str(s).strip()}

        locs: List[str] = []
        if isinstance(self.profile, CandidateProfileData) and self.profile.career:
            locs = self.profile.career.preferred_locations or []
        elif hasattr(self.profile, "preferred_locations") and self.profile.preferred_locations:
            locs = self.profile.preferred_locations

        self._locations = {str(l).lower().strip() for l in locs if str(l).strip()}
        # Normalize general Indian hubs
        self._locations.update({"india", "remote", "bangalore", "bengaluru", "kochi", "kerala", "chennai"})


    def calculate_relevance(self, lead: LinkedInLeadBase) -> Tuple[float, Dict[str, Any]]:
        """Calculate composite 0-100 relevance score and factor breakdown."""
        post_text = (lead.post_text or "").lower()
        headline = (lead.author_headline or "").lower()
        combined_text = f"{post_text} {headline}"

        # 1. Role Similarity (30%)
        role_score, matched_roles = self._score_roles(combined_text)

        # 2. Skill Overlap (25%)
        skill_score, matched_skills = self._score_skills(combined_text)

        # 3. Hiring Intent (20%)
        intent_score, intent_signals = self._score_intent(lead, post_text)

        # 4. Experience & Level Fit (10%)
        exp_score, exp_notes = self._score_experience_fit(combined_text)

        # 5. Location & Remote Fit (10%)
        loc_score, loc_matches = self._score_location(combined_text)

        # 6. Author Authority / Recruiter Signal (5%)
        auth_score, auth_signal = self._score_author(lead.lead_type, headline)

        # Composite Calculation
        raw_composite = (
            (role_score * WEIGHT_ROLE)
            + (skill_score * WEIGHT_SKILL)
            + (intent_score * WEIGHT_INTENT)
            + (exp_score * WEIGHT_EXPERIENCE)
            + (loc_score * WEIGHT_LOCATION)
            + (auth_score * WEIGHT_AUTHORITY)
        )

        final_score = round(max(0.0, min(100.0, raw_composite)), 2)

        details: Dict[str, Any] = {
            "final_score": final_score,
            "role_similarity": {"score": role_score, "matched_roles": matched_roles},
            "skill_overlap": {"score": skill_score, "matched_skills": matched_skills},
            "hiring_intent": {"score": intent_score, "signals": intent_signals},
            "experience_fit": {"score": exp_score, "notes": exp_notes},
            "location_fit": {"score": loc_score, "matched_locations": loc_matches},
            "author_authority": {"score": auth_score, "signal": auth_signal},
            "weights": {
                "role": WEIGHT_ROLE,
                "skill": WEIGHT_SKILL,
                "intent": WEIGHT_INTENT,
                "experience": WEIGHT_EXPERIENCE,
                "location": WEIGHT_LOCATION,
                "authority": WEIGHT_AUTHORITY,
            },
        }

        return final_score, details

    def _score_roles(self, text: str) -> Tuple[float, List[str]]:
        """Score based on candidate target roles present in text."""
        matched: List[str] = []
        for role in self._target_roles:
            if role in text:
                matched.append(role)

        if not matched:
            # Check for generic AI / ML keywords
            for fallback in ["ai engineer", "machine learning", "data scientist", "generative ai", "ml"]:
                if fallback in text:
                    matched.append(fallback)

        if len(matched) >= 2:
            return 100.0, matched
        elif len(matched) == 1:
            return 80.0, matched
        return 20.0, []

    def _score_skills(self, text: str) -> Tuple[float, List[str]]:
        """Score based on candidate skills present in text."""
        matched: List[str] = []
        for skill in self._skills:
            # Only match words with boundaries or distinct terms
            if len(skill) <= 2:
                if f" {skill} " in f" {text} ":
                    matched.append(skill)
            elif skill in text:
                matched.append(skill)

        if len(matched) >= 4:
            return 100.0, matched
        elif len(matched) == 3:
            return 85.0, matched
        elif len(matched) == 2:
            return 65.0, matched
        elif len(matched) == 1:
            return 45.0, matched
        return 10.0, []

    def _score_intent(self, lead: LinkedInLeadBase, text: str) -> Tuple[float, List[str]]:
        """Score the clarity and directness of hiring intent."""
        signals: List[str] = []
        score = 50.0

        if lead.lead_type in (LeadType.HIRING_POST, LeadType.RECRUITER_POST):
            score += 30.0
            signals.append(f"lead_type:{lead.lead_type.value}")
        elif lead.lead_type == LeadType.REFERRAL_POST:
            score += 25.0
            signals.append("referral_offer")

        for phrase in ["we're hiring", "we are hiring", "i'm hiring", "immediate joiner", "urgent requirement"]:
            if phrase in text:
                score += 15.0
                signals.append(phrase)
                break

        if lead.contact_email:
            score += 10.0
            signals.append(f"public_email:{lead.contact_email}")

        return min(100.0, score), signals

    def _score_experience_fit(self, text: str) -> Tuple[float, str]:
        """Score entry-level / junior / fresher alignment without unverified inference."""
        has_junior = any(k in text for k in JUNIOR_KEYWORDS)
        has_senior = any(k in text for k in SENIOR_KEYWORDS)

        if has_junior and not has_senior:
            return 100.0, "Explicit entry-level / junior / intern match"
        elif has_junior and has_senior:
            return 75.0, "Multiple experience levels mentioned"
        elif has_senior and not has_junior:
            return 40.0, "Senior / Lead level indicated (potential experience stretch)"
        return 70.0, "Standard general requirements"

    def _score_location(self, text: str) -> Tuple[float, List[str]]:
        """Score location and remote compatibility."""
        matched: List[str] = []
        for loc in self._locations:
            if loc in text:
                matched.append(loc)

        if "remote" in matched:
            return 100.0, matched
        elif len(matched) >= 1:
            return 90.0, matched
        return 50.0, []

    def _score_author(self, lead_type: LeadType, headline: str) -> Tuple[float, str]:
        """Score recruiter or hiring authority signal."""
        if lead_type == LeadType.RECRUITER_POST:
            return 100.0, "Verified recruiter / talent acquisition post"
        if any(t in headline for t in ["founder", "cto", "ceo", "engineering manager", "director"]):
            return 95.0, "Hiring manager / Leadership author"
        if lead_type == LeadType.REFERRAL_POST:
            return 80.0, "Internal employee referral"
        return 50.0, "Standard author"
