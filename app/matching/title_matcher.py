"""Deterministic Title Matcher.

Calculates semantic token overlap, role alias similarity, and exact match bonuses
between job titles and candidate preferred roles without using an LLM.
"""

import re
from typing import List, Set, Tuple

ROLE_EQUIVALENCE_GROUPS = [
    {"ai", "ml", "ai/ml", "machine learning", "artificial intelligence", "deep learning"},
    {"backend", "back-end", "server-side", "api"},
    {"frontend", "front-end", "ui", "web"},
    {"fullstack", "full-stack", "full stack"},
    {"devops", "sre", "platform", "infrastructure", "cloud"},
    {"data scientist", "data science", "data analyst"},
    {"software engineer", "software developer", "programmer", "sde"},
]


def _clean_text_for_comparison(text: str) -> str:
    """Normalize text by collapsing punctuation and extra whitespace."""
    if not text:
        return ""
    t = re.sub(r"[/\\_+-]+", " ", text.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _tokenize(text: str) -> Set[str]:
    """Clean text into a set of lowercased alphanumeric tokens, stripping common filler words."""
    if not text:
        return set()
    cleaned = _clean_text_for_comparison(text)
    tokens = set(cleaned.split())
    stopwords = {"and", "or", "the", "a", "an", "in", "at", "for", "with", "senior", "lead", "junior", "staff", "principal", "sr", "jr"}
    return {t for t in tokens if t not in stopwords and len(t) > 1}


def _group_matches(tokens1: Set[str], tokens2: Set[str]) -> bool:
    """Check if tokens from both sides fall into the same role equivalence group."""
    for group in ROLE_EQUIVALENCE_GROUPS:
        has_t1 = any(t in group for t in tokens1)
        has_t2 = any(t in group for t in tokens2)
        if has_t1 and has_t2:
            return True
    return False


class TitleMatcher:
    """Evaluates job title similarity against candidate preferred roles."""

    def match(self, job_title: str, preferred_roles: List[str]) -> Tuple[float, str]:
        """Compute deterministic title match score (0-100).

        Returns:
            Tuple: (score_0_to_100, matched_role_name)
        """
        if not job_title or not preferred_roles:
            return 50.0, ""

        clean_title = _clean_text_for_comparison(job_title)
        job_tokens = _tokenize(job_title)

        best_score = 0.0
        best_role = ""

        for role in preferred_roles:
            clean_role = _clean_text_for_comparison(role)
            role_tokens = _tokenize(role)

            # 1. Exact string match
            if clean_title == clean_role:
                return 100.0, role

            # 2. Substring containment
            if clean_role in clean_title or clean_title in clean_role:
                score = 95.0
                if score > best_score:
                    best_score = score
                    best_role = role
                continue

            # 3. Token Jaccard overlap
            if job_tokens and role_tokens:
                intersection = job_tokens.intersection(role_tokens)
                union = job_tokens.union(role_tokens)
                jaccard = len(intersection) / len(union) if union else 0.0

                # Check equivalence group boost
                group_boost = 0.25 if _group_matches(job_tokens, role_tokens) else 0.0
                token_score = min(90.0, (jaccard + group_boost) * 100.0)

                if token_score > best_score:
                    best_score = token_score
                    best_role = role

        return round(best_score, 2), best_role

    def is_domain_relevant(self, job_title: str, preferred_roles: List[str]) -> bool:
        """Check if the job title has any domain overlap with candidate preferred roles.

        Filters out generic professional filler words (engineer, developer, lead, etc.)
        to prevent unrelated engineering roles (e.g. 'Principal Layout Engineer') from
        matching candidates looking for AI/ML or Software roles.
        """
        if not job_title or not preferred_roles:
            return True

        score, _ = self.match(job_title, preferred_roles)
        if score >= 20.0:
            return True

        job_tokens = _tokenize(job_title)
        cand_tokens: Set[str] = set()
        for r in preferred_roles:
            cand_tokens.update(_tokenize(r))

        generic_role_words = {
            "engineer", "developer", "analyst", "specialist", "consultant",
            "manager", "associate", "intern", "lead", "senior", "junior",
            "principal", "staff", "head", "director", "architect",
        }

        job_domain_tokens = {t for t in job_tokens if t not in generic_role_words}
        cand_domain_tokens = {t for t in cand_tokens if t not in generic_role_words}

        if job_domain_tokens and cand_domain_tokens:
            if job_domain_tokens.intersection(cand_domain_tokens):
                return True
            if _group_matches(job_domain_tokens, cand_domain_tokens):
                return True
            return False

        return score > 0.0
