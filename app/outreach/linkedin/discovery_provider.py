"""LinkedIn Discovery Provider via Apify (Phase 6.2).

Dispatches bounded searches to harvestapi~linkedin-post-search and harvestapi~linkedin-profile-posts:
- Validates Apify token via /v2/users/me before launching Actor
- Strictly isolates candidate personal data and resume (queries use only public search terms)
- Normalizes raw LinkedIn post items
- Extracts publicly listed professional emails via strict RFC regex
- Rejects placeholder and guessed emails
- Deduplicates on canonical post_url
- Returns structured diagnostics
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import httpx

from app.config.settings import Settings, get_settings
from app.outreach.linkedin.budget_guard import LinkedInBudgetGuard
from app.outreach.linkedin.config import get_keywords_config
from app.outreach.linkedin.models import (
    LeadType,
    LinkedInDiscoveryDiagnostics,
    LinkedInLeadCreate,
)

logger = logging.getLogger(__name__)

APIFY_BASE_URL = "https://api.apify.com/v2"
TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "TIMED-OUT", "ABORTED"}
NON_TERMINAL_STATUSES = {"READY", "RUNNING"}

# Strict RFC-compliant email regex pattern
EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    re.IGNORECASE,
)

# Blocked dummy or invalid domains
DISALLOWED_DOMAINS = {
    "example.com",
    "domain.com",
    "email.com",
    "company.com",
    "yourcompany.com",
    "test.com",
    "sample.com",
    "linkedin.com",
    "sentry.io",
}


def extract_public_email(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Extract a legitimate publicly listed email address from text.

    Returns:
        Tuple of (clean_email, source) or (None, None).
    """
    if not text:
        return None, None

    matches = EMAIL_REGEX.findall(text)
    for email in matches:
        clean = email.strip(".,;:()<>[]'\"").lower()
        parts = clean.split("@")
        if len(parts) != 2:
            continue
        user, domain = parts
        if len(user) < 2 or len(domain) < 4 or "." not in domain:
            continue
        if domain in DISALLOWED_DOMAINS or domain.endswith(".invalid"):
            continue
        # Avoid common image or asset extensions falsely matching email regex
        if any(clean.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")):
            continue
        return clean, "POST_TEXT"

    return None, None


def normalize_post_url(raw_url: Optional[str], urn_or_id: Optional[str] = None) -> str:
    """Produce a canonical, stable post URL for deduplication."""
    if raw_url and "linkedin.com" in raw_url:
        clean = raw_url.split("?")[0].rstrip("/")
        return clean
    if urn_or_id:
        activity_id = str(urn_or_id).split(":")[-1]
        return f"https://www.linkedin.com/feed/update/urn:li:activity:{activity_id}"
    return ""


def classify_lead_type(text: str, author_headline: Optional[str] = None) -> LeadType:
    """Classify the post into a LeadType based on contextual intent signals."""
    low_text = (text or "").lower()
    low_headline = (author_headline or "").lower()

    if any(k in low_headline for k in ["recruiter", "talent acquisition", "sourcer", "staffing"]):
        return LeadType.RECRUITER_POST

    if any(k in low_text for k in ["referral", "can refer", "refer you", "dm for referral"]):
        return LeadType.REFERRAL_POST

    if any(k in low_text for k in ["we're hiring", "we are hiring", "i'm hiring", "our team is hiring", "open positions", "join our team"]):
        return LeadType.HIRING_POST

    if any(k in low_text for k in ["looking for", "job opening", "opening for", "vacancy"]):
        return LeadType.JOB_OPPORTUNITY

    return LeadType.GENERAL_PROFESSIONAL


class LinkedInApifyDiscoveryProvider:
    """Discovery provider interfacing with Apify LinkedIn scraping actors."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        budget_guard: Optional[LinkedInBudgetGuard] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.budget_guard = budget_guard or LinkedInBudgetGuard(self.settings)
        self.keywords_config = get_keywords_config()

    def _get_api_token(self) -> Optional[str]:
        """Retrieve sanitized Apify API token."""
        return self.settings.apify_token_value

    async def verify_token_validity(self, token: str) -> bool:
        """Verify token against GET /v2/users/me."""
        url = f"{APIFY_BASE_URL}/users/me"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                return resp.status_code == 200
        except Exception as exc:
            logger.warning("Apify token validation network exception: %s", exc)
            return False

    async def run_post_search(
        self,
        queries: Optional[List[str]] = None,
        max_posts: int = 10,
        date_posted: str = "past-week",
    ) -> Tuple[List[LinkedInLeadCreate], LinkedInDiscoveryDiagnostics]:
        """Execute post discovery using harvestapi~linkedin-post-search Actor."""
        diag = LinkedInDiscoveryDiagnostics(
            provider="apify",
            actor=getattr(self.settings, "LINKEDIN_POST_SEARCH_ACTOR_ID", "harvestapi~linkedin-post-search"),
        )

        # 1. Budget Guard Check
        allowed, clamped_posts, budget_reason = self.budget_guard.can_execute_discovery(max_posts)
        if not allowed:
            logger.warning("LinkedIn discovery halted by budget guard: %s", budget_reason)
            diag.errors.append(f"Budget guard rejection: {budget_reason}")
            return [], diag

        # 2. Token Check
        token = self._get_api_token()
        if not token:
            diag.errors.append("APIFY_API_TOKEN is not configured or is empty")
            return [], diag

        if not await self.verify_token_validity(token):
            diag.errors.append("APIFY_API_TOKEN is invalid (failed /v2/users/me)")
            return [], diag

        # 3. Formulate Search Queries
        effective_queries = queries or self.keywords_config.build_search_queries(max_queries=2)
        diag.queries_requested = len(effective_queries)

        actor_id = getattr(self.settings, "LINKEDIN_POST_SEARCH_ACTOR_ID", "harvestapi~linkedin-post-search")
        norm_actor_id = actor_id.replace("/", "~").strip()

        # 4. Construct Payload
        payload = {
            "searchQueries": effective_queries,
            "maxPosts": clamped_posts,
            "datePosted": date_posted if date_posted in ("past-24h", "past-week", "past-month") else "past-month",
            "sortBy": "date",
            "scrapeReactions": False,
            "scrapeComments": False,

        }

        run_url = f"{APIFY_BASE_URL}/acts/{norm_actor_id}/runs"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        leads: List[LinkedInLeadCreate] = []
        raw_items: List[Dict[str, Any]] = []

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                logger.info("Launching Apify Actor %s (maxPosts=%d, queries=%d)...", norm_actor_id, clamped_posts, len(effective_queries))
                run_resp = await client.post(run_url, headers=headers, json=payload)
                if run_resp.status_code not in (200, 201):
                    err_msg = f"Apify Actor launch failed with HTTP {run_resp.status_code}: {run_resp.text[:200]}"
                    logger.error(err_msg)
                    diag.errors.append(err_msg)
                    return [], diag

                run_data = run_resp.json().get("data", {})
                run_id = run_data.get("id")
                dataset_id = run_data.get("defaultDatasetId")

                if not run_id or not dataset_id:
                    diag.errors.append("Apify run did not return run_id or dataset_id")
                    return [], diag

                # 5. Poll for Completion
                poll_url = f"{APIFY_BASE_URL}/actor-runs/{run_id}"
                start_time = asyncio.get_event_loop().time()
                timeout = 120.0

                while (asyncio.get_event_loop().time() - start_time) < timeout:
                    await asyncio.sleep(5)
                    poll_resp = await client.get(poll_url, headers=headers)
                    if poll_resp.status_code != 200:
                        continue
                    status_val = poll_resp.json().get("data", {}).get("status")
                    if status_val in TERMINAL_STATUSES:
                        logger.info("Apify Actor %s completed with status: %s", norm_actor_id, status_val)
                        if status_val != "SUCCEEDED":
                            diag.errors.append(f"Apify Actor finished with non-success status: {status_val}")
                        break

                # 6. Fetch Dataset Items
                dataset_url = f"{APIFY_BASE_URL}/datasets/{dataset_id}/items?clean=true&format=json"
                data_resp = await client.get(dataset_url, headers=headers)
                if data_resp.status_code == 200:
                    raw_items = data_resp.json()
                else:
                    diag.errors.append(f"Failed to fetch dataset items (HTTP {data_resp.status_code})")

        except Exception as exc:
            logger.error("Exception during Apify LinkedIn discovery: %s", exc, exc_info=True)
            diag.errors.append(f"Network / Execution error: {exc}")

        # 7. Normalize & Deduplicate Items
        seen_urls: Set[str] = set()
        diag.posts_returned = len(raw_items)

        for item in raw_items:
            try:
                post_text = item.get("content") or item.get("text") or item.get("postText") or ""
                if not post_text or len(post_text.strip()) < 20:
                    diag.leads_rejected += 1
                    continue

                raw_url = item.get("linkedinUrl") or item.get("url") or item.get("postUrl")
                urn = item.get("id") or item.get("urn")
                canonical_url = normalize_post_url(raw_url, urn)
                if not canonical_url or canonical_url in seen_urls:
                    diag.posts_deduplicated += 1
                    continue
                seen_urls.add(canonical_url)

                author_obj = item.get("author") or {}
                if isinstance(author_obj, dict):
                    author_name = author_obj.get("name")
                    author_profile = author_obj.get("linkedinUrl") or author_obj.get("profileUrl")
                    author_headline = author_obj.get("headline")
                else:
                    author_name = str(author_obj) if author_obj else None
                    author_profile = None
                    author_headline = None

                company = item.get("companyName") or item.get("company")
                company_url = item.get("companyUrl")

                # Extract publication timestamp
                pub_at = None
                raw_pub = item.get("postedAt") or item.get("publishedAt")
                if raw_pub:
                    try:
                        pub_at = datetime.fromisoformat(str(raw_pub).replace("Z", "+00:00"))
                    except Exception:
                        pub_at = None

                matched_kws = self.keywords_config.find_matched_keywords(f"{post_text} {author_headline or ''}")
                lead_type = classify_lead_type(post_text, author_headline)

                # Public email extraction
                email, source = extract_public_email(post_text)
                if not email and author_headline:
                    email, source = extract_public_email(author_headline)
                if email:
                    diag.emails_found += 1

                lead_create = LinkedInLeadCreate(
                    post_url=canonical_url,
                    post_text=post_text.strip(),
                    author_name=author_name,
                    author_profile_url=author_profile,
                    author_headline=author_headline,
                    company=company,
                    company_url=company_url,
                    published_at=pub_at,
                    matched_keywords=matched_kws,
                    lead_type=lead_type,
                    contact_email=email,
                    contact_email_source=source,
                    contact_email_verified=bool(email),
                )
                leads.append(lead_create)
                diag.leads_created += 1

            except Exception as norm_exc:
                logger.debug("Failed normalizing item: %s", norm_exc)
                diag.leads_rejected += 1

        self.budget_guard.record_discovery_run(
            posts_fetched=diag.posts_returned,
            leads_created=diag.leads_created,
        )

        return leads, diag
