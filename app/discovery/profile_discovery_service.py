"""Profile-Driven Broad Job Discovery Service.

Coordinates candidate-driven search planning, resilient JobSpy multi-query execution,
cross-query deduplication, Supabase persistence, automated matching, and ranked URL delivery.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from app.config.settings import Settings, get_settings
from app.database.repositories.job_repository import JobRepository
from app.discovery.india_validator import normalize_query_location, validate_india_location
from app.discovery.jobspy_client import JobSpyClient, SourceResult
from app.discovery.normalizer import JobNormalizer
from app.discovery.providers.apify_provider import ApifyDiscoveryProvider
from app.discovery.search_planner import ProfileSearchPlanner
from app.matching.service import MatchService
from app.models.job import (
    ApplicationMethod,
    AvailabilityStatus,
    DiscoveredJobItem,
    DiscoveryQueryResult,
    DiscoveryStats,
    Job,
    JobCreate,
    MatchedJobItem,
    MatchingStats,
    ProfileJobSearchRequest,
    ProfileJobSearchResponse,
    SearchPlan,
    SearchPlanSummary,
    SearchQuery,
)
from app.models.profile import CandidateProfileData
from app.profile.service import ProfileService

logger = logging.getLogger(__name__)


def is_valid_canonical_job_url(source: str, url: Optional[str]) -> bool:
    """Ensure URL is valid, non-empty, and matches source platform canonical listing patterns."""
    if not url or not isinstance(url, str) or not url.strip():
        return False
    u = url.strip().lower()
    if source == "indeed":
        if "indeed." not in u:
            return False
        clean_path = u.split("?")[0].rstrip("/")
        if clean_path in (
            "https://indeed.com",
            "http://indeed.com",
            "https://www.indeed.com",
            "http://www.indeed.com",
            "https://in.indeed.com",
            "http://in.indeed.com",
        ):
            return False
        if "/jobs" in clean_path and not ("/job/" in clean_path or "/viewjob" in clean_path):
            return False
        if "/q-" in u:
            return False
        return True
    elif source == "glassdoor":
        if "glassdoor." not in u or "apify.com" in u:
            return False
        clean_path = u.split("?")[0].rstrip("/")
        if clean_path in (
            "https://glassdoor.com",
            "http://glassdoor.com",
            "https://www.glassdoor.com",
            "http://www.glassdoor.com",
            "https://www.glassdoor.co.in",
            "http://www.glassdoor.co.in",
        ):
            return False
        if "/search" in u or "/job/search" in u:
            return False
        return True
    return True


class ProfileDiscoveryService:
    """Service orchestrating profile-based multi-query discovery and matched ranking."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        profile_service: Optional[ProfileService] = None,
        search_planner: Optional[ProfileSearchPlanner] = None,
        jobspy_client: Optional[JobSpyClient] = None,
        apify_provider: Optional[ApifyDiscoveryProvider] = None,
        job_repository: Optional[JobRepository] = None,
        match_service: Optional[MatchService] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.profile_service = profile_service or ProfileService()
        self.planner = search_planner or ProfileSearchPlanner(settings=self.settings)
        self.jobspy_client = jobspy_client or JobSpyClient(settings=self.settings)
        self.apify_provider = apify_provider or ApifyDiscoveryProvider(settings=self.settings)
        self.job_repository = job_repository or JobRepository()
        self.match_service = match_service or MatchService(settings=self.settings)

    async def discover_jobs_from_profile(
        self,
        request: ProfileJobSearchRequest,
        profile_id: Optional[UUID] = None,
    ) -> ProfileJobSearchResponse:
        """Execute automated broad discovery derived from candidate career profile.

        Args:
            request: ProfileJobSearchRequest parameters
            profile_id: Optional UUID of candidate profile in database

        Returns:
            ProfileJobSearchResponse with search plan, discovery metrics, matching breakdown, and ranked URLs.
        """
        start_time = time.perf_counter()
        logger.info(
            "Starting Profile-Driven Discovery (max_queries=%d, results_per_query=%d, hours_old=%d, matching=%s)",
            request.max_queries,
            request.results_per_query,
            request.hours_old,
            request.run_matching,
        )

        # 1. Load Active Candidate Profile Data
        profile_data: CandidateProfileData = self.profile_service.get_active_profile_data(profile_id)

        # 2. Generate Bounded Search Plan across requested sites
        sites = request.sites if request.sites else ["indeed"]
        queries_per_site = max(1, request.max_queries // len(sites))
        all_planned_queries: List[SearchQuery] = []

        for site in sites:
            sub_plan = self.planner.create_search_plan(
                profile_data=profile_data,
                max_queries=queries_per_site,
                site=site,
                use_nvidia=request.use_nvidia_expansion,
            )
            all_planned_queries.extend(sub_plan.queries)

        all_planned_queries = all_planned_queries[:request.max_queries]
        search_plan = SearchPlan(
            queries=all_planned_queries,
            generated_from="candidate_profile",
            query_count=len(all_planned_queries),
        )

        # 3. Execute Multi-Query Searches via Routed Providers with Bounded Concurrency
        apify_concurrency = getattr(self.settings, "APIFY_GLASSDOOR_MAX_CONCURRENCY", 1)
        apify_sem = asyncio.Semaphore(max(1, apify_concurrency))
        indeed_sem = asyncio.Semaphore(3)

        async def _execute_single_query(q: SearchQuery) -> Tuple[DiscoveryQueryResult, List[dict]]:
            q_start = time.perf_counter()
            target_source = (q.source or "indeed").lower()
            norm_location, is_remote = normalize_query_location(q.location, provider=target_source)
            query_id = getattr(q, "query_id", None) or f"query_{uuid4().hex[:8]}"
            provider_name = "apify" if target_source == "glassdoor" else "jobspy"

            try:
                if target_source == "glassdoor":
                    async with apify_sem:
                        sr: SourceResult = await self.apify_provider.search(
                            source="glassdoor",
                            search_term=q.search_term,
                            location=norm_location,
                            is_remote=is_remote,
                            results_wanted=request.results_per_query,
                            hours_old=request.hours_old,
                            country="India",
                        )
                else:
                    async with indeed_sem:
                        sr: SourceResult = await self.jobspy_client.search_single_source(
                            source=target_source,
                            search_term=q.search_term,
                            location=norm_location,
                            is_remote=is_remote,
                            results_wanted=request.results_per_query,
                            hours_old=request.hours_old,
                            country_indeed=request.country_indeed,
                        )

                duration_ms = int((time.perf_counter() - q_start) * 1000)
                status_code = sr.provider_status or ("SUCCESS" if sr.status == "success" else "FAILED")
                err_clean = sr.error
                if err_clean and self.settings.apify_token_value:
                    err_clean = err_clean.replace(self.settings.apify_token_value, "[REDACTED]")

                qr = DiscoveryQueryResult(
                    query_id=query_id,
                    source=target_source,
                    provider=provider_name,
                    search_term=q.search_term,
                    location=norm_location or ("Remote" if is_remote else q.location),
                    remote=is_remote,
                    status=status_code,
                    jobs_returned=sr.jobs_returned,
                    error=err_clean,
                    duration_ms=duration_ms,
                )
                return qr, sr.raw_jobs if sr.status == "success" else []
            except Exception as exc:
                duration_ms = int((time.perf_counter() - q_start) * 1000)
                logger.error("Unexpected error executing query '%s' (%s): %s", q.search_term, target_source, exc)
                qr = DiscoveryQueryResult(
                    query_id=query_id,
                    source=target_source,
                    provider=provider_name,
                    search_term=q.search_term,
                    location=norm_location or ("Remote" if is_remote else q.location),
                    remote=is_remote,
                    status="FAILED",
                    jobs_returned=0,
                    error=str(exc),
                    duration_ms=duration_ms,
                )
                return qr, []

        query_execution_results = await asyncio.gather(*[_execute_single_query(q) for q in search_plan.queries])

        query_results: List[DiscoveryQueryResult] = []
        all_raw_jobs: List[dict] = []
        queries_succeeded = 0
        queries_failed = 0

        for qr, raw_list in query_execution_results:
            query_results.append(qr)
            if qr.status == "SUCCESS" or len(raw_list) > 0:
                queries_succeeded += 1
                all_raw_jobs.extend(raw_list)
            else:
                queries_failed += 1

        total_raw_jobs = len(all_raw_jobs)
        logger.info(
            "All queries completed: %d succeeded, %d failed. Total raw jobs: %d",
            queries_succeeded,
            queries_failed,
            total_raw_jobs,
        )

        # 4. Normalize and Cross-Query In-Memory Deduplication
        seen_source_ext_ids: Set[str] = set()
        seen_urls: Set[str] = set()
        seen_signatures: Set[str] = set()

        unique_jobs_to_persist: List[JobCreate] = []
        batch_duplicates = 0

        is_india_search = (
            bool(request.country_indeed and request.country_indeed.lower() == "india")
            or "india" in [str(loc).lower() for loc in (getattr(profile_data, "preferred_locations", None) or [])]
            or any("india" in str(loc).lower() for loc in (getattr(profile_data, "preferred_locations", None) or []))
        )

        for raw_item in all_raw_jobs:
            try:
                job_create, dedup_keys = JobNormalizer.normalize(raw_item)

                # URL integrity assertion: reject non-canonical / invalid / cross-platform URLs
                if not is_valid_canonical_job_url(job_create.source, job_create.url):
                    logger.debug("Skipping job '%s' with invalid URL: %s", job_create.title, job_create.url)
                    continue

                # Location validation
                if job_create.source == "glassdoor":
                    is_india, norm_loc, filter_status = validate_india_location(job_create.location, allow_unknown=False)
                    if not is_india:
                        logger.info("Glassdoor job '%s' rejected by India filter: %s", job_create.title, job_create.location)
                        continue
                    job_create.location = norm_loc
                elif is_india_search:
                    is_india, norm_loc, filter_status = validate_india_location(job_create.location, allow_unknown=True)
                    if not is_india:
                        logger.info("Job '%s' (%s) rejected by India filter: %s", job_create.title, job_create.source, job_create.location)
                        continue
                    job_create.location = norm_loc

                k_id = dedup_keys["source_external_id"]
                k_url = dedup_keys["canonical_url"]
                k_sig = dedup_keys["signature"]

                # 3-Tier Deduplication across all queries
                if k_id in seen_source_ext_ids:
                    batch_duplicates += 1
                    continue

                if k_url and k_url in seen_urls:
                    batch_duplicates += 1
                    continue

                if k_sig and k_sig in seen_signatures:
                    batch_duplicates += 1
                    continue

                seen_source_ext_ids.add(k_id)
                if k_url:
                    seen_urls.add(k_url)
                if k_sig:
                    seen_signatures.add(k_sig)

                unique_jobs_to_persist.append(job_create)
            except Exception as norm_err:
                logger.warning("Failed to normalize raw job payload: %s", norm_err)
                continue

        logger.info(
            "Cross-query deduplication complete: %d unique jobs, %d batch duplicates removed",
            len(unique_jobs_to_persist),
            batch_duplicates,
        )

        # 5. Persist Unique Jobs to Database & Format Output Jobs with URLs
        persisted_jobs: List[Job] = []
        db_duplicates = 0
        if unique_jobs_to_persist:
            persisted_jobs, _, db_duplicates = self.job_repository.upsert_jobs(unique_jobs_to_persist)

        discovered_jobs_output: List[DiscoveredJobItem] = [
            DiscoveredJobItem(
                id=pj.id,
                source=pj.source,
                external_id=pj.external_id,
                title=pj.title,
                company=pj.company,
                location=pj.location or "India",
                url=pj.url or "",
            )
            for pj in persisted_jobs
            if pj.url and is_valid_canonical_job_url(pj.source, pj.url)
        ]

        total_duplicates_removed = batch_duplicates + db_duplicates
        discovery_duration_ms = int((time.perf_counter() - start_time) * 1000)
        discovery_stats = DiscoveryStats(
            raw_jobs=total_raw_jobs,
            unique_jobs=len(persisted_jobs),
            duplicates_removed=total_duplicates_removed,
            queries_succeeded=queries_succeeded,
            queries_failed=queries_failed,
            query_results=query_results,
            execution_time_ms=discovery_duration_ms,
        )

        # 6. Execute Matching Engine & Rank URLs if Requested
        matching_stats: Optional[MatchingStats] = None
        top_matches: List[MatchedJobItem] = []

        if request.run_matching and persisted_jobs:
            matching_start_time = time.perf_counter()
            matching_stats, top_matches = self._run_matching_pipeline(
                jobs=persisted_jobs,
                profile_id=profile_id,
                use_semantic=request.use_semantic,
                minimum_score=request.minimum_score,
                top_k=request.top_k,
            )
            matching_duration_ms = int((time.perf_counter() - matching_start_time) * 1000)
            matching_stats.matching_time_ms = matching_duration_ms

        total_execution_duration = int((time.perf_counter() - start_time) * 1000)

        # Determine overall execution status
        if queries_succeeded > 0 and queries_failed == 0:
            final_status = "success"
        elif queries_succeeded > 0 and queries_failed > 0:
            final_status = "partial_success"
        else:
            final_status = "failed" if queries_failed > 0 else "success"

        logger.info(
            "Profile Discovery run finished with status='%s' in %d ms (discovery: %d ms, matching: %d ms): %d unique jobs, %d top matches (top_k=%d)",
            final_status,
            total_execution_duration,
            discovery_duration_ms,
            matching_stats.matching_time_ms if matching_stats else 0,
            len(persisted_jobs),
            len(top_matches),
            request.top_k,
        )

        return ProfileJobSearchResponse(
            status=final_status,
            search_plan=SearchPlanSummary(
                queries_generated=search_plan.query_count,
                queries=search_plan.queries,
            ),
            discovery=discovery_stats,
            jobs=discovered_jobs_output,
            matching=matching_stats,
            top_matches=top_matches,
            total_execution_time_ms=total_execution_duration,
        )

    def _run_matching_pipeline(
        self,
        jobs: List[Job],
        profile_id: Optional[UUID],
        use_semantic: bool,
        minimum_score: float,
        top_k: int,
    ) -> Tuple[MatchingStats, List[MatchedJobItem]]:
        """Evaluate and rank discovered jobs against candidate profile."""
        stats = MatchingStats()
        candidate_items: List[MatchedJobItem] = []

        # Use batch in-memory matching path (loads profile & profile_data once)
        match_results = self.match_service.match_discovered_jobs(
            jobs=jobs,
            profile_id=profile_id,
            use_semantic=use_semantic,
        )

        RECOMMENDED_DECISION_TIERS = {"excellent", "strong_match", "review"}

        for job, res in zip(jobs, match_results):
            stats.jobs_scored += 1

            # Standardized tier counts
            dec = (res.decision or "").lower()
            if dec == "excellent":
                stats.excellent += 1
            elif dec == "strong_match":
                stats.strong_match += 1
            elif dec == "review":
                stats.review += 1
            elif dec in ("low_match", "weak_match"):
                stats.low_match += 1
                stats.weak_match += 1
            else:
                stats.reject += 1
                stats.unqualified += 1

            # Filter: Must be in a genuine recommended tier AND meet minimum score
            if dec in RECOMMENDED_DECISION_TIERS and res.final_score >= minimum_score:
                app_method = ApplicationMethod.UNKNOWN.value
                if getattr(job, "easy_apply", None) is True:
                    app_method = ApplicationMethod.EASY_APPLY.value
                elif getattr(job, "easy_apply", None) is False and (getattr(job, "source", "") in ("indeed", "glassdoor")):
                    app_method = ApplicationMethod.EXTERNAL_APPLY.value

                avail_status = AvailabilityStatus.AVAILABLE.value if app_method == ApplicationMethod.EASY_APPLY.value else AvailabilityStatus.UNKNOWN.value
                require_easy = getattr(self.settings, "APPLICATION_REQUIRE_EASY_APPLY", True)
                is_eligible = (app_method == ApplicationMethod.EASY_APPLY.value) or (not require_easy and app_method != ApplicationMethod.EXTERNAL_APPLY.value)

                candidate_items.append(
                    MatchedJobItem(
                        job_id=job.id,
                        title=job.title,
                        company=job.company,
                        location=job.location,
                        source=job.source,
                        url=job.url,  # Preserved original job URL
                        deterministic_score=res.deterministic_score,
                        semantic_score=res.embedding_score or res.llm_score,
                        final_score=round(res.final_score, 1),
                        decision=res.decision,
                        reason=res.reasons[0] if res.reasons else None,
                        application_method=app_method,
                        availability_status=avail_status,
                        application_eligible=is_eligible,
                    )
                )

        # Sort descending by final combined match score
        candidate_items.sort(key=lambda x: x.final_score, reverse=True)
        top_matches = candidate_items[:top_k]

        return stats, top_matches
