"""Job Discovery Service orchestrating search, resilience, normalization, and persistence."""

import logging
import time
from typing import Dict, List, Optional, Set
from app.config.settings import Settings, get_settings
from app.database.repositories.job_repository import JobRepository
from app.discovery.india_validator import validate_india_location
from app.discovery.jobspy_client import SUPPORTED_JOB_SOURCES, JobSpyClient, SourceResult
from app.discovery.normalizer import JobNormalizer
from app.discovery.providers.apify_provider import ApifyDiscoveryProvider
from app.models.job import Job, JobCreate, JobSearchRequest, JobSearchResponse, SourceStatus

logger = logging.getLogger(__name__)


class JobDiscoveryService:
    """Service that coordinates resilient multi-source job discovery, normalization, deduplication, and persistence."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        jobspy_client: Optional[JobSpyClient] = None,
        apify_provider: Optional[ApifyDiscoveryProvider] = None,
        job_repository: Optional[JobRepository] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.jobspy_client = jobspy_client or JobSpyClient(settings=self.settings)
        self.apify_provider = apify_provider or ApifyDiscoveryProvider(settings=self.settings)
        self.job_repository = job_repository or JobRepository()

    def validate_sources(self, requested_sources: List[str]) -> List[str]:
        """Validate and filter requested sources against supported job boards.

        Args:
            requested_sources: List of requested source strings

        Returns:
            Validated list of supported sources

        Raises:
            ValueError: If no valid supported sources are provided
        """
        valid = [s.lower().strip() for s in requested_sources if s.lower().strip() in SUPPORTED_JOB_SOURCES]
        if not valid:
            raise ValueError(
                f"No valid sources provided in request. Supported sources are: {sorted(SUPPORTED_JOB_SOURCES)}"
            )
        return valid

    async def discover_jobs(self, request: JobSearchRequest) -> JobSearchResponse:
        """Execute resilient job discovery across all requested sources.

        Args:
            request: JobSearchRequest containing parameters and target sites

        Returns:
            JobSearchResponse with complete source breakdowns, deduplication stats, and stored jobs.
        """
        start_time = time.perf_counter()
        validated_sites = self.validate_sources(request.sites)

        logger.info(
            "Starting Job Discovery: query='%s', location='%s', sites=%s",
            request.search_term,
            request.location,
            validated_sites,
        )

        # 1. Query routed providers independently for each source
        source_results: List[SourceResult] = []

        if "glassdoor" in validated_sites:
            # Route Glassdoor to Apify discovery provider exclusively
            other_sites = [s for s in validated_sites if s != "glassdoor"]
            if other_sites:
                jobspy_results = await self.jobspy_client.search_jobs(
                    site_names=other_sites,
                    search_term=request.search_term,
                    location=request.location,
                    is_remote=request.is_remote,
                    results_wanted=request.results_wanted,
                    hours_old=request.hours_old,
                    country_indeed=request.country_indeed,
                )
                source_results.extend(jobspy_results)

            logger.info("Routing 'glassdoor' to ApifyDiscoveryProvider (Apify ONLY)")
            apify_res = await self.apify_provider.search(
                source="glassdoor",
                search_term=request.search_term,
                location=request.location,
                is_remote=request.is_remote,
                results_wanted=request.results_wanted,
                hours_old=request.hours_old,
                country="India",
            )

            # Strict provider rule: Never invoke JobSpy as a fallback for Glassdoor
            source_results.append(apify_res)
        else:
            # Standard JobSpy discovery across requested non-Glassdoor sites
            source_results = await self.jobspy_client.search_jobs(
                site_names=validated_sites,
                search_term=request.search_term,
                location=request.location,
                is_remote=request.is_remote,
                results_wanted=request.results_wanted,
                hours_old=request.hours_old,
                country_indeed=request.country_indeed,
            )

        # 2. Process source breakdowns and collect raw job items
        sources_status_map: Dict[str, SourceStatus] = {}
        all_raw_jobs: List[dict] = []
        gd_received = 0
        gd_rejected = 0
        gd_rejection_reasons: List[str] = []

        for sr in source_results:
            sources_status_map[sr.source] = SourceStatus(
                status=sr.status,
                provider_status=sr.provider_status,
                http_status=sr.http_status,
                jobs_returned=sr.jobs_returned,
                error=sr.error,
                duration_ms=sr.duration_ms,
                diagnostics=sr.diagnostics,
            )
            if sr.source.lower() == "glassdoor":
                gd_received += sr.jobs_returned
                if sr.error:
                    gd_rejection_reasons.append(sr.error)

            if sr.status == "success" and sr.raw_jobs:
                all_raw_jobs.extend(sr.raw_jobs)

        logger.info(
            "JobSpy query completed. %d raw jobs collected across %d successful sources",
            len(all_raw_jobs),
            sum(1 for s in source_results if s.status == "success"),
        )

        # 3. Normalize and apply 3-tier in-memory deduplication
        seen_source_ext_ids: Set[str] = set()
        seen_urls: Set[str] = set()
        seen_signatures: Set[str] = set()

        unique_jobs_to_persist: List[JobCreate] = []
        batch_duplicates = 0
        gd_normalized = 0
        gd_batch_dup = 0
        gd_india_passed = 0
        gd_india_rejected = 0
        gd_india_unknown = 0

        is_india_search = (
            bool(request.country_indeed and request.country_indeed.lower() == "india")
            or "india" in (request.location or "").lower()
        )

        for raw_item in all_raw_jobs:
            item_source = str(raw_item.get("site") or raw_item.get("source") or "").lower().strip()
            try:
                job_create, dedup_keys = JobNormalizer.normalize(raw_item)
                if job_create.source == "glassdoor":
                    gd_normalized += 1

                # Location validation
                if job_create.source == "glassdoor":
                    is_india, norm_loc, filter_status = validate_india_location(job_create.location, allow_unknown=False)
                    if filter_status == "india_filter_passed":
                        gd_india_passed += 1
                    elif filter_status == "india_filter_rejected":
                        gd_india_rejected += 1
                        gd_rejected += 1
                        gd_rejection_reasons.append(f"Rejected non-India location: {job_create.location}")
                    elif filter_status == "india_location_unknown":
                        gd_india_unknown += 1
                        gd_rejected += 1
                        gd_rejection_reasons.append(f"Rejected unknown location: {job_create.location}")

                    if not is_india:
                        logger.info("Glassdoor job '%s' rejected by India filter (%s): %s", job_create.title, filter_status, job_create.location)
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

                # Deduplication Check 1: source + external_id
                if k_id in seen_source_ext_ids:
                    batch_duplicates += 1
                    if job_create.source == "glassdoor":
                        gd_batch_dup += 1
                    continue

                # Deduplication Check 2: Canonical URL
                if k_url and k_url in seen_urls:
                    batch_duplicates += 1
                    if job_create.source == "glassdoor":
                        gd_batch_dup += 1
                    continue

                # Deduplication Check 3: Normalized title + company + location signature
                if k_sig and k_sig in seen_signatures:
                    batch_duplicates += 1
                    if job_create.source == "glassdoor":
                        gd_batch_dup += 1
                    continue

                # Register seen keys
                seen_source_ext_ids.add(k_id)
                if k_url:
                    seen_urls.add(k_url)
                if k_sig:
                    seen_signatures.add(k_sig)

                unique_jobs_to_persist.append(job_create)

            except Exception as norm_err:
                logger.warning("Failed to normalize raw job: %s. Payload: %s", norm_err, raw_item)
                if item_source == "glassdoor":
                    gd_rejected += 1
                    gd_rejection_reasons.append(f"Normalization error: {norm_err}")
                continue

        logger.info(
            "Normalization & In-memory deduplication: %d unique jobs, %d batch duplicates",
            len(unique_jobs_to_persist),
            batch_duplicates,
        )

        # 4. Persist to Supabase database repository
        persisted_jobs: List[Job] = []
        new_jobs_count = 0
        db_duplicates_count = 0
        gd_inserted = 0
        gd_db_dup = 0
        gd_failed = 0

        if unique_jobs_to_persist:
            for job_in in unique_jobs_to_persist:
                saved_job, is_new = self.job_repository.upsert_job(job_in)
                if saved_job is not None:
                    persisted_jobs.append(saved_job)
                    if is_new:
                        new_jobs_count += 1
                        if job_in.source == "glassdoor":
                            gd_inserted += 1
                    else:
                        db_duplicates_count += 1
                        if job_in.source == "glassdoor":
                            gd_db_dup += 1
                else:
                    if job_in.source == "glassdoor":
                        gd_failed += 1

        total_duplicates = batch_duplicates + db_duplicates_count
        execution_time_ms = int((time.perf_counter() - start_time) * 1000)

        glassdoor_stats = None
        if "glassdoor" in validated_sites:
            budget_status = self.apify_provider.budget_guard.get_budget_status()
            glassdoor_stats = {
                "glassdoor_results_received": gd_received,
                "glassdoor_results_normalized": gd_normalized,
                "glassdoor_results_rejected": gd_rejected,
                "india_filter_passed": gd_india_passed,
                "india_filter_rejected": gd_india_rejected,
                "india_location_unknown": gd_india_unknown,
                "glassdoor_jobs_inserted": gd_inserted,
                "glassdoor_jobs_updated": gd_db_dup,
                "glassdoor_jobs_duplicate": gd_batch_dup + gd_db_dup,
                "glassdoor_jobs_failed": gd_failed,
                "rejection_reasons": gd_rejection_reasons,
                "budget_status": budget_status,
            }

        logger.info(
            "Job Discovery completed in %d ms. %d jobs found, %d new, %d duplicates",
            execution_time_ms,
            len(persisted_jobs),
            new_jobs_count,
            total_duplicates,
        )

        return JobSearchResponse(
            jobs_found=len(persisted_jobs),
            new_jobs=new_jobs_count,
            duplicates=total_duplicates,
            sources=sources_status_map,
            execution_time_ms=execution_time_ms,
            jobs=persisted_jobs,
            glassdoor_stats=glassdoor_stats,
        )
