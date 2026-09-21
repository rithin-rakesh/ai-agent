# AI Job Application Agent — Master Project Context & Full Prompt History

> **INSTRUCTION FOR DOWNSTREAM LLM (ChatGPT / Claude / Gemini / etc.)**:
> You are continuing development on the **AI Job Application Agent**, an enterprise-grade autonomous job discovery, matching, intelligence, and browser automation system built in Python / FastAPI.
> 
> Read this entire document carefully. It contains:
> 1. Complete System Architecture, Design Decisions, and Directory Layout.
> 2. Database Schema (7 Supabase PostgreSQL tables).
> 3. Core Algorithms, Scoring Formulas, and Math.
> 4. All API Endpoints & Data Contracts.
> 5. The **EXACT, COMPLETE CHRONOLOGICAL ARCHIVE OF EVERY USER PROMPT** given to Antigravity from Phase 1 through Phase 5.1.
> 6. Current Test Suite Status (70/70 passing tests).
> 7. Strict Safety, Anti-Bot, and Scope Boundaries.
> 
> You must preserve all existing architectures, maintain 100% test pass rates, never leak credentials, and adhere strictly to platform automation safety rules.

---

## Table of Contents
1. [Executive Summary & High-Level Architecture](#1-executive-summary--high-level-architecture)
2. [Complete Codebase Directory Inventory](#2-complete-codebase-directory-inventory)
3. [Database Architecture & Supabase PostgreSQL Schema](#3-database-architecture--supabase-postgresql-schema)
4. [Core Algorithms, Scoring Formulas & Math](#4-core-algorithms-scoring-formulas--math)
5. [Complete API Endpoints & Request/Response Contracts](#5-complete-api-endpoints--requestresponse-contracts)
6. [Configuration & Environment Variables (.env)](#6-configuration--environment-variables-env)
7. [Comprehensive Verbatim Archive of All User Prompts](#7-comprehensive-verbatim-archive-of-all-user-prompts)
8. [Current Test Suite Status (70 Tests Passing)](#8-current-test-suite-status-70-tests-passing)
9. [Strict Safety, Compliance & Anti-Bot Boundaries](#9-strict-safety-compliance--anti-bot-boundaries)
10. [Roadmap for Phase 5.2 and Future Phases](#10-roadmap-for-phase-52-and-future-phases)

---

## 1. Executive Summary & High-Level Architecture

The **AI Job Application Agent** is designed to streamline and automate the end-to-end job discovery, scoring, and application lifecycle for candidates.

### Core Pipelines:
1. **Multi-Source Job Discovery (Phase 2)**:
   - Queries a standalone **JobSpy FastMCP Server** (running on port 9423) across 4 major job boards: **LinkedIn, Indeed, Naukri, Glassdoor**.
   - **Fault-Tolerant Isolation**: Failures from one platform (e.g. HTTP 406 / CAPTCHA on Naukri, location errors on Glassdoor) do NOT crash the discovery run.
   - **Unified Schema Normalization**: Normalizes varied raw fields (titles, companies, salary strings, location strings, remote flags, easy-apply tags) into a standardized `Job` domain model.
   - **Automated Deduplication**: SHA-256 content hashing prevents duplicate job postings across runs and platforms.
   - **Supabase Persistence**: Persists new unique jobs into Supabase PostgreSQL `jobs` table with `(source, external_id)` uniqueness.

2. **Candidate Profile System (Phase 3)**:
   - Reads candidate preferences, technical skill levels, experience history, and education from a local JSON file (`data/profile/candidate_profile.json`).
   - Normalizes skills against standard aliases (e.g. `node.js` -> `Node.js`, `postgres` -> `PostgreSQL`).
   - Syncs profile data into Supabase `profiles` and `skills` tables.

3. **Deterministic Match Engine (Phase 3)**:
   - Multi-category weighted rule-based compatibility scoring ($100\%$ total):
     - **Skills Matcher (40%)**: Technical keyword overlap, alias resolution, required vs preferred coverage, importance weighting (`critical: 1.5`, `high: 1.2`, `medium: 1.0`, `low: 0.7`).
     - **Title Matcher (20%)**: Exact title bonus, role equivalence groups, Jaccard token overlap.
     - **Experience Matcher (15%)**: Regex bounds parsing (`0-2y`, `3+y`, freshers) vs candidate years.
     - **Location Matcher (10%)**: Remote compatibility, city proximity, preferred regions.
     - **Education Matcher (5%)**: Degree hierarchy (B.Tech/BS, MS, PhD) with neutral fallback.
     - **Salary Matcher (5%)**: Compensation interval overlap with neutral fallback.
     - **Job Type Matcher (5%)**: Full-time, contract, internship alignment.
   - Produces structured `MatchResult` with breakdown, human-readable explanations, strengths, and caveats.

4. **NVIDIA Semantic AI Matching (Phase 4)**:
   - Integration with NVIDIA NIM cloud APIs (`https://integrate.api.nvidia.com/v1`).
   - **Asymmetric Embeddings**: `nvidia/nemotron-3-embed-1b` generating 2048-dimensional vectors with `input_type="query"` for candidate profiles (PII strictly excluded) and `input_type="passage"` for job descriptions.
   - **Cosine Similarity**: Vector similarity normalized into a $0–100$ score.
   - **Reasoning Model**: `nvidia/nemotron-3.5-lightning-30b-a3b` structured evaluation analyzing transferable skills, domain mismatches, strengths, and concerns.
   - **Cost Gating**: LLM reasoning is only executed when Deterministic Score $\ge 60.0$.
   - **Semantic Cache**: In-memory SHA-256 cache preventing redundant API calls.
   - **Combined Scoring Synthesis**:
     $$\text{Final Match Score} = 0.70 \times \text{Deterministic} + 0.20 \times \text{Embedding} + 0.10 \times \text{Reasoning}$$
   - Persists final match score and populated `llm_score` in Supabase `job_matches` table.

5. **Indeed Playwright Browser Foundation (Phase 5.1)**:
   - Persistent Chromium browser context preserving cookies and local storage at `browser_sessions/indeed`.
   - DOM authentication state detection (account menu vs sign-in button).
   - Manual login workflow: opens visible browser for manual user login (MFA/CAPTCHA) and persists session without storing passwords.
   - Captures non-sensitive verification screenshots to `screenshots/indeed/indeed_home.png`.
   - `PlatformAdapter` abstraction for clean platform decoupling.

---

## 2. Complete Codebase Directory Inventory

```
D:\Job Agent\
├── AI-job-application-agent\                 # Main Application Repository
│   ├── .env                                   # Local credentials & configuration (git ignored)
│   ├── .env.example                           # Configuration reference template
│   ├── .gitignore                             # Git ignore rules for secrets, sessions, traces
│   ├── Dockerfile                             # Container image specification
│   ├── docker-compose.yml                     # Docker service orchestration
│   ├── requirements.txt                       # Python dependencies (FastAPI, Supabase, Playwright, etc.)
│   ├── README.md                              # Project documentation
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── api/                               # FastAPI Layer
│   │   │   ├── __init__.py
│   │   │   ├── main.py                        # App entrypoint, lifespan, router mounting (v0.5.0)
│   │   │   └── routers/
│   │   │       ├── __init__.py
│   │   │       ├── automation.py              # /automation/health, /automation/indeed/*
│   │   │       ├── jobs.py                    # /jobs/search, /jobs, /jobs/{id}, /jobs/stats
│   │   │       ├── llm.py                     # /llm/health
│   │   │       ├── matches.py                 # /matches/run, /matches/{job_id}, /matches/stats
│   │   │       └── profile.py                 # /profile, /profile/sync, /profile/skills
│   │   │
│   │   ├── automation/                        # Playwright Browser Automation (Phase 5)
│   │   │   ├── __init__.py
│   │   │   ├── browser_profiles.py            # Persistent context directory management
│   │   │   ├── playwright_manager.py          # Playwright lifecycle, viewport, trace abstraction
│   │   │   └── session_manager.py             # SessionState enums & SessionStatus model
│   │   │
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   └── settings.py                    # Pydantic BaseSettings with SecretStr & path resolvers
│   │   │
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── supabase.py                    # Supabase client singleton & IPv4 patch
│   │   │   └── repositories/
│   │   │       ├── __init__.py
│   │   │       ├── job_repository.py          # CRUD for 'jobs' table
│   │   │       ├── match_repository.py        # CRUD & upsert for 'job_matches' table
│   │   │       └── profile_repository.py      # CRUD for 'profiles' & 'skills' tables
│   │   │
│   │   ├── discovery/                         # Job Discovery Pipeline (Phase 2)
│   │   │   ├── __init__.py
│   │   │   ├── client.py                      # HTTP client to JobSpy MCP Server
│   │   │   ├── deduplicator.py                # SHA256 signature deduplication
│   │   │   ├── normalizer.py                  # Unified schema normalizer across 4 boards
│   │   │   └── service.py                     # Fault-tolerant multi-source discovery pipeline
│   │   │
│   │   ├── llm/                               # NVIDIA AI Semantic Matching (Phase 4)
│   │   │   ├── __init__.py
│   │   │   ├── cache.py                       # SHA256 in-memory semantic cache
│   │   │   ├── embeddings.py                  # NVIDIA Nemotron embeddings (query vs passage)
│   │   │   ├── nvidia_client.py               # NVIDIA NIM API client with auth error handling
│   │   │   ├── reasoning.py                   # NVIDIA Nemotron reasoning & JSON parsing
│   │   │   ├── similarity.py                  # Cosine similarity & 0-100 normalization
│   │   │   └── prompts/
│   │   │       ├── semantic_match_system.txt  # System prompt demanding JSON schema
│   │   │       └── semantic_match_user.txt    # User prompt injecting candidate, job & scores
│   │   │
│   │   ├── matching/                          # Deterministic & Combined Matching (Phases 3 & 4)
│   │   │   ├── __init__.py
│   │   │   ├── education_matcher.py           # Degree level hierarchy matching (5%)
│   │   │   ├── experience_matcher.py          # Regex bounds parsing & experience years (15%)
│   │   │   ├── explanation.py                 # Human-readable match reasons & caveats generator
│   │   │   ├── job_type_matcher.py            # Employment type alignment (5%)
│   │   │   ├── location_matcher.py            # Proximity & remote preference matching (10%)
│   │   │   ├── matcher.py                     # DeterministicMatchEngine orchestration
│   │   │   ├── salary_matcher.py              # Compensation range overlap (5%)
│   │   │   ├── scorer.py                      # ScoringConfig & calculate_combined_score (70/20/10)
│   │   │   ├── semantic_matcher.py            # NVIDIASemanticMatcher implementation
│   │   │   ├── service.py                     # MatchService pipeline with threshold gating
│   │   │   ├── skill_matcher.py               # Weighted technical skill matcher & aliases (40%)
│   │   │   └── title_matcher.py               # Role equivalence & Jaccard token overlap (20%)
│   │   │
│   │   ├── models/                            # Pydantic v2 Domain Models
│   │   │   ├── __init__.py
│   │   │   ├── application.py                 # Application, ApplicationAnswer, AutomationLog
│   │   │   ├── job.py                         # JobBase, JobCreate, Job, JobSearchRequest
│   │   │   ├── llm.py                         # LLMHealthResponse, ReasoningOutput, SemanticMatchEvaluation
│   │   │   ├── match.py                       # MatchBreakdown, MatchResult, JobMatch, JobMatchCreate
│   │   │   └── profile.py                     # CandidateProfileData, Profile, Skill
│   │   │
│   │   ├── platforms/                         # Platform Adapters (Phase 5)
│   │   │   ├── __init__.py
│   │   │   ├── base.py                        # PlatformAdapter abstract base class
│   │   │   └── indeed/
│   │   │       ├── __init__.py
│   │   │       ├── adapter.py                 # IndeedPlatformAdapter implementation
│   │   │       └── session.py                 # IndeedSessionManager (persistent context & login)
│   │   │
│   │   └── profile/                           # Candidate Profile Normalizer & Service (Phase 3)
│   │       ├── __init__.py
│   │       ├── loader.py                      # ProfileLoader reading local JSON
│   │       ├── normalizer.py                  # Skill aliases & casing normalizer
│   │       └── service.py                     # ProfileService synchronizing local to Supabase
│   │
│   ├── data/
│   │   ├── profile/
│   │   │   └── candidate_profile.json         # Local candidate profile configuration
│   │   └── resumes/                           # Local resumes (PDF/DOCX)
│   │
│   ├── sql/
│   │   └── schema.sql                         # Complete PostgreSQL DDL for Supabase
│   │
│   └── tests/                                 # Full Pytest Test Suite (70 tests)
│       ├── __init__.py
│       ├── test_automation.py                 # Browser settings, paths, Indeed session mocks (13 tests)
│       ├── test_discovery.py                  # Normalizer, deduplication, fault tolerance (9 tests)
│       ├── test_health.py                     # /health, settings, client validation (4 tests)
│       ├── test_llm.py                        # NVIDIA client, embeddings, reasoning, cache (16 tests)
│       ├── test_matching.py                   # 7 deterministic sub-matchers & pipeline (14 tests)
│       ├── test_profile.py                    # Profile loading, skill normalization, sync (7 tests)
│       └── test_semantic_matching.py          # Combined scoring, threshold gating, API (7 tests)
│
└── jobspy-mcp-server/                         # Standalone JobSpy FastMCP Discovery Service (Port 9423)
```

---

## 3. Database Architecture & Supabase PostgreSQL Schema

The database is built on PostgreSQL inside Supabase, managing 7 interconnected tables with UUID primary keys, foreign key cascading, and indexes:

```sql
-- 1. PROFILES TABLE
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    location TEXT,
    experience_years NUMERIC(4, 1) DEFAULT 0.0,
    preferred_roles TEXT[] DEFAULT '{}',
    preferred_locations TEXT[] DEFAULT '{}',
    remote_preference TEXT DEFAULT 'any',
    salary_min NUMERIC(12, 2),
    salary_max NUMERIC(12, 2),
    job_type TEXT DEFAULT 'full-time',
    education_degree TEXT,
    education_field TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. SKILLS TABLE
CREATE TABLE IF NOT EXISTS skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    category TEXT DEFAULT 'technical',
    years_experience NUMERIC(4, 1) DEFAULT 0.0,
    importance TEXT DEFAULT 'medium',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. JOBS TABLE
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT,
    description TEXT,
    url TEXT,
    salary_min NUMERIC(12, 2),
    salary_max NUMERIC(12, 2),
    experience_text TEXT,
    job_type TEXT,
    remote BOOLEAN DEFAULT FALSE,
    posted_at TIMESTAMPTZ,
    easy_apply BOOLEAN DEFAULT FALSE,
    raw_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_jobs_source_external_id UNIQUE (source, external_id)
);

-- 4. JOB MATCHES TABLE
CREATE TABLE IF NOT EXISTS job_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    match_score NUMERIC(5, 2) NOT NULL,
    skill_score NUMERIC(5, 2),
    title_score NUMERIC(5, 2),
    experience_score NUMERIC(5, 2),
    location_score NUMERIC(5, 2),
    salary_score NUMERIC(5, 2),
    llm_score NUMERIC(5, 2),
    reason TEXT,
    decision TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_job_matches_job_profile UNIQUE (job_id, profile_id)
);

-- 5. APPLICATIONS TABLE
CREATE TABLE IF NOT EXISTS applications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'draft',
    submission_type TEXT DEFAULT 'manual_review',
    submitted_at TIMESTAMPTZ,
    error_log TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. APPLICATION ANSWERS TABLE
CREATE TABLE IF NOT EXISTS application_answers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id UUID NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    question_type TEXT DEFAULT 'text',
    answer_text TEXT NOT NULL,
    confidence_score NUMERIC(4, 2) DEFAULT 1.0,
    source TEXT DEFAULT 'profile',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 7. AUTOMATION LOGS TABLE
CREATE TABLE IF NOT EXISTS automation_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id UUID REFERENCES applications(id) ON DELETE SET NULL,
    platform TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    screenshot_url TEXT,
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 4. Core Algorithms, Scoring Formulas & Math

### 1. Deterministic Multi-Category Match Scoring
The deterministic engine evaluates 7 independent categories against the candidate profile:
$$\text{Deterministic Composite Score} = \sum_{i=1}^{7} w_i \cdot S_i$$
Where the weights $w_i$ sum to $1.00$ ($100\%$):
- **Skills Score ($S_{\text{skills}}$)**: **$40\%$** (Weighted keyword extraction, alias resolution, required vs preferred coverage, weighted by `critical: 1.5`, `high: 1.2`, `medium: 1.0`, `low: 0.7`).
- **Title Score ($S_{\text{title}}$)**: **$20\%$** (Exact title match bonus, role equivalence groups, Jaccard token overlap).
- **Experience Score ($S_{\text{exp}}$)**: **$15\%$** (Regex boundary extraction from job text: freshers, year intervals, senior roles vs candidate years).
- **Location Score ($S_{\text{loc}}$)**: **$10\%$** (Remote compatibility, candidate city match, preferred regions, country match).
- **Education Score ($S_{\text{edu}}$)**: **$5\%$** (Degree hierarchy: B.Tech/BS, MS, PhD; neutral fallback when unspecified).
- **Salary Score ($S_{\text{sal}}$)**: **$5\%$** (Compensation interval overlap; neutral fallback when undisclosed).
- **Job Type Score ($S_{\text{type}}$)**: **$5\%$** (Full-time, contract, internship alignment).

### 2. Semantic Similarity & Normalization (NVIDIA Nemotron Embeddings)
- Query text: formatted from candidate profile excluding PII (`input_type="query"`).
- Passage text: formatted from job details (`input_type="passage"`).
- Cosine similarity:
  $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
- Normalized to 0–100 score:
  $$S_{\text{embedding}} = \max\left(0.0, \min\left(100.0, \cos(\theta) \times 100.0\right)\right)$$

### 3. Combined Scoring Synthesis (Phase 4)
$$\text{Final Match Score} = 0.70 \cdot S_{\text{deterministic}} + 0.20 \cdot S_{\text{embedding}} + 0.10 \cdot S_{\text{reasoning}}$$
- **Cost Gating**: Only jobs with $S_{\text{deterministic}} \ge 60.0$ trigger LLM reasoning calls ($S_{\text{reasoning}}$). Jobs below 60 skip reasoning calls, saving token cost.
- **Dynamic Weight Redistribution**: If embedding or reasoning fails or is disabled, remaining active weights are scaled proportionally to guarantee a $100\%$ scale.

### 4. Decision Tiers
| Final Score Range | Decision Category |
| :--- | :--- |
| $\ge 85.0$ | `excellent` |
| $\ge 70.0$ | `strong_match` |
| $\ge 55.0$ | `review` |
| $\ge 40.0$ | `low_match` |
| $< 40.0$ | `reject` |

### 5. Deduplication Signature
Unique jobs are identified and deduplicated across platforms via SHA-256:
$$\text{Signature} = \text{SHA256}\left(\text{normalize}(company) \parallel \text{normalize}(title) \parallel \text{normalize}(location)\right)$$

---

## 5. Complete API Endpoints & Request/Response Contracts

| HTTP Method | Route | Description | Phase Added |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Root service identifier and phase status | Phase 1 |
| `GET` | `/health` | Core FastAPI health check | Phase 1 |
| `POST` | `/jobs/search` | Multi-source discovery across LinkedIn, Indeed, Naukri, Glassdoor | Phase 2.2 |
| `GET` | `/jobs` | Paginated job list with filters (source, title, remote, company) | Phase 2.2 |
| `GET` | `/jobs/stats` | Discovery statistics and count by source | Phase 2.2 |
| `GET` | `/jobs/{id}` | Single job posting details | Phase 2.2 |
| `GET` | `/profile` | Fetch active candidate profile | Phase 3 |
| `POST` | `/profile/sync` | Sync local JSON profile to Supabase database | Phase 3 |
| `GET` | `/profile/skills` | List normalized skills for active profile | Phase 3 |
| `POST` | `/matches/run` | Run batch matching across jobs (`use_semantic=True/False`) | Phase 3 & 4 |
| `GET` | `/matches/stats` | Match tier distribution and average score | Phase 3 |
| `GET` | `/matches/{job_id}` | Detailed match breakdown, sub-scores, and reasons for a job | Phase 3 & 4 |
| `GET` | `/llm/health` | NVIDIA NIM model availability and configuration status | Phase 4 |
| `GET` | `/automation/health` | Playwright & Chromium readiness and profile metrics | Phase 5.1 |
| `GET` | `/automation/indeed/session` | Inspect Indeed persistent session authentication status | Phase 5.1 |
| `POST` | `/automation/indeed/open` | Open Indeed homepage in persistent browser and capture screenshot | Phase 5.1 |

---

## 6. Configuration & Environment Variables (.env)

```env
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key-here
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key-here

# Application Configuration
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO

# JobSpy MCP Configuration
JOBSPY_ENABLED=true
JOBSPY_HOST=127.0.0.1
JOBSPY_PORT=9423
JOBSPY_TIMEOUT_SECONDS=120

# Candidate Profile Configuration
CANDIDATE_PROFILE_PATH=data/profile/candidate_profile.json

# NVIDIA AI Configuration (Phase 4 Semantic Matching)
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_API_KEY=your-nvidia-api-key-here
NVIDIA_REASONING_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
NVIDIA_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b
SEMANTIC_MATCHING_ENABLED=true
DETERMINISTIC_SEMANTIC_THRESHOLD=60.0

# Browser Automation Configuration (Phase 5 Playwright Foundation)
PLAYWRIGHT_HEADLESS=false
PLAYWRIGHT_BROWSER=chromium
PLAYWRIGHT_TIMEOUT_MS=30000
PLAYWRIGHT_TRACE=false
INDEED_BROWSER_PROFILE_PATH=browser_sessions/indeed
SCREENSHOTS_PATH=screenshots
PLAYWRIGHT_ARTIFACTS_PATH=playwright-artifacts
```

---

## 7. Comprehensive Verbatim Archive of All User Prompts

Below is the complete, unaltered, verbatim archive of all prompts submitted by the project owner from the very beginning of the project to the current moment.

### Prompt #1 (Conversation Step 0)

```text
We are starting Phase 1 of a new project called:

AI Job Application Agent

This is a completely new project. Do NOT reuse or copy code from my old Job Scout AI project.

PHASE 1 OBJECTIVE:
Build only the project foundation, configuration, Supabase integration, initial database schema, API health check, Docker setup, and GitHub-ready structure.

DO NOT implement:
- JobSpy integration
- Job discovery
- NVIDIA LLM integration
- matching
- Playwright
- pywinauto
- browser login
- auto-apply
- application agents

Those will be implemented in later phases.

TECH STACK FOR PHASE 1:

Python 3.11+
FastAPI
Pydantic Settings
Supabase Python client
pytest
Docker
Docker Compose

PROJECT STRUCTURE:

AI-job-application-agent/
│
├── app/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   └── supabase.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── profile.py
│   │   ├── job.py
│   │   ├── match.py
│   │   └── application.py
│   │
│   └── api/
│       ├── __init__.py
│       └── main.py
│
├── tests/
│   ├── __init__.py
│   └── test_health.py
│
├── data/
│   ├── profile/
│   └── resumes/
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md

CONFIGURATION:

Use Pydantic Settings.

Environment variables should include:

SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=

APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

Do not hard-code secrets.

Never print the Supabase service-role key.

DATABASE:

Use Supabase PostgreSQL.

Create an initial SQL schema for these tables:

1. profiles
<truncated 1511 bytes>
database client module that reads configuration from environment variables.

Do not put business logic in the Supabase client module.

API:

Create a FastAPI application.

Implement:

GET /health

Response:

{
    "status": "ok",
    "environment": "development"
}

Also implement:

GET /

with a simple message identifying the application.

TESTING:

Create a pytest test for /health.

The test must not require the real Supabase credentials.

DOCKER:

Create a Dockerfile for the FastAPI application.

Create docker-compose.yml for local development.

Expose port 8000.

Use environment variables rather than hard-coded secrets.

GIT:

Create a .gitignore that excludes:

.env
__pycache__/
.pytest_cache/
.venv/
*.pyc
browser_sessions/
screenshots/
logs/

Do NOT ignore .env.example.

README:

Explain:
1. Project purpose
2. Phase 1 architecture
3. Local Python setup
4. Environment variable setup
5. How to start FastAPI
6. How to run tests
7. How to run Docker
8. How to connect the project to Supabase
9. What Phase 1 does NOT contain yet

CODE QUALITY:

Use type hints.
Use clean modular structure.
Use logging rather than print statements.
Handle configuration errors clearly.
Do not introduce unnecessary frameworks or dependencies.
Do not implement future phases prematurely.

When complete:
1. Show all files created.
2. Show requirements.txt.
3. Show the Supabase SQL schema.
4. Show the commands to run locally.
5. Run the test suite.
6. Start the FastAPI application and verify /health.
7. Report any errors before considering Phase 1 complete.
```

---

### Prompt #2 (Conversation Step 15)

```text
proceed
```

---

### Prompt #3 (Conversation Step 120)

```text
The Supabase credentials have now been added to the local .env file.

Verify the Supabase connection without exposing or printing any secret values.

Tasks:
1. Load the existing .env configuration.
2. Run the existing Supabase connection test.
3. Run the complete pytest suite:
   python -m pytest tests -v
4. Start FastAPI if necessary.
5. Verify GET /health.
6. Confirm whether Supabase is reachable.
7. Do not modify the architecture.
8. Do not start Phase 2.
9. Do not add JobSpy, NVIDIA, Playwright, or pywinauto yet.

If the connection fails, report only the non-sensitive error message and explain the fix.
```

---

### Prompt #4 (Conversation Step 153)

```text
Phase 1 final verification.

The Supabase database schema has now been successfully executed and the following 7 tables are confirmed:

profiles
skills
jobs
job_matches
applications
application_answers
automation_logs

Now verify the application-to-Supabase connection.

Tasks:
1. Load the existing .env configuration.
2. Do not print any secret values.
3. Use the existing Supabase client implementation.
4. Perform a harmless read-only query against the jobs table.
5. Verify that the query succeeds.
6. Run:
   python -m pytest tests -v
7. Verify GET /health.
8. Do not modify the architecture.
9. Do not start Phase 2.
10. Report:
   - Supabase connection success/failure
   - pytest result
   - /health result

Do not expose credentials in logs or output.
```

---

### Prompt #5 (Conversation Step 165)

```text
I want to reorganize my local project folders.

Current folders:
D:\AI-job-application-agent\
D:\jobspy-mcp-server\

Target structure:
D:\Job Agent\
    ├── AI-job-application-agent\
    └── jobspy-mcp-server\

IMPORTANT:

1. Move the two existing directories into:
   D:\Job Agent\

2. Do NOT copy or duplicate the projects.
3. Do NOT delete any project files.
4. Preserve all Git repositories and their .git directories.
5. Preserve the .env file inside AI-job-application-agent.
6. Do not expose, print, or modify any secrets in .env.
7. Do not modify source code.
8. Do not change Git configuration.
9. Do not modify the JobSpy repository contents.
10. Before moving, verify that both source directories exist.
11. Create D:\Job Agent if it does not exist.
12. Move:
      D:\AI-job-application-agent
      → D:\Job Agent\AI-job-application-agent

      D:\jobspy-mcp-server
      → D:\Job Agent\jobspy-mcp-server

13. After moving, verify:
      D:\Job Agent\AI-job-application-agent
      D:\Job Agent\jobspy-mcp-server

14. Run `git status` inside BOTH repositories to confirm their Git repositories are intact.
15. Report the final folder structure and Git status.
16. Do not start or modify Phase 2.
```

---

### Prompt #6 (Conversation Step 189)

```text
PHASE 2.1 — JOBSPY MCP STANDALONE VERIFICATION

We have completed Phase 1 successfully.

Main application:
D:\Job Agent\AI-job-application-agent

JobSpy MCP repository:
D:\Job Agent\jobspy-mcp-server

IMPORTANT:
This phase is ONLY for verifying and running the standalone JobSpy MCP Server.

DO NOT modify:
D:\Job Agent\AI-job-application-agent

Do NOT:
- copy JobSpy source code into the main application
- add JobSpy integration code to the main application
- modify the main application's architecture
- add NVIDIA LLM
- add matching logic
- add Playwright
- add pywinauto
- add application agents
- add auto-apply
- connect JobSpy to Supabase
- store jobs in the main application's database

The JobSpy repository must remain a separate service/project.

REPOSITORY:

D:\Job Agent\jobspy-mcp-server

Official repository:
https://github.com/borgius/jobspy-mcp-server

TASK 1 — INSPECT THE REPOSITORY

1. Inspect the actual cloned repository contents.
2. Read its README and relevant configuration files.
3. Determine the exact current:
   - Node.js requirements
   - Python requirements
   - package manager
   - installation commands
   - startup commands
   - MCP transport
   - MCP endpoint/port
   - available tools
4. Do not rely on assumptions from previous instructions if the actual repository differs.
5. Follow the repository's current code/documentation as the source of truth.

TASK 2 — VERIFY LOCAL ENVIRONMENT

Check and report:

- Node.js version
- npm version
- pnpm version if available
- Python version
- git version

Do not change the user's global environment unnecessarily.

TASK 3 — INSTALL DEPENDENCIES

Install the dependencies required by the JobSpy MCP repository according to its current documentation/package configuration.

Keep all dependencies isolated to the JobSpy project where possible.

Do not install JobSpy dependencies into:

D:\Job Agent\AI-job-application-agent

TASK 4 — CONFIGURATION

Inspect whethe
<truncated 1925 bytes>
LITY

For successful results, verify whether the returned data contains fields such as:

- title
- company
- location
- description
- URL
- source
- job type
- salary if available
- posted date if available
- external/job ID if available

Do not modify or normalize the results yet.

This phase is only verification.

TASK 9 — NO DATABASE INTEGRATION

Do NOT connect the JobSpy server to:

D:\Job Agent\AI-job-application-agent

Do NOT insert test jobs into Supabase.

Do NOT create new tables.

Do NOT change the existing Supabase schema.

TASK 10 — PRESERVE REPOSITORY STATE

Do not modify tracked source files unless required to run the repository according to its own documented configuration.

Before finishing:

Run:

git status

inside:

D:\Job Agent\jobspy-mcp-server

Report whether the working tree is clean.

Do not commit changes.

TASK 11 — FINAL REPORT

At the end provide a concise technical report containing:

1. Repository path
2. Node.js version
3. Python version
4. Package manager
5. Dependencies installed
6. Required environment variables, without values
7. Startup command
8. MCP transport
9. MCP host
10. MCP port
11. MCP endpoint
12. Available search tool name
13. Exact search parameters used
14. Total results returned
15. Results by source:
    LinkedIn:
    Indeed:
    Naukri:
    Glassdoor:
16. Example structure of one returned job
17. Any failures and their causes
18. Whether the JobSpy MCP server is ready for Phase 2.2
19. `git status` result

IMPORTANT FINAL BOUNDARY:

Do not move forward into Phase 2.2.

Do not connect JobSpy to the AI Job Application Agent yet.

Do not implement any future functionality.

The only success criterion for Phase 2.1 is:

JOBSPY MCP SERVER RUNS LOCALLY
+
SEARCH_JOBS CAN BE CALLED
+
THE FOUR REQUESTED SOURCES ARE TESTED
+
RESULTS/FAILURES ARE CLEARLY REPORTED.
```

---

### Prompt #7 (Conversation Step 562)

```text
PHASE 2.2 — JOBSPY MCP INTEGRATION + RESILIENT JOB DISCOVERY

Phase 2.1 has been completed successfully.

MAIN APPLICATION:
D:\Job Agent\AI-job-application-agent

JOBSPY MCP SERVER:
D:\Job Agent\jobspy-mcp-server

The JobSpy MCP server has already been verified independently.

Current verified source behavior:

- LinkedIn → SUCCESS
- Indeed → SUCCESS
- Naukri → HTTP 406 / reCAPTCHA required
- Glassdoor → location parsing error for "Kochi, Kerala"

IMPORTANT:
We are NOT abandoning Naukri or Glassdoor.

The application must continue requesting all four sources, but a failure from one source must NOT stop successful sources from returning results.

Desired behavior:

LinkedIn succeeds  → use results
Indeed succeeds    → use results
Naukri fails       → record failure and continue
Glassdoor fails    → record failure and continue

Do NOT attempt to bypass CAPTCHA, anti-bot systems, authentication restrictions, or access controls.

--------------------------------------------------
SCOPE OF PHASE 2.2
--------------------------------------------------

Implement ONLY:

JobSpy MCP
    ↓
JobSpy client
    ↓
Source-level error handling
    ↓
Job normalization
    ↓
Deduplication
    ↓
Supabase jobs table
    ↓
Discovery API

DO NOT IMPLEMENT:

- NVIDIA LLM
- resume matching
- semantic matching
- scoring
- Playwright
- pywinauto
- browser login
- application agents
- auto-apply
- application submission

--------------------------------------------------
1. CREATE DISCOVERY MODULE
--------------------------------------------------

Inside:

D:\Job Agent\AI-job-application-agent

create:

app/discovery/
    __init__.py
    jobspy_client.py
    normalizer.py
    service.py

Create:

app/database/repositories/
    __init__.py
    job_repository.py

Keep JobSpy communication isolated in jobspy_client.py.

Keep Supabase/database logic isolated in job_repository.py.

Keep orchestration in service
<truncated 8377 bytes>
malization count
- duplicate count
- Supabase insert count
- total execution time

Never log:

- Supabase secret key
- API keys
- passwords
- cookies
- session tokens

--------------------------------------------------
16. JOBSPY REPOSITORY
--------------------------------------------------

Do not copy the JobSpy repository into the main application.

Do not modify:

D:\Job Agent\jobspy-mcp-server

unless absolutely required for communication.

Do not commit anything to the original JobSpy repository.

The existing local modifications from Phase 2.1 must remain separate.

--------------------------------------------------
17. GIT
--------------------------------------------------

At the end run:

git status

inside:

D:\Job Agent\AI-job-application-agent

Confirm only intended Phase 2.2 files are changed.

Do not commit automatically.

--------------------------------------------------
18. FINAL REPORT
--------------------------------------------------

Report:

1. Files created.
2. Files modified.
3. JobSpy transport used.
4. JobSpy endpoint used.
5. Unit tests.
6. Real integration test.
7. Total jobs received.
8. Total jobs inserted into Supabase.
9. LinkedIn result count.
10. Indeed result count.
11. Naukri result count/error.
12. Glassdoor result count/error.
13. Duplicate count.
14. Supabase verification.
15. Git status.
16. Confirmation that NVIDIA, Playwright, pywinauto and auto-apply were NOT implemented.

STOP after Phase 2.2.

Do not proceed to Phase 3.

SUCCESS CRITERIA:

The phase is complete when:

JobSpy MCP
    ↓
All four requested sources are attempted independently
    ↓
Successful sources produce jobs
    ↓
Failed sources produce structured errors
    ↓
Successful jobs are normalized
    ↓
Duplicates are removed
    ↓
Jobs are stored in Supabase
    ↓
GET /jobs can retrieve them
```

---

### Prompt #8 (Conversation Step 566)

```text
proceed
```

---

### Prompt #9 (Conversation Step 916)

```text
PHASE 2.2 CLEANUP AND RELEASE CHECKPOINT

Phase 2.2 functional implementation is complete.

Do not add any new functionality.

Perform only a release/cleanup review.

MAIN PROJECT:
D:\Job Agent\AI-job-application-agent

JOBSPY PROJECT:
D:\Job Agent\jobspy-mcp-server

Tasks:

1. Review all Phase 2.2 changes in AI-job-application-agent.

2. Verify:
   - all 13 tests still pass
   - no secrets are tracked
   - .env remains ignored
   - no browser automation was added
   - no NVIDIA integration was added
   - no matching/scoring was added

3. Review:
   - app/discovery/
   - app/database/repositories/
   - app/api/routers/jobs.py
   - app/models/job.py
   - app/config/settings.py

4. Check live_integration_results.json.
   Determine whether it is:
   - a temporary generated test artifact
   - a useful permanent fixture
   Do not commit it unless there is a clear reason.

5. Inspect the modification made to:
   D:\Job Agent\jobspy-mcp-server\src\index.js

   Explain exactly what changed and why it was necessary.

6. DO NOT discard the JobSpy change automatically.

7. Do not commit the JobSpy repository.

8. Do not modify the JobSpy repository further.

9. Check whether the main project's .env.example contains only variable names/placeholders and no real secrets.

10. Run:

    git status

    inside:
    D:\Job Agent\AI-job-application-agent

    and:

    git status

    inside:
    D:\Job Agent\jobspy-mcp-server

11. Report:
    - files that should be committed to the main project
    - files that should remain untracked/ignored
    - exact JobSpy source modification
    - whether the JobSpy repository should be restored to clean upstream state
    - final test result

Do not commit anything automatically.
Do not start Phase 3.
```

---

### Prompt #10 (Conversation Step 926)

```text
PHASE 3 — CANDIDATE PROFILE SYSTEM + DETERMINISTIC MATCH ENGINE

We have completed:

Phase 1:
- Project foundation
- Supabase
- PostgreSQL schema
- FastAPI
- Tests
- Docker
- GitHub

Phase 2.1:
- Standalone JobSpy MCP verified

Phase 2.2:
- JobSpy MCP integrated
- LinkedIn/Indeed discovery working
- Naukri/Glassdoor failures isolated
- Job normalization
- Deduplication
- Supabase persistence
- 13 tests passing

MAIN PROJECT:

D:\Job Agent\AI-job-application-agent

PHASE 3 OBJECTIVE:

Build:

Candidate Profile
      +
Deterministic Job Matching Engine
      +
Match explanation
      +
Match persistence

IMPORTANT:

Do NOT yet implement:
- NVIDIA LLM API calls
- NVIDIA embeddings
- Playwright
- pywinauto
- application agents
- browser login
- auto-apply
- application submission

We will integrate NVIDIA AFTER the deterministic engine is working.

--------------------------------------------------
1. CANDIDATE PROFILE SYSTEM
--------------------------------------------------

Extend the candidate profile architecture.

Create:

app/profile/
    __init__.py
    service.py
    repository.py
    profile_loader.py

Use structured profile data.

The profile must support:

PERSONAL:
- name
- email
- phone
- location

CAREER:
- total experience
- education
- preferred job titles
- preferred locations
- remote preference
- job type preference
- salary preference

SKILLS:
For each skill:
- skill name
- category
- proficiency
- years experience
- importance/weight

EXPERIENCE:
Support structured experience records:
- company
- title
- start date
- end date
- responsibilities
- skills used

EDUCATION:
- degree
- field
- institution
- start date
- end date

Do not hard-code candidate information in Python source files.

Store the editable candidate profile in a structured local file and persist the canonical profile in Supabase.

--------------------------------------------------
2. SUPABASE PROF
<truncated 9162 bytes>
--------------------------------------

After unit tests pass:

Use the jobs currently stored in Supabase.

Create/load a test candidate profile from:

data/profile/candidate_profile.json

Populate it with the candidate profile information available in the existing project context, but DO NOT invent personal information.

Run deterministic matching over the existing jobs.

Verify:
- matches are created
- scores are between 0 and 100
- explanations are generated
- ranking works
- job_matches are stored
- llm_score remains NULL

Do NOT call NVIDIA yet.

--------------------------------------------------
22. NVIDIA PREPARATION ONLY
--------------------------------------------------

Add configuration placeholders only:

NVIDIA_API_KEY=
NVIDIA_REASONING_MODEL=nemotron-3.5-lightning-30b-a3b
NVIDIA_EMBEDDING_MODEL=nemotron-3-embed-1b

Do not call the NVIDIA API during Phase 3.

Do not install unnecessary NVIDIA SDKs yet.

Create an interface/abstraction so NVIDIA can be added later without changing the matching engine.

Example:

class SemanticMatcher:
    ...

Do not implement the actual NVIDIA provider yet.

--------------------------------------------------
23. FINAL REPORT
--------------------------------------------------

Report:

1. Files created
2. Files modified
3. Candidate profile schema
4. Matching weights
5. Decision thresholds
6. Number of jobs matched
7. Top 10 matches
8. Unit test count
9. Unit test result
10. Supabase match persistence result
11. Confirmation that llm_score remains NULL
12. Confirmation that NVIDIA API was NOT called
13. Git status

Do not commit automatically.

STOP after Phase 3.

SUCCESS CRITERIA:

Candidate profile
    ↓
Deterministic matcher
    ↓
Scores
    ↓
Explanations
    ↓
job_matches
    ↓
Top ranked jobs

NVIDIA integration will be implemented only in the next phase.
```

---

### Prompt #11 (Conversation Step 936)

```text
proceed
```

---

### Prompt #12 (Conversation Step 1120)

```text
PHASE 4 — NVIDIA SEMANTIC MATCHING

We have completed and committed Phases 1–3.

Current main project:

D:\Job Agent\AI-job-application-agent

Current state:

- JobSpy discovery working
- Jobs persisted in Supabase
- Candidate profile system working
- Deterministic matching working
- 34 tests passing
- job_matches persisted
- llm_score is currently NULL
- SemanticMatcher abstraction already exists
- NVIDIA API has NOT yet been called

IMPORTANT:
Do NOT implement Playwright, pywinauto, browser login, application agents, or auto-apply.

PHASE 4 ONLY adds NVIDIA-powered semantic matching.

--------------------------------------------------
1. NVIDIA CONFIGURATION
--------------------------------------------------

Use environment variables:

NVIDIA_API_KEY=
NVIDIA_REASONING_MODEL=<use the exact reasoning model ID configured by the project owner>
NVIDIA_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b

Do not hard-code the NVIDIA API key.

Do not log the API key.

Do not expose it through API responses.

Use:

NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

Make the base URL configurable through settings rather than hard-coded.

--------------------------------------------------
2. NVIDIA PROVIDER
--------------------------------------------------

Create:

app/llm/
    __init__.py
    nvidia_client.py
    embeddings.py
    reasoning.py

The rest of the application must not directly call the NVIDIA SDK/API.

Create a clean provider abstraction.

For example:

class NVIDIAClient:
    ...

class NVIDIAEmbeddingProvider:
    ...

class NVIDIAReasoningProvider:
    ...

The SemanticMatcher should depend on abstractions/interfaces, not raw HTTP calls.

Use an OpenAI-compatible Python client or direct HTTP only if necessary.

Prefer the simplest stable implementation supported by the current NVIDIA API.

--------------------------------------------------
3. EMBEDDING MODEL
--------------------------------------------------

<truncated 10566 bytes>
:
https://integrate.api.nvidia.com/v1

Embedding model:
nvidia/nemotron-3-embed-1b

Reasoning model:
the exact model ID provided in NVIDIA Build and configured in .env

Do not silently substitute another model.

If the reasoning model ID returns a model-not-found error:

STOP the live integration and report the exact non-secret error.

Do not switch models automatically.

--------------------------------------------------
21. NO APPLICATION AUTOMATION
--------------------------------------------------

Phase 4 must NOT contain:

Playwright
pywinauto
browser login
application submission
application agents
auto-apply

Those are later phases.

--------------------------------------------------
22. GIT
--------------------------------------------------

Do not commit automatically.

At completion run:

git status

inside:

D:\Job Agent\AI-job-application-agent

Verify:

.env is not tracked.

Do not include:
- API keys
- test secrets
- generated credentials
- live private responses

--------------------------------------------------
23. FINAL REPORT
--------------------------------------------------

Report:

1. Files created
2. Files modified
3. NVIDIA base URL
4. reasoning model ID
5. embedding model ID
6. deterministic weight
7. embedding weight
8. reasoning weight
9. number of tests
10. test result
11. live NVIDIA test result
12. embedding dimension
13. sample semantic score
14. sample reasoning score
15. sample final score
16. job_matches persistence result
17. NVIDIA failure handling status
18. caching implementation
19. Git status

DO NOT PRINT THE API KEY.

STOP AFTER PHASE 4.

SUCCESS CRITERIA:

Existing deterministic matcher
        ↓
NVIDIA embedding similarity
        ↓
NVIDIA reasoning analysis
        ↓
Combined match score
        ↓
Supabase job_matches

All existing tests remain green.
```

---

### Prompt #13 (Conversation Step 1140)

```text
proceed
```

---

### Prompt #14 (Conversation Step 1395)

```text
Continue
```

---

### Prompt #15 (Conversation Step 1411)

```text
PHASE 5.1 — INDEED PLAYWRIGHT BROWSER FOUNDATION

We are beginning Phase 5 of the AI Job Application Agent.

MAIN PROJECT:

D:\Job Agent\AI-job-application-agent

Current completed phases:

Phase 1:
- Project foundation
- Supabase
- PostgreSQL
- FastAPI
- Docker
- tests

Phase 2:
- JobSpy MCP
- job discovery
- normalization
- deduplication
- Supabase persistence

Phase 3:
- candidate profile
- deterministic matching
- job ranking

Phase 4:
- NVIDIA embeddings
- NVIDIA reasoning
- semantic matching
- combined scoring

Current system is stable and committed to GitHub.

IMPORTANT SAFETY / SCOPE RULE:

Indeed currently prohibits third-party bots or automated tools from applying for jobs.

Therefore Phase 5 MUST NOT implement automated final submission.

This phase is ONLY browser/session foundation.

DO NOT IMPLEMENT:
- final application submission
- unattended auto-apply
- CAPTCHA bypass
- anti-bot bypass
- rate-limit bypass
- credential scraping
- password storage
- application form auto-filling
- generated application answers
- application agent reasoning
- Naukri automation
- other platform automation

--------------------------------------------------
1. PLAYWRIGHT DEPENDENCY
--------------------------------------------------

Add Playwright to the project using the appropriate Python package.

Use a pinned/stable Playwright version compatible with the existing Python environment.

Do not unnecessarily upgrade unrelated project dependencies.

Install the required browser:

Chromium

Verify that Playwright can launch Chromium successfully.

--------------------------------------------------
2. AUTOMATION MODULE
--------------------------------------------------

Create:

app/automation/
    __init__.py
    playwright_manager.py
    browser_profiles.py
    session_manager.py

Create a clean abstraction around Playwright.

The rest of the application should not directly instantiate Playwright browsers everywhe
<truncated 6458 bytes>
ation filling yet.

Create:

app/platforms/indeed/
    __init__.py
    adapter.py
    session.py

The adapter may implement only:
- session status
- open job page

Do NOT implement Apply clicking yet.

--------------------------------------------------
17. APPLICATION STATE
--------------------------------------------------

Do not create a submitted application state.

For this phase only use states such as:

SESSION_UNKNOWN
LOGIN_REQUIRED
AUTHENTICATED
SESSION_ERROR

--------------------------------------------------
18. DOCKER PREPARATION
--------------------------------------------------

Do not require the production Docker environment to use the local Windows browser profile yet.

For Phase 5.1:

- local Windows development is primary
- visible Chromium is primary
- persistent local browser profile is primary

Keep the architecture portable so a future remote worker can be added.

--------------------------------------------------
19. GIT / SECRETS
--------------------------------------------------

Do not commit automatically.

Run:

git status

Confirm:

.env is not tracked.

browser_sessions/ is ignored.

screenshots/ is ignored.

playwright-artifacts/ is ignored.

No credentials are staged.

--------------------------------------------------
20. FINAL REPORT
--------------------------------------------------

Report:

1. Playwright version
2. Chromium installation status
3. Files created
4. Files modified
5. Browser profile path
6. Indeed session implementation
7. Manual login test result
8. Persistent-session test result
9. Screenshot result
10. Unit tests
11. Total test count
12. Git status

Confirm:

- no Indeed passwords stored
- no credentials automated
- no CAPTCHA bypass
- no application submission
- no Apply button automation

STOP AFTER PHASE 5.1.

Do not proceed to Phase 5.2 automatically.
```

---

### Prompt #16 (Conversation Step 1425)

```text
PHASE 5.1 — INDEED PLAYWRIGHT BROWSER FOUNDATION

We are beginning Phase 5 of the AI Job Application Agent.

MAIN PROJECT:

D:\Job Agent\AI-job-application-agent

Current completed phases:

Phase 1:
- Project foundation
- Supabase
- PostgreSQL
- FastAPI
- Docker
- tests

Phase 2:
- JobSpy MCP
- job discovery
- normalization
- deduplication
- Supabase persistence

Phase 3:
- candidate profile
- deterministic matching
- job ranking

Phase 4:
- NVIDIA embeddings
- NVIDIA reasoning
- semantic matching
- combined scoring

Current system is stable and committed to GitHub.

IMPORTANT SAFETY / SCOPE RULE:

Indeed currently prohibits third-party bots or automated tools from applying for jobs.

Therefore Phase 5 MUST NOT implement automated final submission.

This phase is ONLY browser/session foundation.

DO NOT IMPLEMENT:
- final application submission
- unattended auto-apply
- CAPTCHA bypass
- anti-bot bypass
- rate-limit bypass
- credential scraping
- password storage
- application form auto-filling
- generated application answers
- application agent reasoning
- Naukri automation
- other platform automation

--------------------------------------------------
1. PLAYWRIGHT DEPENDENCY
--------------------------------------------------

Add Playwright to the project using the appropriate Python package.

Use a pinned/stable Playwright version compatible with the existing Python environment.

Do not unnecessarily upgrade unrelated project dependencies.

Install the required browser:

Chromium

Verify that Playwright can launch Chromium successfully.

--------------------------------------------------
2. AUTOMATION MODULE
--------------------------------------------------

Create:

app/automation/
    __init__.py
    playwright_manager.py
    browser_profiles.py
    session_manager.py

Create a clean abstraction around Playwright.

The rest of the application should not directly instantiate Playwright browsers everywhe
<truncated 6458 bytes>
ation filling yet.

Create:

app/platforms/indeed/
    __init__.py
    adapter.py
    session.py

The adapter may implement only:
- session status
- open job page

Do NOT implement Apply clicking yet.

--------------------------------------------------
17. APPLICATION STATE
--------------------------------------------------

Do not create a submitted application state.

For this phase only use states such as:

SESSION_UNKNOWN
LOGIN_REQUIRED
AUTHENTICATED
SESSION_ERROR

--------------------------------------------------
18. DOCKER PREPARATION
--------------------------------------------------

Do not require the production Docker environment to use the local Windows browser profile yet.

For Phase 5.1:

- local Windows development is primary
- visible Chromium is primary
- persistent local browser profile is primary

Keep the architecture portable so a future remote worker can be added.

--------------------------------------------------
19. GIT / SECRETS
--------------------------------------------------

Do not commit automatically.

Run:

git status

Confirm:

.env is not tracked.

browser_sessions/ is ignored.

screenshots/ is ignored.

playwright-artifacts/ is ignored.

No credentials are staged.

--------------------------------------------------
20. FINAL REPORT
--------------------------------------------------

Report:

1. Playwright version
2. Chromium installation status
3. Files created
4. Files modified
5. Browser profile path
6. Indeed session implementation
7. Manual login test result
8. Persistent-session test result
9. Screenshot result
10. Unit tests
11. Total test count
12. Git status

Confirm:

- no Indeed passwords stored
- no credentials automated
- no CAPTCHA bypass
- no application submission
- no Apply button automation

STOP AFTER PHASE 5.1.

Do not proceed to Phase 5.2 automatically.
```

---

### Prompt #17 (Conversation Step 1543)

```text
proceed
```

---

### Prompt #18 (Conversation Step 1552)

```text
wait i need the full cotext , all prompts that is given for the project in and md file so that i can learn another cghatgot model , i need the context till now
```

---

### Prompt #19 (Conversation Step 1569)

```text
what i need is the full project context including all the prompts i ve given antigravty as a md file so that i can teach another llm about the full project
```

---

### Prompt #20 (Conversation Step 1579)

```text
# AI JOB APPLICATION AGENT — MASTER PROJECT GOAL & PLATFORM-SAFE AUTOMATION POLICY

## PROJECT CONTEXT

We are building a personal-use project called:

**AI Job Application Agent**

Main project:

`D:\Job Agent\AI-job-application-agent`

This application is designed for a **single user running locally on their own computer**.

The user will manually authenticate to job platforms such as Indeed, LinkedIn, Naukri, Glassdoor, and external employer Applicant Tracking Systems.

The application must NEVER require or store the user's job-platform passwords.

Persistent browser sessions may be used only where doing so is compatible with the applicable platform rules.

---

# PRIMARY PROJECT GOAL

The long-term goal is to automate as much of the job-search and job-application workflow as is reasonably and contractually permitted by each individual website.

Desired overall pipeline:

```text
Job Discovery
    ↓
Normalization
    ↓
Candidate Matching
    ↓
AI Semantic Ranking
    ↓
Select Strong Jobs
    ↓
Open Application Workflow
    ↓
Inspect Application
    ↓
Prepare Application Information
    ↓
Fill Permitted Fields
    ↓
Prepare Answers
    ↓
Upload Resume Where Permitted
    ↓
Navigate Multi-Step Application Where Permitted
    ↓
STOP AT THE FURTHEST SAFE POINT
    ↓
USER REVIEW / USER ACTION WHEN REQUIRED
```

The system must NOT assume that every website permits the same level of browser automation.

---

# CORE AUTOMATION PRINCIPLE

The governing rule for this project is:

> AUTOMATE TO THE MAXIMUM LEVEL CURRENTLY PERMITTED BY THE TARGET PLATFORM, THEN HAND CONTROL TO THE USER.

Platform terms, policies, technical restrictions, and official APIs take priority over the desire for full automation.

The application must NEVER attempt to circumvent a website's restrictions in order to reach a greater level of automation.

---

# PLATFORM-SPECIFIC AUTOMATION LEVEL

Introduce a platform automation pol
<truncated 10688 bytes>
eed job navigation + application-flow inspection
```

All existing passing tests must remain green.

Do not rewrite completed architecture unnecessarily.

---

# LONG-TERM END STATE

The final application should behave like a policy-aware personal job-application assistant.

Conceptually:

```text
Discover Jobs
      ↓
Rank Jobs
      ↓
Select Good Matches
      ↓
Determine Platform
      ↓
Load Platform Policy
      ↓
Determine Maximum Allowed Automation
      ↓
Prepare Application
      ↓
Perform Allowed Actions
      ↓
Reach Platform-Specific Boundary
      ↓
User Review / User Action
      ↓
Application Completed
```

The target is NOT:

```text
blind auto-apply everywhere
```

The target IS:

```text
maximum useful automation
+
platform-specific compliance
+
human control at restricted or consequential steps
```

---

# DEVELOPMENT RULE

For every future phase:

1. State exactly what new capability is being added.
2. Identify the platform affected.
3. Identify the platform's current automation policy.
4. Define the maximum allowed automation boundary.
5. Implement only actions inside that boundary.
6. Add tests proving the boundary cannot accidentally be crossed.
7. Preserve existing tests.
8. Never weaken CAPTCHA, credential, anti-bot, or access-control protections.
9. Never silently enable submission on a platform.
10. Stop at the end of the requested phase.

The architecture should make it easy to increase automation later if a platform introduces an official supported mechanism without rewriting the complete system.

# FINAL PROJECT PRINCIPLE

**Automate everything that is permitted and useful.**

**When the platform requires a human action, stop cleanly and hand control to the user.**

**Never trade account safety, platform compliance, or candidate-data accuracy for additional automation.**
```

---

### Prompt #21 (Conversation Step 1615)

```text
# PHASE 5.2 — INDEED JOB NAVIGATION AND APPLICATION FLOW INSPECTION

Phase 5.1 is complete and committed architecture is stable.

MAIN PROJECT:

`D:\Job Agent\AI-job-application-agent`

Current capabilities:

* Playwright installed
* Chromium installed
* Persistent Indeed browser profile
* Manual login flow
* Session detection
* Indeed platform adapter
* Browser profile persistence
* 70 tests passing
* No credentials stored
* No auto-submit

## OBJECTIVE

Inspect the application workflow of ONE selected Indeed job using the existing authenticated persistent browser session.

This phase is ONLY for:

* opening a matched Indeed job
* inspecting the page
* detecting the application method
* identifying application controls
* collecting a structured representation of the page/form

DO NOT:

* click final Submit
* submit any application
* bypass CAPTCHA
* bypass anti-bot systems
* automate login
* enter passwords
* modify cookies
* auto-answer application questions
* use NVIDIA reasoning for application answers yet
* implement unattended auto-apply
* implement Naukri automation

---

## 1. SELECT TEST JOB

---

Add a development/test method that accepts a job URL.

Do NOT hard-code a specific production job URL into source code.

Allow input such as:

```http
POST /automation/indeed/open
```

```json
{
    "job_url": "https://www.indeed.com/..."
}
```

Use the existing job URL from Supabase if available.

Validate that the URL belongs to the expected Indeed domain before opening it.

---

## 2. AUTHENTICATED SESSION

---

Use the existing persistent Indeed browser profile:

`browser_sessions/indeed`

Launch visible Chromium.

Before opening the job:

* check session state
* if not authenticated, return `LOGIN_REQUIRED`
* do not automate login

---

## 3. JOB NAVIGATION

---

Open the job URL.

Collect:

* final URL
* page title
* current URL
* HTTP/navigation status where available
* page load timi
<truncated 2256 bytes>
hot.

---

## 9. TRACE

---

If `PLAYWRIGHT_TRACE=true`:

* capture a Playwright trace
* save under `playwright-artifacts/`

Keep tracing disabled by default.

---

## 10. API RESPONSE

---

Extend the existing response with:

```json
{
    "platform": "indeed",
    "authenticated": true,
    "job_url": "...",
    "job_title": "...",
    "company": "...",
    "location": "...",
    "application_method": "...",
    "apply_button_found": true,
    "form_detected": false,
    "fields": [],
    "status": "review_required"
}
```

Never return:

* cookies
* session tokens
* passwords
* browser storage
* authorization headers

---

## 11. TESTING

---

Create mocked tests for:

* valid Indeed URL
* invalid domain
* unauthenticated session
* authenticated session
* Indeed-hosted application detection
* external application detection
* missing apply button
* form field extraction
* application method classification

All previous tests must continue passing.

---

## 12. LIVE TEST

---

Use ONE real Indeed job.

Workflow:

1. Launch persistent browser.
2. If required, manually log into Indeed.
3. Verify authenticated session.
4. Open the supplied Indeed job URL.
5. Inspect the page.
6. Detect application method.
7. Capture safe screenshot.
8. Return structured inspection data.
9. STOP.

Do NOT:

* submit
* fill
* upload resume
* answer questions
* click final Submit

---

## 13. FINAL REPORT

---

Report:

1. Files created/modified
2. Job URL used, if safe to report
3. Authentication status
4. Job title
5. Company
6. Application method
7. Whether Apply was detected
8. Whether a form was detected
9. Field count
10. External application detected or not
11. Screenshot created
12. Trace status
13. Unit test result
14. Git status

STOP after Phase 5.2.

Do not proceed to Phase 5.3 automatically.
```

---

### Prompt #22 (Conversation Step 1703)

```text
PHASE 5.2 LIVE-TEST REPAIR — INDEED INSPECTOR SCOPING + INDENTATION FIX

MAIN PROJECT:

D:\Job Agent\AI-job-application-agent

IMPORTANT:

Phase 5.2 implementation already exists.

Do NOT redesign Phase 5.2.
Do NOT create a new phase.
Do NOT modify the platform-policy architecture.
Do NOT implement form filling.
Do NOT implement Apply clicking.
Do NOT implement application submission.
Do NOT bypass CAPTCHA / Cloudflare / anti-bot systems.
Do NOT proceed to Phase 6.

This task is ONLY to repair the existing Indeed inspector based on issues discovered during the real browser smoke test.

CURRENT TEST BASELINE BEFORE THE MANUAL EDIT:

90 / 90 tests passing.

CURRENT PROBLEM:

A manual edit to:

app/platforms/indeed/inspector.py

introduced:

TabError: inconsistent use of tabs and spaces in indentation

around line 380.

Because inspector.py cannot import, pytest currently reports collection errors across multiple test modules.

The file must first be restored to valid Python indentation using 4 spaces only.

--------------------------------------------------
1. INSPECT CURRENT FILE
--------------------------------------------------

Open and inspect:

app/platforms/indeed/inspector.py

Do not blindly overwrite the entire file.

Preserve:

- existing models
- existing policy checks
- URL validation
- authentication gating
- persistent Playwright usage
- screenshot support
- trace support
- response models
- adapter contracts
- existing public methods

Make minimal targeted changes only.

--------------------------------------------------
2. FIX ALL INDENTATION
--------------------------------------------------

Normalize the entire file to:

4 spaces per indentation level

No tabs.

Ensure there is no mixture of tabs and spaces anywhere in:

app/platforms/indeed/inspector.py

The immediate syntax error is around the security-challenge detection section near:

content_lower = content.lower()

Fix indentation according 
<truncated 6535 bytes>
this exact order:

python -m tabnanny app/platforms/indeed/inspector.py

Expected:
no output / no indentation errors.

Then:

python -m py_compile app/platforms/indeed/inspector.py

Expected:
no output / successful compilation.

Then run the focused test module:

python -m pytest tests/test_indeed_inspector.py -v

Then run the complete suite:

python -m pytest tests -v

The original baseline was:

90 passed

If new regression tests are added, total tests may be greater than 90.

Required:

100% passing.

Do not hide, skip, delete, xfail, or weaken tests merely to obtain a green result.

--------------------------------------------------
11. SAVE CHANGES
--------------------------------------------------

Save all modified source/test files normally in the existing repository.

Do NOT commit automatically.

Do NOT push to GitHub automatically.

Do NOT change .env.

Do NOT modify:

browser_sessions/
screenshots/
playwright-artifacts/

Keep runtime/session artifacts git-ignored.

--------------------------------------------------
12. FINAL REPORT
--------------------------------------------------

After completing the repair, report:

1. Exact files modified
2. Whether tabs were removed
3. tabnanny result
4. py_compile result
5. Job-title extraction fix
6. Search-form exclusion fix
7. Security-challenge detection fix
8. Response-reason fix
9. Regression tests added, if any
10. Focused test result
11. Full test result
12. Total passing test count
13. Git status
14. Confirmation that no Apply click was added
15. Confirmation that no form filling was added
16. Confirmation that no submission was added
17. Confirmation that no CAPTCHA / anti-bot bypass was added

STOP HERE.

Do NOT perform a live Indeed test automatically.

Do NOT proceed to Phase 6.

Wait for the user to perform the next controlled live smoke test manually.
```

---

### Prompt #23 (Conversation Step 1732)

```text
PHASE 5.3 — PROFILE-DRIVEN BROAD JOB DISCOVERY + MATCHED URL PIPELINE

MAIN PROJECT:

D:\Job Agent\AI-job-application-agent

JOBSPY MCP:

D:\Job Agent\jobspy-mcp-server


OBJECTIVE

Implement a profile-driven broad discovery layer so the user does NOT need
to manually enter one job search term such as "AI ML".

The system should:

candidate_profile.json
        ↓
generate a broad but bounded discovery plan
        ↓
perform multiple JobSpy searches
        ↓
collect jobs from Indeed
        ↓
normalize
        ↓
deduplicate across all queries
        ↓
persist jobs in Supabase
        ↓
run the existing matching engine
        ↓
run existing NVIDIA semantic matching
        ↓
rank jobs
        ↓
return the highest-quality jobs WITH their original URLs

IMPORTANT:

This phase is ONLY about DISCOVERY + MATCHING + URL OUTPUT.

Do NOT implement:
- Playwright
- pywinauto
- browser automation
- Apply button clicking
- form filling
- login automation
- CAPTCHA bypass
- application submission
- Phase 6 application-question intelligence

The final output of this phase is a ranked set of job URLs.

--------------------------------------------------
1. INSPECT EXISTING CODE FIRST
--------------------------------------------------

Before changing anything, inspect the existing repository.

Reuse existing components where possible:

app/discovery/jobspy_client.py
app/discovery/service.py
app/discovery/normalizer.py

app/profile/
app/matching/
app/llm/

app/database/repositories/job_repository.py
app/database/repositories/match_repository.py

app/api/routers/jobs.py
app/api/routers/matches.py

data/profile/candidate_profile.json

Do NOT duplicate existing services.

Do NOT create a second JobSpy client.

Do NOT create a second matching engine.

Do NOT create another NVIDIA client.

Use the existing architecture.

--------------------------------------------------
2. IMPORTANT DESIGN RULE
----------
<truncated 12940 bytes>
manual /jobs/search still works
19. API endpoint test
20. no Playwright dependency introduced

Do not weaken, delete or skip existing tests.

Run:

python -m pytest tests -v

All tests must pass.

--------------------------------------------------
24. LIVE TEST
--------------------------------------------------

After unit tests pass, perform ONE controlled live test using:

source:
indeed

candidate profile:
existing active candidate profile

Use conservative values:

results_per_query = 10
max_queries = 3
hours_old = 168

This is only to verify the end-to-end pipeline.

Verify:

profile
→ queries
→ JobSpy
→ Indeed jobs
→ normalization
→ deduplication
→ persistence
→ matching
→ ranked URLs

Do NOT open any returned URLs automatically.

Do NOT launch Playwright.

Do NOT launch pywinauto.

Do NOT interact with Indeed browser pages.

--------------------------------------------------
25. FINAL REPORT
--------------------------------------------------

Report:

1. files created
2. files modified
3. search planner architecture
4. deterministic query-generation behavior
5. optional NVIDIA query expansion behavior
6. maximum query protection
7. location strategy
8. JobSpy execution strategy
9. cross-query deduplication behavior
10. matching integration
11. URL preservation
12. new API request contract
13. new API response contract
14. live test query count
15. live test raw jobs count
16. live test unique jobs count
17. live test matched jobs count
18. top 10 match titles/scores/source WITHOUT exposing sensitive data
19. confirmation that valid job URLs were returned
20. unit-test result
21. final total test count
22. git status
23. confirmation no browser automation was added

Do not commit automatically.

Do not push automatically.

STOP after completing this phase.

Do NOT start Phase 6 automatically.
```

---

## 8. Current Test Suite Status (70 Tests Passing)

As of **Phase 5.1**, all 70 unit and integration tests across 7 test suites pass in ~5.8s:
- `tests/test_automation.py` (13 tests): Playwright config, path resolution, gitignore security, SessionStatus models, BrowserProfileManager, Indeed DOM auth detection, session status inspection, open home workflow, platform adapter contracts, `/automation/health`, `/automation/indeed/session`.
- `tests/test_discovery.py` (9 tests): URL normalization, Job normalizer mapping, missing fields handling, multi-source resilience (LinkedIn, Indeed, Naukri, Glassdoor), SHA-256 deduplication, `/jobs/search`, `/jobs/stats`.
- `tests/test_health.py` (4 tests): Health check endpoint `/health`, root endpoint `/`, secret masking in settings, Supabase client error handling.
- `tests/test_llm.py` (16 tests): NVIDIA client settings, auth error handling, cosine similarity math, privacy-safe candidate embedding formatting, job embedding text formatting, semantic cache hit/miss, NVIDIA embedding provider, reasoning prompt JSON cleaning and parsing, `/llm/health`.
- `tests/test_matching.py` (14 tests): Skill matcher, Title matcher, Experience matcher, Location matcher, Salary matcher, Education matcher, Job type matcher, weighted total calculation, Explanation generator, DeterministicMatchEngine consistency, `/matches/run`, `/matches/{job_id}`.
- `tests/test_profile.py` (7 tests): Skill alias normalization, string cleaning, ProfileLoader JSON parsing, ProfileRepository Supabase mocks, ProfileService sync pipeline, `/profile`, `/profile/sync`, `/profile/skills`.
- `tests/test_semantic_matching.py` (7 tests): Combined scoring formula math ($70/20/10$), weight redistribution fallbacks, NVIDIASemanticMatcher integration, embedding failure resilience, threshold gating ($< 60.0$ skips reasoning), qualifying execution, `/matches/run` with semantic flag.

---

## 9. Strict Safety, Compliance & Anti-Bot Boundaries

1. **Zero Automated Final Submission**: Indeed prohibits automated tools from applying to jobs. The system strictly stops prior to submission.
2. **Zero Credential Scraping or Storage**: Usernames, passwords, and 2FA credentials are never stored in plaintext or database columns.
3. **No CAPTCHA / Anti-Bot Bypass**: All security challenges must be resolved manually by the user in the visible browser window.
4. **Persistent Profile Security**: Browser profile directories (`browser_sessions/`) are treated like credentials and excluded via `.gitignore`.

---

## 10. Roadmap for Phase 5.2 and Future Phases

- **Phase 5.2 (Next)**: Indeed Job Navigation, Page Inspection & Form Detection (without auto-fill / submit).
  - Open live job posting URLs in persistent browser.
  - Distinguish between **Indeed Apply** (modal / in-platform application) vs **External Employer URL** (redirect to external ATS).
  - Read-only inspection of application fields and screening questions.
  - Initialize application draft records in Supabase `applications` table (`status: 'draft'`).
- **Phase 6**: Application Intelligence & Answer Synthesis (LLM-assisted question answering for candidate review).
- **Phase 7**: End-to-End Orchestrator & Audit Telemetry (`automation_logs`).
