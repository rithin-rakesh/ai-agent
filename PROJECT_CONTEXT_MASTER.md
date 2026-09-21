# AI Job Application Agent — Master Project Context & Prompt Archive

> **Document Purpose**: This file is a comprehensive, standalone context document for the **AI Job Application Agent** project. It includes the complete system architecture, database design, directory structure, core algorithms, API contracts, safety rules, test results, and the exact chronological sequence of all user prompts/specifications from **Phase 1 through Phase 5.1**.
> 
> Use this document to initialize context for any new LLM session (ChatGPT, Claude, Gemini, etc.) so the model has complete, precise knowledge of the entire codebase and its history.

---

## Table of Contents
1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Current Codebase Directory Tree](#2-current-codebase-directory-tree)
3. [Database Architecture & Supabase Schema](#3-database-architecture--supabase-schema)
4. [Core Algorithms, Scoring Formulas & Math](#4-core-algorithms-scoring-formulas--math)
5. [Complete API Endpoints & Contracts](#5-complete-api-endpoints--contracts)
6. [Configuration & Environment Variables](#6-configuration--environment-variables)
7. [Comprehensive Chronological User Prompts Archive](#7-comprehensive-chronological-user-prompts-archive)
   - [Phase 1: Project Foundation & Initial Core](#phase-1-project-foundation--initial-core)
   - [Phase 1: Supabase Connection Verification](#phase-1-supabase-connection-verification)
   - [Phase 1: Database Schema Final Verification](#phase-1-database-schema-final-verification)
   - [Local Workspace Reorganization](#local-workspace-reorganization)
   - [Phase 2.1: JobSpy MCP Standalone Verification](#phase-21-jobspy-mcp-standalone-verification)
   - [Phase 2.2: JobSpy MCP Integration & Resilient Discovery](#phase-22-jobspy-mcp-integration--resilient-discovery)
   - [Phase 2.2: Cleanup & Checkpoint](#phase-22-cleanup--checkpoint)
   - [Phase 3: Candidate Profile & Deterministic Match Engine](#phase-3-candidate-profile--deterministic-match-engine)
   - [Phase 4: NVIDIA Semantic Matching & Combined Scoring](#phase-4-nvidia-semantic-matching--combined-scoring)
   - [Phase 5.1: Indeed Playwright Browser Foundation](#phase-51-indeed-playwright-browser-foundation)
8. [Current Verification & Test Status (70/70 Passing)](#8-current-verification--test-status-7070-passing)
9. [Strict Safety, Compliance & Anti-Bot Rules](#9-strict-safety-compliance--anti-bot-rules)
10. [Instructions for Downstream AI Models](#10-instructions-for-downstream-ai-models)

---

## 1. Project Overview & Architecture

### High-Level Purpose
The **AI Job Application Agent** is an enterprise-grade autonomous system built in Python / FastAPI that handles:
1. **Multi-Source Job Discovery**: Scraping jobs via JobSpy MCP server across LinkedIn, Indeed, Naukri, and Glassdoor with failure isolation and automated deduplication.
2. **Candidate Profile Intelligence**: Structured normalization of technical skills, experience hierarchy, education, and career preferences from local JSON files to Supabase.
3. **Deterministic Match Engine**: Multi-category weighted rule-based compatibility scoring (Skills 40%, Title 20%, Experience 15%, Location 10%, Education 5%, Salary 5%, Job Type 5%).
4. **NVIDIA AI Semantic Matching**: Integration with NVIDIA NIM models (`nvidia/nemotron-3-embed-1b` embeddings and `nvidia/nemotron-3.5-lightning-30b-a3b` reasoning) to evaluate domain relevance, qualitative fit, transferable skills, strengths, and concerns.
5. **Combined Scoring Formula**: Mathematical synthesis ($70\%$ Deterministic + $20\%$ Embedding Similarity + $10\%$ LLM Reasoning) producing a final 0–100 score and tier decision (`excellent`, `strong_match`, `review`, `low_match`, `reject`).
6. **Playwright Browser Automation (Phase 5)**: Persistent browser contexts preserving cookies and local storage for Indeed, supporting manual login workflows and non-sensitive screenshot generation.

### Key Architectural Guidelines
- **Modularity & Decoupling**: Business logic does not directly call external drivers. Layered repository, service, client, and engine patterns are strictly enforced.
- **Fail-Safe Resilience**: Individual platform/network failures (e.g. CAPTCHA on Naukri, LLM timeout) never crash the pipeline; fallback mechanisms preserve partial results.
- **Secret Protection**: API keys, database credentials, passwords, cookies, and tokens are protected via Pydantic `SecretStr` and never logged or serialized to client responses.
- **Strict Compliance**: No automated CAPTCHA bypassing, no password scraping, and no automated final submission on platforms where prohibited.

---

## 2. Current Codebase Directory Tree

```
D:\Job Agent\
├── AI-job-application-agent\         # Main Application Repository
│   ├── .env                           # Local environment configuration (git ignored)
│   ├── .env.example                   # Environment configuration template
│   ├── .gitignore                     # Git exclusions for secrets, sessions, logs, traces
│   ├── Dockerfile                     # Container image definition
│   ├── docker-compose.yml             # Container orchestration
│   ├── requirements.txt               # Pinned Python dependencies
│   ├── README.md                      # Comprehensive project documentation
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── api/                       # FastAPI Layer
│   │   │   ├── __init__.py
│   │   │   ├── main.py                # App entrypoint, lifespan, router mounting
│   │   │   └── routers/
│   │   │       ├── __init__.py
│   │   │       ├── automation.py      # /automation/health, /automation/indeed/*
│   │   │       ├── jobs.py            # /jobs/search, /jobs, /jobs/{id}, /jobs/stats
│   │   │       ├── llm.py             # /llm/health
│   │   │       ├── matches.py         # /matches/run, /matches/{job_id}, /matches/stats
│   │   │       └── profile.py         # /profile, /profile/sync, /profile/skills
│   │   │
│   │   ├── automation/                # Browser Automation (Phase 5)
│   │   │   ├── __init__.py
│   │   │   ├── browser_profiles.py    # Persistent context directory management
│   │   │   ├── playwright_manager.py  # Playwright async lifecycle, viewport, trace abstraction
│   │   │   └── session_manager.py     # SessionState enums & SessionStatus model
│   │   │
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   └── settings.py            # Pydantic BaseSettings with SecretStr & path resolvers
│   │   │
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── supabase.py            # Supabase client singleton & IPv4 patch
│   │   │   └── repositories/
│   │   │       ├── __init__.py
│   │   │       ├── job_repository.py      # CRUD for 'jobs' table
│   │   │       ├── match_repository.py    # CRUD & upsert for 'job_matches' table
│   │   │       └── profile_repository.py  # CRUD for 'profiles' & 'skills' tables
│   │   │
│   │   ├── discovery/                 # Job Discovery (Phase 2)
│   │   │   ├── __init__.py
│   │   │   ├── client.py              # HTTP client to JobSpy MCP Server
│   │   │   ├── deduplicator.py        # SHA256 signature deduplication
│   │   │   ├── normalizer.py          # Unified schema normalizer across 4 boards
│   │   │   └── service.py             # Fault-tolerant multi-source discovery pipeline
│   │   │
│   │   ├── llm/                       # NVIDIA AI Semantic Matching (Phase 4)
│   │   │   ├── __init__.py
│   │   │   ├── cache.py               # SHA256 in-memory semantic embedding/reasoning cache
│   │   │   ├── embeddings.py          # NVIDIA Nemotron embeddings (query vs passage)
│   │   │   ├── nvidia_client.py       # NVIDIA NIM API client with auth error handling
│   │   │   ├── reasoning.py           # NVIDIA Nemotron-3.5-lightning reasoning & JSON parsing
│   │   │   ├── similarity.py          # Cosine similarity & 0-100 normalization
│   │   │   └── prompts/
│   │   │       ├── semantic_match_system.txt  # System prompt demanding strict JSON schema
│   │   │       └── semantic_match_user.txt    # User prompt injecting candidate, job & scores
│   │   │
│   │   ├── matching/                  # Deterministic & Combined Scoring (Phases 3 & 4)
│   │   │   ├── __init__.py
│   │   │   ├── education_matcher.py   # Degree level hierarchy matching (5%)
│   │   │   ├── experience_matcher.py  # Regex bounds parsing & experience years (15%)
│   │   │   ├── explanation.py         # Human-readable match reasons & caveats generator
│   │   │   ├── job_type_matcher.py    # Employment type alignment (5%)
│   │   │   ├── location_matcher.py    # Proximity & remote preference matching (10%)
│   │   │   ├── matcher.py             # DeterministicMatchEngine orchestration
│   │   │   ├── salary_matcher.py      # Compensation range overlap (5%)
│   │   │   ├── scorer.py              # ScoringConfig & calculate_combined_score (70/20/10)
│   │   │   ├── semantic_matcher.py    # NVIDIASemanticMatcher implementation
│   │   │   ├── service.py             # MatchService pipeline with threshold gating
│   │   │   ├── skill_matcher.py       # Weighted technical skill matcher & aliases (40%)
│   │   │   └── title_matcher.py       # Role equivalence & Jaccard token overlap (20%)
│   │   │
│   │   ├── models/                    # Pydantic v2 Domain Models
│   │   │   ├── __init__.py
│   │   │   ├── application.py         # Application, ApplicationAnswer, AutomationLog
│   │   │   ├── job.py                 # JobBase, JobCreate, Job, JobSearchRequest
│   │   │   ├── llm.py                 # LLMHealthResponse, ReasoningOutput, SemanticMatchEvaluation
│   │   │   ├── match.py               # MatchBreakdown, MatchResult, JobMatch, JobMatchCreate
│   │   │   └── profile.py             # CandidateProfileData, Profile, Skill
│   │   │
│   │   ├── platforms/                 # Platform Adapters (Phase 5)
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # PlatformAdapter abstract base class
│   │   │   └── indeed/
│   │   │       ├── __init__.py
│   │   │       ├── adapter.py         # IndeedPlatformAdapter implementation
│   │   │       └── session.py         # IndeedSessionManager (persistent context & login)
│   │   │
│   │   └── profile/                   # Candidate Profile Normalizer & Service (Phase 3)
│   │       ├── __init__.py
│   │       ├── loader.py              # ProfileLoader reading local JSON
│   │       ├── normalizer.py          # Skill aliases & casing normalizer
│   │       └── service.py             # ProfileService synchronizing local to Supabase
│   │
│   ├── data/
│   │   ├── profile/
│   │   │   └── candidate_profile.json # Local candidate profile configuration
│   │   └── resumes/                   # Local resumes (PDF/DOCX)
│   │
│   ├── sql/
│   │   └── schema.sql                 # Complete PostgreSQL DDL for Supabase
│   │
│   └── tests/                         # Full Pytest Test Suite (70 tests)
│       ├── __init__.py
│       ├── test_automation.py         # Browser settings, paths, Indeed session mocks (13 tests)
│       ├── test_discovery.py          # Normalizer, deduplication, fault tolerance (9 tests)
│       ├── test_health.py             # /health, settings, client validation (4 tests)
│       ├── test_llm.py                # NVIDIA client, embeddings, reasoning, cache (16 tests)
│       ├── test_matching.py           # 7 deterministic sub-matchers & pipeline (14 tests)
│       ├── test_profile.py            # Profile loading, skill normalization, sync (7 tests)
│       └── test_semantic_matching.py  # Combined scoring, threshold gating, API (7 tests)
│
└── jobspy-mcp-server/                 # Standalone JobSpy FastMCP Discovery Service (Port 9423)
```

---

## 3. Database Architecture & Supabase Schema

The database is built on PostgreSQL inside Supabase, managing 7 interconnected tables with UUID primary keys, foreign key cascading, and indexes:

### Entity Breakdown:
1. `profiles`:
   - `id` (UUID, PK), `name` (TEXT), `email` (TEXT, UNIQUE), `phone` (TEXT), `location` (TEXT), `experience_years` (NUMERIC), `preferred_roles` (TEXT[]), `preferred_locations` (TEXT[]), `remote_preference` (TEXT), `salary_min` (NUMERIC), `salary_max` (NUMERIC), `job_type` (TEXT), `education_degree` (TEXT), `education_field` (TEXT), `created_at`, `updated_at`.
2. `skills`:
   - `id` (UUID, PK), `profile_id` (UUID, FK -> profiles.id ON DELETE CASCADE), `name` (TEXT), `category` (TEXT), `years_experience` (NUMERIC), `importance` (TEXT: critical/high/medium/low), `created_at`.
3. `jobs`:
   - `id` (UUID, PK), `source` (TEXT: linkedin/indeed/naukri/glassdoor), `external_id` (TEXT), `title` (TEXT), `company` (TEXT), `location` (TEXT), `description` (TEXT), `url` (TEXT), `salary_min` (NUMERIC), `salary_max` (NUMERIC), `experience_text` (TEXT), `job_type` (TEXT), `remote` (BOOLEAN), `posted_at` (TIMESTAMPTZ), `easy_apply` (BOOLEAN), `raw_data` (JSONB), `created_at`.
   - Unique Constraint: `(source, external_id)`.
4. `job_matches`:
   - `id` (UUID, PK), `job_id` (UUID, FK -> jobs.id ON DELETE CASCADE), `profile_id` (UUID, FK -> profiles.id ON DELETE CASCADE), `match_score` (NUMERIC, 0-100), `skill_score` (NUMERIC), `title_score` (NUMERIC), `experience_score` (NUMERIC), `location_score` (NUMERIC), `salary_score` (NUMERIC), `llm_score` (NUMERIC, NULL for un-reasoned jobs), `reason` (TEXT), `decision` (TEXT: excellent/strong_match/review/low_match/reject), `created_at`, `updated_at`.
   - Unique Constraint: `(job_id, profile_id)`.
5. `applications`:
   - `id` (UUID, PK), `job_id` (UUID, FK -> jobs.id ON DELETE CASCADE), `profile_id` (UUID, FK -> profiles.id ON DELETE CASCADE), `status` (TEXT: draft/pending/submitted/rejected/interview), `submission_type` (TEXT), `submitted_at`, `error_log` (TEXT), `created_at`, `updated_at`.
6. `application_answers`:
   - `id` (UUID, PK), `application_id` (UUID, FK -> applications.id ON DELETE CASCADE), `question_text` (TEXT), `question_type` (TEXT), `answer_text` (TEXT), `confidence_score` (NUMERIC), `source` (TEXT), `created_at`.
7. `automation_logs`:
   - `id` (UUID, PK), `application_id` (UUID, FK -> applications.id ON DELETE SET NULL), `platform` (TEXT), `action` (TEXT), `status` (TEXT), `screenshot_url` (TEXT), `details` (JSONB), `created_at`.

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

## 5. Complete API Endpoints & Contracts

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

## 6. Configuration & Environment Variables

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

## 7. Comprehensive Chronological User Prompts Archive

Below is the complete, unaltered archive of all major prompts that directed the construction of this codebase from day 1 to the present.


### Phase 1: Project Foundation & Initial Core

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

### Phase 1: Supabase Connection Verification

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

### Phase 1: Database Schema Final Verification

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

### Local Workspace Reorganization

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

### Phase 2.1: JobSpy MCP Standalone Verification

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

### Phase 2.2: JobSpy MCP Integration & Resilient Discovery

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

### Phase 2.2: Cleanup & Checkpoint

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

### Phase 3: Candidate Profile & Deterministic Match Engine

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

### Phase 4: NVIDIA Semantic Matching & Combined Scoring

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

### Phase 5.1: Indeed Playwright Browser Foundation

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

## 8. Current Verification & Test Status (70/70 Passing)

As of **Phase 5.1**, the automated test suite has **70 passing tests** across 7 test modules:

```text
tests/test_automation.py::test_browser_settings_defaults PASSED
tests/test_automation.py::test_path_resolution_relative_to_root PASSED
tests/test_automation.py::test_gitignore_contains_browser_security_exclusions PASSED
tests/test_automation.py::test_session_status_model PASSED
tests/test_automation.py::test_browser_profile_manager PASSED
tests/test_automation.py::test_detect_auth_state_authenticated PASSED
tests/test_automation.py::test_detect_auth_state_unauthenticated PASSED
tests/test_automation.py::test_indeed_session_manager_check_session_status PASSED
tests/test_automation.py::test_indeed_session_manager_open_home_and_screenshot PASSED
tests/test_automation.py::test_indeed_platform_adapter_interface PASSED
tests/test_automation.py::test_api_automation_health PASSED
tests/test_automation.py::test_api_indeed_session_status_endpoint PASSED
tests/test_automation.py::test_api_root_endpoint_updated_to_phase5 PASSED
tests/test_discovery.py (9 tests passed)
tests/test_health.py (4 tests passed)
tests/test_llm.py (16 tests passed)
tests/test_matching.py (14 tests passed)
tests/test_profile.py (7 tests passed)
tests/test_semantic_matching.py (7 tests passed)

====================== 70 passed in 5.81s =======================
```

---

## 9. Strict Safety, Compliance & Anti-Bot Rules

1. **Zero Automated Final Submission**: Indeed prohibits automated tools from applying to jobs. The system strictly stops prior to submission.
2. **Zero Credential Scraping or Storage**: Usernames, passwords, and 2FA credentials are never stored in plaintext or database columns.
3. **No CAPTCHA / Anti-Bot Bypass**: All security challenges must be resolved manually by the user in the visible browser window.
4. **Persistent Profile Security**: Browser profile directories (`browser_sessions/`) are treated like credentials and excluded via `.gitignore`.

---

## 10. Instructions for Downstream AI Models

When continuing development of this project:
1. **Never break existing architecture or tests**: Any new feature must maintain 100% pass rate across the 70 existing tests.
2. **Adhere to modular directory boundaries**:
   - Platform specific code -> `app/platforms/<platform_name>/`
   - Playwright lifecycle -> `app/automation/`
   - LLM calls -> `app/llm/`
   - Matching logic -> `app/matching/`
   - Database queries -> `app/database/repositories/`
3. **Respect Safety Rules**: Never add automatic form submissions, Apply button clicks, or CAPTCHA bypasses unless explicitly authorized by platform policies and user directives.
