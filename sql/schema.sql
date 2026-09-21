-- ==============================================================================
-- AI Job Application Agent - Database Schema (Supabase PostgreSQL)
-- Phase 1 Initial Schema
-- ==============================================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ==============================================================================
-- 1. PROFILES TABLE
-- ==============================================================================
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT,
    location TEXT,
    experience_years NUMERIC(4, 1),
    preferred_locations JSONB DEFAULT '[]'::jsonb,
    preferred_roles JSONB DEFAULT '[]'::jsonb,
    salary_min NUMERIC(12, 2),
    salary_max NUMERIC(12, 2),
    resume_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for searching profiles
CREATE INDEX IF NOT EXISTS idx_profiles_email ON profiles(email);

-- ==============================================================================
-- 2. SKILLS TABLE
-- ==============================================================================
CREATE TABLE IF NOT EXISTS skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    skill TEXT NOT NULL,
    importance TEXT DEFAULT 'medium',
    years_experience NUMERIC(4, 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for skills by profile
CREATE INDEX IF NOT EXISTS idx_skills_profile_id ON skills(profile_id);
CREATE INDEX IF NOT EXISTS idx_skills_skill ON skills(skill);

-- ==============================================================================
-- 3. JOBS TABLE
-- ==============================================================================
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
    -- Uniqueness Strategy: Avoid duplicate jobs for the same source and external_id
    CONSTRAINT uq_jobs_source_external_id UNIQUE (source, external_id)
);

-- Indexes for job search and filtering
CREATE INDEX IF NOT EXISTS idx_jobs_source_external_id ON jobs(source, external_id);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company);
CREATE INDEX IF NOT EXISTS idx_jobs_posted_at ON jobs(posted_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_remote ON jobs(remote);

-- ==============================================================================
-- 4. JOB_MATCHES TABLE
-- ==============================================================================
CREATE TABLE IF NOT EXISTS job_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    match_score NUMERIC(5, 2),
    skill_score NUMERIC(5, 2),
    title_score NUMERIC(5, 2),
    experience_score NUMERIC(5, 2),
    location_score NUMERIC(5, 2),
    salary_score NUMERIC(5, 2),
    llm_score NUMERIC(5, 2),
    reason TEXT,
    decision TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Prevent duplicate match calculations for same job and profile
    CONSTRAINT uq_job_matches_job_profile UNIQUE (job_id, profile_id)
);

-- Indexes for job matches
CREATE INDEX IF NOT EXISTS idx_job_matches_job_id ON job_matches(job_id);
CREATE INDEX IF NOT EXISTS idx_job_matches_profile_id ON job_matches(profile_id);
CREATE INDEX IF NOT EXISTS idx_job_matches_score ON job_matches(match_score DESC);

-- ==============================================================================
-- 5. APPLICATIONS TABLE
-- ==============================================================================
CREATE TABLE IF NOT EXISTS applications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    platform TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    submitted_at TIMESTAMPTZ,
    failure_reason TEXT,
    application_url TEXT,
    confirmation_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for applications
CREATE INDEX IF NOT EXISTS idx_applications_job_id ON applications(job_id);
CREATE INDEX IF NOT EXISTS idx_applications_profile_id ON applications(profile_id);
CREATE INDEX IF NOT EXISTS idx_applications_status ON applications(status);
CREATE INDEX IF NOT EXISTS idx_applications_platform ON applications(platform);

-- ==============================================================================
-- 6. APPLICATION_ANSWERS TABLE
-- ==============================================================================
CREATE TABLE IF NOT EXISTS application_answers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id UUID NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    source TEXT,
    confidence NUMERIC(4, 2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for answers by application
CREATE INDEX IF NOT EXISTS idx_application_answers_application_id ON application_answers(application_id);

-- ==============================================================================
-- 7. AUTOMATION_LOGS TABLE
-- ==============================================================================
CREATE TABLE IF NOT EXISTS automation_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id UUID REFERENCES applications(id) ON DELETE SET NULL,
    platform TEXT,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    screenshot_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for automation logs
CREATE INDEX IF NOT EXISTS idx_automation_logs_application_id ON automation_logs(application_id);
CREATE INDEX IF NOT EXISTS idx_automation_logs_status ON automation_logs(status);
CREATE INDEX IF NOT EXISTS idx_automation_logs_created_at ON automation_logs(created_at DESC);

-- ==============================================================================
-- 8. AGENT_RUNS TABLE (Phase 5.5 Multi-Agent Orchestration)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS agent_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform TEXT NOT NULL,
    profile_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'CREATED',
    current_job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    configuration_json JSONB DEFAULT '{}'::jsonb,
    summary_json JSONB DEFAULT '{}'::jsonb,
    pause_reason TEXT,
    manual_action_required BOOLEAN DEFAULT FALSE,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for agent runs
CREATE INDEX IF NOT EXISTS idx_agent_runs_platform ON agent_runs(platform);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs(status);
CREATE INDEX IF NOT EXISTS idx_agent_runs_profile_id ON agent_runs(profile_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_created_at ON agent_runs(created_at DESC);

-- ==============================================================================
-- TRIGGER FUNCTION: update updated_at timestamp automatically
-- ==============================================================================
CREATE OR REPLACE FUNCTION update_timestamp_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply timestamp triggers
DROP TRIGGER IF EXISTS trg_profiles_updated_at ON profiles;
CREATE TRIGGER trg_profiles_updated_at
    BEFORE UPDATE ON profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp_column();

DROP TRIGGER IF EXISTS trg_applications_updated_at ON applications;
CREATE TRIGGER trg_applications_updated_at
    BEFORE UPDATE ON applications
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp_column();

DROP TRIGGER IF EXISTS trg_agent_runs_updated_at ON agent_runs;
CREATE TRIGGER trg_agent_runs_updated_at
    BEFORE UPDATE ON agent_runs
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp_column();
