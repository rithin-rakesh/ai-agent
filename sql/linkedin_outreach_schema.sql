-- ==============================================================================
-- LinkedIn Outreach Agent - Database Schema (Phase 6.2)
-- Supabase PostgreSQL Schema for LinkedIn Leads and Outreach Drafts
-- ==============================================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ==============================================================================
-- 1. LINKEDIN_LEADS TABLE
-- ==============================================================================
CREATE TABLE IF NOT EXISTS linkedin_leads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_url TEXT NOT NULL UNIQUE,
    post_text TEXT NOT NULL,
    author_name TEXT,
    author_profile_url TEXT,
    author_headline TEXT,
    company TEXT,
    company_url TEXT,
    published_at TIMESTAMPTZ,
    matched_keywords JSONB DEFAULT '[]'::jsonb,
    relevance_score NUMERIC(5, 2) DEFAULT 0.0,
    relevance_details JSONB DEFAULT '{}'::jsonb,
    lead_type TEXT NOT NULL DEFAULT 'JOB_OPPORTUNITY',
    contact_email TEXT,
    contact_email_source TEXT,
    contact_email_verified BOOLEAN DEFAULT FALSE,
    status TEXT NOT NULL DEFAULT 'NEW',
    last_contacted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for searching and deduplicating LinkedIn leads
CREATE INDEX IF NOT EXISTS idx_linkedin_leads_post_url ON linkedin_leads(post_url);
CREATE INDEX IF NOT EXISTS idx_linkedin_leads_status ON linkedin_leads(status);
CREATE INDEX IF NOT EXISTS idx_linkedin_leads_relevance_score ON linkedin_leads(relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_linkedin_leads_contact_email ON linkedin_leads(contact_email);
CREATE INDEX IF NOT EXISTS idx_linkedin_leads_company ON linkedin_leads(company);
CREATE INDEX IF NOT EXISTS idx_linkedin_leads_created_at ON linkedin_leads(created_at DESC);

-- ==============================================================================
-- 2. LINKEDIN_OUTREACH_DRAFTS TABLE
-- ==============================================================================
CREATE TABLE IF NOT EXISTS linkedin_outreach_drafts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id UUID NOT NULL REFERENCES linkedin_leads(id) ON DELETE CASCADE,
    recipient_email TEXT NOT NULL,
    recipient_name TEXT NOT NULL DEFAULT 'Hiring Team',
    subject TEXT NOT NULL,
    body_text TEXT NOT NULL,
    attachment_path TEXT NOT NULL,
    attachment_verified BOOLEAN DEFAULT FALSE,
    personalization_fields JSONB DEFAULT '{}'::jsonb,
    generation_metadata JSONB DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'DRAFT',
    sent_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for outreach drafts
CREATE INDEX IF NOT EXISTS idx_outreach_drafts_lead_id ON linkedin_outreach_drafts(lead_id);
CREATE INDEX IF NOT EXISTS idx_outreach_drafts_recipient_email ON linkedin_outreach_drafts(recipient_email);
CREATE INDEX IF NOT EXISTS idx_outreach_drafts_status ON linkedin_outreach_drafts(status);
CREATE INDEX IF NOT EXISTS idx_outreach_drafts_created_at ON linkedin_outreach_drafts(created_at DESC);

-- Apply automatic updated_at timestamp triggers
DROP TRIGGER IF EXISTS trg_linkedin_leads_updated_at ON linkedin_leads;
CREATE TRIGGER trg_linkedin_leads_updated_at
    BEFORE UPDATE ON linkedin_leads
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp_column();

DROP TRIGGER IF EXISTS trg_linkedin_outreach_drafts_updated_at ON linkedin_outreach_drafts;
CREATE TRIGGER trg_linkedin_outreach_drafts_updated_at
    BEFORE UPDATE ON linkedin_outreach_drafts
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp_column();
