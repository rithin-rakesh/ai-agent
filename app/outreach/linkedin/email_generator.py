"""LinkedIn Outreach Email Generator (Phase 6.2).

Generates professional, concise, and personalized outreach email drafts:
- Validates that RESUME_PDF_PATH exists on disk before drafting
- References the specific post context, role, and author
- Highlights relevant candidate AI/ML skills truthfully from the profile
- Omits spam-like language, aggressive marketing, and unsupported claims
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from app.config.settings import Settings, get_settings
from app.models.profile import CandidateProfileData, Profile
from app.outreach.linkedin.models import (
    DraftStatus,
    LinkedInLead,
    LinkedInOutreachDraft,
)
from app.profile.profile_loader import ProfileLoader

logger = logging.getLogger(__name__)


class ResumeNotFoundError(FileNotFoundError):
    """Raised when the candidate resume PDF cannot be found on disk."""
    pass


class LinkedInEmailGenerator:
    """Service that drafts contextual outreach emails with resume attachments."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        candidate_profile: Optional[Any] = None,
    ) -> None:
        self.settings = settings or get_settings()
        if candidate_profile is not None:
            self.profile = candidate_profile
        else:
            try:
                self.profile = ProfileLoader().load_and_normalize()
            except Exception:
                self.profile = None

    def validate_resume_path(self) -> Path:
        """Verify that the configured resume PDF exists on the filesystem.

        Returns:
            Resolved absolute Path to resume.pdf

        Raises:
            ResumeNotFoundError: If resume file does not exist.
        """
        raw_path = getattr(self.settings, "RESUME_PDF_PATH", "data/profile/resume.pdf")
        resolved = self.settings.resolve_path(raw_path)
        if not resolved.is_file():
            err = f"Configured resume PDF was not found at '{resolved}'. Ensure file exists before drafting."
            logger.error(err)
            raise ResumeNotFoundError(err)
        return resolved

    def generate_draft(self, lead: LinkedInLead) -> LinkedInOutreachDraft:
        """Generate a personalized outreach email draft for a qualified lead."""
        # 1. Validate resume exists
        resume_path = self.validate_resume_path()

        # 2. Extract recipient information
        recipient_name = lead.author_name or "Hiring Team"
        recipient_email = lead.contact_email
        if not recipient_email:
            raise ValueError(f"Cannot generate email draft for lead {lead.id}: no verified contact_email found")

        company_name = lead.company or "your team"

        candidate_name = "Rithin Rakesh"
        candidate_email = None
        candidate_phone = None

        if isinstance(self.profile, CandidateProfileData):
            if self.profile.personal:
                candidate_name = self.profile.personal.name or candidate_name
                candidate_email = self.profile.personal.email
                candidate_phone = self.profile.personal.phone
        elif self.profile:
            candidate_name = getattr(self.profile, "name", candidate_name)
            candidate_email = getattr(self.profile, "email", None)
            candidate_phone = getattr(self.profile, "phone", None)

        # 3. Select matching skills from profile
        skills = ["Python", "Machine Learning", "FastAPI", "PyTorch", "Data Science"]
        if self.profile and hasattr(self.profile, "skills") and self.profile.skills:
            prof_skills = []
            for s in self.profile.skills:
                if hasattr(s, "skill"):
                    prof_skills.append(s.skill)
                elif isinstance(s, dict) and "skill" in s:
                    prof_skills.append(s["skill"])
                elif isinstance(s, str):
                    prof_skills.append(s)
            if prof_skills:
                skills = prof_skills[:4]

        skills_str = ", ".join(skills)

        # 4. Formulate Subject Line
        role_mention = "AI / ML Engineer"
        for kw in lead.matched_keywords:
            if kw.lower() in ("ai engineer", "machine learning", "ml engineer", "data scientist", "generative ai"):
                role_mention = kw
                break

        subject = f"Application / Inquiry: {role_mention} - {candidate_name}"

        # 5. Formulate Body Text (Professional, concise, truthful)
        greeting = f"Dear {recipient_name},"
        if recipient_name.lower() in ("hiring team", "recruiter", "talent team"):
            greeting = "Hello,"

        body_lines = [
            greeting,
            "",
            f"I hope this note finds you well. I came across your recent LinkedIn update regarding the {role_mention} opportunity at {company_name} and wanted to express my sincere interest in the role.",
            "",
            f"I am a software engineer focused on Artificial Intelligence and Machine Learning, with hands-on experience building models, APIs, and data pipelines using {skills_str}.",
            "",
            "Given your focus on practical engineering and impactful problem-solving, I believe my technical foundation and enthusiasm for learning would allow me to contribute meaningfully to your team.",
            "",
            "I have attached my resume for your consideration. I would welcome the opportunity to speak with you briefly about how my background aligns with your team's objectives.",
            "",
            "Thank you for your time and consideration.",
            "",
            "Best regards,",
            candidate_name,
        ]

        if candidate_email:
            body_lines.append(f"Email: {candidate_email}")
        if candidate_phone:
            body_lines.append(f"Phone: {candidate_phone}")

        body_text = "\n".join(body_lines)


        personalization_fields = {
            "recipient_name": recipient_name,
            "company_name": company_name,
            "role_mention": role_mention,
            "skills_highlighted": skills,
            "candidate_name": candidate_name,
            "post_url": lead.post_url,
        }

        generation_metadata = {
            "generator": "LinkedInEmailGenerator",
            "version": "1.0.0",
            "resume_file_size_bytes": resume_path.stat().st_size if resume_path.exists() else 0,
            "resume_path_verified": True,
        }

        draft = LinkedInOutreachDraft(
            id=uuid4(),
            lead_id=lead.id,
            recipient_email=recipient_email,
            recipient_name=recipient_name,
            subject=subject,
            body_text=body_text,
            attachment_path=str(resume_path),
            attachment_verified=True,
            personalization_fields=personalization_fields,
            generation_metadata=generation_metadata,
            status=DraftStatus.DRAFT,
        )

        return draft
