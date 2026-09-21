"""Platform Automation Policy & Safe Execution Governance.

Defines the platform automation policy layer, automation levels, action states,
and permission enforcement to guarantee strict compliance with individual website
terms of service and platform-safe automation boundaries.
"""

from enum import Enum
import logging
from typing import Dict, Optional
from urllib.parse import urlparse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AutomationLevel(str, Enum):
    """Platform automation capability classification."""

    FULL_AUTOMATION_ALLOWED = "full_automation_allowed"
    ASSISTED_AUTOMATION_ALLOWED = "assisted_automation_allowed"
    INSPECTION_ONLY = "inspection_only"
    MANUAL_ONLY = "manual_only"
    UNKNOWN = "unknown"


class ActionState(str, Enum):
    """Execution and decision states when evaluating platform automation."""

    CONTINUE_ALLOWED = "continue_allowed"
    REVIEW_REQUIRED = "review_required"
    USER_ACTION_REQUIRED = "user_action_required"
    LOGIN_REQUIRED = "login_required"
    EXTERNAL_SITE_REVIEW_REQUIRED = "external_site_review_required"
    SUBMISSION_REVIEW_REQUIRED = "submission_review_required"
    AUTOMATION_NOT_PERMITTED = "automation_not_permitted"
    POLICY_UNKNOWN = "policy_unknown"
    BLOCKED = "blocked"


class AutomationAction(str, Enum):
    """Granular browser automation actions subject to policy verification."""

    NAVIGATE = "navigate"
    INSPECT_JOB = "inspect_job"
    DETECT_FORM = "detect_form"
    FILL_FORM = "fill_form"
    GENERATE_ANSWERS = "generate_answers"
    SUBMIT_APPLICATION = "submit_application"
    EXTERNAL_REDIRECT = "external_redirect"


class PolicyEvaluationResult(BaseModel):
    """Structured result returned when evaluating an automation action against platform policy."""

    allowed: bool = Field(..., description="Whether the action is permitted by platform policy")
    action_state: ActionState = Field(..., description="Action state code")
    platform: str = Field(..., description="Target platform identifier")
    action: str = Field(..., description="The action evaluated")
    reason: str = Field(..., description="Detailed explanation of the policy decision")
    allowed_until: str = Field(..., description="The furthest boundary currently allowed for this platform")
    requires_user_action: bool = Field(default=False, description="Whether immediate human user intervention is required")


class PlatformAutomationPolicy(BaseModel):
    """Configuration model specifying the precise automation boundary for a target platform."""

    platform: str = Field(..., description="Platform identifier (e.g. 'indeed', 'linkedin')")
    automation_level: AutomationLevel = Field(..., description="Overall automation classification")
    login_mode: str = Field(default="manual", description="Authentication mode ('manual', 'none')")
    navigation_allowed: bool = Field(default=False, description="Whether browser navigation to job URLs is allowed")
    inspection_allowed: bool = Field(default=False, description="Whether read-only DOM inspection is allowed")
    form_detection_allowed: bool = Field(default=False, description="Whether read-only application form detection is allowed")
    form_fill_allowed: bool = Field(default=False, description="Whether automated form field input is allowed")
    answer_generation_allowed: bool = Field(default=False, description="Whether AI answer generation is allowed")
    submission_allowed: bool = Field(default=False, description="Whether final application submission is allowed")
    external_redirect_allowed: bool = Field(default=False, description="Whether following redirects to external ATS is allowed")
    requires_user_review: bool = Field(default=True, description="Whether user confirmation is mandatory before submission")
    allowed_until: str = Field(default="discovery_only", description="Human-readable boundary delimiter")
    notes: Optional[str] = Field(default=None, description="Policy commentary or reference to Terms of Service")

    def is_action_allowed(self, action: AutomationAction) -> bool:
        """Check whether a specific action is permitted under this policy."""
        mapping = {
            AutomationAction.NAVIGATE: self.navigation_allowed,
            AutomationAction.INSPECT_JOB: self.inspection_allowed,
            AutomationAction.DETECT_FORM: self.form_detection_allowed,
            AutomationAction.FILL_FORM: self.form_fill_allowed,
            AutomationAction.GENERATE_ANSWERS: self.answer_generation_allowed,
            AutomationAction.SUBMIT_APPLICATION: self.submission_allowed,
            AutomationAction.EXTERNAL_REDIRECT: self.external_redirect_allowed,
        }
        return mapping.get(action, False)

    def evaluate_action(self, action: AutomationAction) -> PolicyEvaluationResult:
        """Evaluate an action and return a structured governance decision."""
        if self.automation_level == AutomationLevel.UNKNOWN:
            return PolicyEvaluationResult(
                allowed=False,
                action_state=ActionState.POLICY_UNKNOWN,
                platform=self.platform,
                action=action.value,
                reason=f"Automation policy for platform '{self.platform}' is unknown. Review required before automation.",
                allowed_until="discovery_only",
                requires_user_action=True,
            )

        if self.automation_level == AutomationLevel.MANUAL_ONLY and not self.is_action_allowed(action):
            return PolicyEvaluationResult(
                allowed=False,
                action_state=ActionState.AUTOMATION_NOT_PERMITTED,
                platform=self.platform,
                action=action.value,
                reason=f"Platform '{self.platform}' is classified as MANUAL_ONLY. Active browser automation is prohibited.",
                allowed_until=self.allowed_until,
                requires_user_action=True,
            )

        allowed = self.is_action_allowed(action)
        if allowed:
            return PolicyEvaluationResult(
                allowed=True,
                action_state=ActionState.CONTINUE_ALLOWED,
                platform=self.platform,
                action=action.value,
                reason=f"Action '{action.value}' is permitted under '{self.automation_level.value}' policy.",
                allowed_until=self.allowed_until,
                requires_user_action=False,
            )

        # Prohibited action handling
        if action == AutomationAction.SUBMIT_APPLICATION:
            action_state = ActionState.SUBMISSION_REVIEW_REQUIRED
            reason = f"Final application submission is not permitted on '{self.platform}'. Manual user review and submission required."
        elif action == AutomationAction.EXTERNAL_REDIRECT:
            action_state = ActionState.EXTERNAL_SITE_REVIEW_REQUIRED
            reason = f"External redirect detected from '{self.platform}'. Separate ATS policy verification required."
        else:
            action_state = ActionState.USER_ACTION_REQUIRED
            reason = f"Action '{action.value}' exceeds the allowed boundary ('{self.allowed_until}') for '{self.platform}'."

        return PolicyEvaluationResult(
            allowed=False,
            action_state=action_state,
            platform=self.platform,
            action=action.value,
            reason=reason,
            allowed_until=self.allowed_until,
            requires_user_action=True,
        )


class PlatformPolicyRegistry:
    """Registry maintaining active automation policies for all supported job boards and ATS platforms."""

    def __init__(self) -> None:
        self._policies: Dict[str, PlatformAutomationPolicy] = {}
        self._load_default_policies()

    def _load_default_policies(self) -> None:
        """Register default conservative platform automation policies."""
        # 1. Indeed: Inspection and read-only form discovery allowed; NO automated form-fill, NO automated submission
        self.register_policy(
            PlatformAutomationPolicy(
                platform="indeed",
                automation_level=AutomationLevel.INSPECTION_ONLY,
                login_mode="manual",
                navigation_allowed=True,
                inspection_allowed=True,
                form_detection_allowed=True,
                form_fill_allowed=False,
                answer_generation_allowed=False,
                submission_allowed=False,
                external_redirect_allowed=False,
                requires_user_review=True,
                allowed_until="application_inspection",
                notes="Indeed prohibits automated application tools. Automation stops at form/page inspection.",
            )
        )

        # 2. LinkedIn: Manual only; no unauthorized browser automation
        self.register_policy(
            PlatformAutomationPolicy(
                platform="linkedin",
                automation_level=AutomationLevel.MANUAL_ONLY,
                login_mode="manual",
                navigation_allowed=False,
                inspection_allowed=False,
                form_detection_allowed=False,
                form_fill_allowed=False,
                answer_generation_allowed=False,
                submission_allowed=False,
                external_redirect_allowed=False,
                requires_user_review=True,
                allowed_until="discovery_only",
                notes="LinkedIn prohibits unauthorized automated scraping and bot activity. Discovery and ranking only.",
            )
        )

        # 3. Naukri: Unknown / Manual only until terms explicitly verified
        self.register_policy(
            PlatformAutomationPolicy(
                platform="naukri",
                automation_level=AutomationLevel.UNKNOWN,
                login_mode="manual",
                navigation_allowed=False,
                inspection_allowed=False,
                form_detection_allowed=False,
                form_fill_allowed=False,
                answer_generation_allowed=False,
                submission_allowed=False,
                external_redirect_allowed=False,
                requires_user_review=True,
                allowed_until="discovery_only",
                notes="Naukri automation terms require verification. Default to manual.",
            )
        )

        # 4. Glassdoor: Unknown / Manual only until terms explicitly verified
        self.register_policy(
            PlatformAutomationPolicy(
                platform="glassdoor",
                automation_level=AutomationLevel.UNKNOWN,
                login_mode="manual",
                navigation_allowed=False,
                inspection_allowed=False,
                form_detection_allowed=False,
                form_fill_allowed=False,
                answer_generation_allowed=False,
                submission_allowed=False,
                external_redirect_allowed=False,
                requires_user_review=True,
                allowed_until="discovery_only",
                notes="Glassdoor automation terms require verification. Default to manual.",
            )
        )

    def register_policy(self, policy: PlatformAutomationPolicy) -> None:
        """Register or update a platform automation policy."""
        self._policies[policy.platform.lower()] = policy
        logger.debug("Registered platform policy: %s -> %s", policy.platform, policy.automation_level.value)

    def get_policy(self, platform: str) -> PlatformAutomationPolicy:
        """Fetch policy for a given platform. Returns conservative UNKNOWN policy if not registered."""
        norm_platform = platform.lower().strip()
        if norm_platform in self._policies:
            return self._policies[norm_platform]

        logger.warning("Unregistered platform '%s' requested. Returning default UNKNOWN policy.", platform)
        return PlatformAutomationPolicy(
            platform=norm_platform,
            automation_level=AutomationLevel.UNKNOWN,
            login_mode="manual",
            navigation_allowed=False,
            inspection_allowed=False,
            form_detection_allowed=False,
            form_fill_allowed=False,
            submission_allowed=False,
            requires_user_review=True,
            allowed_until="discovery_only",
            notes=f"Unregistered platform '{platform}'. Requires policy configuration.",
        )

    def detect_ats_from_url(self, url: str) -> str:
        """Detect known Applicant Tracking System (ATS) platform from target URL."""
        domain = urlparse(url).netloc.lower()
        if "myworkdayjobs.com" in domain or "workday.com" in domain:
            return "workday"
        if "greenhouse.io" in domain:
            return "greenhouse"
        if "lever.co" in domain:
            return "lever"
        if "smartrecruiters.com" in domain:
            return "smartrecruiters"
        if "ashbyhq.com" in domain:
            return "ashby"
        if "icims.com" in domain:
            return "icims"
        if "oraclecloud.com" in domain or "taleo.net" in domain:
            return "oracle_recruiting"
        if "successfactors.com" in domain:
            return "successfactors"
        if "indeed.com" in domain:
            return "indeed"
        if "linkedin.com" in domain:
            return "linkedin"
        if "naukri.com" in domain:
            return "naukri"
        if "glassdoor.com" in domain:
            return "glassdoor"
        return "external_ats"


# Global singleton registry
default_policy_registry = PlatformPolicyRegistry()


def get_platform_policy(platform: str) -> PlatformAutomationPolicy:
    """Convenience getter for platform policy using default registry."""
    return default_policy_registry.get_policy(platform)
