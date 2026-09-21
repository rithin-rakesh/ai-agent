"""Unit tests for Platform Automation Policy, Registry, and Safe Governance."""

import pytest
from app.platforms.base import PlatformAdapter
from app.platforms.indeed.adapter import IndeedPlatformAdapter
from app.platforms.policy import (
    ActionState,
    AutomationAction,
    AutomationLevel,
    PlatformAutomationPolicy,
    PlatformPolicyRegistry,
    default_policy_registry,
    get_platform_policy,
)


def test_platform_automation_policy_evaluation_allowed():
    """Verify evaluation of allowed actions under inspection policy."""
    policy = PlatformAutomationPolicy(
        platform="test_platform",
        automation_level=AutomationLevel.INSPECTION_ONLY,
        navigation_allowed=True,
        inspection_allowed=True,
        allowed_until="application_inspection",
    )
    result = policy.evaluate_action(AutomationAction.NAVIGATE)
    assert result.allowed is True
    assert result.action_state == ActionState.CONTINUE_ALLOWED
    assert result.requires_user_action is False


def test_platform_automation_policy_submission_prohibited():
    """Verify that attempting to submit application triggers SUBMISSION_REVIEW_REQUIRED."""
    policy = PlatformAutomationPolicy(
        platform="indeed",
        automation_level=AutomationLevel.INSPECTION_ONLY,
        navigation_allowed=True,
        submission_allowed=False,
        allowed_until="application_inspection",
    )
    result = policy.evaluate_action(AutomationAction.SUBMIT_APPLICATION)
    assert result.allowed is False
    assert result.action_state == ActionState.SUBMISSION_REVIEW_REQUIRED
    assert result.requires_user_action is True
    assert "submission is not permitted" in result.reason.lower()


def test_platform_automation_policy_form_fill_prohibited():
    """Verify that attempting to fill forms when prohibited returns USER_ACTION_REQUIRED."""
    policy = PlatformAutomationPolicy(
        platform="indeed",
        automation_level=AutomationLevel.INSPECTION_ONLY,
        form_fill_allowed=False,
        allowed_until="application_inspection",
    )
    result = policy.evaluate_action(AutomationAction.FILL_FORM)
    assert result.allowed is False
    assert result.action_state == ActionState.USER_ACTION_REQUIRED
    assert result.requires_user_action is True


def test_platform_automation_policy_manual_only():
    """Verify MANUAL_ONLY classification blocks active automation with AUTOMATION_NOT_PERMITTED."""
    policy = PlatformAutomationPolicy(
        platform="linkedin",
        automation_level=AutomationLevel.MANUAL_ONLY,
        allowed_until="discovery_only",
    )
    result = policy.evaluate_action(AutomationAction.INSPECT_JOB)
    assert result.allowed is False
    assert result.action_state == ActionState.AUTOMATION_NOT_PERMITTED
    assert result.requires_user_action is True


def test_platform_automation_policy_unknown():
    """Verify UNKNOWN classification blocks automation with POLICY_UNKNOWN."""
    policy = PlatformAutomationPolicy(
        platform="custom_site",
        automation_level=AutomationLevel.UNKNOWN,
        allowed_until="discovery_only",
    )
    result = policy.evaluate_action(AutomationAction.NAVIGATE)
    assert result.allowed is False
    assert result.action_state == ActionState.POLICY_UNKNOWN
    assert result.requires_user_action is True


def test_policy_registry_default_platform_configurations():
    """Verify default configurations for indeed, linkedin, naukri, and glassdoor."""
    registry = PlatformPolicyRegistry()

    # Indeed: Inspection only
    indeed_policy = registry.get_policy("indeed")
    assert indeed_policy.automation_level == AutomationLevel.INSPECTION_ONLY
    assert indeed_policy.navigation_allowed is True
    assert indeed_policy.inspection_allowed is True
    assert indeed_policy.form_detection_allowed is True
    assert indeed_policy.form_fill_allowed is False
    assert indeed_policy.submission_allowed is False

    # LinkedIn: Manual only
    linkedin_policy = registry.get_policy("linkedin")
    assert linkedin_policy.automation_level == AutomationLevel.MANUAL_ONLY
    assert linkedin_policy.navigation_allowed is False
    assert linkedin_policy.submission_allowed is False

    # Naukri: Unknown
    naukri_policy = registry.get_policy("naukri")
    assert naukri_policy.automation_level == AutomationLevel.UNKNOWN

    # Glassdoor: Unknown
    glassdoor_policy = registry.get_policy("glassdoor")
    assert glassdoor_policy.automation_level == AutomationLevel.UNKNOWN


def test_policy_registry_case_insensitive_and_fallback():
    """Verify registry handles case variations and provides safe fallback for unknown platforms."""
    registry = PlatformPolicyRegistry()
    assert registry.get_policy("InDeEd").platform == "indeed"
    assert registry.get_policy("LINKEDIN").platform == "linkedin"

    fallback = registry.get_policy("unseen_portal")
    assert fallback.platform == "unseen_portal"
    assert fallback.automation_level == AutomationLevel.UNKNOWN
    assert fallback.navigation_allowed is False


def test_policy_registry_detect_ats_from_url():
    """Verify ATS domain detection from target application URLs."""
    registry = PlatformPolicyRegistry()
    assert registry.detect_ats_from_url("https://company.wd5.myworkdayjobs.com/careers/job/123") == "workday"
    assert registry.detect_ats_from_url("https://boards.greenhouse.io/techcorp/jobs/456") == "greenhouse"
    assert registry.detect_ats_from_url("https://jobs.lever.co/startup/789") == "lever"
    assert registry.detect_ats_from_url("https://jobs.smartrecruiters.com/Acme/001") == "smartrecruiters"
    assert registry.detect_ats_from_url("https://jobs.ashbyhq.com/scale/002") == "ashby"
    assert registry.detect_ats_from_url("https://careers-us.icims.com/jobs/003") == "icims"
    assert registry.detect_ats_from_url("https://customcompany.com/apply") == "external_ats"


def test_platform_adapter_policy_queries():
    """Verify PlatformAdapter base class helper queries reflect registered policy."""
    adapter = IndeedPlatformAdapter()
    assert adapter.can_navigate() is True
    assert adapter.can_inspect() is True
    assert adapter.can_detect_forms() is True
    assert adapter.can_fill_form() is False
    assert adapter.can_generate_answers() is False
    assert adapter.can_submit() is False


@pytest.mark.asyncio
async def test_indeed_adapter_blocks_navigation_if_policy_prohibits():
    """Verify adapter open_job returns structured user_action_required if policy restricts navigation."""
    from unittest.mock import AsyncMock

    mock_session_mgr = AsyncMock()
    adapter = IndeedPlatformAdapter(session_manager=mock_session_mgr)

    # Temporarily register restrictive policy for testing
    restrictive_policy = PlatformAutomationPolicy(
        platform="indeed",
        automation_level=AutomationLevel.MANUAL_ONLY,
        navigation_allowed=False,
        allowed_until="discovery_only",
    )
    default_policy_registry.register_policy(restrictive_policy)

    try:
        result = await adapter.open_job("https://www.indeed.com/viewjob?jk=123")
        assert result["allowed"] is False
        assert result["status"] == ActionState.AUTOMATION_NOT_PERMITTED.value
        assert result["requires_user_action"] is True
        mock_session_mgr.playwright_manager.launch_persistent_context.assert_not_called()
    finally:
        # Restore default indeed policy
        default_policy_registry._load_default_policies()
