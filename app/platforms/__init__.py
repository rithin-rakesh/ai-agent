"""Platform-specific browser automation adapters & policy governance."""

from app.platforms.base import PlatformAdapter
from app.platforms.policy import (
    ActionState,
    AutomationAction,
    AutomationLevel,
    PlatformAutomationPolicy,
    PlatformPolicyRegistry,
    PolicyEvaluationResult,
    default_policy_registry,
    get_platform_policy,
)

__all__ = [
    "PlatformAdapter",
    "AutomationLevel",
    "ActionState",
    "AutomationAction",
    "PlatformAutomationPolicy",
    "PolicyEvaluationResult",
    "PlatformPolicyRegistry",
    "default_policy_registry",
    "get_platform_policy",
]
