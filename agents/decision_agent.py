"""
Decision Agent — Now delegates to Policy Agent.

This module maintains backward compatibility while using the new
Policy Agent for centralized rule management.

Phase 2.3: Rules are now in agents/policy.py
"""

import structlog
from pipeline.models import Track, MatchResult, DecisionResult
from agents.policy import PolicyAgent

logger = structlog.get_logger(__name__)

# Singleton policy agent
_policy_agent = None


def _get_policy_agent() -> PolicyAgent:
    global _policy_agent
    if _policy_agent is None:
        _policy_agent = PolicyAgent()
    return _policy_agent


def decide(track: Track, match_result: MatchResult,
           recognition_result: dict = None, memory_context: dict = None) -> DecisionResult:
    """Make a decision using the centralized Policy Agent.

    Args:
        track: Current track object
        match_result: Result from Matching Agent
        recognition_result: Optional result from Recognition Agent
        memory_context: Optional context from Memory Agent

    Returns:
        DecisionResult with status, alert_level, and actions.
    """
    policy = _get_policy_agent()

    result = policy.run({
        "recognition_result": recognition_result or {},
        "memory_context": memory_context or {},
        "match_result": {
            "matched": match_result.matched,
            "person_id": match_result.person_id,
            "name": match_result.name,
            "role": match_result.role,
            "tags": match_result.tags,
            "similarity_score": match_result.similarity_score,
            "verified": match_result.verified,
            "alert_level": match_result.alert_level,
        },
        "track": track,
    })

    return DecisionResult(
        status=result["status"],
        alert_level=result["alert_level"],
        person_id=result["person_id"],
        name=result["name"],
        reason=result["reason"],
        should_alert=result["should_alert"],
        should_register=result["should_register"],
    )
