"""
Phase 2.3 — Policy Agent

Centralizes ALL business rules in one place.
Instead of scattered if/else in decision_agent.py, this agent
considers multiple signals to make a final decision.

Input:
{
    "recognition_result": {"status": "known", "confidence": 76, ...},
    "memory_context": {"visit_count": 5, "is_known": true, ...},
    "match_result": {"matched": true, "tags": ["authorized"], ...},
    "track": {"visibility": "visible", "is_masked": false, ...},
    "camera_id": "cam_01",
    "current_hour": 14
}

Output:
{
    "status": "authorized",
    "alert_level": "none",
    "should_alert": false,
    "should_register": false,
    "reason": "Authorized person: John Doe"
}
"""

import structlog
from datetime import datetime
from typing import Any, Dict, Optional
from agents.base import BaseAgent
from pipeline.models import Track, MatchResult, DecisionResult
from config import settings

logger = structlog.get_logger(__name__)


class PolicyAgent(BaseAgent):
    """Centralizes all business rules for surveillance decisions.

    Rules are organized by priority:
    1. Blacklist (highest priority)
    2. Authorized
    3. Verified
    4. Known visitor
    5. Hidden/person avoiding detection
    6. Masked unknown
    7. Unknown (default)

    This agent also considers:
    - Time of day (office hours vs after hours)
    - Camera location (restricted areas)
    - Visit history (from Memory Agent)
    - Confidence level (from Recognition Agent)
    """

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute policy decision.

        Args:
            input_data: Must contain:
                - recognition_result (dict): From Recognition Agent
                - memory_context (dict): From Memory Agent
                - match_result (dict): From Matching Agent
                - track (Track): Current track object
                - camera_id (str): Current camera

        Returns:
            Dict with decision result.
        """
        recognition = input_data.get("recognition_result", {})
        memory = input_data.get("memory_context", {})
        match_data = input_data.get("match_result", {})
        track = input_data.get("track")
        camera_id = input_data.get("camera_id", settings.CAMERA_ID)

        # Get current time context
        now = datetime.utcnow()
        current_hour = now.hour
        is_office_hours = settings.OFFICE_HOURS_START <= current_hour <= settings.OFFICE_HOURS_END
        is_weekday = now.weekday() in settings.OFFICE_DAYS

        result = self._decide(
            recognition=recognition,
            memory=memory,
            match_data=match_data,
            track=track,
            camera_id=camera_id,
            current_hour=current_hour,
            is_office_hours=is_office_hours,
            is_weekday=is_weekday,
        )

        logger.info("policy_decision",
                    status=result.status,
                    alert_level=result.alert_level,
                    should_alert=result.should_alert,
                    reason=result.reason)

        return {
            "status": result.status,
            "alert_level": result.alert_level,
            "person_id": result.person_id,
            "name": result.name,
            "reason": result.reason,
            "should_alert": result.should_alert,
            "should_register": result.should_register,
        }

    def _decide(self, recognition: Dict, memory: Dict, match_data: Dict,
                track: Optional[Track], camera_id: str,
                current_hour: int, is_office_hours: bool, is_weekday: bool) -> DecisionResult:
        """Core decision logic with all rules centralized."""

        # Extract recognition info
        rec_status = recognition.get("status", "unknown")
        confidence = recognition.get("confidence", 0)
        is_masked = recognition.get("is_masked", False)

        # Extract match info
        matched = match_data.get("matched", False)
        person_id = match_data.get("person_id")
        name = match_data.get("name")
        tags = match_data.get("tags", [])
        verified = match_data.get("verified", False)
        alert_level_from_match = match_data.get("alert_level", "low")
        # Use match_data similarity_score as the authoritative source (not recognition.similarity
        # which can be 0 when recognition_result is None or stale during finalization)
        similarity = match_data.get("similarity_score", 0)

        # Extract memory info
        visit_count = memory.get("visit_count", 0)
        is_known_from_memory = memory.get("is_known", False)

        # Extract track info
        visibility = track.visibility if track else "unknown"
        track_lifetime = (datetime.now().timestamp() - track.first_seen) if track else 0

        # ═══════════════════════════════════════════════════════
        # RULE 1: Blacklist (highest priority)
        # ═══════════════════════════════════════════════════════
        if "blacklist" in tags:
            return DecisionResult(
                status="blacklist",
                alert_level="critical",
                person_id=person_id,
                name=name,
                reason=f"Blacklisted person detected: {name}",
                should_alert=True,
                should_register=False
            )

        # ═══════════════════════════════════════════════════════
        # RULE 2: Authorized
        # ═══════════════════════════════════════════════════════
        if "authorized" in tags:
            return DecisionResult(
                status="authorized",
                alert_level="none",
                person_id=person_id,
                name=name,
                reason=f"Authorized person: {name}",
                should_alert=False,
                should_register=False
            )

        # ═══════════════════════════════════════════════════════
        # RULE 3: Verified
        # ═══════════════════════════════════════════════════════
        if verified:
            return DecisionResult(
                status="verified",
                alert_level="none",
                person_id=person_id,
                name=name,
                reason=f"Verified visitor: {name}",
                should_alert=False,
                should_register=False
            )

        # ═══════════════════════════════════════════════════════
        # RULE 4: Known visitor (matched + memory confirms)
        # ═══════════════════════════════════════════════════════
        if matched and is_known_from_memory:
            # Higher confidence from memory
            return DecisionResult(
                status="known_visitor",
                alert_level="low",
                person_id=person_id,
                name=name,
                reason=f"Known visitor: {name}. {visit_count} previous visits.",
                should_alert=False,
                should_register=False
            )

        # ═══════════════════════════════════════════════════════
        # RULE 5: Matched but not verified (new or uncertain)
        # ═══════════════════════════════════════════════════════
        if matched:
            # Check similarity or confidence — high similarity alone is sufficient
            if similarity >= 0.85 or confidence >= 80:
                return DecisionResult(
                    status="known_visitor",
                    alert_level="low",
                    person_id=person_id,
                    name=name,
                    reason=f"Known visitor: {name}. High confidence match (similarity={similarity:.2%}).",
                    should_alert=False,
                    should_register=False
                )
            elif similarity >= settings.MATCH_THRESHOLD:
                return DecisionResult(
                    status="known_visitor",
                    alert_level="low",
                    person_id=person_id,
                    name=name,
                    reason=f"Known visitor: {name}. Match above threshold (similarity={similarity:.2%}).",
                    should_alert=False,
                    should_register=False
                )
            else:
                return DecisionResult(
                    status="uncertain",
                    alert_level="low",
                    person_id=person_id,
                    name=name,
                    reason=f"Uncertain match: {name}. Confidence {confidence:.0f}%.",
                    should_alert=False,
                    should_register=True
                )

        # ═══════════════════════════════════════════════════════
        # RULE 6: Intentionally hidden (avoiding detection)
        # ═══════════════════════════════════════════════════════
        if visibility == "hidden":
            return DecisionResult(
                status="intentionally_hidden",
                alert_level="high",
                reason="Person avoided face detection during track",
                should_alert=True,
                should_register=False
            )

        # ═══════════════════════════════════════════════════════
        # RULE 7: Masked unknown
        # ═══════════════════════════════════════════════════════
        if is_masked or visibility == "partial":
            # Escalate if loitering
            if track_lifetime > settings.LOITER_SECS:
                alert_level = "high"
                reason = "Masked unknown person loitering"
            else:
                alert_level = "medium"
                reason = "Unknown person with partial visibility or mask"

            return DecisionResult(
                status="masked_unknown",
                alert_level=alert_level,
                reason=reason,
                should_alert=True,
                should_register=True
            )

        # ═══════════════════════════════════════════════════════
        # RULE 8: After-hours unknown (higher alert)
        # ═══════════════════════════════════════════════════════
        if not is_office_hours or not is_weekday:
            return DecisionResult(
                status="unknown",
                alert_level="high",
                reason="Unknown person detected after hours",
                should_alert=True,
                should_register=True
            )

        # ═══════════════════════════════════════════════════════
        # RULE 9: Unknown during office hours (default)
        # ═══════════════════════════════════════════════════════
        return DecisionResult(
            status="unknown",
            alert_level="medium",
            reason="Unknown person detected",
            should_alert=True,
            should_register=True
        )
