"""
Phase 2.3 — Policy Agent

Centralizes ALL business rules in one place.
Considers multiple signals to make a final decision:

Input:
{
    "recognition_result": {"status": 3, "confidence": 76, ...},
    "memory_context": {"visit_count": 5, "is_known": true, ...},
    "match_result": {"matched": true, "tags": ["authorized"], ...},
    "track": {"visibility": "visible", "is_masked": false, ...},
    "camera_id": "cam_01",
    "current_hour": 14
}

Output: DecisionResult
"""

import structlog
from datetime import datetime
from typing import Any, Dict
from agents.base import BaseAgent
from pipeline.models import Track, MatchResult, DecisionResult
from config import settings
from config.status import Status, STATUS_LABELS, LABEL_TO_STATUS

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
        """Execute policy decision. Returns dict for BaseAgent compatibility."""
        result = self._decide_from_input(input_data)
        return {
            "status": result.status,
            "alert_level": result.alert_level,
            "person_id": result.person_id,
            "name": result.name,
            "reason": result.reason,
            "should_alert": result.should_alert,
            "should_register": result.should_register,
        }

    def decide(self, track: Track, match_result: MatchResult,
               recognition_result: dict = None,
               memory_context: dict = None) -> DecisionResult:
        """Make a decision using the centralized Policy Agent.

        Preferred entry point over run() — returns structured DecisionResult
        without dict round-trip.
        """
        return self._decide_from_input({
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

    def _decide_from_input(self, input_data: Dict[str, Any]) -> DecisionResult:
        """Extract fields from input dict and delegate to _decide()."""
        recognition = input_data.get("recognition_result", {})
        memory = input_data.get("memory_context", {})
        match_data = input_data.get("match_result", {})
        track = input_data.get("track")
        camera_id = input_data.get("camera_id", settings.CAMERA_ID)

        now = datetime.now()
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
                    track_id=track.track_id,
                    status=result.status,
                    alert_level=result.alert_level,
                    should_alert=result.should_alert,
                    visit_count=memory.get("visit_count", 0),
                    person_id=result.person_id,
                    name=result.name,
                    reason=result.reason)

        return result

    def _decide(self, recognition: Dict, memory: Dict, match_data: Dict,
                track: Track, camera_id: str,
                current_hour: int, is_office_hours: bool, is_weekday: bool) -> DecisionResult:
        """Core decision logic with all rules centralized."""

        # Extract recognition info
        rec_status = recognition.get("status", Status.UNKNOWN)
        if isinstance(rec_status, str):
            rec_status = LABEL_TO_STATUS.get(rec_status, Status.UNKNOWN)
        confidence = recognition.get("confidence", 0)
        is_masked = recognition.get("is_masked", False)

        # Extract match info
        matched = match_data.get("matched", False)
        person_id = match_data.get("person_id")
        name = match_data.get("name")
        tags = match_data.get("tags", [])
        verified = match_data.get("verified", False)
        # Use match_data similarity_score as the authoritative source (not recognition.similarity
        # which can be 0 when recognition_result is None or stale during finalization)
        similarity = match_data.get("similarity_score", 0)

        # Extract memory info
        visit_count = memory.get("visit_count", 0)
        is_known_from_memory = memory.get("is_known", False)

        # Extract track info
        visibility = track.visibility
        track_lifetime = datetime.now().timestamp() - track.first_seen

        display_name = name or "Unknown"

        # ═══════════════════════════════════════════════════════
        # RULE 1: Blacklist (highest priority)
        # ═══════════════════════════════════════════════════════
        if "blacklist" in tags:
            return DecisionResult(
                status=Status.BLACKLIST,
                alert_level="critical",
                person_id=person_id,
                name=name,
                reason=f"Blacklisted person detected: {display_name}",
                should_alert=True,
                should_register=False
            )

        # ═══════════════════════════════════════════════════════
        # RULE 2: Authorized
        # ═══════════════════════════════════════════════════════
        if "authorized" in tags:
            return DecisionResult(
                status=Status.AUTHORIZED,
                alert_level="none",
                person_id=person_id,
                name=name,
                reason=f"Authorized person: {display_name}",
                should_alert=False,
                should_register=False
            )

        # ═══════════════════════════════════════════════════════
        # RULE 3: Verified
        # ═══════════════════════════════════════════════════════
        if verified and (rec_status >= Status.KNOWN or similarity > settings.VERIFIED_SIMILARITY_THRESHOLD):
            return DecisionResult(
                status=Status.VERIFIED,
                alert_level="none",
                person_id=person_id,
                name=name,
                reason=f"Verified visitor: {display_name}",
                should_alert=False,
                should_register=False
            )

        # ═══════════════════════════════════════════════════════
        # RULE 4a: Auto-registered self-match (similarity > 0.65)
        # ═══════════════════════════════════════════════════════
        if "auto_registered" in tags and similarity > settings.AUTO_REGISTERED_SIMILARITY:
            return DecisionResult(
                status=Status.KNOWN_VISITOR,
                alert_level="low",
                person_id=person_id,
                name=name,
                reason=f"Auto-registered visitor: {display_name} (similarity={similarity:.2%}).",
                should_alert=False,
                should_register=False
            )

        # ═══════════════════════════════════════════════════════
        # RULE 4b: Known visitor (matched + memory confirms)
        # ═══════════════════════════════════════════════════════
        if matched and is_known_from_memory:
            return DecisionResult(
                status=Status.KNOWN_VISITOR,
                alert_level="low",
                person_id=person_id,
                name=name,
                reason=f"Known visitor: {display_name}. {visit_count} previous visits.",
                should_alert=False,
                should_register=False
            )

        # ═══════════════════════════════════════════════════════
        # RULE 5: Matched but not verified (new or uncertain)
        # ═══════════════════════════════════════════════════════
        if matched:
            # If recognition agent already classified as "known", respect that
            # — it considered multi-signal scoring (quality, track duration, memory).
            # This prevents RECOG=KNOWN but POLICY=UNKNOWN divergence.
            if rec_status >= Status.KNOWN:
                return DecisionResult(
                    status=Status.KNOWN_VISITOR,
                    alert_level="low",
                    person_id=person_id,
                    name=name,
                    reason=f"Known visitor: {display_name}. Recognition confidence {confidence}%",
                    should_alert=False,
                    should_register=False
                )
            # Check similarity or confidence — high similarity alone is sufficient
            if similarity >= settings.KNOWN_VISITOR_SIMILARITY or confidence >= settings.KNOWN_VISITOR_CONFIDENCE:
                return DecisionResult(
                    status=Status.KNOWN_VISITOR,
                    alert_level="low",
                    person_id=person_id,
                    name=name,
                    reason=f"Known visitor: {display_name}. High confidence match (similarity={similarity:.2%}).",
                    should_alert=False,
                    should_register=False
                )
            if similarity >= settings.MATCH_THRESHOLD:
                # Matched an existing identity but not memory-confirmed as known
                return DecisionResult(
                    status=Status.UNKNOWN,
                    alert_level="low",
                    person_id=person_id,
                    name=name,
                    reason=f"Matched identity but not confirmed known (similarity={similarity:.2%}).",
                    should_alert=False,
                    should_register=False   # already linked to existing record
                )
            # Unreachable: vector_search filters by MATCH_THRESHOLD,
            # so matched=True implies similarity >= MATCH_THRESHOLD.
            logger.error("unreachable_policy_branch",
                         similarity=similarity,
                         threshold=settings.MATCH_THRESHOLD,
                         person_id=person_id)
            return DecisionResult(
                status=Status.UNKNOWN,
                alert_level="low",
                person_id=person_id,
                name=name,
                reason="Internal policy invariant violated (similarity below MATCH_THRESHOLD).",
                should_alert=False,
                should_register=False,
            )

        # ═══════════════════════════════════════════════════════
        # RULE 6: Intentionally hidden (avoiding detection)
        # ═══════════════════════════════════════════════════════
        if visibility == "hidden":
            return DecisionResult(
                status=Status.HIDDEN,
                alert_level="high",
                name="Unidentified Person",
                reason="Person avoided face detection during track",
                should_alert=True,
                should_register=False
            )

        # ═══════════════════════════════════════════════════════
        # RULE 7: Masked unknown
        # ═══════════════════════════════════════════════════════
        if is_masked or visibility == "partial":
            if track_lifetime > settings.LOITER_SECS:
                alert_level = "high"
                reason = "Masked unknown person loitering"
            else:
                alert_level = "medium"
                reason = "Unknown person with partial visibility or mask"

            return DecisionResult(
                status=Status.MASKED_UNKNOWN,
                alert_level=alert_level,
                name="Masked Person",
                reason=reason,
                should_alert=True,
                should_register=True
            )

        # ═══════════════════════════════════════════════════════
        # RULE 8: After-hours unknown (higher alert)
        # ═══════════════════════════════════════════════════════
        if not is_office_hours or not is_weekday:
            return DecisionResult(
                status=Status.UNKNOWN,
                alert_level="high",
                name="After-Hours Unknown",
                reason="Unknown person detected after hours",
                should_alert=True,
                should_register=True
            )

        # ═══════════════════════════════════════════════════════
        # RULE 9: Unknown during office hours (default)
        # ═══════════════════════════════════════════════════════
        return DecisionResult(
            status=Status.UNKNOWN,
            alert_level="medium",
            name="Unidentified Person",
            reason="Unknown person detected",
            should_alert=True,
            should_register=True
        )


_policy_agent = None
_policy_lock = None


def _get_policy_agent() -> PolicyAgent:
    global _policy_agent, _policy_lock
    if _policy_lock is None:
        import threading
        _policy_lock = threading.Lock()
    with _policy_lock:
        if _policy_agent is None:
            _policy_agent = PolicyAgent()
        return _policy_agent


def decide(track: Track, match_result: MatchResult,
           recognition_result: dict = None, memory_context: dict = None) -> DecisionResult:
    """Make a decision using the centralized Policy Agent."""
    return _get_policy_agent().decide(
        track=track,
        match_result=match_result,
        recognition_result=recognition_result,
        memory_context=memory_context,
    )
