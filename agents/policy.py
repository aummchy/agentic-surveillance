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
from typing import Any, Dict, Optional
from agents.base import BaseAgent
from pipeline.models import Track, MatchResult, DecisionResult
from config import settings
from config.status import Status, STATUS_LABELS, LABEL_TO_STATUS, AlertLevel, Visibility

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
        if track is None:
            logger.warning("decide_called_without_track")
            return DecisionResult(status=Status.UNKNOWN, alert_level=AlertLevel.LOW)

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
        """Core decision logic — evaluates rules in priority order.

        Each rule method returns DecisionResult if it matches, or None to
        let the next rule evaluate.  First match wins.
        """
        ctx = self._build_decision_context(
            recognition, memory, match_data, track, current_hour, is_office_hours, is_weekday)

        rules = [
            self._rule_blacklist,
            self._rule_authorized,
            self._rule_verified,
            self._rule_auto_registered,
            self._rule_known_visitor_memory,
            self._rule_matched,
            self._rule_hidden,
            self._rule_masked,
            self._rule_after_hours,
        ]

        for rule in rules:
            result = rule(ctx)
            if result is not None:
                return result

        return self._rule_unknown_default(ctx)

    # ── Context extraction ──────────────────────────────────────

    def _build_decision_context(self, recognition: Dict[str, Any], memory: Dict[str, Any],
                                match_data: Dict[str, Any], track: Track,
                                current_hour: int, is_office_hours: bool, is_weekday: bool) -> dict:
        """Extract all decision signals into a single dict for rule methods."""
        rec_status = recognition.get("status", Status.UNKNOWN)
        if isinstance(rec_status, str):
            rec_status = LABEL_TO_STATUS.get(rec_status, Status.UNKNOWN)

        return {
            "rec_status": rec_status,
            "confidence": recognition.get("confidence", 0),
            "is_masked": recognition.get("is_masked", False),
            "matched": match_data.get("matched", False),
            "person_id": match_data.get("person_id"),
            "name": match_data.get("name"),
            "tags": match_data.get("tags", []),
            "verified": match_data.get("verified", False),
            "similarity": match_data.get("similarity_score", 0),
            "visit_count": memory.get("visit_count", 0),
            "is_known_from_memory": memory.get("is_known", False),
            "visibility": track.visibility,
            "track_lifetime": datetime.now().timestamp() - track.first_seen,
            "display_name": match_data.get("name") or "Unknown",
            "is_office_hours": is_office_hours,
            "is_weekday": is_weekday,
        }

    # ── Rules (priority order) ─────────────────────────────────

    def _rule_blacklist(self, ctx: dict) -> Optional[DecisionResult]:
        """RULE 1: Blacklisted person (highest priority, critical alert)."""
        if "blacklist" not in ctx["tags"]:
            return None
        return DecisionResult(
            status=Status.BLACKLIST,
            alert_level=AlertLevel.CRITICAL,
            person_id=ctx["person_id"],
            name=ctx["name"],
            reason=f"Blacklisted person detected: {ctx['display_name']}",
            should_alert=True,
            should_register=False,
        )

    def _rule_authorized(self, ctx: dict) -> Optional[DecisionResult]:
        """RULE 2: Authorized person (no alert)."""
        if "authorized" not in ctx["tags"]:
            return None
        return DecisionResult(
            status=Status.AUTHORIZED,
            alert_level=AlertLevel.NONE,
            person_id=ctx["person_id"],
            name=ctx["name"],
            reason=f"Authorized person: {ctx['display_name']}",
            should_alert=False,
            should_register=False,
        )

    def _rule_verified(self, ctx: dict) -> Optional[DecisionResult]:
        """RULE 3: Verified visitor (no alert)."""
        if not ctx["verified"]:
            return None
        if ctx["rec_status"] < Status.KNOWN and ctx["similarity"] <= settings.VERIFIED_SIMILARITY_THRESHOLD:
            return None
        return DecisionResult(
            status=Status.VERIFIED,
            alert_level=AlertLevel.NONE,
            person_id=ctx["person_id"],
            name=ctx["name"],
            reason=f"Verified visitor: {ctx['display_name']}",
            should_alert=False,
            should_register=False,
        )

    def _rule_auto_registered(self, ctx: dict) -> Optional[DecisionResult]:
        """RULE 4a: Auto-registered self-match (similarity > threshold)."""
        if "auto_registered" not in ctx["tags"]:
            return None
        if ctx["similarity"] <= settings.AUTO_REGISTERED_SIMILARITY:
            return None
        return DecisionResult(
            status=Status.KNOWN_VISITOR,
            alert_level=AlertLevel.LOW,
            person_id=ctx["person_id"],
            name=ctx["name"],
            reason=f"Auto-registered visitor: {ctx['display_name']} (similarity={ctx['similarity']:.2%}).",
            should_alert=False,
            should_register=False,
        )

    def _rule_known_visitor_memory(self, ctx: dict) -> Optional[DecisionResult]:
        """RULE 4b: Known visitor (matched + memory confirms)."""
        if not ctx["matched"] or not ctx["is_known_from_memory"]:
            return None
        return DecisionResult(
            status=Status.KNOWN_VISITOR,
            alert_level=AlertLevel.LOW,
            person_id=ctx["person_id"],
            name=ctx["name"],
            reason=f"Known visitor: {ctx['display_name']}. {ctx['visit_count']} previous visits.",
            should_alert=False,
            should_register=False,
        )

    def _rule_matched(self, ctx: dict) -> Optional[DecisionResult]:
        """RULE 5: Matched but not memory-confirmed (new or uncertain)."""
        if not ctx["matched"]:
            return None

        display_name = ctx["display_name"]
        person_id = ctx["person_id"]
        name = ctx["name"]
        similarity = ctx["similarity"]
        confidence = ctx["confidence"]

        # If recognition already classified as "known", respect that
        if ctx["rec_status"] >= Status.KNOWN:
            return DecisionResult(
                status=Status.KNOWN_VISITOR,
                alert_level=AlertLevel.LOW,
                person_id=person_id,
                name=name,
                reason=f"Known visitor: {display_name}. Recognition confidence {confidence}%",
                should_alert=False,
                should_register=False,
            )

        # High similarity or confidence is sufficient
        if similarity >= settings.KNOWN_VISITOR_SIMILARITY or confidence >= settings.KNOWN_VISITOR_CONFIDENCE:
            return DecisionResult(
                status=Status.KNOWN_VISITOR,
                alert_level=AlertLevel.LOW,
                person_id=person_id,
                name=name,
                reason=f"Known visitor: {display_name}. High confidence match (similarity={similarity:.2%}).",
                should_alert=False,
                should_register=False,
            )

        if similarity >= settings.MATCH_THRESHOLD:
            return DecisionResult(
                status=Status.UNKNOWN,
                alert_level=AlertLevel.LOW,
                person_id=person_id,
                name=name,
                reason=f"Matched identity but not confirmed known (similarity={similarity:.2%}).",
                should_alert=False,
                should_register=False,
            )

        # Unreachable: vector_search filters by MATCH_THRESHOLD
        logger.error("unreachable_policy_branch",
                     similarity=similarity,
                     threshold=settings.MATCH_THRESHOLD,
                     person_id=person_id)
        return DecisionResult(
            status=Status.UNKNOWN,
            alert_level=AlertLevel.LOW,
            person_id=person_id,
            name=name,
            reason="Internal policy invariant violated (similarity below MATCH_THRESHOLD).",
            should_alert=False,
            should_register=False,
        )

    def _rule_hidden(self, ctx: dict) -> Optional[DecisionResult]:
        """RULE 6: Intentionally hidden (avoiding detection)."""
        if ctx["visibility"] != Visibility.HIDDEN:
            return None
        return DecisionResult(
            status=Status.HIDDEN,
            alert_level=AlertLevel.HIGH,
            name="Unidentified Person",
            reason="Person avoided face detection during track",
            should_alert=True,
            should_register=False,
        )

    def _rule_masked(self, ctx: dict) -> Optional[DecisionResult]:
        """RULE 7: Masked or partial visibility unknown."""
        if not ctx["is_masked"] and ctx["visibility"] != Visibility.PARTIAL:
            return None
        if ctx["track_lifetime"] > settings.LOITER_SECS:
            return DecisionResult(
                status=Status.MASKED_UNKNOWN,
                alert_level=AlertLevel.HIGH,
                name="Masked Person",
                reason="Masked unknown person loitering",
                should_alert=True,
                should_register=True,
            )
        return DecisionResult(
            status=Status.MASKED_UNKNOWN,
            alert_level=AlertLevel.MEDIUM,
            name="Masked Person",
            reason="Unknown person with partial visibility or mask",
            should_alert=True,
            should_register=True,
        )

    def _rule_after_hours(self, ctx: dict) -> Optional[DecisionResult]:
        """RULE 8: After-hours unknown (higher alert)."""
        if ctx["is_office_hours"] and ctx["is_weekday"]:
            return None
        return DecisionResult(
            status=Status.UNKNOWN,
            alert_level=AlertLevel.HIGH,
            name="After-Hours Unknown",
            reason="Unknown person detected after hours",
            should_alert=True,
            should_register=True,
        )

    def _rule_unknown_default(self, ctx: dict) -> DecisionResult:
        """RULE 9: Unknown during office hours (default)."""
        return DecisionResult(
            status=Status.UNKNOWN,
            alert_level=AlertLevel.MEDIUM,
            name="Unidentified Person",
            reason="Unknown person detected",
            should_alert=True,
            should_register=True,
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
