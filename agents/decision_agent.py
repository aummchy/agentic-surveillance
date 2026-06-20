import time
from pipeline.models import Track, MatchResult, DecisionResult
from config import settings


def decide(track: Track, match_result: MatchResult) -> DecisionResult:
    if match_result.matched:
        if "blacklist" in match_result.tags:
            return DecisionResult(
                status="blacklist",
                alert_level="critical",
                person_id=match_result.person_id,
                name=match_result.name,
                reason=f"Blacklisted person detected: {match_result.name}",
                should_alert=True,
                should_register=False
            )

        if "authorized" in match_result.tags:
            return DecisionResult(
                status="authorized",
                alert_level="none",
                person_id=match_result.person_id,
                name=match_result.name,
                reason=f"Authorized person: {match_result.name}",
                should_alert=False,
                should_register=False
            )

        return DecisionResult(
            status="known_visitor",
            alert_level="low",
            person_id=match_result.person_id,
            name=match_result.name,
            reason=f"Known visitor: {match_result.name}",
            should_alert=False,
            should_register=False
        )

    visibility = track.visibility

    if visibility == "hidden":
        return DecisionResult(
            status="intentionally_hidden",
            alert_level="high",
            reason="Person avoided face detection during track",
            should_alert=True,
            should_register=False
        )

    if visibility == "partial" or track.is_masked:
        track_lifetime = time.time() - track.first_seen
        alert_level = "high" if track_lifetime > settings.LOITER_SECS else "medium"

        return DecisionResult(
            status="masked_unknown",
            alert_level=alert_level,
            reason="Unknown person with partial visibility or mask",
            should_alert=True,
            should_register=True
        )

    return DecisionResult(
        status="unknown",
        alert_level="medium",
        reason="Unknown person detected",
        should_alert=True,
        should_register=True
    )
