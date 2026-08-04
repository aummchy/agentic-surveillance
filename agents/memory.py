"""
Phase 2.2 — Memory Agent

Instead of only searching MongoDB Vector Search, this agent checks
visit history to provide context for better recognition decisions.

Input:
{
    "person_id": "cam_01_123_42",
    "camera_id": "cam_01",
    "similarity": 0.83,
    "status": "unknown"
}

Output:
{
    "person_id": "cam_01_123_42",
    "is_known": true,
    "visit_count": 5,
    "last_seen": "2024-01-15T10:30:00",
    "days_since_last_visit": 2,
    "typical_hours": [9, 10, 14, 15],
    "is_typical_time": true,
    "confidence_boost": 10,
    "reason": "Returning visitor, seen 5 times before"
}
"""

import structlog
from datetime import datetime
from typing import Any, Dict, Optional
from agents.base import BaseAgent
from utils.db_utils import get_or_create_memory, update_visit_memory

logger = structlog.get_logger(__name__)


class MemoryAgent(BaseAgent):
    """Tracks visit history and provides context for recognition.

    This agent answers questions like:
    - Has this person been here before?
    - How often do they visit?
    - Is this a typical visit time?
    - Should we be more confident in the recognition?
    """

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory lookup and update.

        Args:
            input_data: Must contain:
                - person_id (str): The person's ID
                - camera_id (str): Current camera
                - similarity (float): Current similarity score
                - status (str): Current recognition status

        Returns:
            Dict with memory context for the Recognition Agent.
        """
        person_id = input_data.get("person_id")
        camera_id = input_data.get("camera_id", "unknown")
        similarity = input_data.get("similarity", 0.0)
        status = input_data.get("status", "unknown")

        if not person_id:
            return self._empty_result("No person_id provided")

        result = self._analyze(person_id, camera_id, similarity, status)

        logger.info("memory_lookup",
                    person_id=person_id,
                    visit_count=result["visit_count"],
                    is_known=result["is_known"],
                    confidence_boost=result["confidence_boost"])

        return result

    def _analyze(self, person_id: str, camera_id: str,
                 similarity: float, status: str) -> dict:
        """Core memory analysis logic."""
        try:
            memory = get_or_create_memory(person_id)
        except Exception as e:
            logger.error("memory_lookup_failed", person_id=person_id, error=str(e))
            return self._empty_result(f"Memory lookup failed: {e}")

        now = datetime.now()
        visit_count = memory.get("visit_count", 0)
        last_seen = memory.get("last_seen")
        first_seen = memory.get("first_seen")
        avg_similarity = memory.get("avg_similarity", 0.0)
        typical_hours = memory.get("typical_hours", [])
        typical_cameras = memory.get("typical_cameras", [])
        last_status = memory.get("last_status")

        # Calculate days since last visit
        days_since_last = None
        if last_seen:
            if isinstance(last_seen, datetime):
                days_since_last = (now - last_seen).days

        # Check if this is a typical visit time
        current_hour = now.hour
        is_typical_time = False
        if typical_hours:
            # Check if current hour is within 2 hours of typical hours
            hour_counts = {}
            for h in typical_hours:
                hour_counts[h] = hour_counts.get(h, 0) + 1
            # Get the most common hours
            common_hours = sorted(hour_counts.keys(), key=lambda x: hour_counts[x], reverse=True)[:3]
            is_typical_time = any(min(abs(current_hour - h), 24 - abs(current_hour - h)) <= 2 for h in common_hours)

        # Check if this is a typical camera
        is_typical_camera = camera_id in typical_cameras

        # Calculate confidence boost based on memory
        confidence_boost = self._calculate_confidence_boost(
            visit_count, days_since_last, avg_similarity,
            is_typical_time, is_typical_camera, similarity
        )

        # Determine if this is a returning visitor
        is_known = visit_count > 0 and last_status in ["known", "verified", "authorized"]

        # Build reason
        reason = self._build_reason(visit_count, days_since_last, is_typical_time, is_known)

        return {
            "person_id": person_id,
            "is_known": is_known,
            "visit_count": visit_count,
            "first_seen": first_seen.isoformat() if isinstance(first_seen, datetime) else None,
            "last_seen": last_seen.isoformat() if isinstance(last_seen, datetime) else None,
            "days_since_last_visit": days_since_last,
            "avg_similarity": round(avg_similarity, 4),
            "typical_hours": typical_hours[-10:] if typical_hours else [],
            "typical_cameras": typical_cameras,
            "is_typical_time": is_typical_time,
            "is_typical_camera": is_typical_camera,
            "last_status": last_status,
            "confidence_boost": confidence_boost,
            "reason": reason,
        }

    def _calculate_confidence_boost(self, visit_count: int, days_since_last: int,
                                    avg_similarity: float, is_typical_time: bool,
                                    is_typical_camera: bool, current_similarity: float) -> float:
        """Calculate how much memory should boost recognition confidence.

        Returns a value between -10 and +20.
        """
        boost = 0.0

        # Boost for returning visitors
        if visit_count > 0:
            boost += min(10, visit_count * 2)  # +2 per visit, max +10

        # Boost for recent visits (within 7 days)
        if days_since_last is not None and days_since_last <= 7:
            boost += 5
        elif days_since_last is not None and days_since_last <= 30:
            boost += 2

        # Boost for consistent similarity
        if avg_similarity > 0.8:
            boost += 3
        elif avg_similarity > 0.6:
            boost += 1

        # Boost for typical visit patterns
        if is_typical_time:
            boost += 2
        if is_typical_camera:
            boost += 1

        # Penalty if current similarity is much lower than average
        if avg_similarity > 0 and current_similarity < avg_similarity * 0.7:
            boost -= 5

        return max(-10, min(20, boost))

    def _build_reason(self, visit_count: int, days_since_last: int,
                      is_typical_time: bool, is_known: bool) -> str:
        """Build a human-readable reason string."""
        if visit_count == 0:
            return "First-time visitor, no memory available."

        parts = []

        if is_known:
            parts.append(f"Returning visitor seen {visit_count} times")
        else:
            parts.append(f"Previously seen {visit_count} times")

        if days_since_last is None:
            parts.append("recency unknown")
        elif days_since_last == 0:
            parts.append("visited today")
        elif days_since_last == 1:
            parts.append("visited yesterday")
        elif days_since_last <= 7:
            parts.append(f"visited {days_since_last} days ago")
        else:
            parts.append(f"last seen {days_since_last} days ago")

        if is_typical_time:
            parts.append("at a typical time")

        return ". ".join(parts) + "."

    def _empty_result(self, reason: str) -> dict:
        """Return an empty memory result."""
        return {
            "person_id": None,
            "is_known": False,
            "visit_count": 0,
            "first_seen": None,
            "last_seen": None,
            "days_since_last_visit": None,
            "avg_similarity": 0.0,
            "typical_hours": [],
            "typical_cameras": [],
            "is_typical_time": False,
            "is_typical_camera": False,
            "last_status": None,
            "confidence_boost": 0,
            "reason": reason,
        }

    def record_visit(self, person_id: str, camera_id: str, status: str,
                     similarity: float, is_masked: bool = False,
                     visit_action: str = "recorded") -> dict:
        """Record a visit in memory. Call after recognition is complete.

        Args:
            person_id: The person's ID
            camera_id: Current camera
            status: Recognition status
            similarity: Similarity score
            is_masked: Whether person is masked
            visit_action: Action taken (recorded/suppressed/skipped)

        Returns:
            Updated memory document.
        """
        if not person_id:
            logger.debug("visit_skipped_no_person_id")
            return {}
        try:
            result = update_visit_memory(person_id, camera_id, status, similarity, is_masked)
            suppressed = result.get("suppressed", False)
            event = "visit_suppressed_duplicate" if suppressed else "visit_recorded"
            logger.info(event,
                        person_id=person_id,
                        visit_count=result.get("visit_count", 0),
                        camera=camera_id,
                        visit_action="suppressed_duplicate" if suppressed else visit_action)
            return result
        except Exception as e:
            logger.error("visit_recording_failed", person_id=person_id, error=str(e))
            return {}
