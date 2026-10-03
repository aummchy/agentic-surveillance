"""
Phase 2.5 — Report Agent

Generates human-readable reports and summaries.
Instead of just logging events, this agent creates actionable insights.

Input:
{
    "report_type": "incident",
    "person_id": "cam_01_123_42",
    "track_data": {...},
    "recognition_result": {...},
    "memory_context": {...},
    "policy_result": {...}
}

Output:
{
    "title": "Unknown Visitor at Gate 2",
    "summary": "Person detected at 09:41, stayed for 38 seconds...",
    "recommendation": "Manual verification recommended",
    "severity": "medium",
    "details": {...}
}
"""

import structlog
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from agents.base import BaseAgent
from config.status import Status, STATUS_LABELS, AlertLevel
from utils.db_utils import (
    get_events_with_faces, get_stats, get_visit_history
)
from utils import llm_client

logger = structlog.get_logger(__name__)


class ReportAgent(BaseAgent):
    """Generates reports and summaries for surveillance events.

    Report types:
    - incident: Single event report
    - summary: Daily/weekly summary
    - person: Person visit history
    - stats: Dashboard statistics
    - anomaly: Unusual activity detection
    """

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report.

        Args:
            input_data: Must contain:
                - report_type (str): "incident" | "summary" | "person" | "stats"
                - Additional data based on report_type

        Returns:
            Dict with report content.
        """
        report_type = input_data.get("report_type", "summary")

        if report_type == "incident":
            return self._incident_report(input_data)
        elif report_type == "summary":
            return self._summary_report(input_data)
        elif report_type == "person":
            return self._person_report(input_data)
        elif report_type == "stats":
            return self._stats_report(input_data)
        else:
            return {"error": f"Unknown report type: {report_type}"}

    def _incident_report(self, data: Dict) -> Dict:
        """Generate a report for a single incident."""
        track_id = data.get("track_id", "unknown")
        camera_id = data.get("camera_id", "unknown")
        status = data.get("status", Status.UNKNOWN)
        alert_level = data.get("alert_level", AlertLevel.MEDIUM)
        reason = data.get("reason", "")
        timestamp = data.get("timestamp", datetime.now(timezone.utc).isoformat())
        image_url = data.get("image_url")
        person_id = data.get("person_id")
        name = data.get("name", "Unknown")

        # Memory context arrives pre-batched: routes/reports.py fetches all
        # visit histories in one query and passes them here as "visit_history".
        # No per-incident DB query (avoids N+1).
        memory_context = data.get("visit_history") or {}

        visit_count = memory_context.get("visit_count", 0)
        last_seen = memory_context.get("last_seen")

        # Build template-based content first (fallback)
        title = self._build_title(status, name, camera_id)
        summary = self._build_summary(status, name, camera_id, timestamp, visit_count, last_seen)
        recommendation = self._build_recommendation(status, alert_level, visit_count)

        # Try LLM-enhanced report (non-blocking, falls back to templates)
        llm_result = llm_client.generate_incident_summary({
            "status": status,
            "alert_level": alert_level,
            "name": name,
            "camera_id": camera_id,
            "reason": reason,
            "visit_count": visit_count,
            "timestamp": timestamp,
        })

        if llm_result:
            lines = llm_result.strip().split("\n")
            for i, line in enumerate(lines):
                lower = line.lower().strip()
                if lower.startswith("summary:"):
                    summary = line.split(":", 1)[1].strip()
                elif lower.startswith("recommendation:"):
                    recommendation = line.split(":", 1)[1].strip()

        return {
            "report_type": "incident",
            "title": title,
            "summary": summary,
            "recommendation": recommendation,
            "severity": alert_level,
            "details": {
                "track_id": track_id,
                "camera_id": camera_id,
                "status": status,
                "person_id": person_id,
                "name": name,
                "timestamp": timestamp,
                "image_url": image_url,
                "visit_count": visit_count,
                "reason": reason,
            }
        }

    def _summary_report(self, data: Dict) -> Dict:
        """Generate a daily/weekly summary."""
        period = data.get("period", "daily")

        try:
            stats = get_stats()
        except Exception as e:
            logger.warning("stats_query_failed", error=str(e))
            stats = {"total_unknown": 0, "total_verified": 0, "events_today": 0, "unknown_today": 0}

        # Get recent events
        try:
            events_data = get_events_with_faces(limit=20)
            events = events_data.get("events", [])
        except Exception as e:
            logger.warning("events_query_failed", error=str(e))
            events = []

        # Analyze events
        status_counts = {}
        camera_counts = {}
        for event in events:
            status = event.get("status", Status.UNKNOWN)
            camera = event.get("camera_id", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            camera_counts[camera] = camera_counts.get(camera, 0) + 1

        # Find peak hours
        hour_counts = {}
        for event in events:
            ts = event.get("timestamp")
            if isinstance(ts, datetime):
                hour = ts.hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1

        peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else "N/A"

        # Template-based summary (fallback)
        summary = (f"Processed {stats.get('events_today', 0)} events today. "
                   f"{stats.get('unknown_today', 0)} unknown persons detected.")

        # Try LLM-enhanced executive summary
        llm_summary = llm_client.generate_executive_summary(stats, events)
        if llm_summary:
            summary = llm_summary

        return {
            "report_type": "summary",
            "title": f"{period.title()} Surveillance Summary",
            "summary": summary,
            "period": period,
            "stats": stats,
            "status_breakdown": status_counts,
            "camera_breakdown": camera_counts,
            "peak_hour": peak_hour,
            "recent_events": events[:10],
        }

    def _person_report(self, data: Dict) -> Dict:
        """Generate a report for a specific person."""
        person_id = data.get("person_id")

        if not person_id:
            return {"error": "person_id required"}

        try:
            memory = get_visit_history(person_id)
        except Exception as e:
            logger.warning("visit_history_query_failed", person_id=person_id, error=str(e))
            memory = {}

        visit_count = memory.get("visit_count", 0)
        first_seen = memory.get("first_seen")
        last_seen = memory.get("last_seen")
        typical_hours = memory.get("typical_hours", [])
        typical_cameras = memory.get("typical_cameras", [])
        avg_similarity = memory.get("avg_similarity", 0)

        # Build visit pattern summary
        if typical_hours:
            hour_counts = {}
            for h in typical_hours:
                hour_counts[h] = hour_counts.get(h, 0) + 1
            common_hours = sorted(hour_counts.keys(), key=lambda x: hour_counts[x], reverse=True)[:3]
            visit_pattern = f"Usually visits around {', '.join(str(h) for h in common_hours)}:00"
        else:
            visit_pattern = "No visit pattern established"

        return {
            "report_type": "person",
            "title": f"Person Report: {person_id}",
            "summary": f"Visited {visit_count} times. Last seen: {last_seen or 'Never'}",
            "person_id": person_id,
            "visit_count": visit_count,
            "first_seen": first_seen.isoformat() if isinstance(first_seen, datetime) else None,
            "last_seen": last_seen.isoformat() if isinstance(last_seen, datetime) else None,
            "avg_similarity": round(avg_similarity, 4),
            "visit_pattern": visit_pattern,
            "typical_cameras": typical_cameras,
            "typical_hours": typical_hours[-10:] if typical_hours else [],
        }

    def _stats_report(self, data: Dict) -> Dict:
        """Generate dashboard statistics."""
        try:
            stats = get_stats()
        except Exception as e:
            logger.warning("stats_query_failed", error=str(e))
            stats = {"total_unknown": 0, "total_verified": 0, "events_today": 0, "unknown_today": 0}

        return {
            "report_type": "stats",
            "title": "Dashboard Statistics",
            "stats": stats,
        }

    def _build_title(self, status: int, name: str, camera_id: str) -> str:
        """Build incident report title."""
        status_titles = {
            Status.BLACKLIST: f"CRITICAL: Blacklisted Person at {camera_id}",
            Status.MASKED_UNKNOWN: f"Masked Unknown Person at {camera_id}",
            Status.UNKNOWN: f"Unknown Visitor at {camera_id}",
        }
        return status_titles.get(status, f"Incident at {camera_id}")

    def _build_summary(self, status: int, name: str, camera_id: str,
                       timestamp: str, visit_count: int, last_seen: datetime) -> str:
        """Build incident report summary."""
        parts = []

        if name and name != "Unknown":
            parts.append(f"Person identified as '{name}'")
        else:
            parts.append("Unknown person detected")

        parts.append(f"at camera {camera_id}")

        if isinstance(last_seen, datetime):
            # MongoDB stores naive UTC datetimes; normalize to aware UTC so the
            # subtraction works no matter which form the value arrives in.
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            days_ago = (now_utc - last_seen).days
            if days_ago == 0:
                parts.append(f"(previously seen today, visit #{visit_count + 1})")
            elif days_ago == 1:
                parts.append(f"(previously seen yesterday, visit #{visit_count + 1})")
            elif days_ago <= 7:
                parts.append(f"(last seen {days_ago} days ago, visit #{visit_count + 1})")
            else:
                parts.append(f"(first-time visitor)")
        elif visit_count == 0:
            parts.append("(first-time visitor)")

        return ". ".join(parts) + "."

    def _build_recommendation(self, status: int, alert_level: str, visit_count: int) -> str:
        """Build recommendation based on status."""
        recommendations = {
            Status.BLACKLIST: "IMMEDIATE ACTION REQUIRED. Contact security team.",
            Status.MASKED_UNKNOWN: "Manual verification recommended. Monitor closely.",
            Status.UNKNOWN: "Manual verification recommended if in restricted area.",
        }

        base = recommendations.get(status, "Monitor situation.")

        if visit_count > 3:
            base += " Returning visitor — check if pattern is normal."

        return base
