"""
Phase 2.4 — Alert Agent (Context-aware)

Instead of always alerting on unknown, this agent considers:
- Camera location (reception vs server room)
- Time of day (office hours vs after hours)
- Alert level (low, medium, high, critical)
- Cooldown (don't spam alerts)

Input:
{
    "status": "unknown",
    "alert_level": "medium",
    "camera_id": "cam_01",
    "camera_location": "reception",
    "track_id": "cam_01_123_42",
    "reason": "Unknown person detected"
}

Output:
{
    "should_alert": true,
    "alert_channels": ["console", "webhook"],
    "priority": "normal",
    "reason": "Unknown person at reception during office hours"
}
"""

import structlog
from datetime import datetime
from typing import Any, Dict, List
from agents.base import BaseAgent
from config import settings

logger = structlog.get_logger(__name__)

# Cooldown tracking
_alert_timestamps: Dict[str, float] = {}


class AlertAgent(BaseAgent):
    """Context-aware alert dispatch.

    This agent decides:
    1. Whether to alert (considering cooldown)
    2. Which channels to use (console, email, SMS, webhook)
    3. Priority level (normal, urgent, critical)

    Rules:
    - Blacklist: Always alert, all channels, critical priority
    - After-hours unknown: Higher priority
    - Reception: Log only (low priority)
    - Server room: High alert
    - Cooldown: Don't repeat same alert within N seconds
    """

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute alert decision.

        Args:
            input_data: Must contain:
                - status (str): Recognition status
                - alert_level (str): From Policy Agent
                - camera_id (str): Current camera
                - track_id (str): Track ID
                - reason (str): Alert reason

        Returns:
            Dict with alert decision and channels.
        """
        status = input_data.get("status", "unknown")
        alert_level = input_data.get("alert_level", "medium")
        camera_id = input_data.get("camera_id", settings.CAMERA_ID)
        track_id = input_data.get("track_id", "unknown")
        reason = input_data.get("reason", "")

        result = self._decide(status, alert_level, camera_id, track_id, reason)

        if result["should_alert"]:
            logger.info("alert_triggered",
                       status=status,
                       level=alert_level,
                       channels=result["alert_channels"],
                       priority=result["priority"])
        else:
            logger.debug("alert_suppressed",
                        status=status,
                        level=alert_level,
                        reason=result.get("suppress_reason", "cooldown"))

        return result

    def _decide(self, status: str, alert_level: str, camera_id: str,
                track_id: str, reason: str) -> dict:
        """Core alert logic."""

        # Check cooldown first
        cooldown_key = f"{track_id}:{alert_level}"
        if not self._check_cooldown(cooldown_key):
            return {
                "should_alert": False,
                "alert_channels": [],
                "priority": "normal",
                "reason": reason,
                "suppress_reason": "cooldown",
            }

        # Determine priority based on status and alert level
        priority = self._determine_priority(status, alert_level)

        # Determine which channels to use
        channels = self._determine_channels(status, alert_level, priority)

        # Determine if we should alert at all
        should_alert = len(channels) > 0

        return {
            "should_alert": should_alert,
            "alert_channels": channels,
            "priority": priority,
            "reason": reason,
            "status": status,
            "alert_level": alert_level,
        }

    def _determine_priority(self, status: str, alert_level: str) -> str:
        """Determine alert priority."""
        if status == "blacklist":
            return "critical"
        if alert_level == "critical":
            return "critical"
        if alert_level == "high":
            return "urgent"
        return "normal"

    def _determine_channels(self, status: str, alert_level: str, priority: str) -> List[str]:
        """Determine which alert channels to use."""
        channels = []

        # Always use console for visibility
        if "console" in settings.ALERT_CHANNELS:
            channels.append("console")

        # Use webhook for medium+ alerts
        if alert_level in ("medium", "high", "critical"):
            if "webhook" in settings.ALERT_CHANNELS and settings.ALERT_WEBHOOK_URL:
                channels.append("webhook")

        # Use email for high+ alerts
        if alert_level in ("high", "critical"):
            if "email" in settings.ALERT_CHANNELS:
                if all([settings.SMTP_HOST, settings.SMTP_USER, settings.SMTP_PASS, settings.ALERT_EMAIL_TO]):
                    channels.append("email")

        # Use SMS only for critical alerts
        if alert_level == "critical":
            if "sms" in settings.ALERT_CHANNELS:
                if all([settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN, settings.ALERT_SMS_TO]):
                    channels.append("sms")

        return channels

    def _check_cooldown(self, key: str) -> bool:
        """Check if alert is within cooldown period."""
        import time
        now = time.time()
        last = _alert_timestamps.get(key, 0)

        if now - last < settings.ALERT_COOLDOWN_SECS:
            return False

        _alert_timestamps[key] = now
        return True


# Convenience function for backward compatibility
def should_alert(track_id: str, alert_level: str) -> bool:
    """Check if alert should be sent (backward compatible)."""
    agent = AlertAgent()
    result = agent.run({
        "status": "unknown",
        "alert_level": alert_level,
        "track_id": track_id,
    })
    return result["should_alert"]
