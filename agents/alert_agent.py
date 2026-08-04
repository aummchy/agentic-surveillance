import json
import structlog
import time
import smtplib
import threading
import concurrent.futures
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from config import settings
from pipeline.models import Track, DecisionResult
from utils import llm_client

logger = structlog.get_logger(__name__)

_alert_timestamps = {}
_alert_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="alert")
_last_prune_time = 0.0
_alert_lock = threading.Lock()


def _prune_stale_alerts():
    """Remove entries older than 2x cooldown to prevent unbounded growth.
    Must be called while holding _alert_lock."""
    global _last_prune_time
    now = time.time()
    if now - _last_prune_time < settings.ALERT_COOLDOWN_SECS * 2:
        return
    _last_prune_time = now
    cutoff = now - settings.ALERT_COOLDOWN_SECS * 2
    stale_keys = [k for k, ts in _alert_timestamps.items() if ts < cutoff]
    for k in stale_keys:
        del _alert_timestamps[k]
    if stale_keys:
        logger.debug("alert_timestamps_pruned", count=len(stale_keys))


_UNVERIFIED_STATUSES = {"unknown", "masked_unknown", "uncertain", "intentionally_hidden"}


def should_send_alert(track_id: str, alert_level: str, status: str = None) -> bool:
    now = time.time()
    with _alert_lock:
        _prune_stale_alerts()
        # Per-track, per-level dedup. Include track_id for unverified
        # statuses too, so two different unknown people don't suppress
        # each other's alerts within the cooldown window.
        if status in _UNVERIFIED_STATUSES:
            key = f"{track_id}:unverified:{alert_level}"
        else:
            key = f"{track_id}:{alert_level}"
        last = _alert_timestamps.get(key, 0)
        if now - last < settings.ALERT_COOLDOWN_SECS:
            return False
        _alert_timestamps[key] = now
    return True


def dispatch(track: Track, decision: DecisionResult, image_url: str = None) -> bool:
    if not decision.should_alert:
        return False

    if track.alerted:
        return False

    if not should_send_alert(track.track_id, decision.alert_level, decision.status):
        return False

    payload = {
        "track_id": track.track_id,
        "status": decision.status,
        "alert_level": decision.alert_level,
        "person_id": decision.person_id,
        "name": decision.name,
        "reason": decision.reason,
        "is_masked": track.is_masked,
        "timestamp": time.time(),
        "camera_id": settings.CAMERA_ID,
        "image_url": image_url,
        "memory_context": getattr(track, "pending_memory_context", None) or {},
    }

    # Console is instant, run synchronously (no LLM blocking)
    if "console" in settings.ALERT_CHANNELS:
        try:
            _alert_console(payload, decision)
        except Exception as e:
            logger.error("alert_channel_failed", channel="console", error=str(e))

    # Network-bound alerts run in thread pool (non-blocking)
    async_channels = [c for c in settings.ALERT_CHANNELS if c in ("email", "sms", "webhook")]

    # Generate NL summary synchronously before thread dispatch to avoid
    # mutating DecisionResult from the thread pool.
    if not decision.nl_summary:
        try:
            nl = llm_client.generate_nl_summary(payload)
            if nl:
                decision.nl_summary = nl
        except Exception as e:
            logger.debug("nl_summary_generation_failed", error=str(e))

    def _send_async():
        for channel in async_channels:
            try:
                if channel == "email":
                    _alert_email(payload, decision)
                elif channel == "sms":
                    _alert_sms(payload, decision)
                elif channel == "webhook":
                    _alert_webhook(payload)
            except Exception as e:
                logger.error("alert_channel_failed", channel=channel, error=str(e))

    if async_channels:
        _alert_executor.submit(_send_async)

    return True


def _alert_console(payload: dict, decision: DecisionResult = None):
    level = payload["alert_level"].upper()
    status = payload["status"]
    track_id = payload["track_id"]

    summary = decision.nl_summary if decision and decision.nl_summary else payload["reason"]
    name = payload.get("name")

    logger.warning("alert",
                   alert_level=level.lower(),
                   status=status,
                   track_id=track_id,
                   summary=summary,
                   name=name)


def _alert_email(payload: dict, decision: DecisionResult = None):
    if not all([settings.SMTP_HOST, settings.SMTP_USER, settings.SMTP_PASS, settings.ALERT_EMAIL_TO]):
        return

    msg = MIMEMultipart()
    msg["From"] = settings.SMTP_USER
    msg["To"] = settings.ALERT_EMAIL_TO
    msg["Subject"] = f"[{payload['alert_level'].upper()}] Surveillance Alert: {payload['status']}"

    body = f"""
Surveillance Alert

Status: {payload['status']}
Alert Level: {payload['alert_level']}
Track ID: {payload['track_id']}
Camera: {payload['camera_id']}
Masked: {payload['is_masked']}
Time: {payload['timestamp']}

Person ID: {payload.get('person_id', 'N/A')}
Name: {payload.get('name', 'N/A')}
"""
    if decision and decision.nl_summary:
        body += f"\nSummary: {decision.nl_summary}\n"
    else:
        body += f"\nReason: {payload['reason']}\n"

    if payload.get("image_url"):
        body += f"\nImage: {payload['image_url']}"

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT)
        server.starttls()
        server.login(settings.SMTP_USER, settings.SMTP_PASS)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        logger.error("email_alert_failed", error=str(e))
        raise


def _alert_sms(payload: dict, decision: DecisionResult = None):
    if not all([settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN, settings.ALERT_SMS_TO]):
        return

    try:
        from twilio.rest import Client
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

        if decision and decision.nl_summary:
            body = f"[{payload['alert_level'].upper()}] {decision.nl_summary}"
        else:
            body = f"[{payload['alert_level'].upper()}] {payload['status']}: {payload['reason']}"
        if payload.get("name"):
            body += f" Person: {payload['name']}"

        client.messages.create(
            body=body,
            from_=settings.TWILIO_FROM,
            to=settings.ALERT_SMS_TO
        )
    except ImportError:
        logger.warning("twilio_not_installed")
    except Exception as e:
        logger.error("sms_alert_failed", error=str(e))
        raise


def _alert_webhook(payload: dict):
    if not settings.ALERT_WEBHOOK_URL:
        return

    try:
        resp = requests.post(
            settings.ALERT_WEBHOOK_URL,
            json=payload,
            timeout=10
        )
        resp.raise_for_status()
    except Exception as e:
        logger.error("webhook_alert_failed", error=str(e))
        raise


def shutdown():
    """Shut down the alert executor on process exit."""
    _alert_executor.shutdown(wait=False)
