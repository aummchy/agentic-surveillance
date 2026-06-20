import json
import logging
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from config import settings
from pipeline.models import Track, DecisionResult

logger = logging.getLogger(__name__)

_alert_timestamps = {}


def should_send_alert(track_id: str, alert_level: str) -> bool:
    key = f"{track_id}:{alert_level}"
    now = time.time()
    last = _alert_timestamps.get(key, 0)
    if now - last < settings.ALERT_COOLDOWN_SECS:
        return False
    _alert_timestamps[key] = now
    return True


def dispatch(track: Track, decision: DecisionResult, image_url: str = None) -> bool:
    if not decision.should_alert:
        return True

    if track.alerted:
        return True

    if not should_send_alert(track.track_id, decision.alert_level):
        return True

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
        "image_url": image_url
    }

    success = True
    for channel in settings.ALERT_CHANNELS:
        try:
            if channel == "console":
                _alert_console(payload)
            elif channel == "email":
                _alert_email(payload)
            elif channel == "sms":
                _alert_sms(payload)
            elif channel == "webhook":
                _alert_webhook(payload)
        except Exception as e:
            logger.error(f"Alert channel '{channel}' failed: {e}")
            success = False

    return success


def _alert_console(payload: dict):
    level = payload["alert_level"].upper()
    status = payload["status"]
    reason = payload["reason"]
    track_id = payload["track_id"]

    emoji_map = {
        "critical": "!!!",
        "high": "!!",
        "medium": "!",
        "low": "*",
        "none": ""
    }
    prefix = emoji_map.get(level, "")

    print(f"\n{prefix} ALERT [{level}] {status} | Track: {track_id}")
    print(f"    Reason: {reason}")
    if payload.get("name"):
        print(f"    Person: {payload['name']}")
    print()


def _alert_email(payload: dict):
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
Reason: {payload['reason']}
Masked: {payload['is_masked']}
Time: {payload['timestamp']}

Person ID: {payload.get('person_id', 'N/A')}
Name: {payload.get('name', 'N/A')}
"""
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
        logger.error(f"Email alert failed: {e}")
        raise


def _alert_sms(payload: dict):
    if not all([settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN, settings.ALERT_SMS_TO]):
        return

    try:
        from twilio.rest import Client
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

        body = f"[{payload['alert_level'].upper()}] {payload['status']}: {payload['reason']}"
        if payload.get("name"):
            body += f" Person: {payload['name']}"

        client.messages.create(
            body=body,
            from_=settings.TWILIO_FROM,
            to=settings.ALERT_SMS_TO
        )
    except ImportError:
        logger.warning("Twilio not installed, SMS alert skipped")
    except Exception as e:
        logger.error(f"SMS alert failed: {e}")
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
        logger.error(f"Webhook alert failed: {e}")
        raise
