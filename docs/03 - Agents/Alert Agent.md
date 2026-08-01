# Alert Agent

> Dispatches alerts through multiple channels (console, email, SMS, webhook) with per-track deduplication and cooldown. Generates NL summaries via local LLM.

**File**: `agents/alert_agent.py` (225 lines)

## Role in system

```
Policy Agent → DecisionResult(should_alert=True, alert_level="critical")
                    │
                    ▼
            Alert Agent:dispatch()
                    │
                    ├─ Console (synchronous, immediate)
                    ├─ Email (async, via executor)
                    ├─ SMS (async, via executor)
                    └─ Webhook (async, via executor)
```

## Function: `dispatch(track, decision, image_url)`

```python
def dispatch(track: Track, decision: DecisionResult, image_url: str = None) -> bool:
```

### Processing

```
1. IF not decision.should_alert → return True (nothing to do)
2. IF track.alerted → return True (already sent)
3. IF not should_send_alert(track_id, alert_level, status) → return False (cooldown)

4. Build payload:
   {track_id, status, alert_level, person_id, name, reason,
    is_masked, timestamp, camera_id, image_url, memory_context}

5. Console alert (synchronous):
   IF "console" in ALERT_CHANNELS → _alert_console(payload)

6. Generate NL summary (synchronous, before thread dispatch):
   IF not decision.nl_summary:
       nl = llm_client.generate_nl_summary(payload)
       IF nl: decision.nl_summary = nl

7. Network alerts (async, via executor):
   IF any of ["email", "sms", "webhook"] in ALERT_CHANNELS:
       Submit _send_async() to _alert_executor
```

## Alert deduplication: `should_send_alert(track_id, alert_level, status)`

```
IF status in UNVERIFIED_STATUSES (unknown, masked_unknown, uncertain, intentionally_hidden):
    key = "unverified:{alert_level}"
    — All unknowns of same level share one cooldown timer
    — A routine unknown does NOT suppress a critical (blacklist) alert

ELSE:
    key = "{track_id}:{alert_level}"
    — Per-track, per-level dedup

IF now - last_alert_time < ALERT_COOLDOWN_SECS (60):
    → suppress (return False)
ELSE:
    → allow, update timestamp
```

## Stale pruning

Every `ALERT_COOLDOWN_SECS × 2` seconds:
```
Remove entries older than ALERT_COOLDOWN_SECS × 2 from _alert_timestamps
— Prevents unbounded memory growth
```

## Channel implementations

### Console (`_alert_console`)
```
logger.warning("alert", alert_level=level, status=status, ...)
```
Synchronous, instant. Uses LLM summary if available, else reason string.

### Email (`_alert_email`)
```
- Build MIMEText message with status, alert_level, track_id, camera, person, summary
- Connect to SMTP server (STARTTLS)
- Send message
```
Requires: `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS`, `ALERT_EMAIL_TO`

### SMS (`_alert_sms`)
```
- Import twilio.rest.Client
- Send body: "[LEVEL] summary. Person: name"
```
Requires: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM`, `ALERT_SMS_TO`

### Webhook (`_alert_webhook`)
```
- POST JSON payload to ALERT_WEBHOOK_URL
- 10s timeout
```

## Progressive critical alerts

**File**: `agents/camera_agent.py:373-385`

During progressive recognition (NOT at finalization):
```
IF decision.should_alert
   AND decision.alert_level == "critical"    ← ONLY critical fires early
   AND NOT track.alerted
   AND NOT already_finalized:
    → dispatch immediately
    → set_decision(track_id, status)
```

All other alerts are deferred to track finalization.

## Thread safety

- `_alert_lock` protects `_alert_timestamps` dict
- `_alert_executor` has 2 workers for async channels
- `track.mark_alerted_once()` is atomic (uses Track._lock)

## See also
- [[Policy Rules]] — what triggers alerts
- [[Policy Agent]] — decides should_alert + alert_level
- [[LLM Client]] — generates NL summaries
- [[3-Tier Logging]] — how alerts appear in logs
