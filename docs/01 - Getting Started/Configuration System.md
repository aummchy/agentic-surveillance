# Configuration System

## Priority chain

```
.env  (secrets only)
  ↓ fallback to
config/config.jsonc  (tunables)
  ↓ fallback to
hardcoded defaults in config/settings.py
```

The `_get(env_key, config_key, default, cast)` function in `config/settings.py:109-122` resolves each setting by checking these three sources in order.

## Two files, two purposes

### `.env` — Secrets only
Never put tunables here. Contains:
- `MONGODB_URI`, `MONGODB_DATABASE`
- `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET`
- `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS`, `ALERT_EMAIL_TO`
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM`, `ALERT_SMS_TO`
- `ALERT_WEBHOOK_URL`

### `config/config.jsonc` — Tunables
JSON with comments (supports `//` and `/* */`). Contains:
- Detection thresholds (`PERSON_CONF_THRESHOLD`, `DET_SCORE_MIN`, etc.)
- Quality gates (`QUALITY_VALID_BLUR_MIN`, etc.)
- Confidence formula weights (`WEIGHT_SIMILARITY`, etc.)
- Recognition intervals (`RECOGNITION_INTERVAL_FRAMES`, etc.)
- Camera settings (`FRAME_WIDTH`, `FRAME_HEIGHT`, etc.)
- Alert cooldowns, office hours, ByteTrack params

## How `_get()` works

```python
def _get(env_key, config_key, default, cast):
    val = os.getenv(env_key)
    if val is not None:
        # Cast env var string to the right type
        if cast is list: return [elem_type(v) for v in val.split(",")]
        if cast is bool: return val.lower() not in ("false", "0", "no", "off", "")
        return cast(val)
    return cast(_config.get(config_key, default))
```

**List handling**: `OFFICE_DAYS=0,1,2,3,4` → `[0, 1, 2, 3, 4]` (ints, not strings).

## Validation

`validate_config()` in `config/settings.py:422-471` runs at startup:
- `MONGODB_URI` must be set
- `MATCH_THRESHOLD` ≤ 0.45
- `TRACK_TIMEOUT_SECS` > 0
- `MAX_TRACK_SECS` > 0
- `DET_SCORE_MIN` and `DET_SCORE_RELAXITY` between 0 and 1
- Quality weights sum to ~1.0
- Alert channels must be valid (`console`, `email`, `sms`, `webhook`)

## Effective settings log

At startup, every tunable setting's effective value and source (env/config.jsonc/default) is logged to `logs/surveillance.jsonl`.

## See also
- [[Environment Variables]] — full list of `.env` keys
- `config/config.jsonc` — every tunable with defaults
- [[CURRENT_ARCHITECTURE]] Appendix B — every threshold with purpose
