# 3-Tier Logging

> Three logging outputs: compact terminal, JSON forensic file, verbose debug file.

**File**: `config/settings.py:296-383`

## Three tiers

| Tier | Handler | Level | Format | File |
|------|---------|-------|--------|------|
| **Terminal** | StreamHandler | INFO+ | Compact one-liner with ANSI colors | stdout |
| **JSON file** | RotatingFileHandler | DEBUG+ | Machine-readable JSON lines | `logs/surveillance.jsonl` |
| **Debug file** | RotatingFileHandler | DEBUG+ | Full verbose output | `logs/surveillance.debug.log` |

## Terminal: CompactTerminalRenderer

Only events in `TERMINAL_ALLOWLIST` appear in terminal. One colored line per event:
```
2026-07-09T07:02:24 RECOG   aum        KNOWN        sim=0.608 top2=0.500 gap=0.109 q=0.70 dur=8s conf=71
```

Color coding:
- Status: GREEN (known/verified), YELLOW (uncertain), RED (unknown/blacklist)
- Prefix: CYAN (CAM/RECOG), BLUE (MATCH), MAGENTA (MEMORY/VISIT), GREEN (POLICY/FINAL), RED (ALERT)

## JSON file: JSONFileRenderer

Full structured JSON for machine parsing:
```json
{"event": "recognition_decision", "track_id": "...", "status": "known", "confidence": 71, "similarity": 0.608, ...}
```

Rotation: 5MB × 5 files.

## Debug file

Uses structlog's `ConsoleRenderer` for human-readable verbose output.

Rotation: 10MB × 3 files.

## Terminal allowlist

Only these events appear in terminal output:
```
startup_complete, camera_started, camera_stopped, camera_reconnected,
track_finalized, recognition_decision, match_found, memory_lookup,
visit_recorded, visit_suppressed_duplicate, alert, alert_dispatched,
alert_broadcast_sent, progressive_critical_alert, policy_decision,
duplicate_suppressed, face_crop_saved, pipeline_stats,
no_embedding_after_retries, llm_connect_failed, llm_unavailable, shutdown
```

All other events go to files only.

## Suppressed loggers

| Logger | Level | Why |
|--------|-------|-----|
| `pymongo` | WARNING | Too verbose |
| `insightface` | WARNING | Model loading spam |
| `ultralytics` | ERROR | Detection noise |
| `cloudinary` | WARNING | Connection pool warnings |

## structlog configuration

```python
structlog.configure(
    processors=[
        merge_contextvars,
        add_log_level,
        TimeStamper(fmt="iso"),
        add_byte_track_id,        # extracts byte_track_id from composite track_id
        StackInfoRenderer(),
        format_exc_info,
        wrap_for_formatter,
    ],
)
```

## See also
- [[Terminal Output Reference]] — all event formats
- [[Calculation Log]] — confidence formula breakdown log
- [[System Overview]] — logging architecture
