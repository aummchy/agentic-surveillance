# Terminal Output Reference

## Overview

The surveillance system uses a 3-tier logging architecture:
- **Terminal** — Compact one-liner with ANSI colors (human operator)
- **JSON file** (`logs/surveillance.jsonl`) — Machine-readable JSON lines
- **Debug file** (`logs/surveillance.debug.log`) — Full verbose output
- **Calculation log** (`logs/calculation.log`) — Full tabular confidence formula breakdown per recognition

## Event Flow (Per Track)

```
Camera detects person
    │
    ├─► MATCH    — Vector search finds best match
    ├─► MEMORY   — Lookup visit history
    ├─► RECOG    — Recognition decision (every 10 frames)
    ├─► POLICY   — Policy decision (alert/no-alert)
    ├─► FACE     — Face crop saved
    │
    │   [Person leaves frame]
    │
    ├─► FINAL    — Track finalized
    └─► VISIT    — Visit recorded in memory
```

## Event Formats

### Camera Events

| Event | Format | Example |
|-------|--------|---------|
| `CAM started` | `{ts} CAM     started source={src} backend={ backend} res={WxH}` | `2026-07-09T07:02:13 CAM     started source=0 backend=dshow res=1280x720` |
| `CAM stopped` | `{ts} CAM     stopped` | `2026-07-09T07:04:11 CAM     stopped` |
| `CAM reconnected` | `{ts} CAM     reconnected res={WxH}` | `2026-07-09T07:05:00 CAM     reconnected res=1280x720` |

### Recognition Pipeline Events

| Event | Format | Example |
|-------|--------|---------|
| `MATCH` | `{ts} MATCH   {name} sim={sim} top2={top2} gap={gap} verified={bool} tags=[...]` | `2026-07-09T07:02:24 MATCH   aum        sim=0.608 top2=0.499 gap=0.109 verified=True tags=['auto_registered', 'verified']` |
| `MEMORY` | `{ts} MEMORY  {person_id} visits={count} known={bool} boost={float}` | `2026-07-09T07:02:24 MEMORY  cam_01_1782040060_1 visits=218 known=True boost=18.0` |
| `RECOG` | `{ts} RECOG   {name} {status} sim={sim} top2={top2} gap={gap} q={quality} dur={secs}s conf={conf}` | `2026-07-09T07:02:24 RECOG   aum        KNOWN        sim=0.608 top2=0.500 gap=0.109 q=0.70 dur=8s conf=71` |
| `POLICY` | `{ts} POLICY  {status} alert={level} {ok/ALERT} vis={count}` | `2026-07-09T07:02:24 POLICY  VERIFIED     alert=none  ok vis=218` |

### Face Events

| Event | Format | Example |
|-------|--------|---------|
| `FACE saved` | `{ts} FACE    saved track={track_id} path={path}` | `2026-07-09T07:02:23 FACE    saved track=cam_01_1783580530_1 path=captures/face_crops/.../53.jpg` |
| `FACE no_embedding` | `{ts} FACE    no_embedding track={track_id} vis={visibility} frames={count}` | `2026-07-09T07:03:05 FACE    no_embedding track=cam_01_1783580530_2 vis=unknown frames=0` |

### Finalization Events

| Event | Format | Example |
|-------|--------|---------|
| `FINAL` | `{ts} FINAL   {name} {status} sim={sim} top2={top2} gap={gap} conf={conf} frames={N} face={N} vis={N}` | `2026-07-09T07:03:13 FINAL   aum        VERIFIED     sim=0.592 top2=0.457 gap=0.136 conf=71  frames=13 face=1 vis=218` |
| `VISIT` | `{ts} VISIT   {name} {action} total={count}` | `2026-07-09T07:03:13 VISIT   aum        recorded total=219` |

### Alert Events

| Event | Format | Example |
|-------|--------|---------|
| `ALERT` | `{ts} ALERT   {status} name={name} level={level}` | `2026-07-09T07:03:13 ALERT   UNKNOWN    name=Unknown level=high` |
| `ALERT CRITICAL` | `{ts} ALERT   CRITICAL {name} reason={reason}` | `2026-07-09T07:03:13 ALERT   CRITICAL Blacklisted Person reason=Blacklisted person detected` |

### System Events

| Event | Format | Example |
|-------|--------|---------|
| `LLM unavailable` | `{ts} LLM     unavailable model={model} url={url}` | `2026-07-09T07:03:17 LLM     unavailable model=gemma3:4b url=http://localhost:11434` |
| `SHUTDOWN` | `{ts} SHUTDOWN  {reason}` | `2026-07-09T07:04:11 SHUTDOWN  shutdown_complete` |

## Field Reference

| Field | Description | Values |
|-------|-------------|--------|
| `sim` | Cosine similarity to best match | 0.0 - 1.0, `---` if no match |
| `top2` | Second-best similarity | 0.0 - 1.0, `---` if < 2 candidates |
| `gap` | Margin (top1 - top2) | 0.0 - 1.0, `---` if unavailable |
| `q` | Face quality score | 0.0 - 1.0, `---` if unavailable |
| `dur` | Track duration in seconds | Integer |
| `conf` | Recognition confidence | 1 - 100 |
| `status` | Recognition/policy status | `KNOWN`, `VERIFIED`, `UNCERTAIN`, `UNKNOWN`, `BLACKLISTED` |
| `alert` | Alert level | `none`, `low`, `medium`, `high`, `critical` |
| `vis` | Visit count in memory | Integer |
| `frames` | Total frames track was alive | Integer |
| `face` | Frames with detectable face | Integer |

## Status Colors (ANSI)

| Status | Color | Meaning |
|--------|-------|---------|
| `KNOWN`, `VERIFIED`, `AUTHORIZED`, `KNOWN_VISITOR` | Green | Person identified |
| `UNCERTAIN` | Yellow | Low confidence match |
| `UNKNOWN`, `BLACKLIST`, `MASKED_UNKNOWN`, `INTENTIONALLY_HIDDEN` | Red | Unidentified/alert |

## Event Prefix Colors (ANSI)

| Prefix | Color | Events |
|--------|-------|--------|
| `CAM` | Cyan | camera_started, camera_stopped, camera_reconnected |
| `MATCH` | Blue | match_found |
| `MEMORY` | Magenta | memory_lookup, visit_recorded |
| `RECOG` | Cyan | recognition_decision |
| `POLICY` | Green | policy_decision |
| `VISIT` | Magenta | visit_recorded |
| `FINAL` | Green | track_finalized |
| `FACE` | Yellow | face_crop_saved, no_embedding_after_retries |
| `ALERT` | Red | alert_dispatched, progressive_critical_alert |
| `SHUTDOWN` | Dim | shutdown |

## Configuration

### Terminal Allowlist

Only events in `TERMINAL_ALLOWLIST` (config/settings.py) appear in terminal output. Events not in the list are only written to JSON/debug log files.

### Log Levels

- **Terminal**: INFO and above
- **JSON file**: DEBUG and above
- **Debug file**: DEBUG and above

### Suppressed Loggers

These loggers are silenced to reduce noise:
- `pymongo`, `motor` — WARNING level
- `insightface` — WARNING level
- `ultralytics` — ERROR level
- `cloudinary` — WARNING level

## Example Session Output

```
2026-07-09T07:02:13 CAM     started source=0 backend=dshow res=1280x720
2026-07-09T07:02:23 FACE    saved track=cam_01_1783580530_1 path=captures/face_crops/.../53.jpg
2026-07-09T07:02:24 MATCH   aum        sim=0.608 top2=0.499 gap=0.109 verified=True tags=['auto_registered', 'verified']
2026-07-09T07:02:24 MEMORY  cam_01_1782040060_1 visits=218 known=True boost=18.0
2026-07-09T07:02:24 RECOG   aum        KNOWN        sim=0.608 top2=0.500 gap=0.109 q=0.70 dur=8s conf=71
2026-07-09T07:02:24 POLICY  VERIFIED     alert=none  ok vis=218
2026-07-09T07:03:13 FINAL   aum        VERIFIED     sim=0.592 top2=0.457 gap=0.136 conf=71  frames=13 face=1 vis=218
2026-07-09T07:03:13 VISIT   aum        recorded total=219
2026-07-09T07:04:11 CAM     stopped
```
