# Terminal Output Reference

> Every event format that appears in terminal output, with examples.

> **Note:** This file supersedes the former root `docs/TERMINAL_OUTPUT.md` (deleted during docs consolidation).

## Event flow per track

```
Camera detects person
    │
    ├─► MATCH    — Vector search finds best match
    ├─► MEMORY   — Lookup visit history
    ├─► RECOG    — Recognition decision (every 20 frames)
    ├─► POLICY   — Policy decision (alert/no-alert)
    ├─► FACE     — Face crop saved
    │
    │   [Person leaves frame]
    │
    ├─► FINAL    — Track finalized
    └─► VISIT    — Visit recorded in memory
```

## Event formats

| Prefix | Event | Format |
|--------|-------|--------|
| `CAM` | started | `{ts} CAM     source={src} backend={backend} res={WxH} yolo={model} device={dev} face={model}` |
| `CAM` | stopped | `{ts} CAM     stopped` |
| `MATCH` | match_found | `{ts} MATCH   {name} sim={sim} top2={top2} gap={gap} verified={bool} tags=[...]` |
| `MEMORY` | memory_lookup | `{ts} MEMORY  {person_id} visits={count} known={bool} boost={float}` |
| `RECOG` | recognition_decision | `{ts} RECOG   {name} {status} sim={sim} top2={top2} gap={gap} q={quality} dur={secs}s conf={conf}` |
| `POLICY` | policy_decision | `{ts} POLICY  {status} alert={level} {ok/ALERT} vis={count}` |
| `FINAL` | track_finalized | `{ts} FINAL   {name} {status} sim={sim} top2={top2} gap={gap} conf={conf} frames={N} face={N} vis={N}` |
| `VISIT` | visit_recorded | `{ts} VISIT   {name} {action} total={count}` |
| `ALERT` | alert | `{ts} ALERT   {status} name={name} level={level} summary={summary}` |
| `FACE` | face_crop_saved | `{ts} FACE    saved track={track_id} path={path}` |

## Field reference

| Field | Description | Values |
|-------|-------------|--------|
| `sim` | Cosine similarity to best match | 0.0–1.0, `---` if no match |
| `top2` | Second-best similarity | 0.0–1.0, `---` if <2 candidates |
| `gap` | Margin (top1 - top2) | 0.0–1.0, `---` if unavailable |
| `q` | Face quality score | 0.0–1.0, `---` if unavailable |
| `dur` | Track duration in seconds | Integer |
| `conf` | Recognition confidence | 1–100 |
| `status` | Recognition/policy status | KNOWN, VERIFIED, UNCERTAIN, UNKNOWN, BLACKLIST |
| `alert` | Alert level | none, low, medium, high, critical |
| `vis` | Visit count | Integer |
| `frames` | Total frames track alive | Integer |
| `face` | Frames with detectable face | Integer |

## Status colors (ANSI)

| Status | Color |
|--------|-------|
| KNOWN, VERIFIED, AUTHORIZED, KNOWN_VISITOR | Green |
| UNCERTAIN | Yellow |
| UNKNOWN, BLACKLIST, MASKED_UNKNOWN, INTENTIONALLY_HIDDEN | Red |

## Example session

```
2026-07-09T07:02:13 CAM     started source=0 backend=dshow res=1280x720
2026-07-09T07:02:23 FACE    saved track=cam_01_1783580530_1 path=captures/...
2026-07-09T07:02:24 MATCH   aum        sim=0.608 top2=0.499 gap=0.109 verified=True tags=['auto_registered', 'verified']
2026-07-09T07:02:24 MEMORY  cam_01_... visits=218 known=True boost=18.0
2026-07-09T07:02:24 RECOG   aum        KNOWN        sim=0.608 top2=0.500 gap=0.109 q=0.70 dur=8s conf=71
2026-07-09T07:02:24 POLICY  VERIFIED     alert=none  ok vis=218
2026-07-09T07:03:13 FINAL   aum        VERIFIED     sim=0.592 top2=0.457 gap=0.136 conf=71  frames=13 face=1 vis=218
2026-07-09T07:03:13 VISIT   aum        recorded total=219
```

## See also
- [[3-Tier Logging]] — logging architecture
- [[Calculation Log]] — confidence formula breakdown
