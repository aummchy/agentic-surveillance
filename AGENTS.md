# AGENTS.md — agentic_ai_singlecam

## What this is

AI-powered surveillance system: YOLOv8 person detection → ByteTrack tracking → InsightFace face recognition → autonomous decision engine → alerts. Python 3.11, MongoDB Atlas vector search, FastAPI + React dashboard.

## Quick start

```bash
# Terminal 1 — surveillance pipeline
python main.py

# Terminal 2 — dashboard
cd dashboard/frontend
npm install
npm run dev
```

Dashboard: http://localhost:5173. API: http://localhost:8000.

## Prerequisites

- Python 3.11 (InsightFace/onnxruntime wheels unreliable on other versions)
- MongoDB Atlas cluster with `surveillance` database, `faces`/`events`/`visit_memory` collections
- Atlas Vector Search index named `vector_index` on `faces.latest_embedding` (512 dims, cosine)
- `.env` with at minimum `MONGODB_URI` (copy from `.env.example`)
- Camera device at `CAMERA_INDEX=0` (or adjust in `.env`)

## Commands

| Task | Command |
|------|---------|
| Run surveillance | `python main.py` |
| Run dashboard API | FastAPI starts automatically on port 8000 inside `main.py` |
| Run dashboard frontend | `cd dashboard/frontend && npm run dev` |
| Install Python deps | `pip install -r requirements.txt` |
| Install frontend deps | `cd dashboard/frontend && npm install` |

No test suite, linter, or CI pipeline exists in this repo.

## Architecture

```
main.py (entry point, wires everything)
├── agents/camera_agent.py    — capture loop + ByteTrack + progressive recognition
├── agents/matching_agent.py  — embedding + MongoDB vector search
├── agents/decision_agent.py  — delegates to PolicyAgent
├── agents/memory.py          — visit history tracking
├── agents/alert_agent.py     — alert dispatch (console/email/sms/webhook)
├── pipeline/tracker.py       — YOLOv8 + ByteTrack (single model instance)
├── pipeline/face.py          — SCRFD detection + ArcFace embedding + mask heuristic
├── pipeline/models.py        — Track, MatchResult, DecisionResult, etc.
├── pipeline/track_state.py   — per-track accumulation with threading.Lock
├── utils/db_utils.py         — MongoDB CRUD + vector search + Python fallback
├── utils/embedding_utils.py  — InsightFace singleton (load once, never per-frame)
├── utils/image_utils.py      — crop, save, upload to Cloudinary
├── config/settings.py        — loads .env, validate_config()
└── dashboard/
    ├── backend/main.py       — FastAPI app (REST + WebSocket)
    └── frontend/             — React + Vite
```

## Critical patterns

- **Load models once.** YOLO in `tracker.py`, InsightFace as singleton in `embedding_utils.py`. Never reload in per-frame loops.
- **Thread safety.** `TrackState` uses `threading.Lock`. Camera thread and worker pool (2 threads) run concurrently.
- **I/O decoupled from camera.** MongoDB, Cloudinary, alerts run via `queue.Queue` + workers. Camera loop must never block.
- **Threshold conversion.** Atlas `vectorSearchScore = (1+cosine)/2`. Always convert back: `raw_cosine = (atlas_score * 2) - 1` before comparing against `MATCH_THRESHOLD`. Use `compare_similarity()` in `db_utils.py`.
- **Match threshold.** Keep `MATCH_THRESHOLD` ≤ 0.45. Default is 0.25. Higher rejects genuine same-person matches under indoor lighting.
- **One decision per track.** Recognition + decision runs once when track ends (or progressively every 20 frames). Not per-frame.
- **Composite track IDs.** Format: `{camera_id}_{session_epoch}_{byte_track_id}` — unique across camera restarts.

## Known issues (4 remaining)

| # | Issue | Location |
|---|-------|----------|
| 9 | Progressive recognition blocks camera loop (sync InsightFace + MongoDB on camera thread) | `camera_agent.py:79-83` |
| 14 | `_python_cosine_scan` silently caps at 500 records with no log warning | `db_utils.py:116` |
| 19 | `backfill_missing_embeddings` full collection scan on every startup | `db_utils.py:641` |
| 23 | `memory_agent.run()` blocks camera thread (MongoDB query) | `camera_agent.py:200` |

## Gotchas

- `axios` pinned to `1.7.9` in `dashboard/frontend/` — versions ≥1.7.10 break Vite's esbuild
- Console shows INFO+ only; full debug logs go to `logs/surveillance.log` (5MB × 5 backups)
- `FutureWarning` from insightface is harmless (deprecated `estimate` in 0.26)
- Camera streams via WebSocket, not a local OpenCV window — open browser to see feed
- Mask detection uses geometric landmark heuristic (lower_face/upper_face ratio < 0.3), not a classifier
- Blacklisted persons always trigger critical alert even if also authorized (tag priority: blacklist > authorized > known_visitor)
- `agent.md` is the build specification (architecture, data contracts, acceptance criteria), not an OpenCode config file
