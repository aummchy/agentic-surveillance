# AGENTS.md — agentic_ai_singlecam

## What this is

AI-powered surveillance system: YOLOv8 person detection → ByteTrack tracking → InsightFace face recognition → autonomous decision engine → alerts + **local LLM** for NL summaries and conversational dashboard. Python 3.11, MongoDB Atlas vector search, FastAPI + React dashboard, Ollama (Gemma 3 4B / Qwen 3.5 4B).

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
- Camera device at `CAMERA_INDEX=0` or `CAMERA_SOURCE=http://<phone-ip>:8080/video` for mobile (adjust in `.env`)
- For Intel Arc iGPU acceleration: `pip install openvino` and export YOLO (optional — CPU works too)

## Commands

| Task | Command |
|------|---------|
| Run surveillance | `python main.py` |
| Run dashboard API | FastAPI starts automatically on port 8000 inside `main.py` |
| Run dashboard frontend | `cd dashboard/frontend && npm run dev` |
| Install Python deps | `pip install -r requirements.txt` |
| Install frontend deps | `cd dashboard/frontend && npm install` |
| Export YOLO to OpenVINO IR | `yolo export model=models/yolov8s.pt format=openvino half=True` |
| Run tests | `python -m pytest tests/ -v` |
| Run specific test | `python -m pytest tests/test_recognition.py -v` |

## Configuration system

**Priority chain:** `.env` (secrets) > `config/config.jsonc` (tunables) > hardcoded defaults

- `config/config.jsonc` — Centralized tunable parameters (JSON with comments). Edit this file for detection, quality, and recognition settings.
- `.env` — Secrets only (API keys, URIs, passwords). Never put tunables here.
- `config/settings.py` — Loads both files via `_get(env_key, config_key, default, cast)`.

## Architecture

```
main.py (entry point, wires everything)
├── agents/camera_agent.py    — capture loop + ByteTrack + progressive recognition
├── agents/matching_agent.py  — embedding + MongoDB vector search
├── agents/decision_agent.py  — delegates to PolicyAgent
├── agents/memory.py          — visit history tracking
├── agents/alert_agent.py     — alert dispatch (console/email/sms/webhook)
├── agents/recognition.py     — multi-signal identity classification
├── agents/policy.py          — business rule evaluation
├── pipeline/tracker.py       — YOLOv8 + ByteTrack (single model instance)
├── pipeline/face.py          — SCRFD detection + ArcFace embedding + mask heuristic
├── pipeline/models.py        — Track, MatchResult, DecisionResult, etc.
├── pipeline/track_state.py   — per-track accumulation with threading.Lock
├── utils/db_utils.py         — MongoDB CRUD + vector search + Python fallback
├── utils/embedding_utils.py  — InsightFace singleton (load once, never per-frame)
├── utils/llm_client.py       — Ollama HTTP client (generate, chat, NL summaries)
├── utils/image_utils.py      — crop, save, upload to Cloudinary
├── config/settings.py        — loads .env + config.jsonc, validate_config()
├── config/config.jsonc       — tunable parameters (edit this, not settings.py)
├── tests/                    — pytest test suite
└── dashboard/
    ├── backend/main.py       — FastAPI app (REST + WebSocket)
    └── frontend/             — React + Vite
```

## Critical patterns

- **Load models once.** YOLO in `tracker.py`, InsightFace as singleton in `embedding_utils.py`. Never reload in per-frame loops.
- **Thread safety.** `TrackState` uses `threading.Lock`. Camera thread and worker pool (2 threads) run concurrently.
- **I/O decoupled from camera.** MongoDB, Cloudinary, alerts run via `queue.Queue` + workers. Camera loop must never block.
- **Threshold conversion.** Atlas `vectorSearchScore = (1+cosine)/2`. Always convert back: `raw_cosine = (atlas_score * 2) - 1` before comparing against `MATCH_THRESHOLD`. Use `compare_similarity()` in `db_utils.py`.
- **Match threshold.** Keep `MATCH_THRESHOLD` ≤ 0.45. Default is 0.45. Higher rejects genuine same-person matches under indoor lighting.
- **One decision per track.** Recognition + decision runs once when track ends (or progressively every 10 frames). Not per-frame.
- **Composite track IDs.** Format: `{camera_id}_{session_epoch}_{byte_track_id}` — unique across camera restarts.

## Quality scoring

Two-tier system in `pipeline/face.py`:

1. **Validity gates** — Reject unusable faces (too blurry, too dark, too small):
   - Blur: Laplacian variance ≥ `QUALITY_VALID_BLUR_MIN` (40)
   - Brightness: V-channel mean in [35, 255]
   - Face area: ≥ `QUALITY_VALID_FACE_AREA_MIN` (1200 px²)

2. **Quality score** — Weighted composite [0, 1] for embedding comparison:
   - Blur (50%): `min(Laplacian_var / 350, 1.0)`
   - Brightness (25%): center-radius model (peak at 145, radius 110)
   - Area (25%): `min(area / 10000, 1.0)`

Quality gates prevent low-quality embeddings from overwriting high-quality ones in MongoDB.

## Gotchas

- `axios` pinned to `1.7.9` in `dashboard/frontend/` — versions ≥1.7.10 break Vite's esbuild
- Console shows INFO+ only; full debug logs go to `logs/surveillance.jsonl` and `logs/surveillance.debug.log`
- `FutureWarning` from insightface is harmless (deprecated `estimate` in 0.26)
- Camera streams via WebSocket, not a local OpenCV window — open browser to see feed
- Mask detection uses geometric landmark heuristic (lower_face/upper_face ratio < 0.3), not a classifier
- Blacklisted persons always trigger critical alert even if also authorized (tag priority: blacklist > authorized > known_visitor)
- LLM (Ollama) must be running for chat and NL summaries — system falls back to template strings if unavailable
- Swap LLM model via single env var: `OLLAMA_MODEL=qwen3.5:4b` in `.env` (Qwen 3.5 4B has better reasoning)
- Both Gemma 3 4B and Qwen 3.5 4B fit in 4GB VRAM at Q4_K_M quantization (~2.7GB weights)
- On Windows, if MSMF drops frames (error -1072875772), set `CAMERA_BACKEND=dshow` in `.env`

## Logging system

3-tier logging architecture in `config/settings.py`:

- **Terminal** — Compact one-liner with ANSI colors (`CompactTerminalRenderer`)
- **JSON file** (`logs/surveillance.jsonl`) — Machine-readable JSON lines
- **Debug file** (`logs/surveillance.debug.log`) — Full verbose output

Events in `TERMINAL_ALLOWLIST` appear in terminal. Others go to files only.

Suppressed loggers: `pymongo`, `insightface` (WARNING), `ultralytics` (ERROR), `cloudinary` (WARNING).

See `TERMINAL_OUTPUT.md` for full event format reference.

## Recent fixes

- **2026-07-09**: Fixed auto-registered unknowns being promoted to `known_visitor` — Policy Rule 5 mid-range match now returns `unknown` instead of `known_visitor`; memory `is_known` gate no longer includes `known_visitor` status. Fixes permanent feedback loop where repeated sightings cemented incorrect classification.
- **2026-07-09**: Fixed `OFFICE_DAYS` env var type mismatch — `_get()` now casts list elements to the default's element type, so `OFFICE_DAYS=0,1,2,3,4` produces `[0,1,2,3,4]` (ints) not strings.
- **2026-07-09**: Fixed PyMongo `return_document=True` → `ReturnDocument.AFTER` in `get_or_create_memory()` and `update_visit_memory()`.
- **2026-07-09**: Added WebSocket origin validation to `/ws/live` endpoint to prevent cross-origin hijacking.
- **2026-07-09**: Added lock-protected setters for `cached_embedding` and `pending_recognition` in `TrackState`.
- **2026-07-09**: Fixed LLM client shutdown — replaced deprecated `asyncio.get_event_loop()` with `get_running_loop()` + `asyncio.run()` fallback.
- **2026-07-09**: Tightened CORS to explicit methods/headers.
