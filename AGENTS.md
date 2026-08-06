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

| Task                       | Command                                                         |
| -------------------------- | --------------------------------------------------------------- |
| Run surveillance           | `python main.py`                                                |
| Run dashboard API          | FastAPI starts automatically on port 8000 inside `main.py`      |
| Run dashboard frontend     | `cd dashboard/frontend && npm run dev`                          |
| Install Python deps        | `pip install -r requirements.txt`                               |
| Install frontend deps      | `cd dashboard/frontend && npm install`                          |
| Export YOLO to OpenVINO IR | `yolo export model=models/yolov8s.pt format=openvino half=True` |
| Run tests                  | `python -m pytest tests/ -v`                                    |
| Run specific test          | `python -m pytest tests/test_recognition.py -v`                 |

## Configuration system

**Priority chain:** `.env` (secrets) > `config/config.jsonc` (tunables) > hardcoded defaults

- `config/config.jsonc` — Centralized tunable parameters (JSON with comments). Edit this file for detection, quality, and recognition settings.
- `.env` — Secrets only (API keys, URIs, passwords). Never put tunables here.
- `config/settings.py` — Loads both files via `_get(env_key, config_key, default, cast)`.

## Architecture

```
main.py (entry point, wires everything)
├── agents/camera_agent.py        — capture loop + ByteTrack + progressive recognition
├── agents/matching_agent.py      — embedding + MongoDB vector search
├── agents/memory.py              — visit history tracking
├── agents/alert_agent.py         — alert dispatch (console/email/sms/webhook)
├── agents/recognition.py         — multi-signal identity classification
├── agents/scoring.py             — confidence scoring (weighted normalization + logging)
├── agents/policy.py              — business rule evaluation + decide() entry point
├── agents/track_processor.py     — track finalization + dashboard broadcasting
├── agents/finalizer.py           — final embedding retry on track expiry
├── agents/timing.py              — thread-safe timing diagnostics collector
├── agents/report.py              — incident/stats/summary reports via LLM
├── agents/base.py                — BaseAgent abstract class (ABC)
├── pipeline/tracker.py           — YOLOv8 + ByteTrack (single model instance)
├── config/bytetrack_surveillance.yaml — ByteTrack params tuned for fixed-camera surveillance
├── pipeline/recognition_pipeline.py — orchestrates detect → quality → embed → match → decide
├── pipeline/quality_agent.py     — face quality scoring + compute_face_ratio()
├── pipeline/models.py            — Track, MatchResult, DecisionResult, QualityResult, etc.
├── pipeline/track_state.py       — per-track accumulation with threading.Lock
├── utils/db_client.py            — MongoDB connection singleton + collection getters
├── utils/db_faces.py             — face CRUD, deduplication, embedding history
├── utils/db_events.py            — event logging + stats queries
├── utils/db_memory.py            — visit memory CRUD
├── utils/db_search.py            — vector search (Atlas + Python fallback) + backfill
├── utils/db_utils.py             — re-export facade (all consumers import from here)
├── utils/embedding_utils.py      — InsightFace singleton (load once, never per-frame)
├── utils/llm_client.py           — Ollama HTTP client (generate, chat, NL summaries)
├── utils/image_utils.py          — crop, save, upload to Cloudinary
├── config/settings.py            — loads .env + config.jsonc, validate_config()
├── config/logging_setup.py       — Colors, CompactTerminalRenderer, JSONFileRenderer, setup_logging()
├── config/config.jsonc           — tunable parameters (edit this, not settings.py)
├── tests/                        — pytest test suite (84 tests)
└── dashboard/
    ├── backend/main.py           — FastAPI app (REST + WebSocket)
    └── frontend/                 — React + Vite
```

## Critical patterns

- **Load models once.** YOLO in `tracker.py`, InsightFace as singleton in `embedding_utils.py`. Never reload in per-frame loops.
- **Thread safety.** `TrackState` uses `threading.Lock`. Camera thread and worker pool (4 threads) run concurrently.
- **I/O decoupled from camera.** MongoDB, Cloudinary, alerts run via `queue.Queue` + workers. Camera loop must never block.
- **Threshold conversion.** Atlas `vectorSearchScore = (1+cosine)/2`. Always convert back: `raw_cosine = (atlas_score * 2) - 1` before comparing against `MATCH_THRESHOLD`. Use `compare_similarity()` in `utils/embedding_utils.py`.
- **Match threshold.** Keep `MATCH_THRESHOLD` ≤ 0.45. Default is 0.45. Higher rejects genuine same-person matches under indoor lighting.
- **One decision per track.** Recognition + decision runs once when track ends (or progressively every 20 frames). Not per-frame.
- **Composite track IDs.** Format: `{camera_id}_{session_epoch}_{byte_track_id}` — unique across camera restarts.

## Quality scoring

Two-tier system in `pipeline/quality_agent.py`:

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

- `axios` uses `^1.7.9` (compatible range) in `dashboard/frontend/` — versions ≥1.7.10 break Vite's esbuild
- Console shows INFO+ only; full debug logs go to `logs/surveillance.jsonl` and `logs/surveillance.debug.log`
- `FutureWarning` from insightface is harmless (deprecated `estimate` in 0.26)
- Camera streams via WebSocket, not a local OpenCV window — open browser to see feed
- Mask detection uses geometric landmark heuristic (lower_face/upper_face ratio < 0.3), not a classifier
- Blacklisted persons always trigger critical alert even if also authorized (tag priority: blacklist > authorized > known_visitor)
- LLM (Ollama) must be running for chat and NL summaries — system falls back to template strings if unavailable
- Swap LLM model via single env var: `OLLAMA_MODEL=qwen3.5:4b` in `.env` (Qwen 3.5 4B has better reasoning)
- Both Gemma 3 4B and Qwen 3.5 4B fit in 4GB VRAM at Q4_K_M quantization (~2.7GB weights)
- On Windows, if MSMF drops frames (error -1072875772), set `CAMERA_BACKEND=dshow` in `.env`

## Status system

Centralized in `config/status.py` — single source of truth for all identity status values.

```python
class Status(IntEnum):
    UNKNOWN = 1           # Unrecognized person
    UNCERTAIN = 2         # Weak match, low confidence
    KNOWN = 3             # Matched identity, confidence above threshold
    KNOWN_VISITOR = 4     # Confirmed returning visitor
    VERIFIED = 5          # Verified visitor (manual or high-similarity)
    AUTHORIZED = 6        # Employee / authorized person
    BLACKLIST = 7         # Blacklisted person (highest priority)
    MASKED_UNKNOWN = 8    # Masked / partial-visibility unknown
    HIDDEN = 9            # Intentionally hidden
```

- **Higher number = more trusted.** `is_known` becomes `status >= 3`.
- Numeric everywhere: MongoDB, API, logs, frontend.
- `alert_level` and `visibility` remain strings (separate concepts).
- Display labels via `STATUS_LABELS` dict; reverse lookup via `LABEL_TO_STATUS`.
- Migration script: `scripts/migrate_status_ints.py [--dry-run]`.

## Logging system

3-tier logging architecture in `config/logging_setup.py`:

- **Terminal** — Compact one-liner with ANSI colors (`CompactTerminalRenderer`)
- **JSON file** (`logs/surveillance.jsonl`) — Machine-readable JSON lines
- **Debug file** (`logs/surveillance.debug.log`) — Full verbose output

Events in `TERMINAL_ALLOWLIST` appear in terminal. Others go to files only.

Suppressed loggers: `pymongo`, `insightface` (WARNING), `ultralytics` (ERROR), `cloudinary` (WARNING).

See `docs/08 - Logging/Terminal Output Reference.md` for full event format reference.

## Recent fixes

- **2026-07-09**: Added weighted normalization confidence formula — 5 components (sim 65%, quality 15%, track 10%, memory 5%, margin 5%) in `agents/scoring.py`. Replaces old point-based formula.
- **2026-07-09**: Added calculation log — full tabular breakdown per confidence computation written to `logs/calculation.log`. Formula header written at startup.
- **2026-07-09**: Added quality-gated recognition throttle — skips recognition if face quality didn't improve by ≥ 0.10 (`MIN_QUALITY_IMPROVEMENT`). No timers or counters.
- **2026-07-09**: Added max-confidence gate — recognition never downgrades, only upgrades confidence. Critical alerts always update.
- **2026-07-09**: Fixed auto-registered unknowns being promoted to `known_visitor` — Policy Rule 5 mid-range match now returns `unknown` instead of `known_visitor`; memory `is_known` gate no longer includes `known_visitor` status.
- **2026-07-09**: Fixed `OFFICE_DAYS` env var type mismatch — `_get()` now casts list elements to the default's element type, so `OFFICE_DAYS=0,1,2,3,4` produces `[0,1,2,3,4]` (ints) not strings.
- **2026-07-09**: Fixed PyMongo `return_document=True` → `ReturnDocument.AFTER` in `get_or_create_memory()` and `update_visit_memory()`.
- **2026-07-09**: Added WebSocket origin validation to `/ws/live` endpoint to prevent cross-origin hijacking.
- **2026-07-09**: Added lock-protected setters for `cached_embedding` and `pending_recognition` in `TrackState`.
- **2026-07-09**: Fixed LLM client shutdown — replaced deprecated `asyncio.get_event_loop()` with `get_running_loop()` + `asyncio.run()` fallback.
- **2026-07-09**: Tightened CORS to explicit methods/headers.
- **2026-07-09**: Removed dead code (unused imports, dead fields, `get_embedding_for_track` method) and fixed `JPEG_QUALITY_BROADCAST` 50→90.
- **2026-07-30**: Added quality-gated recognition skip — `else: return` in `camera_agent.py:321` prevents storing/searching embeddings from low-quality (blurry/dark/small) faces.
- **2026-07-30**: Performance — unconditional `cv2.resize()` to 1280×720 before YOLO inference (`camera_agent.py:146`); `FRAME_SKIP` skips every other frame before JPEG encode (`track_processor.py:46`); WebSocket broadcast resized to 640×360 preview (`track_processor.py:52`).
- **2026-08-04**: Updated documentation — refreshed AGENTS.md with current architecture, added base.py to architecture tree, updated test count to 84.
- **2026-08-05**: Refactored codebase — deleted `pipeline/face.py` (inlined `compute_face_ratio()` into `quality_agent.py`), deleted `agents/decision_agent.py` (replaced with `decide()` in `policy.py`), extracted `config/logging_setup.py` from `settings.py`, split `utils/db_utils.py` (879 lines) into 5 domain modules (`db_client.py`, `db_faces.py`, `db_events.py`, `db_memory.py`, `db_search.py`) + re-export facade. Updated test mocks to target domain modules directly.
- **2026-08-05**: Status refactor — replaced all string statuses with centralized `Status(IntEnum)` enum in `config/status.py`. Updated 16 source files, 6 test files, and all MongoDB queries. Numeric everywhere (MongoDB, API, logs, frontend). Higher number = more trusted. Migration script: `scripts/migrate_status_ints.py`.

## Common entry points

| File                        | What it starts                                | How to run                        |
| --------------------------- | --------------------------------------------- | --------------------------------- |
| `main.py`                   | Full pipeline + API server                    | `python main.py`                  |
| `agents/camera_agent.py`    | Capture loop, recognition, track finalization | Loaded by main.py                 |
| `dashboard/backend/main.py` | FastAPI REST + WebSocket                      | Starts automatically on port 8000 |
| `pipeline/tracker.py`       | YOLO + ByteTrack model loading                | Loaded by camera_agent            |
| `config/settings.py`        | All configuration loading                     | Imported by every module          |

## Folder responsibilities

| Folder     | Purpose                     | Key files                                                                                                                                                          |
| ---------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| agents/    | AI orchestration (7 agents) | camera_agent, matching, recognition, scoring, policy, memory, alerts, report                                                                                       |
| pipeline/  | Computer vision pipeline    | tracker, models, track_state, quality_agent, recognition_pipeline                                                                                                  |
| utils/     | Shared utilities            | db_client, db_faces, db_events, db_memory, db_search, db_utils (facade), embedding_utils, llm_client, image_utils                                                 |
| config/    | Configuration               | settings.py, status.py, logging_setup.py, config.jsonc, bytetrack_surveillance.yaml                                                                                 |
| dashboard/ | Web dashboard               | backend/main.py, frontend/src/                                                                                                                                     |
| tests/     | 84 pytest tests             | test_recognition, test_recognition_pipeline, test_recognition_pipeline_stages, test_embedding_history, test_thread_safety, test_track_finalizer                    |
| scripts/   | Session query tools         | query\_\*.py (11 files)                                                                                                                                            |
| docs/      | Documentation               | ARCHITECTURE, 00-Home, 01-Getting Started, 02-Architecture, 03-Agents, 04-Pipeline, 05-Utilities, 06-Formulas, 07-Dashboard, 08-Logging, 09-Reference, 10-Problems |

## Coding rules

- **Load models once.** YOLO in `tracker.py`, InsightFace as singleton in `embedding_utils.py`. Never reload in per-frame loops.
- **Thread safety.** `TrackState` uses `threading.Lock`. Camera thread and worker pool (4 threads) run concurrently.
- **I/O decoupled from camera.** MongoDB, Cloudinary, alerts run via `queue.Queue` + workers. Camera loop must never block.
- **Quality gates before embedding storage.** Invalid faces (blur < 40, brightness outside 35-255, area < 1200px²) never get embeddings stored or searched.
- **Confidence never downgrades.** Only upgrades across recognition passes. Critical alerts always update.
- **Status is numeric.** Use `Status.X` from `config/status.py` — never raw strings. `is_known` = `status >= 3`.
- **Edit config/config.jsonc** for tunables (detection, quality, recognition). Never edit config/settings.py defaults.
- **Run tests before committing.** `python -m pytest tests/ -v`

## Do / Don'ts

Do:

- Edit `config/config.jsonc` for detection, quality, recognition settings
- Edit `.env` for secrets (MONGODB_URI, API keys, passwords)
- Run `python -m pytest tests/ -v` before committing
- Use `compare_similarity()` in `utils/embedding_utils.py` for threshold checks
- Convert Atlas scores: `raw_cosine = (atlas_score * 2) - 1`
- Use `track.best_face_crop` for numpy array (not `best_face_crop_path`)
- Use `Status.X` from `config/status.py` for all status comparisons

Don't:

- Edit `config/settings.py` hardcoded defaults
- Reload InsightFace in per-frame loops
- Block camera loop with I/O (MongoDB, Cloudinary, alerts)
- Set `MATCH_THRESHOLD` above 0.45
- Store embeddings from low-quality faces (quality gates block this)
- Use raw string statuses — always use `Status.X` enum values

## Current priorities

1. fix code issues and make the working correct.
