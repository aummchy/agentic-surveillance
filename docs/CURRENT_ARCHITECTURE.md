# Current Architecture

> **Purpose:** Describe the architecture that currently exists in the codebase.
>
> This document is a snapshot of the current implementation, not the target architecture.
>
> **Source of truth:** Current source code and tests. If this document conflicts with the implementation, update this document rather than changing code to match it.
>
> **Last verified:** 2026-10-03
>
> **Not a refactor plan.** "Concerns" and "duplications" below are factual observations only. What is approved to change, and when, lives in `plan.md` and `docs/REFACTOR_PLAN.md`.

## Files in this document

The document is split into six linked files. Section numbers follow one continuous numbering scheme.

| File | Sections |
|------|----------|
| **`docs/CURRENT_ARCHITECTURE.md`** (this file) | 1–3, 25–26, 31 + Appendices A–D |
| [`Current Architecture - Components.md`](02%20-%20Architecture/Current%20Architecture%20-%20Components.md) | 4–7 — major components, detection/tracking, track identity, track state |
| [`Current Architecture - Recognition.md`](02%20-%20Architecture/Current%20Architecture%20-%20Recognition.md) | 8–14 — the two recognition paths, pipeline, quality, embedding, matching, classification, scoring |
| [`Current Architecture - Decision Flow.md`](02%20-%20Architecture/Current%20Architecture%20-%20Decision%20Flow.md) | 15–20 — policy, finalization, finalizer, memory, database, alerts |
| [`Current Architecture - Platform.md`](02%20-%20Architecture/Current%20Architecture%20-%20Platform.md) | 21–24 — dashboard, LLM, configuration, status model |
| [`Current Architecture - Code State.md`](02%20-%20Architecture/Current%20Architecture%20-%20Code%20State.md) | 27–30 — known duplication, concerns, fixed issues, tests |

---

# 1. System Overview

## What the system does

This project is an AI-powered single-camera surveillance system.

At a high level, it:

1. Captures frames from a camera, video file, or stream.
2. Detects people using YOLO.
3. Tracks people across frames using ByteTrack.
4. Maintains per-track state.
5. Selects face observations for recognition.
6. Detects and evaluates face quality.
7. Generates face embeddings using InsightFace.
8. Searches known identities using MongoDB Atlas Vector Search.
9. Combines recognition signals and memory.
10. Produces an identity/recognition result.
11. Applies policy rules.
12. Creates alerts/events when required.
13. Stores relevant information in MongoDB.
14. Streams surveillance information to the dashboard.
15. Uses a local LLM for natural-language summaries and dashboard interaction.

Every one of these steps exists in the current code. The components that own each step are listed in section 4 onward.

---

# 2. High-Level Runtime Flow

```text
Camera / Video / Stream
        |
        v
   Frame Capture  (camera thread: resize to 1280x720)
        |
        v
   YOLO Detection  (pipeline/tracker.py)
        |
        v
     ByteTrack
        |
        v
    Track State  (pipeline/track_state.py, composite IDs, generation suffix)
        |
        +--------------------------+
        |                          |
        |  Progressive Recognition |   every RECOGNITION_INTERVAL_FRAMES (20)
        |  (recognition pool,      |   and only if quality improved by
        |   RECOGNITION_MAX_WORKERS|   MIN_QUALITY_IMPROVEMENT (0.10)
        |   = 4 threads)           |
        v                          |
 Recognition Pipeline              |
        |                          |
        v                          |
 Face Detection                    |
        |                          |
        v                          |
 Face Quality                      |
        |                          |
        v                          |
 Face Embedding                    |
        |                          |
        v                          |
 Vector Search                     |
        |                          |
        v                          |
 Memory / Recognition              |
        |                          |
        v                          |
 Policy Decision                   |
        |                          |
        +-------------+------------+
                      |
                      v
               Track Finalization     (camera worker: final embedding retry,
                      |                then enqueue on track_queue)
                      v
          Final recognition pass      (ONE of TWO queue-consumer threads:
          + events / alerts / storage  TrackProcessor.process)
                      |
             +--------+--------+
             |                 |
             v                 v
         MongoDB           Dashboard (REST + WebSocket)
```

> **Important:** The diagram above describes the current implementation. Recognition has **two execution paths** — progressive (mid-track) and finalization (end-of-track). This duplication is real and verified; it is documented in [Section 8](02%20-%20Architecture/Current%20Architecture%20-%20Recognition.md#8-current-recognition-architecture).

> **Correction note:** An earlier version of this document claimed the track queue had ONE consumer. The code starts **two** (`runtime/track_workers.py`, `NUM_WORKERS = 2`). The inventory in section 25 reflects the code.

---

# 3. Application Entry Points

## `main.py`

### Responsibility

Thin composition root: validates configuration, wires the components in a fixed phase order, starts the camera (which blocks), then runs the shutdown sequence. Extracted 2026-10-03 into the `runtime/` package.

### Currently responsible for

* Phase 1: config validation (`settings.validate_config()`), formula header, startup log.
* Phase 2/5/6: starting the background one-shot tasks — **delegated to `runtime/background.py`** (`start_llm_check`, `start_mongo_checks`, `start_prewarms`).
* Phase 3: creating the track queue + its **two** consumer threads — **`runtime/track_workers.py`** (`TrackWorkers.start`, late-bound `process_fn`).
* Phase 4: starting the dashboard/API (asyncio loop + Uvicorn on a daemon thread) — **`runtime/api_server.py`** (`ApiServer.start`).
* Phase 7: constructing `TrackProcessor` then `CameraAgent`, registering the `handle_track_finalized` / `handle_frame_annotated` closures.
* Phase 8: `camera.start()` (blocks until Ctrl+C).
* Phase 9: shutdown orchestration in fixed order — `workers.stop()` → `api.request_exit()` → `workers.drain()` → `track_processor.shutdown()` → alert/LLM/Mongo close → `api.finalize()` (join thread + close loop).

### `runtime/` package (extracted 2026-10-03)

| Module | Owns |
|--------|------|
| `runtime/background.py` | LLM availability probe, Atlas index check + embedding backfill, YOLO/InsightFace prewarm — each on its own daemon thread |
| `runtime/api_server.py` | `ApiServer`: asyncio loop, Uvicorn config/server, daemon thread; two-phase stop (`request_exit` / `finalize`) |
| `runtime/track_workers.py` | `TrackWorkers`: `queue.Queue` + 2 consumer threads, worker loop (`_worker_loop`), `stop()`/`drain()` |

### Important dependencies

* `agents.camera_agent` (capture loop, progressive recognition)
* `agents.track_processor` (finalization sequence)
* `runtime/` package (lifecycle concerns above)
* configuration system (`config/settings.py`)
* dashboard backend (`dashboard/backend/main.py`)
* logging system (`config/logging_setup.py`)

### Notes

`main.py` wires the system; it does not contain the processing logic. The per-frame work lives in `camera_agent`/`tracker`, the end-of-track work lives in `track_processor`, the API lives under `dashboard/backend/`, and process lifecycle lives in `runtime/`.

---

# 25. Concurrency Model

## Thread inventory (verified from source)

| # | Thread(s) | Where started | What it does |
|---|-----------|---------------|--------------|
| 1 | Main thread | process entry | starts everything, then waits for shutdown |
| 2 | Camera thread | `camera_agent.start()` | capture → detect → track → update state → schedule recognition → enqueue finalizations |
| 4 | Recognition pool (`RECOGNITION_MAX_WORKERS`, default 4) | `camera_agent.py:87-88` | progressive recognition passes; finalization work is also submitted here |
| **2** | Track queue consumers | **`runtime/track_workers.py` `NUM_WORKERS = 2`** | `TrackProcessor.process(track)` — final recognition, registration, alerts, events, broadcasts |
| 2 | JPEG encode pool | `track_processor.py:95-96` | preview JPEG encoding (`thread_name_prefix="jpeg"`) |
| 2 | Alert executor | `alert_agent.py:18` | email / SMS / webhook alert channels (`thread_name_prefix="alert"`) |
| 1 | API thread (daemon) | `runtime/api_server.py` `ApiServer.start()` | Uvicorn + asyncio loop for REST and `/ws/live` |
| 4 | One-shot daemon helpers | `runtime/background.py` (`start_llm_check`, `start_mongo_checks`, `start_prewarms`) | LLM background check, startup checks, YOLO prewarm, InsightFace prewarm |

Conceptual picture:

```text
                    Camera Thread
                         |
                         v
                 Frame / Tracking
                         |
                         v
                    TrackState  (TrackState._lock -> track._lock)
                         |
              +----------+----------+
              |                     |
              v                     v
     Recognition Workers      Finalization path
      (4 threads)            retry_embedding (same pool)
              |                     |
              v                     v
      RecognitionPipeline     track_queue (queue.Queue)
                                       |
                                       v
                              2 Queue-Consumer Threads
                              TrackProcessor.process
                                       |
                                       v
                              JPEG pool (2) / Alert executor (2)
                              / asyncio loop (broadcasts)
```

### Important concurrency-sensitive state

* `TrackState` dictionary + per-track `Track` fields — guarded by `TrackState._lock` and `track._lock`.
* **Lock ordering rule:** `TrackState._lock` may be taken before `track._lock`, never the reverse.
* `pending_recognition` / `pending_match_result` / `pending_memory_context` — lock-protected setters with upgrade-only rules (weaker results never replace stronger ones).
* `_recognizing_tracks` and `_finalized_track_ids` — camera-side bookkeeping that prevents duplicate recognition scheduling and duplicate finalization. `_recognizing_tracks.add()` runs **inside the worker** (`camera_agent.py:538`), not at submit time; the queue-time check alone is not airtight, so the worker-start resolved-check is the real backstop.
* Alert cooldown cache (`agents/alert_agent._alert_timestamps`) — module-global dict, pruned by age.

### Current protection (verified)

* Duplicate recognition scheduling has guards (set membership + `pending_recognition` check).
* Duplicate finalization has guards (`_finalized_track_ids` add/discard protocol, `camera_agent.py:438-439, 734-735, 768`).
* Pending results are upgrade-only (lock-protected setters in `TrackState`).
* Confidence never downgrades across recognition passes (max-upgrade gate lives in `TrackState` setters and `camera_agent` write-back, not in `scoring.py`).

---

# 26. External I/O

The camera loop must not block on slow external operations.

| Operation | Where it runs (verified) |
|-----------|--------------------------|
| MongoDB reads/writes (vector search, faces, events, visit memory) | recognition workers + the **2 queue-consumer threads** — never the camera thread |
| Cloudinary image upload (`resolve_track_image_url` / `resolve_track_person_crop_url`) | recognition worker (`camera_agent.py:678`, mid-track alert path) or queue consumer (`track_processor.py:182-183`) |
| Console alert | synchronous on the calling worker (`alert_agent.py:101-106`) |
| Email / SMS / webhook alerts | alert executor pool (`alert_agent.py:18`), submitted via `_alert_executor.submit` |
| LLM generation (summaries, chat, reports) | inline on the calling worker or route thread; optional — template fallbacks when Ollama is unavailable |
| WebSocket broadcasts | posted onto the asyncio loop with `run_coroutine_threadsafe`, only while the loop is alive |
| JPEG encode for preview | 2-thread JPEG pool (`track_processor.py:95-96`), outside track locks |

Queues: one `queue.Queue` (`TrackWorkers.queue`, owned by `runtime/track_workers.py`) decouples finalization from the camera path; it has **two** consumers (section 25).

> The exact ownership of every queue/worker was verified for the operations listed above. Anything not listed here is `UNKNOWN — NEEDS VERIFICATION`.

---

# 31. Current Architecture — Summary

The system currently has a coherent overall surveillance pipeline:

```text
Capture
  ↓
Detection
  ↓
Tracking
  ↓
Track State
  ↓
Recognition        (two execution paths: progressive + finalization)
  ↓
Identity / Memory
  ↓
Policy
  ↓
Events / Alerts / Storage
  ↓
Dashboard
```

The main architectural difficulty is not that the required components are missing.

The main difficulty is that responsibilities currently overlap, particularly around:

```text
Recognition        (RecognitionPipeline.run vs TrackProcessor.process)
Track finalization (camera worker + queue consumers + finalizer.py)
Track state        (one dataclass holding tracking/face/identity/decision/presentation state)
Persistence        (db_utils facade hides domain-module ownership)
```

The most important verified duplication is between the progressive recognition path (Section 8.1) and the finalization recognition path (Section 8.2).

No target architecture is defined by this document. Future architectural changes are documented separately in `docs/REFACTOR_PLAN.md` and `plan.md`, and must be evaluated against the current implementation and tests before being implemented.

---

# Appendix A — Tech stack

| Component | Technology | Where |
|-----------|-----------|-------|
| Person detection | YOLOv8 `yolov8s` (PyTorch or OpenVINO IR) | `pipeline/tracker.py` singleton |
| Multi-object tracking | ByteTrack, `config/bytetrack_surveillance.yaml` | `track_high_thresh=0.45`, `track_buffer=60`, `new_track_thresh=0.50` |
| Face detection | InsightFace SCRFD (`buffalo_l`), CLAHE first | `utils/embedding_utils.py` singleton |
| Face embedding | ArcFace, 512-dim L2-normalized | quality-gated: overwrites only if `det_score` improves by ≥ 0.05 |
| Mask detection | Geometric landmark heuristic, no classifier | lower/upper face ratio < `MASK_RATIO_THRESHOLD` (0.3) |
| Face quality | Two-tier: validity gates then weighted score | `pipeline/quality_agent.py` |
| Vector search | MongoDB Atlas `$vectorSearch`, index `vector_index`, 512d cosine | `utils/db_search.py` + Python cosine fallback |
| Database | MongoDB Atlas, 3 collections | `utils/db_*` modules |
| Image storage | Cloudinary with local `captures/` fallback | `utils/image_utils.py` |
| Local LLM | Ollama (Gemma 3 4B / Qwen 3.5 4B), `httpx` | `utils/llm_client.py`, optional |
| Dashboard API | FastAPI + Uvicorn, port 8000 | `dashboard/backend/` |
| Dashboard UI | React + Vite, port 5173 | `dashboard/frontend/` |
| Configuration | `.env` + `config/config.jsonc` | `config/settings.py` |
| Logging | structlog, 3 tiers | `config/logging_setup.py` |

Key libraries: `ultralytics`, `insightface`, `onnxruntime`, `opencv-python`, `pymongo`, `fastapi`, `uvicorn`, `structlog`, `numpy`, `httpx`, `requests`, `python-dotenv`, `cloudinary`.

Hardware acceleration: YOLO on CPU by default; OpenVINO IR export for Intel Arc iGPU; CUDA via `YOLO_DEVICE=0`. InsightFace runs on CPU (execution provider not forced).

---

# Appendix B — Key thresholds (read from config)

> Formerly "section 6" of this document. Values read from `config/config.jsonc` (effective values — the file wins over `settings.py` defaults; see Section 23).

| Setting | Value | Note |
|---------|-------|------|
| `MATCH_THRESHOLD` | 0.45 | never raise above 0.45; Atlas scores convert first (`raw_cosine = (atlas_score * 2) - 1`) |
| `HIGH_CONFIDENCE_SIMILARITY` | 0.85 | skips memory + recognition |
| `CONFIDENCE_KNOWN_MIN` | 70 | policy gate for "known" |
| `CONFIDENCE_UNCERTAIN_MIN` | 55 | policy gate for "uncertain" |
| `KNOWN_VISITOR_CONFIDENCE` | 80 | **paired threshold under review** — differs from `CONFIDENCE_KNOWN_MIN` (concern §28.5) |
| `KNOWN_VISITOR_SIMILARITY` | 0.85 | auto-escalate to known_visitor |
| `VERIFIED_SIMILARITY_THRESHOLD` | 0.75 | verified-person path |
| `AUTO_REGISTERED_SIMILARITY` | 0.65 | self-match threshold |
| `DEDUP_SIMILARITY_THRESHOLD` | 0.45 | auto-registration dedup |
| `EMBEDDING_DET_SCORE_MIN` | 0.40 | min score to embed |
| `DET_SCORE_RELAXED` | 0.20 | face-detection entry gate |
| `FACE_SCORE_IMPROVEMENT_MIN` | 0.03 | best-face hysteresis |
| `EMBEDDING_DET_SCORE_IMPROVEMENT_MIN` | 0.05 | per-track embedding overwrite |
| `MIN_QUALITY_IMPROVEMENT` | 0.10 | progressive-recognition throttle |
| Quality validity gates | blur 40 / brightness 35–255 / area 1200 px² | reject before embedding |
| Quality scoring weights | blur 50% / brightness 25% / area 25% | composite score [0,1] |
| Confidence weights | sim 65% / quality 15% / track 10% / memory 5% / margin 5% | `settings.WEIGHT_*`, applied in `agents/scoring.py`; mask penalty ×(1 − 0.15×mask) |
| `TRACK_TIMEOUT_SECS` | 15.0 | person gone this long ends the track (`settings.py` default is 3.0 — see §23 duplication) |
| `MAX_TRACK_SECS` | 300.0 | hard cap on track lifetime |
| `RECOGNITION_INTERVAL_FRAMES` | 20 | progressive recognition cadence |
| `RECOGNITION_MAX_WORKERS` | 4 | recognition pool size |
| `FRAME_SKIP` | 2 | preview frames dropped |
| `BROADCAST_WIDTH/HEIGHT` | 960 × 540 | preview broadcast size |
| `JPEG_QUALITY_BROADCAST` | 85 | preview JPEG quality (`settings.py` default is 65 — see §23 duplication) |
| `ALERT_COOLDOWN_SECS` | 60 | per-level dedup window |
| `LOITER_SECS` | 30 | loitering detection |
| `OVERLAP_IOU_THRESHOLD` | 0.5 | track overlap dedup |
| `MIN_VISIT_GAP_SECS` | 60 | visit dedup window |
| `MASK_RATIO_THRESHOLD` | 0.3 | geometric mask heuristic |

---

# Appendix C — MongoDB collections

| Collection | Vector index | Key fields |
|-----------|--------------|------------|
| `faces` | `vector_index` on `latest_embedding` (512d cosine) | person_id, name, role, tags, verified, alert_level, images[], embeddings[], latest_embedding, mean_embedding, latest_embedding_quality, embedding_model, person_crop_url, source, quality_scores, verified_at, verified_by, created_at, updated_at |
| `events` | none | track_id, camera_id, timestamp, status, alert_level, person_id, similarity_score, name, is_masked, image_url, person_crop_url, reason, alerted |
| `visit_memory` | none | person_id (unique), visit_count, typical_hours[], typical_cameras[], avg_similarity, first_seen, last_seen, last_camera, last_status, best_status, similarity_history[], status_history[], created_at, updated_at |

Detailed field semantics: `docs/02 - Architecture/MongoDB Schema.md`.

---

# Appendix D — Document map

| Document | Role |
|----------|------|
| `AGENTS.md` | Operating contract for AI agents (workflow, invariants, do/don't) |
| `docs/CURRENT_ARCHITECTURE.md` | **this file** — hub: overview, runtime flow, entry points, concurrency, I/O, appendices |
| `docs/02 - Architecture/Current Architecture - *.md` | the five detail files linked from the top of this page |
| `docs/ARCHITECTURE_RULES.md` | semantic boundaries (what must not change) |
| `docs/REFACTOR_PLAN.md` | approval-gated change candidates |
| `docs/02 - Architecture/Data Flow.md` | per-frame step-by-step walkthrough |
| `docs/02 - Architecture/Thread Architecture.md` | every thread and its duties in depth |
| `docs/02 - Architecture/MongoDB Schema.md` | collections and indexes in depth |
| `docs/04 - Pipeline/Data Models.md` | data/state at each pipeline stage |
| `plan.md` | phased roadmap this document belongs to |
