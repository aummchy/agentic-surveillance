# Code Reference for Refactoring

> AI-powered surveillance: YOLOv8 → ByteTrack → InsightFace → Decision Engine → Alerts + LLM

---

## Table of Contents

- [System Data Flow](#system-data-flow)
- [Thread Architecture](#thread-architecture)
- [MongoDB Collections](#mongodb-collections)
- [File-by-File Reference](#file-by-file-reference)
  - [Entry Point](#mainpy--entry-point)
  - [Configuration](#configsettingspy--configuration)
  - [Logging Setup](#configlogging_setuppy--logging-setup)
  - [Agents](#agents)
    - [base.py](#agentsbasepy--abstract-base-agent)
    - [camera_agent.py](#agentscamera_agentpy--camera-loop)
    - [track_processor.py](#agentstrack_processorpy--track-post-processing)
    - [matching_agent.py](#agentsmatching_agentpy--vector-search)
    - [policy.py](#agentspolicypy--business-rules--decide-entry-point)
    - [recognition.py](#agentsrecognitionpy--identity-classification)
    - [scoring.py](#agentsscoringpy--confidence-scoring)
    - [memory.py](#agentsmemorypy--visit-history)
    - [alert_agent.py](#agentsalert_agentpy--alert-dispatch)
    - [finalizer.py](#agentsfinalizerpy--embedding-retry)
    - [timing.py](#agentstimingpy--timing-diagnostics)
    - [report.py](#agentsreportpy--report-generation)
  - [Pipeline](#pipeline)
    - [tracker.py](#pipelinetrackerpy---yolo--bytetrack)
    - [recognition_pipeline.py](#pipelinerecognition_pipelinepy--orchestrator)
    - [quality_agent.py](#pipelinequality_agentpy--face-quality--compute_face_ratio)
    - [models.py](#pipelinemodelspy--data-models)
    - [track_state.py](#pipelinetrack_statepy--track-state-management)
  - [Utilities](#utilities)
    - [db_client.py](#utilsdb_clientpy--connection-singleton)
    - [db_faces.py](#utilsdb_facespy--face-crud)
    - [db_events.py](#utilsdb_eventspy--event-logging)
    - [db_memory.py](#utilsdb_memorypy--visit-memory)
    - [db_search.py](#utilsdb_searchpy--vector-search)
    - [db_utils.py](#utilsdb_utilspy--re-export-facade)
    - [embedding_utils.py](#utilsembedding_utilspy--insightface)
    - [llm_client.py](#utilsllm_clientpy--ollama)
    - [image_utils.py](#utilsimage_utilspy--image-processing)
  - [Dashboard](#dashboard)
    - [main.py](#dashboardbackendmainpy--fastapi)
    - [routes/faces.py](#dashboardbackendroutesfacespy--face-crud)
    - [routes/events.py](#dashboardbackendrouteseventspy--events-api)
    - [routes/live.py](#dashboardbackendrouteslivepy--websocket)
    - [routes/reports.py](#dashboardbackendroutesreportspy--reports-api)
    - [routes/chat.py](#dashboardbackendrouteschatpy--chat-api)
- [Dependency Map](#dependency-map)
- [Refactoring Observations](#refactoring-observations)
- [Project Metrics](#project-metrics)
  - [Summary](#summary)
  - [By Directory](#by-directory)
  - [Top 10 Largest Files](#top-10-largest-files)
  - [Function Inventory](#function-inventory)
- [Refactoring Priority Matrix](#refactoring-priority-matrix)
- [Module Responsibility Map](#module-responsibility-map)

---

## System Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          main.py                                    │
│  (orchestrator: wires components, manages lifecycle, runs loop)     │
└─────────────────────────────────────────────────────────────────────┘
        │                          │                          │
        ▼                          ▼                          ▼
  CameraAgent                FastAPI (8000)             Background Threads
  (main thread)             (daemon thread)          (LLM, DB checks, model warmup)
        │                          │
        │                   ┌──────┴──────┐
        │                   │  Routes      │
        │                   │  faces       │
        │                   │  events      │
        │                   │  reports     │
        │                   │  chat        │
        │                   └──────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │              Camera Loop (_loop)                    │
  │  1. Read frame → resize 1280×720                    │
  │  2. track_persons(frame) → ByteTrack detections     │
  │  3. track_state.update() → Track objects            │
  │  4. Every 20 frames → schedule recognition         │
  │  5. On expiry → schedule finalization               │
  │  6. draw_annotations() → broadcast frame            │
  └─────────────────────────────────────────────────────┘
        │                          │
        ▼                          ▼
  RecognitionPipeline        TrackProcessor
  (progressive)              (on track expiry)
        │                          │
        ├─ detect_face()           ├─ final matching
        ├─ assess_quality()        ├─ final recognition
        ├─ build_embedding()       ├─ final memory
        ├─ lookup_memory()         ├─ decide()
        ├─ run_recognition()       ├─ store_face()
        └─ run_policy()            ├─ record_visit()
                                   ├─ dispatch()
                                   └─ broadcast_event()
```

---

## Thread Architecture

| Thread Pool | Count | Purpose |
|---|---|---|
| Main thread | 1 | CameraAgent._loop() blocking |
| Recognition pool | 4 | Progressive recognition + finalization |
| Worker pool | 2 | Track finalization (from queue) |
| JPEG encode pool | 2 | WebSocket frame encoding |
| Alert pool | 2 | Email/SMS/webhook dispatch |
| FastAPI thread | 1 | uvicorn asyncio event loop |
| Background daemons | 3 | LLM check, DB checks, model warmup |

---

## MongoDB Collections

| Collection | Purpose | Key Fields |
|---|---|---|
| `faces` | Person records | person_id, name, role, tags, latest_embedding, embedding_history |
| `events` | Track finalization events | person_id, status, alert_level, timestamp, camera_id |
| `visit_memory` | Visit history per person | person_id, visit_count, typical_hours, typical_cameras, similarity_history |

---

## File-by-File Reference

### `main.py` — Entry Point

| | |
|---|---|
| **Path** | `main.py` |
| **Purpose** | Orchestrator. Wires all components, manages lifecycle, runs camera loop. |
| **Thread** | Main (blocking), starts daemon threads for FastAPI/workers |
| **Key functions** | `main()`, `handle_track_finalized(track)`, `handle_frame_annotated(frame)`, `worker_process_tracks()` |

**Data flow:**
- Creates `queue.Queue` + 2 worker threads
- Starts FastAPI on port 8000 (daemon)
- Pre-warms YOLO + InsightFace (background)
- Calls `CameraAgent.start()` (BLOCKS until Ctrl+C)
- On exit: joins queue, shuts down everything

**Depends on:** config.settings, agents.camera_agent, agents.track_processor, agents.scoring, utils.db_utils, agents.alert_agent, utils.llm_client, pipeline.models, dashboard.backend.main

---

### `config/status.py` — Status Enum

| | |
|---|---|
| **Path** | `config/status.py` |
| **Purpose** | Centralized `Status(IntEnum)` enum — single source of truth for all identity status values. |
| **Thread** | Module-level (import-time) |

**Status bands (higher = more trusted):**
- `UNKNOWN` (1) — Unrecognized person
- `UNCERTAIN` (2) — Weak match, low confidence
- `KNOWN` (3) — Matched identity, confidence above threshold
- `KNOWN_VISITOR` (4) — Confirmed returning visitor
- `VERIFIED` (5) — Verified visitor (manual or high-similarity)
- `AUTHORIZED` (6) — Employee / authorized person
- `BLACKLIST` (7) — Blacklisted person (highest priority)
- `MASKED_UNKNOWN` (8) — Masked / partial-visibility unknown
- `HIDDEN` (9) — Intentionally avoiding detection

**Key exports:**
- `Status` — The enum class
- `STATUS_LABELS` — `dict[int, str]` display labels (e.g., `{3: "known", ...}`)
- `LABEL_TO_STATUS` — `dict[str, int]` reverse lookup (e.g., `{"known": 3, ...}`)
- `RESOLVED_STATUSES` — `set[int]` of verified/known_visitor/authorized
- `UNVERIFIED_STATUSES` — `set[int]` of unknown/masked_unknown/uncertain/hidden
- `IS_KNOWN_THRESHOLD` — `int` value 3 (Status.KNOWN)

---

### `config/settings.py` — Configuration

| | |
|---|---|
| **Path** | `config/settings.py` |
| **Purpose** | Loads and validates all settings. 3-tier: `.env` > `config.jsonc` > defaults. |
| **Thread** | Module-level (import-time), accessed by all threads |

**Key constants (100+):**
- `MATCH_THRESHOLD` (0.45), `PERSON_CONF_THRESHOLD` (0.40)
- `QUALITY_WEIGHT_BLUR` (0.50), `QUALITY_WEIGHT_BRIGHTNESS` (0.25), `QUALITY_WEIGHT_AREA` (0.25)
- `WEIGHT_SIMILARITY` (0.65), `WEIGHT_QUALITY` (0.15), `WEIGHT_TRACK` (0.10), `WEIGHT_MEMORY` (0.05), `WEIGHT_MARGIN` (0.05)
- `CONFIDENCE_KNOWN_MIN` (70), `CONFIDENCE_UNCERTAIN_MIN` (55)

**Key functions:**
- `_get(env_key, config_key, default, cast)` — resolution function
- `validate_config()` — validates thresholds, ranges

**Note:** Flat namespace, 100+ module-level constants. Consider grouping for refactoring.

---

### `config/logging_setup.py` — Logging Setup

| | |
|---|---|
| **Path** | `config/logging_setup.py` |
| **Purpose** | 3-tier structlog setup. Extracted from settings.py. |
| **Thread** | Module-level (import-time) |

**Key components:**
- `Colors` — ANSI color constants for terminal
- `CompactTerminalRenderer` — one-liner structlog renderer for terminal
- `JSONFileRenderer` — machine-readable JSON lines for `logs/surveillance.jsonl`
- `_BlankFilter` — suppresses empty/duplicate messages
- `setup_logging()` — configures all 3 tiers (terminal, JSON file, debug file)
- `TERMINAL_ALLOWLIST` — events that appear in terminal (others go to files only)

---

#### Agents

### `agents/base.py` — Abstract Base Agent

| | |
|---|---|
| **Path** | `agents/base.py` |
| **Purpose** | ABC defining agent contract. |
| **Class** | `BaseAgent(ABC)` with abstract `run(input_data: Dict) -> Dict` |

Used by: MemoryAgent, RecognitionAgent, PolicyAgent, ReportAgent

---

### `agents/camera_agent.py` — Camera Loop

| | |
|---|---|
| **Path** | `agents/camera_agent.py` |
| **Purpose** | The heart of the system. Capture loop, ByteTrack tracking, progressive recognition, track expiry/finalization. |
| **Thread** | Main thread (blocking), schedules work to recognition pool (4 threads) |
| **Class** | `CameraAgent` |

**Constructor params:** `on_track_finalized`, `on_frame_annotated`, `recognition_pipeline` (optional)

**Internal state:**
- `track_state: TrackState` — manages all active tracks
- `_recognizing_tracks: set` — tracks currently being recognized
- `_finalized_track_ids: set` — tracks already finalized
- `_recognition_executor: ThreadPoolExecutor(max_workers=4)` — recognition tasks
- `_timing: TimingCollector` — diagnostic timing

**Data flow (_loop):**
```
Read frame → resize(1280×720) → track_persons() → track_state.update()
→ IoU dedup → every 20 frames: schedule _progressive_recognition()
→ check expired: schedule _finalize_track() → draw_annotations() → callback
```

**`_progressive_recognition(frame, track)`:**
```
begin_recognition → ensure_fallback_frame → pipeline.run()
→ store best face/embedding/match/recognition/memory on track
→ handle blacklist alerts → end_recognition
```

**`_finalize_track(track)`:**
```
mark_finalized_once → retry_embedding → on_track_finalized(track) → queue
```

**Depends on:** config.settings, pipeline.tracker, pipeline.track_state, pipeline.models, pipeline.recognition_pipeline, utils.image_utils, agents.policy, agents.timing

---

### `agents/track_processor.py` — Track Post-Processing

| | |
|---|---|
| **Path** | `agents/track_processor.py` |
| **Purpose** | Orchestrates everything after a track expires: matching, recognition, memory, decision, registration, alerts, event logging, WebSocket. |
| **Thread** | Worker pool (2 threads), called from queue |
| **Class** | `TrackProcessor` |

**`process(track)` flow:**
```
snapshot → if no embedding: log as unknown → return
if no pending match: vector_search()
if no pending memory: memory_agent.run()
if no pending recognition: RecognitionAgent.run()
decide() → if should_register: deduplicate + store_face()
if matched: record_visit()
if should_alert: dispatch()
broadcast_event() → log_event()
```

**`handle_frame(frame)`:** WebSocket frame broadcasting (every FRAME_SKIP frames)

**Note:** This is the heaviest file — could be decomposed into smaller pipeline stages.

**Depends on:** config.settings, pipeline.models, agents.policy, agents.alert_agent, agents.memory, utils.db_utils, utils.image_utils, dashboard.backend.routes.live

---

### `agents/matching_agent.py` — Vector Search

| | |
|---|---|
| **Path** | `agents/matching_agent.py` |
| **Purpose** | Wraps MongoDB Atlas vector search into clean interface. |
| **Function** | `run_matching_from_embedding(embedding, track_id) -> MatchResult` |

**Data flow:** embedding list → vector_search() → MatchResult (person_id, name, role, tags, similarity_score, margin, candidates)

**Depends on:** numpy, pipeline.models, utils.db_utils

---

### `agents/policy.py` — Business Rules + decide() Entry Point

| | |
|---|---|
| **Path** | `agents/policy.py` |
| **Purpose** | Centralizes ALL business rules. Priority-ordered evaluation. Also exposes `decide()` convenience function. |
| **Class** | `PolicyAgent(BaseAgent)` |
| **Exposed** | `RESOLVED_STATUSES = {"verified", "known_visitor", "authorized"}` |
| **Functions** | `decide(track, match_result, recognition_result, memory_context) -> DecisionResult` (lazy singleton) |

**Rule evaluation order (priority):**

| # | Rule | Condition | Alert |
|---|---|---|---|
| 1 | Blacklist | tag "blacklist" | critical |
| 2 | Authorized | tag "authorized" | none |
| 3 | Verified | verified + (known/high sim) | none |
| 4 | Auto-registered self-match | "auto_registered" + sim > 0.65 | low |
| 5 | Known visitor | matched + memory confirms is_known | low |
| 6 | Matched uncertain | sub-rules (known→known, high sim→known, mid→unknown) | varies |
| 7 | Intentionally hidden | visibility == "hidden" | high |
| 8 | Masked unknown | is_masked or partial visibility | medium/high |
| 9 | After-hours unknown | not office hours or not weekday | high |
| 10 | Unknown default | unknown during office hours | medium |

**Depends on:** pipeline.models, config.settings

---

### `agents/recognition.py` — Identity Classification

| | |
|---|---|
| **Path** | `agents/recognition.py` |
| **Purpose** | Multi-signal identity classification. Delegates scoring to scoring.py. |
| **Class** | `RecognitionAgent(BaseAgent)` |

**`run(input_data)`:** similarity, is_masked, face_quality, track_duration, memory_context, top2, margin, name, track_id → compute_confidence() → confidence_status() → RecognitionResult

**Convenience:** `recognize(...)` — creates agent and calls run

**Depends on:** agents.base, agents.scoring, pipeline.models, config.settings

---

### `agents/scoring.py` — Confidence Scoring

| | |
|---|---|
| **Path** | `agents/scoring.py` |
| **Purpose** | Pure scoring functions. No I/O. |
| **Thread** | Called from recognition threads, writes to logs/calculation.log |

**Formula:**
```
base = 0.65×sim + 0.15×quality + 0.10×track + 0.05×memory + 0.05×margin
adjusted = base × (1 - 0.15×mask)
confidence = int(round(1 + 99×clip(adjusted, 0, 1)))
```

**Functions:**
- `compute_confidence(raw_cosine, face_quality, track_seconds, memory_boost, is_masked, margin)` → int
- `is_match(raw_cosine)` → bool — binary gate against MATCH_THRESHOLD
- `confidence_status(confidence, matched)` → str — "known"/"uncertain"/"unknown"
- Normalizers: `normalize_cosine`, `normalize_quality`, `normalize_track_duration`, `normalize_memory`, `normalize_mask`, `normalize_margin`

**Depends on:** config.settings

---

### `agents/memory.py` — Visit History

| | |
|---|---|
| **Path** | `agents/memory.py` |
| **Purpose** | Tracks visit history in MongoDB, provides confidence boost. |
| **Class** | `MemoryAgent(BaseAgent)` |

**`run(input_data)` → dict:** person_id, camera_id, similarity, status → memory lookup → confidence_boost (-10 to +20) + is_known + reason

**`record_visit(...)`:** Updates visit_count, typical_hours, typical_cameras, similarity_history

**Boost logic:** +2/visit (max +10), +5 recent (≤7 days), +3 high avg sim, +2 typical time, +1 typical camera, -5 if current sim much lower than avg

**Depends on:** agents.base, utils.db_utils

---

### `agents/alert_agent.py` — Alert Dispatch

| | |
|---|---|
| **Path** | `agents/alert_agent.py` |
| **Purpose** | Multi-channel alert dispatch with per-track cooldown dedup. |
| **Thread** | Alert pool (2 threads) for email/SMS/webhook |
| **Functions** | `should_send_alert()`, `dispatch()`, `shutdown()` |

**Dispatch flow:**
```
Check should_alert + not already alerted → cooldown check
→ build payload → console (sync) → LLM generate summary (sync)
→ email/SMS/webhook (async pool)
```

**Depends on:** config.settings, pipeline.models, utils.llm_client

---

### `agents/finalizer.py` — Embedding Retry

| | |
|---|---|
| **Path** | `agents/finalizer.py` |
| **Purpose** | Last-chance InsightFace detection when track expires without embedding. |
| **Function** | `retry_embedding(track, set_embedding)` |

**Flow:** Try best_face_crop → if no face: try best_full_frame (if enabled) → set_embedding()

**Depends on:** pipeline.models, utils.embedding_utils

---

### `agents/timing.py` — Timing Diagnostics

| | |
|---|---|
| **Path** | `agents/timing.py` |
| **Purpose** | Thread-safe collector for recognition timing. |
| **Class** | `TimingCollector` |

**Methods:** `record_submit()`, `pop_submit()`, `submit_pending_count()`, `record_start()`, `pop_start()`, `get_duration_ms()`

---

### `agents/report.py` — Report Generation

| | |
|---|---|
| **Path** | `agents/report.py` |
| **Purpose** | Generates human-readable reports. Uses LLM when available, falls back to templates. |
| **Class** | `ReportAgent(BaseAgent)` |

**Methods:** `_incident_report`, `_summary_report`, `_person_report`, `_stats_report`

**Depends on:** agents.base, utils.db_utils, utils.llm_client

---

#### Pipeline

### `pipeline/tracker.py` — YOLO + ByteTrack

| | |
|---|---|
| **Path** | `pipeline/tracker.py` |
| **Purpose** | Loads YOLO model (singleton) and runs person tracking. |
| **Thread** | Called from camera loop, inference lock for thread safety |

**Functions:**
- `get_model()` → YOLO — lazy singleton with double-checked locking
- `track_persons(frame, persist=True)` → list of `{track_id, box, confidence}`

**Depends on:** ultralytics.YOLO, config.settings

---

### `pipeline/recognition_pipeline.py` — Orchestrator

| | |
|---|---|
| **Path** | `pipeline/recognition_pipeline.py` |
| **Purpose** | Central recognition pipeline. Orchestrates: detect → quality → embed → match → memory → recognize → decide. |
| **Class** | `RecognitionPipeline` |

**`run(frame, track) -> PipelineResult`:**
```
Early exit if high-confidence cached
→ _detect_face(frame, track) — crop person, detect faces (fallback to full frame)
→ _assess_quality(face_crop) — validity gates + scoring
→ _build_embedding(face_det, track) — InsightFace + vector search
→ _lookup_memory(track, match) — visit history
→ _run_recognition(track, match, quality, memory, face_det) — confidence
→ _run_policy(track, match, recognition, memory) — business rules
→ return PipelineResult
```

**Helper classes:** `_FaceResult`, `_EmbedResult`

**Note:** Imports agents lazily in constructor to avoid circular imports. Could be cleaner.

**Depends on:** config.settings, pipeline.models, pipeline.quality_agent, utils.image_utils, utils.embedding_utils, agents.matching_agent, agents.recognition, agents.memory, agents.policy

---

### `pipeline/quality_agent.py` — Face Quality + compute_face_ratio()

| | |
|---|---|
| **Path** | `pipeline/quality_agent.py` |
| **Purpose** | Two-tier quality: validity gates (reject) + scoring (weighted composite). Also contains `compute_face_ratio()` (inlined from deleted `face.py`). |
| **Functions** | `compute_quality(face_crop) -> QualityResult`, `compute_face_ratio(face_bbox, person_box) -> float` |

**Validity gates:**
- Blur: Laplacian variance ≥ 40
- Brightness: V-channel in [35, 255]
- Area: ≥ 1200 px²

**Score:** blur (50%) + brightness center-radius (25%) + area (25%)

**Depends on:** pipeline.models, utils.image_utils, config.settings

---

### `pipeline/models.py` — Data Models

| | |
|---|---|
| **Path** | `pipeline/models.py` |
| **Purpose** | All dataclasses used across the system. |

**Classes:**

| Class | Purpose | Fields |
|---|---|---|
| `DedupStatus` | Enum | MERGED, NEW, FAILED |
| `DedupResult` | Dedup outcome | status, person_id, reason, similarity |
| `Track` | Central mutable state (~40 fields) | person_id, embedding, match_result, recognition_result, quality_score, alerts, etc. |
| `TrackSnapshot` | Immutable copy | ~35 fields (snapshot of Track) |
| `QualityResult` | Quality assessment | blur_score, brightness, face_area, is_valid, overall_score |
| `MatchResult` | Vector search result | person_id, name, role, tags, similarity_score, margin, candidates |
| `DecisionResult` | Policy decision | status, alert_level, should_alert, should_register, nl_summary |
| `RecognitionResult` | Classification | status, confidence, similarity, is_masked, reason |

**Note:** Track is a god object (40+ fields). Consider splitting for refactoring.

---

### `pipeline/track_state.py` — Track State Management

| | |
|---|---|
| **Path** | `pipeline/track_state.py` |
| **Purpose** | Thread-safe track lifecycle. Composite IDs, creation/update/expiry, in-flight recognition. |
| **Class** | `TrackState` |

**Key methods:**
- `update(camera_id, bt_id, box)` → Track
- `get_expired_tracks()` → list (classifies visibility, respects in-flight recognition)
- `begin_recognition()` / `end_recognition()` — refcounting
- `set_best_face(track, face_crop, score, ...)` — quality-gated update
- `ensure_fallback_frame(track, frame)` — guarantees every track gets a photo
- State setters with upgrade-only semantics

**Composite ID format:** `{camera_id}_{session_epoch}_{bt_id}_{generation}`

**Depends on:** config.settings, pipeline.models, utils.image_utils

---

#### Utilities

### `utils/db_client.py` — Connection Singleton

| | |
|---|---|
| **Path** | `utils/db_client.py` |
| **Purpose** | Thread-safe MongoDB connection singleton + collection getters. |
| **Functions** | `get_client()`, `close_client()`, `get_faces_collection()`, `get_events_collection()`, `get_memory_collection()` |

---

### `utils/db_faces.py` — Face CRUD

| | |
|---|---|
| **Path** | `utils/db_faces.py` |
| **Purpose** | Face record CRUD, deduplication, embedding history. |
| **Functions** | `store_face()`, `update_face()`, `verify_person()`, `get_unknown_faces()`, `get_face_by_id()`, `delete_face()`, `deduplicate_identity()` |

**Note:** `update_face()` uses quality-gated embedding writes — lower-quality embeddings never overwrite higher-quality ones.

---

### `utils/db_events.py` — Event Logging

| | |
|---|---|
| **Path** | `utils/db_events.py` |
| **Purpose** | Event logging + statistics queries. |
| **Functions** | `log_event()`, `get_events_with_faces()`, `get_stats()` |

---

### `utils/db_memory.py` — Visit Memory

| | |
|---|---|
| **Path** | `utils/db_memory.py` |
| **Purpose** | Visit memory CRUD — atomic upserts, visit-dedup gate, best_status tracking. |
| **Functions** | `get_or_create_memory()`, `update_visit_memory()`, `get_visit_history()`, `get_memory_stats()` |

---

### `utils/db_search.py` — Vector Search

| | |
|---|---|
| **Path** | `utils/db_search.py` |
| **Purpose** | Atlas Vector Search + Python cosine fallback + index maintenance. |
| **Functions** | `vector_search()`, `_python_cosine_scan()`, `find_similar_unknowns()`, `check_atlas_search_index()`, `backfill_missing_embeddings()` |

---

### `utils/db_utils.py` — Re-export Facade

| | |
|---|---|
| **Path** | `utils/db_utils.py` |
| **Purpose** | Re-export facade. All 12+ consumers import from here; nothing breaks. |
| **Pattern** | Imports all public names from `db_client`, `db_faces`, `db_events`, `db_memory`, `db_search` |

**Depends on:** all 5 db domain modules

---

### `utils/embedding_utils.py` — InsightFace

| | |
|---|---|
| **Path** | `utils/embedding_utils.py` |
| **Purpose** | InsightFace singleton. Face detection, CLAHE, mask detection, similarity. |
| **Thread** | Thread-safe (inference lock) |
| **Class** | `InsightFaceSingleton` |

**Key methods:**
- `detect_faces_raw(image, min_score)` → list of `{det_score, embedding, bbox, is_masked}`
- `_apply_clahe(image)` — CLAHE with early exit
- `_detect_mask_geometric(landmarks)` — lower/upper face ratio heuristic

**Exposed functions:**
- `get_insightface()` → singleton
- `compare_similarity(raw_cosine, threshold)` → bool
- `atlas_score_to_cosine(atlas_score)` → float: `(score * 2) - 1`

---

### `utils/llm_client.py` — Ollama

| | |
|---|---|
| **Path** | `utils/llm_client.py` |
| **Purpose** | Synchronous HTTP client for Ollama API. |
| **Thread** | Called from various threads, httpx connection pool |

**Functions:**
- `generate(prompt, ...)` → str|None
- `chat_completion(message, ...)` → str|None
- `generate_nl_summary(alert_payload)` → str|None
- `generate_incident_summary(incident_data)` → str|None
- `generate_executive_summary(stats, recent_events)` → str|None
- `is_available()` → bool (cached 10s)
- `shutdown()` — close client

**Depends on:** httpx, config.settings

---

### `utils/image_utils.py` — Image Processing

| | |
|---|---|
| **Path** | `utils/image_utils.py` |
| **Purpose** | Image processing, Cloudinary upload, annotation drawing. |
| **Thread** | Called from camera loop and workers |

**Functions:**
- `upload_to_cloudinary(image, folder)` → str|None
- `upload_jpeg_to_cloudinary(jpeg_bytes, folder)` → str|None
- `compute_blur_score(image)` → float — Laplacian variance
- `compute_brightness(image)` → float — V-channel mean
- `crop_person(frame, box)` → np.ndarray
- `save_image(image, path)` → bool
- `resolve_track_image_url(track)` → str|None — priority chain
- `resolve_track_person_crop_url(track)` → str|None
- `compute_iou(box_a, box_b)` → float
- `draw_annotations(frame, tracks)` → np.ndarray

**Depends on:** cv2, numpy, config.settings

---

#### Dashboard

### `dashboard/backend/main.py` — FastAPI

| | |
|---|---|
| **Path** | `dashboard/backend/main.py` |
| **Purpose** | Creates FastAPI app, configures CORS, mounts routers + static files. |

**Routes:** `/api/faces`, `/api/events`, `/api/reports`, `/api/chat`, `/ws/live`, `/captures`

---

### `dashboard/backend/routes/faces.py` — Face CRUD

| | |
|---|---|
| **Path** | `dashboard/backend/routes/faces.py` |
| **Endpoints** | GET /api/faces, GET /api/faces/{id}, POST /verify, PUT /update, DELETE |

All use `asyncio.to_thread()` for MongoDB calls.

---

### `dashboard/backend/routes/events.py` — Events API

| | |
|---|---|
| **Path** | `dashboard/backend/routes/events.py` |
| **Endpoints** | GET /api/events/stats, GET /api/events, GET /api/events/unknown, GET /api/events/alerts |

---

### `dashboard/backend/routes/live.py` — WebSocket

| | |
|---|---|
| **Path** | `dashboard/backend/routes/live.py` |
| **Purpose** | WebSocket live feed. Frame/alert/event broadcasting. |
| **Key state** | `connected_clients: Set[WebSocket]` |

**Functions (called from pipeline via asyncio.run_coroutine_threadsafe):**
- `broadcast_frame(frame_data)` — base64 JPEG
- `broadcast_event(event)` — track events
- `broadcast_alert(alert_data)` — alerts

**WebSocket:** `ws://localhost:8000/ws/live` — origin-validated

---

### `dashboard/backend/routes/reports.py` — Reports API

| | |
|---|---|
| **Path** | `dashboard/backend/routes/reports.py` |
| **Endpoints** | GET /api/reports/stats, GET /api/reports/summary, GET /api/reports/person/{id}, GET /api/reports/incidents |

---

### `dashboard/backend/routes/chat.py` — Chat API

| | |
|---|---|
| **Path** | `dashboard/backend/routes/chat.py` |
| **Purpose** | NL interface to surveillance. Routes queries → data fetch → LLM → response. |
| **Endpoints** | POST /api/chat, GET /api/chat/health |

**Query routing:** Keyword-based intent detection (stats, unknowns, events, memory)

---

## Dependency Map

```
main.py
  ├─ config.settings (everything)
  ├─ agents.camera_agent
  │    ├─ pipeline.tracker (YOLO)
  │    ├─ pipeline.track_state
  │    ├─ pipeline.recognition_pipeline
  │    │    ├─ pipeline.quality_agent → utils.image_utils
  │    │    ├─ utils.embedding_utils (InsightFace singleton)
  │    │    ├─ agents.matching_agent → utils.db_search
  │    │    ├─ agents.recognition → agents.scoring
  │    │    ├─ agents.memory → utils.db_memory
  │    │    └─ agents.policy → decide()
  │    ├─ agents.finalizer → utils.embedding_utils
  │    ├─ agents.timing
  │    └─ utils.image_utils
  ├─ agents.track_processor
  │    ├─ agents.policy → decide()
  │    ├─ agents.alert_agent → utils.llm_client
  │    ├─ agents.memory
  │    ├─ utils.db_utils (facade → db_faces, db_events, db_memory, db_search)
  │    ├─ utils.image_utils
  │    └─ dashboard.backend.routes.live (broadcast_*)
  └─ dashboard.backend.main (FastAPI)
       ├─ dashboard.backend.routes.faces → utils.db_faces
       ├─ dashboard.backend.routes.events → utils.db_events
       ├─ dashboard.backend.routes.reports → agents.report
       ├─ dashboard.backend.routes.chat → utils.llm_client
       └─ dashboard.backend.routes.live (WebSocket)
```

---

## Refactoring Observations

| Issue | Location | Impact | Suggestion |
|---|---|---|---|
| ~~Split `db_utils.py` (880 lines)~~ | ~~`utils/db_utils.py`~~ | ~~All MongoDB CRUD in one file~~ | ~~Split into domain modules~~ **DONE: db_client, db_faces, db_events, db_memory, db_search + facade** |
| ~~Delete `pipeline/face.py`~~ | ~~`pipeline/face.py`~~ | ~~Single function~~ | ~~Inline into `quality_agent.py`~~ **DONE** |
| ~~Delete `agents/decision_agent.py`~~ | ~~`agents/decision_agent.py`~~ | ~~Thin facade~~ | ~~Move `decide()` into `policy.py`~~ **DONE** |
| ~~Extract logging from settings~~ | ~~`config/settings.py`~~ | ~~Mixed concerns~~ | ~~Extract to `config/logging_setup.py`~~ **DONE** |
| TrackProcessor too heavy | `track_processor.py` | Single file does matching + recognition + memory + decision + registration + alerts + broadcasting | Decompose into smaller pipeline stages |
| Track god object | `pipeline/models.py` | 40+ mutable fields, shared across threads | Split into immutable data + separate mutable state |
| Two recognition paths | camera_agent + track_processor | Progressive (pipeline.run) and final (TrackProcessor.process) duplicate logic | Unify into single pipeline, track_processor delegates to recognition_pipeline |
| Circular-ish imports | recognition_pipeline ↔ agents | Lazy imports in constructor | Consider dependency injection pattern |
| Module-level singletons | alert_agent, llm_client | State coupled to module import | Consider explicit lifecycle management |
| Flat config namespace | settings.py | 100+ constants in one file | Group into config dataclasses by domain |
| WebSocket coupling | track_processor → live.py | Hard dependency between pipeline and dashboard | Use event bus or callback pattern |
| Duplicated matching/memory/decision | track_processor.process() vs recognition_pipeline.run() | Same logic in two places | Consolidate into recognition_pipeline only |

---

## Project Metrics

### Summary

| Metric | Total |
|---|---:|
| **Lines of code** | ~7,100 |
| **Functions/methods** | ~210 |
| **Classes** | 37 |
| **Project imports** | ~80 |

### By Directory

| Directory | Lines | Functions | Classes | Imports | Avg complexity |
|---|---:|---:|---:|---:|---|
| `agents/` | ~2,660 | 72 | 7 | 37 | High |
| `pipeline/` | ~1,050 | 47 | 14 | 12 | Medium |
| `utils/` | ~1,350 | 62 | 1 | 8 | Medium (split into 6 files) |
| `config/` | ~900 | 16 | 6 | 0 | Low (no deps) |
| `dashboard/backend/` | 735 | 5 | 10 | 10 | Low |
| `main.py` | 179 | 9 | 0 | 8 | Low (orchestrator) |

### Top 10 Largest Files

| Rank | File | Lines | Functions | Classes | Why it's large |
|---:|---|---:|---:|---:|---|
| 1 | `pipeline/track_state.py` | 374 | 24 | 1 | Track lifecycle + composite IDs + state mutations |
| 2 | `pipeline/recognition_pipeline.py` | 339 | 11 | 5 | Orchestrates 6-step recognition pipeline |
| 3 | `agents/policy.py` | 331 | 2 | 1 | 10 business rules with nested conditions + decide() |
| 4 | `agents/camera_agent.py` | 506 | 9 | 1 | Camera loop + progressive recognition + finalization |
| 5 | `agents/report.py` | 307 | 8 | 1 | 4 report types + LLM fallbacks |
| 6 | `agents/track_processor.py` | 294 | 7 | 1 | Post-processing: match → recognize → decide → alert |
| 7 | `utils/llm_client.py` | 282 | 8 | 0 | Ollama HTTP client + NL summaries |
| 8 | `agents/memory.py` | 268 | 6 | 1 | Visit history + confidence boost calc |
| 9 | `agents/scoring.py` | 259 | 14 | 0 | Confidence formula + normalizers |
| 10 | `utils/db_faces.py` | ~200 | 8 | 0 | Face CRUD, dedup, embedding history |

### Function Inventory

#### agents/

| File | Functions | Purpose |
|---|---|---|
| `alert_agent.py` | `should_send_alert`, `dispatch`, `shutdown`, `_send_console`, `_send_email`, `_send_sms`, `_send_webhook`, `_dispatch_channel`, `_build_payload` | Multi-channel alert dispatch |
| `base.py` | `run` (abstract) | Agent contract |
| `camera_agent.py` | `start`, `_loop`, `_process_frame`, `_progressive_recognition`, `_finalize_track`, `_check_expired`, `_stats_reporter`, `_warmup_models`, `_init_worker_pool` | Camera capture + recognition loop |
| `finalizer.py` | `retry_embedding` | Last-chance InsightFace detection |
| `matching_agent.py` | `run_matching_from_embedding` | Atlas vector search wrapper |
| `memory.py` | `run`, `record_visit`, `_calc_boost`, `_is_known`, `_build_reason`, `_get_or_create` | Visit history + confidence boost |
| `policy.py` | `run`, `_eval_rules`, `decide` (lazy singleton) | 10 priority-ordered business rules + decide() |
| `recognition.py` | `run`, `recognize`, `_classify`, `_format_reason` | Multi-signal identity classification |
| `report.py` | `run`, `_incident_report`, `_summary_report`, `_person_report`, `_stats_report`, `_format_table`, `_llm_or_template`, `_fetch_context` | Report generation |
| `scoring.py` | `log_formula_header`, `clip`, `normalize_cosine`, `normalize_quality`, `normalize_track_duration`, `normalize_memory`, `normalize_mask`, `normalize_margin`, `compute_confidence`, `is_match`, `confidence_status`, `_log_calc`, `_write_formula`, `_ensure_log_dir` | Confidence scoring (pure functions) |
| `timing.py` | `record_submit`, `pop_submit`, `submit_pending_count`, `record_start`, `pop_start`, `get_duration_ms`, `_now_ms` | Timing diagnostics |
| `track_processor.py` | `process`, `handle_frame`, `_run_matching_from_embedding`, `_run_recognition`, `_run_memory`, `_auto_register`, `_should_skip_quality` | Post-track finalization pipeline |

#### pipeline/

| File | Functions | Purpose |
|---|---|---|
| `models.py` | `Track`, `TrackSnapshot`, `QualityResult`, `MatchResult`, `DecisionResult`, `RecognitionResult`, `DedupResult`, `DedupStatus` | Data models |
| `quality_agent.py` | `compute_quality`, `compute_face_ratio`, `_blur_score`, `_brightness_score`, `_area_score` | Face quality assessment + face ratio |
| `recognition_pipeline.py` | `run`, `_detect_face`, `_assess_quality`, `_build_embedding`, `_lookup_memory`, `_run_recognition`, `_run_policy`, `_should_skip`, `_log_result` | 6-step recognition orchestrator |
| `track_state.py` | `update`, `get_expired_tracks`, `begin_recognition`, `end_recognition`, `set_best_face`, `ensure_fallback_frame`, `set_embedding`, `set_decision`, `set_person_name`, `set_pending_match`, `set_pending_memory`, `set_pending_recognition`, `_make_composite_id`, `_find_track`, `_remove_track`, `_classify_visibility`, `_is_bt_reuse`, `_gc_in_flight`, `_quality_improved`, `_log_track_event`, `_log_expiry`, `_log_new_track`, `_log_recognition_start`, `_log_recognition_end`, `_log_quality_update`, `_log_embedding_set` | Track lifecycle management |
| `tracker.py` | `get_model`, `track_persons` | YOLO + ByteTrack inference |

#### utils/

| File | Functions | Purpose |
|---|---|---|
| `db_client.py` | `get_client`, `close_client`, `get_faces_collection`, `get_events_collection`, `get_memory_collection` | MongoDB connection singleton + collection getters |
| `db_faces.py` | `store_face`, `update_face`, `verify_person`, `get_unknown_faces`, `get_face_by_id`, `delete_face`, `deduplicate_identity`, `_compute_mean_embedding` | Face CRUD, dedup, embedding history |
| `db_events.py` | `log_event`, `get_events_with_faces`, `get_stats` | Event logging + statistics |
| `db_memory.py` | `get_or_create_memory`, `update_visit_memory`, `get_visit_history`, `get_memory_stats` | Visit memory CRUD |
| `db_search.py` | `vector_search`, `_python_cosine_scan`, `find_similar_unknowns`, `check_atlas_search_index`, `backfill_missing_embeddings` | Vector search + Python fallback + backfill |
| `db_utils.py` | (re-export facade — all public names from above 5 modules) | Import compatibility layer |
| `embedding_utils.py` | `get_insightface`, `compare_similarity`, `atlas_score_to_cosine`, `_apply_clahe`, `_detect_mask_geometric`, `detect_faces_raw`, `_init_singleton`, `_load_model` | InsightFace singleton + utilities |
| `image_utils.py` | `upload_to_cloudinary`, `upload_jpeg_to_cloudinary`, `compute_blur_score`, `compute_brightness`, `crop_person`, `save_image`, `resolve_track_image_url`, `resolve_track_person_crop_url`, `compute_iou`, `draw_annotations`, `_encode_jpeg`, `_save_local` | Image processing + Cloudinary |
| `llm_client.py` | `generate`, `chat_completion`, `generate_nl_summary`, `generate_incident_summary`, `generate_executive_summary`, `is_available`, `shutdown`, `_get_client`, `_check_available` | Ollama HTTP client |

#### dashboard/backend/

| File | Functions | Purpose |
|---|---|---|
| `main.py` | (creates FastAPI app, mounts routers) | App bootstrap |
| `routes/chat.py` | `chat`, `health`, `_route_query`, `_fetch_stats`, `_fetch_recent_events`, `_fetch_unknown_faces`, `_fetch_visit_memory` | Conversational NL interface |
| `routes/events.py` | `get_stats`, `list_events`, `get_unknown_events`, `get_alert_events` | Events REST API |
| `routes/faces.py` | `list_faces`, `get_face`, `verify_person`, `update_face`, `delete_face` | Faces CRUD API |
| `routes/live.py` | `websocket_endpoint`, `broadcast_frame`, `broadcast_event`, `broadcast_alert`, `_validate_origin`, `_send_safe` | WebSocket live feed |
| `routes/reports.py` | `get_stats`, `get_summary`, `get_person_report`, `get_incidents` | Reports REST API |

---

## Refactoring Priority Matrix

| Priority | Issue | Effort | Impact | Files to change |
|---:|---|---|---|---|
| **P1** | ~~Split `db_utils.py` (880 lines)~~ | ~~Medium~~ | ~~High~~ | ~~`utils/db_utils.py` → `db_client.py`, `db_faces.py`, `db_events.py`, `db_memory.py`, `db_search.py` + facade~~ **DONE** |
| **P2** | Unify recognition paths | High | High — eliminates duplication | `pipeline/recognition_pipeline.py`, `agents/track_processor.py`, `agents/camera_agent.py` |
| **P3** | Split Track god object | High | High — cleaner state management | `pipeline/models.py` → `TrackData` (immutable) + `TrackState` (mutable) |
| **P4** | Decompose TrackProcessor | Medium | Medium — clearer responsibilities | `agents/track_processor.py` → separate matching, registration, alerting |
| **P5** | Group config into dataclasses | Low | Medium — better discoverability | `config/settings.py` → `DetectionConfig`, `QualityConfig`, `RecognitionConfig`, etc. |
| **P6** | Break WebSocket coupling | Low | Medium — decouple pipeline from dashboard | `agents/track_processor.py` → use event bus instead of direct import |
| **P7** | Add dependency injection | Medium | Low — cleaner wiring | `pipeline/recognition_pipeline.py`, `main.py` |
| **P8** | Extract `policy.py` rule definitions | Low | Low — easier to add rules | `agents/policy.py` → rule data + evaluator |

---

## Module Responsibility Map

```
main.py              ─── wires everything, lifecycle
config/settings.py   ─── loads config, exposes 100+ constants
config/logging_setup.py ─── 3-tier structlog setup

agents/
├── camera_agent     ─── capture loop, progressive recognition
├── track_processor  ─── post-track finalization (match/decide/alert)
├── matching_agent   ─── vector search wrapper
├── recognition      ─── identity classification
├── scoring          ─── confidence formula (pure functions)
├── policy           ─── 10 business rules + decide()
├── memory           ─── visit history + boost
├── alert_agent      ─── multi-channel dispatch
├── finalizer        ─── last-chance embedding
├── report           ─── report generation
└── timing           ─── diagnostic timing

pipeline/
├── tracker          ─── YOLO + ByteTrack
├── recognition_pipeline ─── 6-step recognition orchestrator
├── quality_agent    ─── face quality scoring + compute_face_ratio()
├── track_state      ─── track lifecycle (thread-safe)
└── models           ─── all dataclasses

utils/
├── db_client        ─── MongoDB connection singleton
├── db_faces         ─── face CRUD, dedup, embedding history
├── db_events        ─── event logging + stats
├── db_memory        ─── visit memory CRUD
├── db_search        ─── vector search (Atlas + Python fallback)
├── db_utils         ─── re-export facade (all consumers import from here)
├── embedding_utils  ─── InsightFace singleton
├── llm_client       ─── Ollama HTTP client
└── image_utils      ─── image processing + Cloudinary
```
