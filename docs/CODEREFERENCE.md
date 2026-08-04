# Code Reference for Refactoring

> AI-powered surveillance: YOLOv8 → ByteTrack → InsightFace → Decision Engine → Alerts + LLM

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

### `config/settings.py` — Configuration

| | |
|---|---|
| **Path** | `config/settings.py` |
| **Purpose** | Loads and validates all settings. 3-tier: `.env` > `config.jsonc` > defaults. Sets up 3-tier logging. |
| **Thread** | Module-level (import-time), accessed by all threads |

**Key constants (100+):**
- `MATCH_THRESHOLD` (0.45), `PERSON_CONF_THRESHOLD` (0.40)
- `QUALITY_WEIGHT_BLUR` (0.50), `QUALITY_WEIGHT_BRIGHTNESS` (0.25), `QUALITY_WEIGHT_AREA` (0.25)
- `WEIGHT_SIMILARITY` (0.65), `WEIGHT_QUALITY` (0.15), `WEIGHT_TRACK` (0.10), `WEIGHT_MEMORY` (0.05), `WEIGHT_MARGIN` (0.05)
- `CONFIDENCE_KNOWN_MIN` (70), `CONFIDENCE_UNCERTAIN_MIN` (55)

**Key functions:**
- `_get(env_key, config_key, default, cast)` — resolution function
- `validate_config()` — validates thresholds, ranges
- `setup_logging()` — 3-tier structlog

**Note:** Flat namespace, 100+ module-level constants. Consider grouping for refactoring.

---

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

**Depends on:** config.settings, pipeline.models, agents.decision_agent, agents.alert_agent, agents.memory, utils.db_utils, utils.image_utils, dashboard.backend.routes.live

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

### `agents/decision_agent.py` — Decision Facade

| | |
|---|---|
| **Path** | `agents/decision_agent.py` |
| **Purpose** | Backward-compatible facade that delegates to PolicyAgent. |
| **Function** | `decide(track, match_result, recognition_result, memory_context) -> DecisionResult` |

**Depends on:** pipeline.models, agents.policy

---

### `agents/policy.py` — Business Rules

| | |
|---|---|
| **Path** | `agents/policy.py` |
| **Purpose** | Centralizes ALL business rules. Priority-ordered evaluation. |
| **Class** | `PolicyAgent(BaseAgent)` |
| **Exposed** | `RESOLVED_STATUSES = {"verified", "known_visitor", "authorized"}` |

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

**Depends on:** config.settings, pipeline.models, pipeline.quality_agent, pipeline.face, utils.image_utils, utils.embedding_utils, agents.matching_agent, agents.recognition, agents.memory, agents.decision_agent

---

### `pipeline/quality_agent.py` — Face Quality

| | |
|---|---|
| **Path** | `pipeline/quality_agent.py` |
| **Purpose** | Two-tier quality: validity gates (reject) + scoring (weighted composite). |
| **Function** | `compute_quality(face_crop) -> QualityResult` |

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

### `pipeline/face.py` — Face Ratio

| | |
|---|---|
| **Path** | `pipeline/face.py` |
| **Purpose** | Single utility function. |
| **Function** | `compute_face_ratio(face_bbox, person_box) -> float` — face area / person bbox ratio [0, 1] |

---

### `utils/db_utils.py` — MongoDB

| | |
|---|---|
| **Path** | `utils/db_utils.py` |
| **Purpose** | All database operations. Thread-safe singleton client. |
| **Thread** | Called from all threads, safe with pymongo |

**Functions:**

| Function | Purpose |
|---|---|
| `get_client()` / `close_client()` | Singleton lifecycle |
| `vector_search(embedding, ...)` | Atlas Vector Search + Python fallback |
| `_python_cosine_scan(embedding)` | Brute-force numpy cosine |
| `deduplicate_identity(embedding, ...)` | Check if embedding exists |
| `store_face(...)` | Insert new face record |
| `update_face(...)` | Quality-gated embedding writes |
| `verify_person(...)` | Mark person verified |
| `get_events_with_faces(...)` | Events with batch face lookups |
| `log_event(...)` | Insert event document |
| `get_or_create_memory(...)` | Atomic upsert for visit memory |
| `update_visit_memory(...)` | Atomic visit update with dedup gate |
| `check_atlas_search_index()` | Startup validation |
| `backfill_missing_embeddings()` | Fix legacy records |

**Depends on:** pymongo, config.settings, pipeline.models, utils.embedding_utils

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
  │    │    ├─ pipeline.face
  │    │    ├─ utils.embedding_utils (InsightFace singleton)
  │    │    ├─ agents.matching_agent → utils.db_utils
  │    │    ├─ agents.recognition → agents.scoring
  │    │    ├─ agents.memory → utils.db_utils
  │    │    └─ agents.decision_agent → agents.policy
  │    ├─ agents.finalizer → utils.embedding_utils
  │    ├─ agents.timing
  │    └─ utils.image_utils
  ├─ agents.track_processor
  │    ├─ agents.decision_agent
  │    ├─ agents.alert_agent → utils.llm_client
  │    ├─ agents.memory
  │    ├─ utils.db_utils
  │    ├─ utils.image_utils
  │    └─ dashboard.backend.routes.live (broadcast_*)
  └─ dashboard.backend.main (FastAPI)
       ├─ dashboard.backend.routes.faces → utils.db_utils
       ├─ dashboard.backend.routes.events → utils.db_utils
       ├─ dashboard.backend.routes.reports → agents.report
       ├─ dashboard.backend.routes.chat → utils.llm_client
       └─ dashboard.backend.routes.live (WebSocket)
```

---

## Refactoring Observations

| Issue | Location | Impact | Suggestion |
|---|---|---|---|
| TrackProcessor too heavy | `track_processor.py` | Single file does matching + recognition + memory + decision + registration + alerts + broadcasting | Decompose into smaller pipeline stages |
| Track god object | `pipeline/models.py` | 40+ mutable fields, shared across threads | Split into immutable data + separate mutable state |
| Two recognition paths | camera_agent + track_processor | Progressive (pipeline.run) and final (TrackProcessor.process) duplicate logic | Unify into single pipeline, track_processor delegates to recognition_pipeline |
| Circular-ish imports | recognition_pipeline ↔ agents | Lazy imports in constructor | Consider dependency injection pattern |
| Module-level singletons | alert_agent, db_utils, llm_client | State coupled to module import | Consider explicit lifecycle management |
| Flat config namespace | settings.py | 100+ constants in one file | Group into config dataclasses by domain |
| WebSocket coupling | track_processor → live.py | Hard dependency between pipeline and dashboard | Use event bus or callback pattern |
| Duplicated matching/memory/decision | track_processor.process() vs recognition_pipeline.run() | Same logic in two places | Consolidate into recognition_pipeline only |
