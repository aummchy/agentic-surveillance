# Current Architecture (as-built)

> **What this document is.** An as-built inventory of the system as the code
> stands today — where work happens, what each file is responsible for, who
> calls it, what it mutates, and which thread it runs on. It absorbs the
> content of three documents deleted on 2026-10-03 (`docs/ARCHITECTURE.md`,
> `docs/02 - Architecture/System Overview.md`,
> `docs/02 - Architecture/Tech Stack.md`).
>
> **What it is not.** A target architecture or a refactor plan. Proposed
> changes live in `docs/REFACTOR_PLAN.md`; semantic boundaries live in
> `docs/ARCHITECTURE_RULES.md`. Where this document says "Known concerns",
> those are documented observations only — see `plan.md` for what is approved
> to change and when.

Every threshold, ID format and thread count below was read from the running
configuration or the source, not copied from older prose. Where an older
document disagreed, this document follows the code.

---

## 1. Runtime flow

```
main.py
  │
  ├─ Camera thread ──────────────── CameraAgent._loop()
  │    capture → resize → YOLO+ByteTrack (tracker.track_persons)
  │      → TrackState.update()  (composite IDs, generation suffix)
  │      → IoU overlap dedup
  │      → _maybe_schedule_recognition()   every RECOGNITION_INTERVAL_FRAMES
  │      → _finalize_expired_tracks()      on track timeout
  │      → on_frame_annotated → TrackProcessor.handle_frame (preview relay)
  │
  ├─ Recognition pool (RECOGNITION_MAX_WORKERS threads)
  │    _progressive_recognition
  │      → RecognitionPipeline.run(frame, track)
  │           face detect → quality gates → ArcFace embedding
  │           → vector search → memory → confidence → policy decide()
  │      → _handle_pipeline_result   (sole write-back site)
  │      → _cleanup_recognition_track → maybe _finalize_track
  │
  ├─ Queue consumer (ONE thread, main.py worker_process_tracks)
  │    TrackProcessor.process(track)
  │      snapshot → match → recognition+memory → decide()
  │      → auto-register → record visit → dispatch alert
  │      → log_event → broadcast → dashboard
  │
  └─ API thread ─────────────────── Uvicorn (FastAPI :8000)
       REST (faces/events/reports/chat) + WebSocket /ws/live
```

Two recognition implementations exist by design (documented, not a surprise):

- **Progressive** — `RecognitionPipeline.run()` in the recognition pool, runs
  mid-track against a live frame, every 20 frames or on quality improvement.
- **Finalization** — `TrackProcessor.process()` on the queue-consumer thread,
  re-runs matching/recognition/memory/policy from the track snapshot plus any
  `pending_*` results the progressive pass cached, because a finalizing track
  usually has no frame left.

Unifying them is a Phase 4 candidate (`run_final()`), see `plan.md`.

---

## 2. Component inventory

Format: **File → Responsibilities → Called by → Mutates → Threading → Known concerns**

### Entry point

**`main.py`**
- **Responsibilities:** wires everything: loads config, sets up logging and
  MongoDB, constructs `TrackProcessor` then `CameraAgent`, registers the three
  callbacks (`on_track_finalized`, `on_frame_annotated`), starts the queue
  consumer, prewarms models, starts the FastAPI thread, owns shutdown ordering
  (drain queue → executors → MongoDB → HTTP pools).
- **Called by:** process entry (`python main.py`).
- **Mutates:** module globals `track_queue`, `loop`, `track_processor`;
  `_shutdown_event`.
- **Threading:** main thread starts everything; queue consumer is one daemon
  thread; API server is one daemon thread; several one-shot prewarm/check
  daemon threads.
- **Known concerns:** the queue has a **single** consumer thread, not the
  "2 workers" / "4 background threads" the retired docs claimed. The 4-thread
  number refers to the separate recognition pool in `camera_agent`.

### Camera + progressive recognition

**`agents/camera_agent.py`**
- **Responsibilities:** capture loop (open/apply props/resize), runs YOLO +
  ByteTrack per frame, updates `TrackState`, IoU overlap dedup, schedules
  progressive recognition, schedules finalization of expired tracks, runs the
  recognition worker body and its write-back, debug diagnostics, FPS stats.
- **Called by:** `main.py` (constructor + callbacks); `_recognition_executor`
  invokes the worker methods.
- **Mutates:** `TrackState` (via its API), `Track` fields during write-back,
  own bookkeeping sets `_recognizing_tracks` / `_finalized_track_ids`.
- **Threading:** camera thread for the loop; `RECOGNITION_MAX_WORKERS`
  (default 4) recognition threads; finalization also submitted to that pool.
- **Known concerns:** no test file imports `CameraAgent`.
  `_recognizing_tracks` is set at worker start, not submit time, so the
  queue-time check alone is not airtight — the worker-start resolved-check is
  the real backstop. `_should_skip_recognition` mutates `rescan_attempts`
  (predicate with a side effect). `_process_tracks` rebuilds the active-track
  snapshot once per detection (quadratic per frame).

**`pipeline/tracker.py`**
- **Responsibilities:** `get_model()` (YOLO singleton, PyTorch or OpenVINO IR)
  and `track_persons()` (detect + ByteTrack step, returns raw detections).
- **Called by:** `camera_agent._loop` per frame; `_prewarm_yolo` in `main.py`.
- **Mutates:** the Ultralytics tracker's internal state only.
- **Threading:** camera thread; the model object is created once on first use.
- **Known concerns:** none known.

**`pipeline/track_state.py`**
- **Responsibilities:** thread-safe track registry; composite-ID formatting
  (`make_composite_id`); per-track accumulation with lock discipline; best-face
  hysteresis; embedding gate; visibility classification; expired-track sweep;
  `pending_*` recognition fields and their upgrade-only rules; generation
  bookkeeping for ByteTrack ID reuse.
- **Called by:** `camera_agent` (every frame), `track_processor` (read via
  snapshot), `finalizer` (through the `set_embedding` callback).
- **Mutates:** the `TrackState` dictionary and every `Track` it owns, always
  under a lock.
- **Threading:** camera thread writes, recognition workers write, queue
  consumer reads. Two-lock ordering rule: `TrackState._lock` may be taken
  before `track._lock`, never the reverse.
- **Known concerns:** no test file imports `CameraAgent`, but `TrackState`
  itself has dedicated tests (`tests/test_thread_safety.py`). The file reads
  `track._lock` directly from a couple of external sites, which is a
  encapsulation smell, not a bug.

**`pipeline/recognition_pipeline.py`**
- **Responsibilities:** one progressive pass — face detect → validity gates →
  quality scoring → embedding (quality-gated) → vector search → memory →
  confidence scoring → policy decision. Returns `PipelineResult` with metrics.
- **Called by:** `camera_agent._progressive_recognition` only.
- **Mutates:** nothing itself; callers write results back to the `Track`.
- **Threading:** runs entirely on one recognition worker; reads track state
  under lock, writes nothing.
- **Known concerns:** `SkipReason` (StrEnum) and plain `str` are both returned
  for skip reasons — types are inconsistent, documented only. This is the
  mid-track twin of `track_processor`'s finalization chain.

**`agents/finalizer.py`**
- **Responsibilities:** `retry_embedding()` — one last InsightFace attempt for
  a track that never produced an embedding, crop first, frame fallback behind
  `ENABLE_FULL_FRAME_FALLBACK`.
- **Called by:** `camera_agent._finalize_track`.
- **Mutates:** the `Track` via the injected `set_embedding` callback.
- **Threading:** recognition worker.
- **Known concerns:** `ENABLE_FULL_FRAME_FALLBACK` is `False` by default, so
  the frame fallback path is currently dead unless enabled.

**`agents/timing.py`**
- **Responsibilities:** `TimingCollector` — thread-safe submit/start/done
  timing for recognition diagnostics.
- **Called by:** `camera_agent` (submit, worker start, completion).
- **Mutates:** its own timing dictionaries under separate locks.
- **Threading:** called from camera and recognition threads.

### Finalization + dashboard relay

**`agents/track_processor.py`**
- **Responsibilities:** two jobs — `process()` runs the closing sequence
  (match → recognize → decide → register → record visit → dispatch →
  log → broadcast) for a finished track, and `handle_frame()` relays the
  throttled JPEG preview.
- **Called by:** `main.py` — `enqueue` from `camera_agent`'s finalization
  worker, `process` from the single queue-consumer thread, `handle_frame` from
  the camera thread.
- **Mutates:** `Track` (person name, alert timestamp, frame release); MongoDB
  (face registration, events, visit memory); WebSocket payloads.
- **Threading:** finalization is serialized on one consumer thread, so the
  body is written as if single-threaded; JPEG encoding runs on a 2-thread
  pool; broadcasts are posted onto the asyncio loop only if it is alive.
- **Known concerns:** no test file imports `TrackProcessor`. It constructs a
  fresh `RecognitionAgent()` per track (the only such site). `fresh_match`
  gates trust in three pending fields from one flag. `_broadcast_alert` puts
  `track.track_id` under the payload key `"person_id"` — an established
  frontend contract, documented as a hazard. `shutdown()` sets an event
  nothing in the class reads.

**`agents/policy.py`**
- **Responsibilities:** `decide()` — 9-rule priority tree producing a
  `DecisionResult` (status, alert level, reason, should_alert,
  should_register, NL summary), plus the `PolicyAgent` wrapper.
  Priority: blacklist > authorized > verified > auto-registered self-match >
  known visitor > matched-not-confirmed > hidden > masked/partial >
  after-hours unknown > office-hours unknown.
- **Called by:** both recognition chains — `recognition_pipeline.run()` and
  `track_processor.process()`.
- **Mutates:** nothing.
- **Threading:** pure function of its inputs; safe on any thread.
- **Known concerns:** must not grow into a second recognition pipeline (see
  `docs/ARCHITECTURE_RULES.md`). At finalization it defers to a stronger
  accumulated recognition status rather than re-deciding from scratch.

**`agents/recognition.py`** + **`agents/scoring.py`**
- **Responsibilities:** `RecognitionAgent` maps similarity + duration + mask +
  memory context to a status/confidence; `scoring.py` computes the weighted
  confidence (similarity 0.65, quality 0.15, track 0.10, memory 0.05,
  margin 0.05), mask penalty, and the calculation log.
- **Called by:** `recognition_pipeline`, `track_processor` (fresh instance
  per track), and by extension every `decide()` call.
- **Mutates:** nothing.
- **Threading:** stateless per call.
- **Known concerns:** the confidence max-upgrade gate lives in the callers
  (`TrackState` setters + `camera_agent`), not here — worth remembering when
  reading scoring in isolation.

**`agents/matching_agent.py`**
- **Responsibilities:** `run_matching_from_embedding()` — vector search
  against the `faces` collection, score conversion, threshold comparison,
  margin/second-best computation.
- **Called by:** `recognition_pipeline` (injectable, defaults to this) and
  `track_processor._run_matching`.
- **Mutates:** nothing (read-only DB query).
- **Threading:** synchronous; runs on whichever worker calls it.
- **Known concerns:** Atlas score must always be converted
  (`raw_cosine = (atlas_score * 2) - 1`) — see the invariant in `AGENTS.md`.

**`agents/memory.py`**
- **Responsibilities:** `MemoryAgent` — visit lookup, visit recording,
  typical-hours patterns, confidence boost for returning visitors.
- **Called by:** `recognition_pipeline` and `track_processor` (each holds its
  own instance).
- **Mutates:** the `visit_memory` collection.
- **Threading:** synchronous on the calling worker.

**`agents/alert_agent.py`**
- **Responsibilities:** `dispatch()` — per-level cooldown dedup, then console
  (synchronous) / email / SMS / webhook; LLM NL summary generated without
  blocking console output.
- **Called by:** `camera_agent._handle_decision_and_alert` (blacklist only,
  mid-track) and `track_processor._dispatch_alert` (finalization).
- **Mutates:** its own cooldown cache; external channels.
- **Threading:** called from recognition workers and the queue consumer;
  non-console channels are dispatched async.
- **Known concerns:** cooldown pruning is time-based to stop unbounded
  growth — correct, but the cache is module-global.

**`agents/report.py`**
- **Responsibilities:** `ReportAgent` — incident/stats/summary reports via
  the LLM with template fallback.
- **Called by:** `dashboard/backend/routes/reports.py` (via
  `asyncio.to_thread`).
- **Threading:** async route handlers hand it to a thread.

### Storage layer

**`utils/db_client.py`** — lazy thread-safe MongoDB singleton; collection
getters. Called by every `db_*` module.
**`utils/db_faces.py`** — face CRUD, deduplication
(`deduplicate_identity`), embedding history, quality-gated overwrite of
`latest_embedding`.
**`utils/db_events.py`** — event logging + dashboard stats queries.
**`utils/db_memory.py`** — visit memory CRUD.
**`utils/db_search.py`** — Atlas `$vectorSearch` + Python cosine fallback +
backfill.
**`utils/db_utils.py`** — re-export facade; consumers import from here even
though the logic lives in the five domain modules above.

- **Mutates:** MongoDB only.
- **Threading:** shared connection, used concurrently by camera-side workers
  and the queue consumer.
- **Known concerns:** the facade hides which domain module owns a function —
  read `db_utils.py` first when locating logic.

**`utils/embedding_utils.py`** — InsightFace singleton (SCRFD detection +
ArcFace embedding, CLAHE preprocessing, geometric mask heuristic). Loaded
once; never reloaded per frame. CPU execution provider.

**`utils/image_utils.py`** — crop/save, Cloudinary upload (or local
`captures/` fallback), track image URL resolution.

**`utils/llm_client.py`** — pooled `httpx` client for Ollama. Optional: the
system runs on template fallbacks when Ollama is down. 3 retries, cached
health check.

### Configuration + logging

**`config/settings.py`** — loads `.env` (secrets) then `config/config.jsonc`
(tunables) with env > jsonc > default precedence, casts types, warns on
duplicate keys, `validate_config()` on startup, logs effective settings.
- **Known concerns:** some defaults are duplicated between this file and
  `config.jsonc` — a de-duplication candidate parked in `REFACTOR_PLAN.md`.

**`config/status.py`** — `Status(IntEnum)` single source of truth,
`STATUS_LABELS`, `LABEL_TO_STATUS`, `RESOLVED_STATUSES`. Higher = more
trusted, `is_known = status >= 3`.

**`config/logging_setup.py`** — `Colors`, `CompactTerminalRenderer`
(terminal), `JSONFileRenderer` (`logs/surveillance.jsonl`), debug file,
`setup_logging()` with noisy-logger suppression. Terminal is allowlist-based
(see `docs/08 - Logging/Terminal Output Reference.md`).

### Dashboard

**`dashboard/backend/main.py`** — FastAPI app, CORS, startup/shutdown,
WebSocket `/ws/live` with origin validation.
**`routes/live.py`** — `broadcast_frame` / `broadcast_alert` /
`broadcast_event` coroutines; the receiving end of everything posted via
`run_coroutine_threadsafe`.
**`routes/faces.py`, `events.py`, `reports.py`, `chat.py`** — CRUD, stats,
LLM reports, chat with intent routing. All long work goes through
`asyncio.to_thread`.
**`dashboard/frontend/`** — React + Vite on 5173; consumes the WebSocket
(feed + events + alerts) and REST endpoints.

- **Threading:** API thread owns the asyncio loop; producers from other
  threads must post onto it (`track_processor` does so only while the loop is
  alive).

---

## 3. Tech stack (absorbed from `Tech Stack.md` / `ARCHITECTURE.md`)

| Component | Technology | Where |
|-----------|-----------|-------|
| Person detection | YOLOv8 `yolov8s` (PyTorch or OpenVINO IR) | `pipeline/tracker.py` singleton |
| Multi-object tracking | ByteTrack, `config/bytetrack_surveillance.yaml` | `track_high_thresh=0.45`, `track_buffer=60`, `new_track_thresh=0.50` |
| Face detection | InsightFace SCRFD (`buffalo_l`), CLAHE first | `utils/embedding_utils.py` singleton |
| Face embedding | ArcFace, 512-dim L2-normalized | quality-gated: overwrites only if `det_score` improves by ≥ 0.05 |
| Mask detection | Geometric landmark heuristic, no classifier | lower/upper face ratio < `MASK_RATIO_THRESHOLD` (0.3) |
| Face quality | Two-tier: validity gates then weighted score | `pipeline/quality_agent.py` |
| Vector search | MongoDB Atlas `$vectorSearch`, index `vector_index`, 512d cosine | `utils/db_search.py` + Python fallback |
| Database | MongoDB Atlas, 3 collections | `utils/db_*` modules |
| Image storage | Cloudinary with local `captures/` fallback | `utils/image_utils.py` |
| Local LLM | Ollama (Gemma 3 4B / Qwen 3.5 4B), `httpx` | `utils/llm_client.py`, optional |
| Dashboard API | FastAPI + Uvicorn, port 8000 | `dashboard/backend/` |
| Dashboard UI | React + Vite, port 5173 | `dashboard/frontend/` |
| Configuration | `.env` + `config.jsonc` | `config/settings.py` |
| Logging | structlog, 3 tiers | `config/logging_setup.py` |

Key libraries: `ultralytics`, `insightface`, `onnxruntime`, `opencv-python`,
`pymongo`, `fastapi`, `uvicorn`, `structlog`, `numpy`, `httpx`, `requests`,
`python-dotenv`, `cloudinary`.

Hardware acceleration: YOLO on CPU by default; OpenVINO IR export for Intel
Arc iGPU (`yolo export ... format=openvino half=True`, device `intel:GPU`);
CUDA via `YOLO_DEVICE=0`. InsightFace always runs on CPU.

---

## 4. Connection patterns (absorbed from `ARCHITECTURE.md`)

- **Camera loop never blocks** — all slow I/O (MongoDB, Cloudinary, alerts)
  happens on workers behind a `queue.Queue`.
- **Thread safety** — `TrackState` lock for track mutations; two-lock ordering
  `TrackState._lock` → `track._lock`, never reversed; JPEG encoding outside
  the lock.
- **Progressive caching** — mid-track results are cached on the track as
  `pending_*` fields and reused at finalization to avoid duplicate DB calls.
- **Load models once** — YOLO in `tracker.py`, InsightFace in
  `embedding_utils.py`.
- **Quality gates before storage** — invalid faces (blur < 40, brightness
  outside 35–255, area < 1200 px²) never produce an embedding that is stored
  or searched.
- **Confidence never downgrades** — upgrades only, except critical alerts
  which always update.
- **Composite track IDs** — `{camera_id}_{session_epoch}_{bt_id}_{generation}`,
  unique across camera restarts and ByteTrack ID reuse. (The retired docs
  omitted the generation suffix; `make_composite_id` includes it.)
- **Status is numeric** — `Status.X` from `config/status.py`, never raw
  strings.
- **Atlas score conversion** — `raw_cosine = (atlas_score * 2) - 1` before any
  comparison against `MATCH_THRESHOLD`.
- **LLM optional** — template fallbacks keep the system running without Ollama.
- **One decision per track** — not per-frame; progressive passes upgrade, the
  final pass is authoritative.

---

## 5. MongoDB collections (absorbed from `ARCHITECTURE.md`)

| Collection | Vector index | Key fields |
|-----------|--------------|------------|
| `faces` | `vector_index` on `latest_embedding` (512d cosine) | person_id, name, role, tags, verified, alert_level, images[], embeddings[], latest_embedding, mean_embedding, latest_embedding_quality, embedding_model, person_crop_url, source, quality_scores, verified_at, verified_by, created_at, updated_at |
| `events` | none | track_id, camera_id, timestamp, status, alert_level, person_id, similarity_score, name, is_masked, image_url, person_crop_url, reason, alerted |
| `visit_memory` | none | person_id (unique), visit_count, typical_hours[], typical_cameras[], avg_similarity, first_seen, last_seen, last_camera, last_status, best_status, similarity_history[], status_history[], created_at, updated_at |

---

## 6. Key thresholds (read from config)

| Setting | Value | Note |
|---------|-------|------|
| `MATCH_THRESHOLD` | 0.45 | never raise above 0.45; Atlas scores convert first |
| `HIGH_CONFIDENCE_SIMILARITY` | 0.85 | skips memory + recognition |
| `CONFIDENCE_KNOWN_MIN` | 70 | policy gate for "known" |
| `CONFIDENCE_UNCERTAIN_MIN` | 55 | policy gate for "uncertain" |
| `KNOWN_VISITOR_CONFIDENCE` | 80 | **paired threshold under review** — differs from `CONFIDENCE_KNOWN_MIN` |
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
| Confidence weights | sim 65% / quality 15% / track 10% / memory 5% / margin 5% | `agents/scoring.py` |
| `TRACK_TIMEOUT_SECS` | 15.0 | person gone this long ends the track |
| `MAX_TRACK_SECS` | 300.0 | hard cap on track lifetime |
| `RECOGNITION_INTERVAL_FRAMES` | 20 | progressive recognition cadence |
| `RECOGNITION_MAX_WORKERS` | 4 | recognition pool size |
| `FRAME_SKIP` | 2 | preview frames dropped |
| `BROADCAST_WIDTH/HEIGHT` | 960 × 540 | preview broadcast size |
| `JPEG_QUALITY_BROADCAST` | 85 | preview JPEG quality |
| `ALERT_COOLDOWN_SECS` | 60 | per-level dedup window |
| `LOITER_SECS` | 30 | loitering detection |
| `OVERLAP_IOU_THRESHOLD` | 0.5 | track overlap dedup |
| `MIN_VISIT_GAP_SECS` | 60 | visit dedup window |
| `MASK_RATIO_THRESHOLD` | 0.3 | geometric mask heuristic |

---

## 7. Document map

| Document | Role |
|----------|------|
| `AGENTS.md` | Operating contract for AI agents (workflow, invariants, do/don't) |
| `docs/CURRENT_ARCHITECTURE.md` | **this file** — as-built inventory |
| `docs/ARCHITECTURE_RULES.md` | semantic boundaries (what must not change) |
| `docs/REFACTOR_PLAN.md` | approval-gated change candidates |
| `docs/02 - Architecture/Data Flow.md` | per-frame step-by-step walkthrough |
| `docs/02 - Architecture/Thread Architecture.md` | every thread and its duties |
| `docs/02 - Architecture/MongoDB Schema.md` | collections and indexes in depth |
| `docs/04 - Pipeline/Data Models.md` | data/state at each pipeline stage |
| `plan.md` | phased roadmap this document belongs to |
