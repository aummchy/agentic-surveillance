# All Implemented Features

Complete inventory of features currently implemented in `agentic_ai_singlecam`.
Last verified against code: 2026-09-30.

---

## 1. Surveillance Pipeline (capture → decision)

### Camera capture — `agents/camera_agent.py`
- Camera source resolution: `CAMERA_SOURCE` (RTSP/HTTP/file) overrides `CAMERA_INDEX`
- Configurable OpenCV backend (`dshow`/`msmf`/auto) + read/open timeouts for network streams
- Auto-reconnect on consecutive read failures with backoff; file sources end at EOF
- Per-frame resize to configured resolution before tracking
- **IoU overlap de-duplication** — skips scheduling for tracks overlapping an active track above `OVERLAP_IOU_THRESHOLD`
- **Periodic progressive recognition** — every `RECOGNITION_INTERVAL_FRAMES` (default 20) on a dedicated `ThreadPoolExecutor`
- Recognition throttles: already-resolved status, high-confidence match, rescan interval + `MAX_RESCAN_ATTEMPTS`, quality-improvement gate (`MIN_QUALITY_IMPROVEMENT` ≥ 0.10)
- One in-flight recognition task per track; duplicate-alert/finalization guards
- **Confidence never downgrades** (`update_confidence_if_higher`); critical alerts dispatch mid-track, others deferred to finalization; per-track `ALERT_COOLDOWN_SECS`
- Idempotent finalization with one last embedding retry

### Detection & tracking — `pipeline/tracker.py`
- YOLOv8 model singleton (thread-safe, load once) with OpenVINO option
- Custom ByteTrack config `config/bytetrack_surveillance.yaml` tuned for fixed-camera surveillance
- Person-class filter, `PERSON_CONF_THRESHOLD`, NMS IoU, device selection
- Debug dumps of raw detections / ByteTrack output

### Face pipeline — `pipeline/recognition_pipeline.py`
- Stage orchestration: **detect → quality → embed → match → memory → recognize → decide**
- Person-box expansion (`PERSON_BOX_EXPANSION_RATIO`) before cropping
- **Two-pass face detection**: relaxed detection on person crop, then full-frame fallback
- Best-face selection preferring crop over frame; bbox coordinate translation
- Skip reasons: `high_confidence`, `no_face`, `low_quality`, `embedding_failed`
- Per-stage timing metrics (`RecognitionMetrics`)

### Quality gates — `pipeline/quality_agent.py`
- Validity gates: blur (Laplacian variance) ≥ 40, brightness V-channel in [35, 255], face area ≥ 1200 px²
- Composite quality score: blur 50% + brightness 25% (center-radius model) + area 25%
- `compute_face_ratio()` — face area / person area for visibility classification
- Low-quality faces never get embeddings stored or searched

### Embedding & matching — `utils/embedding_utils.py`, `agents/matching_agent.py`, `utils/db_search.py`
- InsightFace singleton with serialized inference lock
- Adaptive CLAHE preprocessing (only when contrast is low)
- **Geometric mask heuristic** — lower_face/upper_face landmark ratio (no classifier)
- Atlas `$vectorSearch` on `latest_embedding` (index `vector_index`) with **Python cosine-scan fallback**
- Atlas score → raw cosine conversion; `compare_similarity()` canonical threshold check
- Top-1 identity + second-best similarity + **margin** + candidate count
- Embedding history capped (`EMBEDDING_HISTORY_CAP`) with L2-normalized `mean_embedding`
- Quality-gated `latest_embedding` overwrite (only when quality improves)
- Startup backfill of empty `latest_embedding` from history; Atlas index existence check

### Identity deduplication — `utils/db_faces.py`
- `deduplicate_identity()` — Atlas-first top-5 decision, Python fallback, MERGED / NEW / FAILED outcomes
- Verified-identity conflict skip

### Visit memory — `agents/memory.py`, `utils/db_memory.py`
- Visit history: visit count, first/last seen, days since last visit, avg similarity
- Typical hours / typical cameras; **±2h typical-visit-time check** (wrap-around aware)
- Memory confidence boost: per-visit (capped), recency, similarity consistency, typical time (+2) / camera (+1), similarity-drop penalty
- **Visit-gap suppression** — duplicates within `MIN_VISIT_GAP_SECS` don't increment
- Best-status tracking (highest-trust status ever seen, backward-compatible with legacy string statuses)
- Atomic upsert (`$setOnInsert`) + single-op visit update

---

## 2. Recognition & Decisions

### Confidence scoring — `agents/scoring.py`
- **Weighted-normalization formula**: `base = 0.65·sim + 0.15·quality + 0.10·track + 0.05·memory + 0.05·margin`
- **Mask penalty**: `base × (1 − 0.15 × mask)`
- Final mapping to 1–100; raw-cosine range guard
- `is_match()` binary gate at `MATCH_THRESHOLD` (≤ 0.45)
- `confidence_status()` matrix → `KNOWN` (≥70) / `UNCERTAIN` (≥55) / `UNKNOWN`
- **Calculation log** — full tabular breakdown per computation in `logs/calculation.log` with formula header at startup, size-based rotation

### Policy rules — `agents/policy.py`
Ordered rule chain (first match wins):

| Rule | Condition | Result |
|------|-----------|--------|
| R1 | Blacklist tag | `BLACKLIST`, CRITICAL, alert |
| R2 | Authorized tag | `AUTHORIZED`, no alert |
| R3 | Verified + sim ≥ threshold | `VERIFIED`, no alert |
| R4a | Auto-registered tag + high sim | `KNOWN_VISITOR` |
| R4b | Matched + memory `is_known` | `KNOWN_VISITOR` with visit count |
| R5 | Matched (rec-known / high sim / ≥ threshold) | `KNOWN_VISITOR` / `KNOWN` |
| R6 | Hidden visibility | `HIDDEN`, HIGH, alert |
| R7 | Masked / partial visibility | `MASKED_UNKNOWN` (HIGH + loitering after `LOITER_SECS`) |
| R8 | After-hours unknown | `UNKNOWN`, HIGH, alert |
| R9 | Default unknown (office hours) | `UNKNOWN`, MEDIUM, alert + register |

- Office-hours window + weekday check; every decision logged as `policy_decision`

### Status system — `config/status.py`
- `Status(IntEnum)` 9 levels: UNKNOWN=1 … HIDDEN=9; higher = more trusted; `is_known = status >= 3`
- `AlertLevel` (none/low/medium/high/critical), `Visibility`, `SkipReason` enums
- `STATUS_LABELS` + reverse lookup; `RESOLVED_STATUSES` / `UNVERIFIED_STATUSES` sets
- Numeric everywhere: MongoDB, API, logs, frontend; migration script `scripts/migrate_status_ints.py [--dry-run]`

---

## 3. Track Processing & Registration

### Finalization — `agents/track_processor.py`
- 2 worker threads drain a `queue.Queue` of finalized tracks (camera loop never blocks)
- Flow: snapshot → image URLs → match → recognition/memory → decide → register → visit → alert → event log → WebSocket broadcast
- **Auto-registration of unknowns** with `auto_registered` tag; quality gate `REGISTRATION_QUALITY_MIN`
- Identity dedup before insert; person-name update only upgrades
- Reuses pending (progressive) match/recognition/memory results
- Frame broadcast pipeline: `FRAME_SKIP` + resize (640×360) + JPEG + WebSocket

### Last-chance embedding — `agents/finalizer.py`
- Retries embedding only when missing; tries best face crop then full frame (relaxed det-score)
- Quality-gated write-through; `no_embedding_after_retries` diagnostic

### Track state — `pipeline/track_state.py`
- **Composite track IDs** `{camera_id}_{session_epoch}_{byte_track_id}` with generation bookkeeping for ByteTrack ID reuse
- Thread-safe global store with `threading.Lock`
- **Expiry with deferred removal** while recognition is in flight
- **Visibility classification**: VISIBLE / PARTIAL / HIDDEN / UNKNOWN
- Quality-gated best-face update and embedding storage (improvement thresholds)
- `ensure_fallback_frame()` — guarantees ≥1 photo per track
- Upgrade-only pending caches (match / memory / recognition)
- IoU dedup snapshot; recognition snapshot for throttling

---

## 4. Alerts & LLM

### Alert dispatch — `agents/alert_agent.py`
- **4 channels** via `ALERT_CHANNELS`: console, email (SMTP+STARTTLS), SMS (Twilio), webhook (POST JSON)
- Per-track / per-level cooldown dedup; distinct keys so different unknowns don't suppress each other
- Stale-timestamp pruning; one-shot `track.alerted` gate
- Rich payload: track, status, level, person, reason, masked, time, camera, image URL, memory context
- Console synchronous; network channels on a 2-worker thread pool
- Email includes NL summary + image link; SMS = level prefix + summary

### LLM integration — `utils/llm_client.py` (Ollama)
- Pooled HTTP client with timeouts and retry (`LLM_MAX_RETRIES`)
- `generate()`, `chat_completion()` (stateless)
- **`generate_nl_summary()`** — 1–2 sentence alert description
- **`generate_incident_summary()`** — structured `Summary:` / `Recommendation:` output
- **`generate_executive_summary()`** — period summary from stats + recent events
- Availability probe (cached 10 s), honors `LLM_ENABLED`; graceful template fallback when offline
- Model swap via `OLLAMA_MODEL` (Gemma 3 4B / Qwen 3.5 4B)

### Reports — `agents/report.py`
- **incident** — status-specific title, visit-history context, recommendation, LLM-enhanced with template fallback
- **summary** — daily/weekly: stats, status/camera breakdowns, peak hour, top events, LLM executive summary
- **person** — visit count, first/last seen, avg similarity, "usually visits around H:00" pattern
- **stats** — dashboard aggregates

---

## 5. Dashboard API — `dashboard/backend/`

### REST endpoints
| Area | Endpoints |
|------|-----------|
| Faces | `GET /api/faces` (status filters, paged), `GET /api/faces/{id}`, `POST /api/faces/{id}/verify`, `PUT /api/faces/{id}`, `DELETE /api/faces/{id}` |
| Events | `GET /api/events/stats`, `GET /api/events` (status filter, joined with face images), `GET /api/events/unknown`, `GET /api/events/alerts` |
| Reports | `GET /api/reports/stats`, `GET /api/reports/summary?period=daily\|weekly`, `GET /api/reports/person/{id}`, `GET /api/reports/incidents` |
| Chat | `POST /api/chat`, `GET /api/chat/health` |
| Misc | `GET /`, `GET /health`, static `GET /captures` |

- CORS with explicit origins/methods/headers; Pydantic request/response models

### Chat (keyword RAG) — `routes/chat.py`
- Intent routing: stats/status → `get_stats`; unknowns → `get_unknown_faces`; recent/alerts → events; memory/visitors → `get_memory_stats`
- System prompt with data-availability rules + 20-turn history
- Fallback chain: LLM → raw JSON/templated summary → graceful "Ollama not available"

### WebSocket — `routes/live.py`
- `WS /ws/live`: base64 JPEG `frame` (1 MB cap), `event`, `alert` JSON, `ping`→`pong`
- **Origin validation** (close code 4003 on mismatch), 3 s send timeout, auto-drop silent clients

---

## 6. Dashboard Frontend — `dashboard/frontend/` (React + Vite)

- **Shell**: stats bar (4 tiles, 10 s poll), two-tab nav (Dashboard / Manage), "System Active" indicator
- **WebSocket client**: auto-reconnect 500 ms, 30 s ping keepalive, handles frame/event/alert
- **Live feed**: connection status, waiting placeholders, red "UNKNOWN PERSON DETECTED" overlay
- **Event log**: 5 tab filters (All/Unknown/Masked/Blacklist/Verified), live prepend with dedupe, 60 s poll, status icons, relative + absolute time, similarity %, thumbnails
- **Alerts**: toast popups (critical/high/masked/unknown variants, auto-dismiss 8 s) + rolling sidebar (20 alerts, dismiss/Clear All)
- **AI Assistant chat**: health badge poll (30 s), typing indicator, data-source badges, 4 suggested queries, Enter-to-send, disabled when offline
- **Unknown Persons grid**: crops, first/last seen, "NEW" highlight (<30 s, sessionStorage), Verify button
- **Verified Persons grid**: name, alert-level badge, delete with inline confirm
- **Verify modal**: preview image, name input, alert-level select, validation
- Status constants mirror backend `Status` enum; axios with timeout + error interceptor; per-panel ErrorBoundary
- Dev proxy for `/api`, `/ws`, `/captures` → :8000

---

## 7. Configuration & Logging

### Configuration — `config/settings.py` + `config/config.jsonc`
- Priority chain: `.env` (secrets) > `config.jsonc` (tunables) > hardcoded defaults
- Custom JSONC parser (comments), list/bool element-type casting (`OFFICE_DAYS` → ints)
- Startup **effective-settings log** (`config.resolved` per setting + source)
- Duplicate-key warning across `.env` / `config.jsonc`
- `validate_config()`: requires `MONGODB_URI`, caps `MATCH_THRESHOLD ≤ 0.45`, validates timeouts, det-score ranges, alert channels, quality weights sum ≈ 1.0
- Tunable groups: matching, models, storage, camera, YOLO, quality gates/scoring, JPEG, vector search, thresholds, confidence weights, debug, CLAHE, mask/loiter, office hours, alerting, LLM, broadcast, etc.

### Logging — `config/logging_setup.py` (3 tiers)
1. **Terminal** — compact colored one-liners, INFO+, gated by `TERMINAL_ALLOWLIST`
2. **JSON file** — `logs/surveillance.jsonl`, DEBUG+, 5 MB × 5 rotation
3. **Debug file** — `logs/surveillance.debug.log`, DEBUG+, 10 MB × 3 rotation
- Plus `logs/calculation.log` (confidence breakdowns)
- Suppressed noisy loggers (pymongo, insightface, ultralytics, cloudinary)

---

## 8. Image & Storage Utilities — `utils/image_utils.py`

- Cloudinary upload (numpy → JPEG → `secure_url`) with lazy config + connection pool; no-op when unconfigured
- Blur score (Laplacian), brightness (HSV V), person crop (boundary-clipped), IoU
- `resolve_track_image_url()` / `resolve_track_person_crop_url()` — priority URL resolution, Cloudinary first, local `captures/` fallback
- `draw_annotations()` — status-colored boxes, name + status label, `MASK` badge

---

## 9. System Lifecycle — `main.py`

- Config validation at startup (exits on error); formula header written to calculation log
- Background LLM availability check (non-blocking)
- Track worker pool (2 threads) + FastAPI/uvicorn on `0.0.0.0:8000` in its own thread
- Background Mongo checks (vector index + embedding backfill) and model pre-warm (YOLO + InsightFace)
- Blocking capture loop on main thread; **ordered graceful shutdown**: stop event → uvicorn exit → drain queue → track processor → alerts → LLM → Mongo client

---

## 10. Tests, Scripts & Docs

### Tests — `tests/` (6 suites, 84 tests)
- `test_recognition.py` — confidence math, clipping, mask penalty, thresholds
- `test_recognition_pipeline.py` — end-to-end decisions, alerts, confidence bounds
- `test_recognition_pipeline_stages.py` — stage skips, agent injection, metrics
- `test_embedding_history.py` — cap, dedup, slicing
- `test_thread_safety.py` — visit counting, concurrent expiry, composite IDs
- `test_track_finalizer.py` — registration/merge/visit/alert rules

### Scripts — `scripts/`
- `migrate_status_ints.py [--dry-run]` — string → `Status` int migration
- 11 session query tools (`query_*.py`) for local session-store introspection

### Docs — `docs/`
- Home, Getting Started, Architecture, Agents (10), Pipeline (6), Utilities (4), Formulas (6), Dashboard (4), Logging (3), Reference (3), Problems (6), Issues (1)
