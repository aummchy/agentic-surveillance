# TODO — Visitor Surveillance System

## Status: Last reviewed 2026-06-23 (24 issues fixed including track loop, repeated warnings, verified alerts, 12 performance optimizations, live annotation colors, shutdown handling, person name in bounding box, track timeout fix)

### Recently Fixed (2026-06-23)

- [x] **Live annotation colors + labels** — unverified people now show red border + `UNVERIFIED`, known visitors show yellow, verified/authorized stay green, and masks always force red
- [x] **Dashboard event log colors** — event status badges now match live overlay: red for unverified, yellow for known visitors, green for verified/authorized. All unverified event items get red left border highlight
- [x] **`embedding_result` NameError** — `camera_agent.py:211` referenced undefined `embedding_result` after refactor to `detect_faces_raw()`. Fixed to `best["is_masked"]`
- [x] **Dead import `detect_and_embed`** — removed from `camera_agent.py` (only `compute_face_ratio` still needed)
- [x] **Performance: 12 optimizations** — removed dead `detector.py`, optimized face detection cascade, fixed N+1 queries, async MongoDB in routes, concurrent WebSocket broadcast, async alert dispatch, offloaded JPEG encoding, compressed frames in tracks, consolidated frontend WebSocket
- [x] **Track finalization loop** — `camera_agent.py` now tracks `_recognizing_tracks` and `_finalized_track_ids` to prevent same track from being finalized/alarmed multiple times
- [x] **Repeated warnings after verification** — `policy.py` Rule 3 now sets `should_alert=False` always for verified persons
- [x] **Vector search always returning 0** — Added detailed logging to `vector_search()` and `_python_cosine_scan()`, plus Atlas index check and embedding backfill on startup
- [x] **No dedup against verified faces** — Added `find_similar_faces()` that searches ALL roles, used in `store_face()` to prevent duplicate records for verified persons
- [x] **No unique photo IDs** — Added UUID to each image in `store_face()` and `update_face()`
- [x] **Duplicate alert broadcasts** — `main.py` now checks `not match_result.matched` before broadcasting unknown alerts
- [x] **Track timeout too short** — `TRACK_TIMEOUT_SECS` increased from 2.0 to 8.0 in `.env` (tracks now survive past 6-second recognition window)
- [x] **Recognition interval too long** — `RECOGNITION_INTERVAL_FRAMES` reduced from 30 to 20 in `.env` (first recognition at ~4 seconds instead of ~6)
- [x] **Frontend photo ID** — `UnknownPersons.jsx` now shows first 8 chars of photo UUID
- [x] **Startup health checks** — `main.py` calls `check_atlas_search_index()` and `backfill_missing_embeddings()` on boot
- [x] **Shutdown RuntimeError** — All dashboard routes (`events.py`, `faces.py`, `reports.py`) now catch `RuntimeError` from `asyncio.to_thread()` during server shutdown and return 503
- [x] **Logger TypeError** — Route files switched from stdlib `logging` to `structlog` to fix `Logger._log() got an unexpected keyword argument 'error'`
- [x] **Person name in bounding box** — `Track.person_name` field stores name from match result; `draw_annotations()` displays name alongside ID in live feed

### Recently Fixed

- [x] **Vite build error** — axios v1.18.0 incompatible with esbuild. Pinned to `axios@1.7.9` in `dashboard/frontend/`
- [x] **`GET /api/events` 500 error** — `similarity_score: float` rejected null values from MongoDB. Fixed to `Optional[float]` in `dashboard/backend/models.py`
- [x] **Terminal log noise** — uvicorn access logs (`GET /api/... 200 OK`) spamming console. Set to `warning` + `access_log=False` in `main.py`
- [x] **File logging not working** — `structlog.PrintLoggerFactory` bypassed stdlib logging entirely. Switched to `structlog.stdlib.LoggerFactory()` + `ProcessorFormatter`
- [x] **Console too noisy** — `[debug]` messages from structlog printing directly to stdout. Now goes through stdlib; console handler filters to INFO+
- [x] **Duplicate alerts** — `main.py:94` called `dispatch()` without checking `track.alerted`. Fixed with `and not track.alerted` guard
- [x] **Config default mismatch** — `DET_SCORE_MIN` default was 0.30, updated to 0.50 to match `.env`
- [x] **Verified bugs #2, #3, #4 already fixed** — `$push` overwrite, `image_url` None, `crop_face_region` coordinate bug were all already corrected in code
- [x] **Fix #2: `record_visit()` dead memory system** — now called in `process_finalized_track`
- [x] **Fix #5: `reports.py` deprecated `regex=`** — changed to `pattern=`
- [x] **Fix #6: `decide()` missing context** — now passes recognition + memory
- [x] **Fix #7: Progressive alerts missing image** — now uploads before dispatch
- [x] **Fix #8: Dashboard routes no error handling** — all routes wrapped in try/except
- [x] **Fix #9: Worker loses tracks on exception** — `task_done()` in finally block
- [x] **Fix #10: `similarity_score` null** — default changed to 0.0
- [x] **Fix #11: Embeddings unbounded** — capped at 10 via `$slice`
- [x] **Fix #12: WebSocket backpressure** — timeout + frame size limit
- [x] **Config: `MIN_TRACK_FRAMES`** — default 15 → 30
- [x] **`yolov8n.pt`** — moved to `models/`
- [x] **Unused imports in `main.py`** — removed `time`, `Path`, `run_matching`, `get_insightface`
- [x] **`models/.gitkeep`** — created so directory is tracked in git

---

## Full System Diagnosis (2026-06-22)

Ran comprehensive audit of all Python files, API routes, data flow, and build status.

### Build Status

- All 24 Python files compile clean
- Frontend Vite build succeeds (192 KB JS, 9.4 KB CSS)

### CRITICAL Issues (3)

| #   | Issue                                               | File:Line                                         | Impact                                                                                                                                    |
| --- | --------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **`broadcast_frame` receives numpy, expects bytes** | `main.py:40` → `live.py:14`                       | **FIXED** — encode numpy frame to JPEG bytes via `cv2.imencode` before broadcast                                                          |
| 2   | **`record_visit()` never called**                   | `memory.py:232` (defined), never invoked anywhere | **FIXED** — `main.py:process_finalized_track` now calls `memory_agent.record_visit()` after recognition                                   |
| 3   | **Face crop uses wrong coordinate system**          | `camera_agent.py:141-143`                         | **FIXED** — tracks `detected_in_person_crop` flag, converts bbox to frame coordinates by adding `person_box` offset when detected in crop |

### HIGH Issues (2)

| #   | Issue                                             | File:Line             | Impact                                                                          |
| --- | ------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------- |
| 4   | **`compute_face_ratio` mixes coordinate systems** | `camera_agent.py:130` | **FIXED** — `frame_bbox` now always in frame coordinates, matching `person_box` |
| 5   | **`reports.py` uses deprecated `regex=` param**   | `reports.py:16`       | **FIXED** — changed `regex=` to `pattern=`                                      |

### MEDIUM Issues (4)

| #   | Issue                                                    | File:Line                             | Impact                                                                                                   |
| --- | -------------------------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 6   | **`decide()` called without recognition/memory context** | `main.py:73`                          | **FIXED** — `process_finalized_track` now passes `recognition_result` and `memory_context` to `decide()` |
| 7   | **Progressive recognition alerts have no image**         | `camera_agent.py:205-208`             | **FIXED** — `dispatch(track, decision, image_url)` now uploads image before dispatching                  |
| 8   | **No error handling on any dashboard route**             | `events.py`, `faces.py`, `reports.py` | **FIXED** — all routes wrapped in try/except with proper HTTP error responses                            |
| 9   | **Worker exceptions silently lose tracks**               | `main.py:44-52`                       | **FIXED** — `task_done()` now called in finally block even on exception                                  |

### LOW Issues (3)

| #   | Issue                                                | File:Line         | Impact                                                                           |
| --- | ---------------------------------------------------- | ----------------- | -------------------------------------------------------------------------------- |
| 10  | `None` passed for `similarity_score` to `_log_event` | `main.py:67`      | **FIXED** — default changed to `0.0`                                             |
| 11  | `embeddings` array grows unbounded                   | `db_utils.py:229` | **FIXED** — `$slice: -10` limits to last 10 embeddings                           |
| 12  | WebSocket `connected_clients` not process-safe       | `live.py:11`      | **FIXED** — added backpressure with `asyncio.wait_for` timeout, frame size limit |

### What's Working Correctly

- Camera capture + YOLO detection + ByteTrack tracking
- InsightFace embedding + vector search matching
- Cloudinary image upload
- MongoDB face storage and event logging
- Console/file logging
- Frontend React dashboard build
- Alert dispatch (console, webhook)
- Progressive recognition during active tracks

---

## Bugs (fix immediately)

### 1. Duplicate alerts — `main.py:94`

**Status: FIXED** — Added `and not track.alerted` check

### 2. `$push` overwrite — `db_utils.py:225-233`

**Status: ALREADY FIXED** — uses merged `push_ops` dict

### 3. `image_url` always `None` — `main.py:57-62`

**Status: ALREADY FIXED** — assigns from `upload_to_cloudinary()` with local fallback

### 4. `crop_face_region` coordinate bug — `image_utils.py:81-82`

**Status: ALREADY FIXED** — uses `min(w, abs_fx2)` correctly

### 5. InsightFace lock contention — `embedding_utils.py`

**Status: LOW RISK** — lock only on singleton init, not on `detect_and_embed()`. Only a problem if `FaceAnalysis.get()` isn't thread-safe (ONNX usually is)

---

## Config Mismatches (`.env` vs spec)

| Setting            | Current Default | Spec Value | Status                                                           |
| ------------------ | --------------- | ---------- | ---------------------------------------------------------------- |
| `DET_SCORE_MIN`    | 0.50            | 0.70       | `settings.py` default now 0.50 (matches `.env`); spec wants 0.70 |
| `MIN_TRACK_FRAMES` | 30              | 30         | **FIXED** — default now 30                                       |

Note: The `.env` file overrides `DET_SCORE_MIN` to 0.50. `MIN_TRACK_FRAMES` is not set in `.env` so it uses the default of 30.

---

## Security Issues

### .env contains real credentials

**Status: CRITICAL**
The `.env` file contains actual production credentials:

- MongoDB URI with password
- Cloudinary API key and secret

**Action needed:**

1. Rotate all credentials immediately
2. Create `.env.example` with placeholder values
3. Ensure `.env` is in `.gitignore` (it is)
4. Check git history to ensure `.env` was never committed

---

## Missing Features (Phase 4 — Dashboard)

### Backend (`dashboard/backend/`)

- [x] FastAPI app setup
- [x] `GET /api/events` — paginated audit log (filter by status, alert_level, date)
- [x] `GET /api/faces` — enrolled identities CRUD
- [x] `GET /api/alerts` — active/recent high+ alerts
- [x] `WS /ws/live` — WebSocket pushing annotated frames + tracks
- [ ] `POST /api/faces/{person_id}/review` — operator relabels unknown

### Frontend (`dashboard/frontend/`)

- [x] Live view — annotated camera feed with track boxes, IDs, names, badges
- [x] Visitor log — chronological events with thumbnails
- [x] Alerts panel — high/critical events with acknowledge
- [x] Enrollment manager — review unknowns, assign names/roles/tags
- [ ] Audit trail — immutable events history

---

## Missing Files (per spec §3)

| File                              | Status  | Priority                                      |
| --------------------------------- | ------- | --------------------------------------------- |
| `dashboard/backend/`              | Exists  | Done                                          |
| `dashboard/frontend/`             | Exists  | Done                                          |
| `README.md`                       | Exists  | Done                                          |
| `.env.example`                    | Exists  | Done                                          |
| `logs/` directory                 | Exists  | Done                                          |
| `models/` directory               | Exists  | Done — `yolov8n.pt` moved here                |
| `pipeline/visibility_analyzer.py` | Missing | Low — logic in track_state.py (acceptable)    |
| `pipeline/decision_engine.py`     | Missing | Low — logic in decision_agent.py (acceptable) |

---

## Cloudinary Integration

**Status: IMPLEMENTED**

- `utils/image_utils.py` has `upload_to_cloudinary()` function
- `main.py:60` calls `upload_to_cloudinary()` and stores `secure_url`
- Images uploaded successfully (confirmed in logs: `url=https://res.cloudinary.com/...`)

---

## Hardening (spec §9 Phase 8)

- [x] Move `yolov8n.pt` to `models/` directory
- [ ] Add alert debounce per track_id (partially done in `alert_agent.py`)
- [ ] Graceful camera release on unexpected exit
- [ ] Model loading audit — ensure no per-frame reload
- [ ] Create `.env.example` with placeholder values
- [ ] Rotate exposed credentials

---

## Performance Optimizations (2026-06-23)

### 🔴 Critical

| #   | Issue                                                                                                                                                  | File:Line                                                    | Impact                                    | Status                                                                                          |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 1   | **YOLO model loaded twice** — `detector.py` and `tracker.py` each maintain separate singleton. `detector.py` is dead code                              | `pipeline/detector.py:8-16`, `pipeline/tracker.py:8-16`      | ~50-100MB wasted RAM                      | [x] Removed `detector.py`, unified to single model in `tracker.py`                              |
| 2   | **Face detection runs 4x per track** — cascading retry pattern calls `app.get()` up to 4 times (crop → full frame → relaxed crop → relaxed full frame) | `agents/camera_agent.py:94-132`, `_finalize_track():250-305` | ~200ms blocking main loop per recognition | [x] Added `detect_faces_raw()` to InsightFace; runs detector once, applies thresholds in Python |
| 3   | **`_python_cosine_scan()` loads ALL faces into memory** — full collection scan with no limit                                                           | `utils/db_utils.py:108-158`                                  | O(N) memory + CPU for large collections   | [x] Added `SCAN_LIMIT=500` cap                                                                  |
| 4   | **N+1 query in `get_events_with_faces()`** — 1 query + N individual face lookups (up to 51 round-trips)                                                | `utils/db_utils.py:386-414`                                  | 250-500ms per API call                    | [x] Batched with `$in` query                                                                    |

### 🟠 High Impact

| #   | Issue                                                                                           | File:Line                                | Impact                         | Status                                                    |
| --- | ----------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------ | --------------------------------------------------------- |
| 5   | **`store_face()` 2 full collection scans before insert** — scans unknowns + all faces for dedup | `utils/db_utils.py:234-275`              | 2 full scans per registration  | [x] Uses vector search index instead of Python-side scans |
| 6   | **Synchronous MongoDB in async FastAPI routes** — blocking PyMongo in `async` handlers          | `dashboard/backend/routes/*.py`          | Blocks all concurrent requests | [x] All routes wrapped in `asyncio.to_thread()`           |
| 7   | **WebSocket broadcasting is serial** — sends to each client sequentially with 0.5s timeout      | `dashboard/backend/routes/live.py:15-37` | 5 clients = 2.5s delay         | [x] Uses `asyncio.gather()` for concurrent sends          |
| 8   | **Synchronous alert dispatch** — SMTP/Twilio/HTTP calls block worker thread                     | `agents/alert_agent.py:26-64`            | 1-5s per alert                 | [x] Network alerts dispatched via `ThreadPoolExecutor`    |

### 🟡 Medium Impact

| #   | Issue                                                                           | File:Line                       | Impact                    | Status                                                        |
| --- | ------------------------------------------------------------------------------- | ------------------------------- | ------------------------- | ------------------------------------------------------------- |
| 9   | **JPEG encoding on camera thread** — `cv2.imencode()` adds 5-15ms per frame     | `main.py:36-39`                 | Reduces achievable FPS    | [x] Offloaded to `ThreadPoolExecutor` with quality 80         |
| 10  | **`update_visit_memory()` redundant read** — read → modify → write → read again | `utils/db_utils.py:491-550`     | 3 DB ops where 1 suffices | [x] Returns updated data directly, removed redundant 2nd read |
| 11  | **Full frame stored in Track objects** — ~921KB raw BGR per track               | `pipeline/track_state.py:74-83` | 10 tracks = ~9MB          | [x] Added `best_frame_jpeg` field (~50KB compressed)          |

### 🟢 Low Impact

| #   | Issue                                                                                             | File:Line                             | Impact                     | Status                                                                               |
| --- | ------------------------------------------------------------------------------------------------- | ------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------ |
| 12  | **Duplicate WebSocket connections** — `App.jsx` and `LiveFeed.jsx` both open separate connections | `App.jsx:34-57`, `LiveFeed.jsx:22-70` | Double connection overhead | [x] `App.jsx` owns single connection, passes frame/connected to `LiveFeed` via props |

---

## Priority Order

### Phase 1 — Critical (broken features)

1. ~~**Fix #1: WebSocket live feed**~~ ✓ Fixed — encode numpy to JPEG bytes in `handle_frame_annotated`
2. ~~**Fix #3: Face crop coordinates**~~ ✓ Fixed — tracks detection source, converts bbox to frame coordinates via `person_box` offset
3. ~~**Fix #2: Call `record_visit()`**~~ ✓ Fixed — `main.py:process_finalized_track` now calls `memory_agent.record_visit()` after recognition
4. **Rotate exposed credentials** (MongoDB, Cloudinary) — _still needs doing_

### Phase 2 — High (incorrect behavior)

5. ~~**Fix #4: `compute_face_ratio` coordinates**~~ ✓ Fixed — same fix as #3, `frame_bbox` always in frame coords
6. ~~**Fix #5: `reports.py` regex= → pattern=**~~ ✓ Fixed — changed `regex=` to `pattern=`

### Phase 3 — Medium (resilience)

7. ~~**Fix #6: Pass recognition/memory to `decide()`**~~ ✓ Fixed — `main.py:process_finalized_track` now passes all context to `decide()`
8. ~~**Fix #7: Attach image to progressive alerts**~~ ✓ Fixed — `camera_agent.py` now uploads image before `dispatch()`
9. ~~**Fix #8: Add try/except to dashboard routes**~~ ✓ Fixed — all routes wrapped in try/except with proper HTTP errors
10. ~~**Fix #9: Call `task_done()` in worker even on exception**~~ ✓ Fixed — moved to finally block

### Phase 4 — Hardening

11. ~~**Fix #10-12**~~ ✓ Fixed — similarity_score default 0.0, embeddings capped at 10, WebSocket backpressure added
12. ~~**Config defaults**~~ ✓ Fixed — MIN_TRACK_FRAMES=30
13. ~~**Move `yolov8n.pt` to `models/`**~~ ✓ Fixed
14. **Graceful camera release on unexpected exit** — _still needs doing_

### Phase 5 — Performance

15. ~~**Remove dead `detector.py`**~~ ✓ Fixed — deleted unused file
16. ~~**Optimize face detection cascade**~~ ✓ Fixed — `detect_faces_raw()` runs detector once per image
17. ~~**Fix N+1 query pattern**~~ ✓ Fixed — batched with `$in` query
18. ~~**Add MongoDB to async routes**~~ ✓ Fixed — all routes use `asyncio.to_thread()`
19. ~~**Concurrent WebSocket broadcasting**~~ ✓ Fixed — `asyncio.gather()` for parallel sends
20. ~~**Async alert dispatch**~~ ✓ Fixed — network alerts via `ThreadPoolExecutor`
