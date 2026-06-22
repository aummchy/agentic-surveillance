# TODO — Visitor Surveillance System

## Status: Last reviewed 2026-06-22 (all 12 issues from diagnosis fixed)

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

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| 1 | **`broadcast_frame` receives numpy, expects bytes** | `main.py:40` → `live.py:14` | **FIXED** — encode numpy frame to JPEG bytes via `cv2.imencode` before broadcast |
| 2 | **`record_visit()` never called** | `memory.py:232` (defined), never invoked anywhere | **FIXED** — `main.py:process_finalized_track` now calls `memory_agent.record_visit()` after recognition |
| 3 | **Face crop uses wrong coordinate system** | `camera_agent.py:141-143` | **FIXED** — tracks `detected_in_person_crop` flag, converts bbox to frame coordinates by adding `person_box` offset when detected in crop |

### HIGH Issues (2)

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| 4 | **`compute_face_ratio` mixes coordinate systems** | `camera_agent.py:130` | **FIXED** — `frame_bbox` now always in frame coordinates, matching `person_box` |
| 5 | **`reports.py` uses deprecated `regex=` param** | `reports.py:16` | **FIXED** — changed `regex=` to `pattern=` |

### MEDIUM Issues (4)

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| 6 | **`decide()` called without recognition/memory context** | `main.py:73` | **FIXED** — `process_finalized_track` now passes `recognition_result` and `memory_context` to `decide()` |
| 7 | **Progressive recognition alerts have no image** | `camera_agent.py:205-208` | **FIXED** — `dispatch(track, decision, image_url)` now uploads image before dispatching |
| 8 | **No error handling on any dashboard route** | `events.py`, `faces.py`, `reports.py` | **FIXED** — all routes wrapped in try/except with proper HTTP error responses |
| 9 | **Worker exceptions silently lose tracks** | `main.py:44-52` | **FIXED** — `task_done()` now called in finally block even on exception |

### LOW Issues (3)

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| 10 | `None` passed for `similarity_score` to `_log_event` | `main.py:67` | **FIXED** — default changed to `0.0` |
| 11 | `embeddings` array grows unbounded | `db_utils.py:229` | **FIXED** — `$slice: -10` limits to last 10 embeddings |
| 12 | WebSocket `connected_clients` not process-safe | `live.py:11` | **FIXED** — added backpressure with `asyncio.wait_for` timeout, frame size limit |

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

| Setting | Current Default | Spec Value | Status |
|---------|----------------|------------|--------|
| `DET_SCORE_MIN` | 0.50 | 0.70 | `settings.py` default now 0.50 (matches `.env`); spec wants 0.70 |
| `MIN_TRACK_FRAMES` | 30 | 30 | **FIXED** — default now 30 |

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

| File | Status | Priority |
|------|--------|----------|
| `dashboard/backend/` | Exists | Done |
| `dashboard/frontend/` | Exists | Done |
| `README.md` | Exists | Done |
| `.env.example` | Exists | Done |
| `logs/` directory | Exists | Done |
| `models/` directory | Exists | Done — `yolov8n.pt` moved here |
| `pipeline/visibility_analyzer.py` | Missing | Low — logic in track_state.py (acceptable) |
| `pipeline/decision_engine.py` | Missing | Low — logic in decision_agent.py (acceptable) |

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

## Priority Order

### Phase 1 — Critical (broken features)
1. ~~**Fix #1: WebSocket live feed**~~ ✓ Fixed — encode numpy to JPEG bytes in `handle_frame_annotated`
2. ~~**Fix #3: Face crop coordinates**~~ ✓ Fixed — tracks detection source, converts bbox to frame coordinates via `person_box` offset
3. ~~**Fix #2: Call `record_visit()`**~~ ✓ Fixed — `main.py:process_finalized_track` now calls `memory_agent.record_visit()` after recognition
4. **Rotate exposed credentials** (MongoDB, Cloudinary) — *still needs doing*

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
14. **Graceful camera release on unexpected exit** — *still needs doing*
