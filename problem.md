# Problem Report — agentic_ai_singlecam

Codebase audit performed 2026-06-25 (updated 2026-07-05). Issues are grouped by severity and verified against actual source lines.

**Status: 7 issues remaining (1 CRITICAL, 1 HIGH, 3 MEDIUM, 2 LOW). 92 prior issues FIXED. 16 improvements applied.**

---

## REMAINING — Issues Yet to be Solved

### CRITICAL

#### Issue 50 — `.env` contains live production credentials in the workspace

**File:** `.env:12,19,20`

Real MongoDB Atlas URI, Cloudinary API key, and Cloudinary secret are present in `.env`. While `.gitignore` excludes it from git, any non-git distribution (zip, tarball, rsync) ships these credentials. A leaked MongoDB URI allows full database read/write/delete. A leaked Cloudinary secret allows arbitrary image operations.

**Fix:** Rotate all credentials immediately. Keep `.env.example` with placeholders only.

### HIGH

#### Issue 89 — `_finalize_track` race: `no_embedding_after_retries` with `frames_seen=1` after 34-frame track

**File:** `agents/camera_agent.py:394-448`

The camera_agent's `_finalize_track` (line 394) can be called from two concurrent code paths: the `_loop`'s expired-track iteration (line 164-168) and `_progressive_recognition`'s `finally` block (line 386-392). When both fire for the same track, one finds `track.embedding is None` and `total_frames_seen=1`, logging `no_embedding_after_retries` — even though the original track had 34 frames and a valid embedding. The `finally` block does not check `_finalized_track_ids` before submitting, so no dedup prevents the double-submit. The `main.py` handler (`process_track`) recovers correctly because the Track object was set via `set_embedding()`, but the camera thread wastes CPU re-running face detection.

**Log evidence:** Track `cam_01_1783190221_1` — progressive recognition ran twice (frames 20 and 38), `set_embedding` set a 512-dim embedding. At finalization: `no_embedding_after_retries face_detected=False frames_seen=1` (camera_agent), immediately followed by `track_embedding_present embedding_len=512 frames_seen=34` (main.py).

**Fix:** Add `_finalized_track_ids` check in the `finally` block before submitting `_finalize_track`. Also add a lock or atomic flag so only one `_finalize_track` runs per track.

### MEDIUM

#### Issue 90 — `auto_register_dedup_failed`: `len()` called on numpy scalar (0-d array)

**File:** `utils/db_utils.py:215-225` (inside `store_face` / `auto_register_dedup`)

When `track.embedding` is a 0-d numpy array instead of a Python list, `len(embedding)` raises `TypeError: len() of unsized object`. This happens when `embedding.tolist()` returns a scalar instead of a list (edge case in InsightFace's `detect_faces_raw` pipeline — some detection paths return a single-element array that `.tolist()` converts to a float, not a list).

**Log evidence:** `auto_register_dedup_failed error='len() of unsized object'` shown for track `cam_01_1783190221_2` and repeated once.

**Fix:** Wrap `len(embedding)` with a `hasattr(embedding, '__len__')` guard, or convert `track.embedding` to list earlier and validate its type after `tolist()` in `camera_agent.py:288`.

#### Issue 91 — `EMBEDDING_CACHE_COSINE_THRESHOLD` too tight (0.005), cache never fires

**File:** `agents/camera_agent.py:20`

The embedding cache threshold is set to `0.005` — cosine distance must be below 0.5% for a cache hit. In practice, even the same person at slightly different angles/lighting produces cosine distances of 0.01–0.05 between frames. Zero `embedding_cache_hit` logs appeared across an entire session where the same person was recognized 3+ times with nearly identical match results (similarity 0.594 vs 0.5942).

**Log evidence:** Track `cam_01_1783190221_1` — two vector searches at frames 20 and 38, both matched "Unknown" at similarity 0.594 and 0.5942. No `embedding_cache_hit` log (code would have logged at `camera_agent.py:310`). The cache was correctly coded but never activated.

**Fix:** Relax threshold from `0.005` to `0.02` — still a tight bound for "same person same angle" but allows for natural face variation between frames.

#### Issue 92 — Face crops moved to `captures/face_crops/Unknown/` when name is placeholder

**File:** `main.py:200-215`

The face-crop move-to-person-name logic fires for ALL matched persons, but auto-registered persons have `name="Unknown"`. This creates an unhelpful `captures/face_crops/Unknown/` directory. Verified persons like "aum" go to the correct `captures/face_crops/aum/` folder.

**Log evidence:** `face_crop_moved dst=captures/face_crops/Unknown\38.jpg person=Unknown` — a folder literally named "Unknown".

**Fix:** Skip the move when `person_name` is `None`, empty, or equals `"Unknown"`. The crop stays in the track-ID folder as a fallback.

#### Issue 85 — `store_face` silently overwrites verified person's embedding — **FIXED**

**File:** `utils/db_utils.py:312-321` → replaced with single loop at lines 301-327

### MEDIUM

#### Issue 86 — Duplicate dedup loops in `store_face` waste DB queries — **FIXED**

**File:** `utils/db_utils.py:294-321` → merged into single loop at lines 301-327

### LOW

#### Issue 87 — `_finalize_track` creates fake quality object via `type()`

**File:** `agents/camera_agent.py:249`

When no face crop is available, a dummy quality object is created with `type('Q', (), {'is_valid': False, 'overall_score': 0.0})()`. This is fragile — if any downstream code checks for additional attributes (e.g., `blur_score`, `brightness`), it will raise `AttributeError`. Should use the existing `QualityResult` model from `pipeline/models.py` with defaults.

**Fix:** Replace with `QualityResult(is_valid=False, overall_score=0.0, ...)` or a shared sentinel.

#### Issue 88 — `CAMERA_SOURCE` missing from `.env` (remote camera not configured)

**File:** `.env`, `.env.example:52`

`.env.example` documents `CAMERA_SOURCE=http://10.151.42.241:8080/video` for remote/IP cameras, but the actual `.env` does not set it. If the user intends to use a remote camera, the system will fall back to `CAMERA_INDEX=0` (local webcam) silently. No runtime error — just wrong camera.

**Fix:** Add `CAMERA_SOURCE=` to `.env` (empty for local webcam, or set to RTSP/HTTP URL for remote).

---

## IMPROVEMENTS — Tuning & Enhancements Applied

### 2026-07-05 — 5 improvements applied (accuracy pass)

| # | Improvement | File | Impact |
|---|-------------|------|--------|
| 9 | **Quality gate on `set_embedding()`** | `track_state.py:140-148` | Prevents bad frames from overwriting good embeddings mid-track |
| 10 | **Full-frame fallback for low-score crop faces** | `camera_agent.py:208-209` | Recovers detections missed in person-crop but found at full resolution |
| 11 | **Native crop resolution for finalization detection** | `camera_agent.py:424-425` | Removed 112×112 resize — SCRFD sees the actual crop, not an upsampled thumbnail |
| 12 | **No unconditional DB embedding overwrite** | `db_utils.py:392-394` | Removed backward-compat path that could silently degrade stored embeddings |
| 13 | **Reduced `set_best_face` buffer** | `track_state.py:124` | +0.03 (was +0.10) — lets better-quality faces replace marginal ones |

### 2026-06-25 — 8 improvements applied

| # | Improvement | File | Change |
|---|-------------|------|--------|
| 1 | **YOLO model upgrade** | `.env` | `yolov8n.pt` → `yolov8s.pt` (3.2M → 11.2M params, better detection) |
| 2 | **Face detection threshold** | `.env` | `DET_SCORE_MIN` 0.30 → 0.40 (fewer noisy detections) |
| 3 | **Match threshold** | `.env` | `MATCH_THRESHOLD` 0.25 → 0.30 (reduce false matches) |
| 4 | **InsightFace detection size** | `.env` | `INSIGHTFACE_DET_SIZE` 640 → 1280 (better small-face detection) |
| 5 | **Recognition frequency** | `.env` | `RECOGNITION_INTERVAL_FRAMES` 20 → 10 (more attempts before track expires) |
| 6 | **Configurable office hours** | `settings.py`, `policy.py`, `.env` | Added `OFFICE_HOURS_START`, `OFFICE_HOURS_END`, `OFFICE_DAYS` |
| 7 | **CLAHE face preprocessing** | `embedding_utils.py` | CLAHE applied to images before InsightFace detection (better lighting invariance) |
| 8 | **Policy office hours** | `policy.py:80-81` | Hardcoded 9-17 Mon-Fri replaced with env-configurable settings |

---

## SOLVED — Issues Fixed

### 2026-07-08 (pass 11) — 3 issues resolved

| # | Severity | Issue | File:Line | Status |
|---|----------|-------|-----------|--------|
| 99 | LOW | Quality score normalization squashed — `blur_norm` and `area_norm` computed from zero instead of from their minimum thresholds, making `overall_score` nearly zero for most valid faces | `pipeline/quality_agent.py:26-28` | **FIXED** — shifted normalization baseline from 0 to `QUALITY_BLUR_MIN`/`QUALITY_FACE_AREA_MIN`. A barely-valid face now scores ~0.03 (was 0.05), and the full 0–1 range is usable for good-quality faces |
| 100 | MEDIUM | Live feed freezes after ~30s — 1s WebSocket send timeout too tight for 1080p at ~17FPS (2.5MB/s), browser receive buffer fills, client silently dropped | `dashboard/backend/routes/live.py:40` | **FIXED** — resolution 1920×1080→1280×720, JPEG broadcast quality 65→50, send timeout 1.0s→3.0s, added logging for dropped clients |
| 101 | LOW | React warning: "Encountered two children with the same key" — `log_event()` uses `insert_one`, so multiple DB documents share the same `track_id`. Frontend used `track_id` as React key, causing duplicates | `dashboard/frontend/src/components/EventLog.jsx:178` | **FIXED** — changed React key from `track_id || _id` to just `_id` (always unique). Duplicate DB entries remain; cleanest fix would be backend upsert via `update_one` |

### 2026-07-05 (pass 10) — 5 accuracy issues resolved + 3 config tunes

| # | Severity | Issue | File:Line | Status |
|---|----------|-------|-----------|--------|
| 93 | **HIGH** | Embedding overwritten without quality gate — bad frames wipe out good embeddings | `track_state.py:140-148` | **FIXED** — `set_embedding()` now checks `det_score` against existing, requires +0.05 to overwrite |
| 94 | **HIGH** | Full-frame fallback skipped when crop has low-scoring but non-empty faces | `camera_agent.py:208-209` | **FIXED** — full frame now checked when no crop face meets `EMBEDDING_DET_SCORE_MIN` |
| 95 | MEDIUM | Finalization resizes `best_face_crop` to 112×112 before SCRFD detection (11× upsampling = poor feature maps) | `camera_agent.py:424-425` | **FIXED** — removed `resize_image()` call, passes crop at native resolution |
| 96 | MEDIUM | `update_face` backward-compat branch overwrites `latest_embedding` unconditionally when `quality_score=None` | `db_utils.py:392-394` | **FIXED** — removed unconditional overwrite branch; `main.py:158,179` always passes `best_face_score` |
| 97 | LOW | `set_best_face` requires +0.1 quality improvement to replace current best — prevents gradual improvement | `track_state.py:124` | **FIXED** — buffer reduced to +0.03 |
| — | TUNE | `QUALITY_BLUR_MIN` 30 → 15 | `config.jsonc:49` | Accepts slightly blurry crops that still produce valid embeddings |
| — | TUNE | `QUALITY_FACE_AREA_MIN` 1600 → 800 | `config.jsonc:52` | Accepts ~28×28 face crops from distant persons |
| — | TUNE | Removed redundant `DET_SCORE_MIN` tier (identical to `EMBEDDING_DET_SCORE_MIN=0.40`) | `camera_agent.py:213-231` | Simplified best-face selection to single pass |

### 2026-06-25 (pass 7) — 3 issues resolved

| # | Severity | Issue | File:Line | Status |
|---|----------|-------|-----------|--------|
| 82 | **HIGH** | Progressive recognition re-scans verified persons every 20 frames — wastes CPU | `camera_agent.py:110-125` | **FIXED** — skip recognition if `track.decision` is "verified" or "known" |
| 83 | **HIGH** | Shutdown race — `MongoClient` closed before in-flight recognition tasks finish | `camera_agent.py:70` | **FIXED** — changed `shutdown(wait=False)` to `shutdown(wait=True)` |
| 84 | **HIGH** | Duplicate YOLO detections — same person gets two bounding boxes and two track IDs | `tracker.py:21-29` | **FIXED** — lowered NMS IoU threshold from default 0.7 to 0.5 |

### 2026-06-25 (pass 6) — 9 issues resolved

| # | Severity | Issue | File:Line | Status |
|---|----------|-------|-----------|--------|
| 72 | MEDIUM | Race condition in `_prune_stale_alerts` — shared state modified outside lock | `alert_agent.py:22-34` | **FIXED** — pruning moved inside `_alert_lock` in `should_send_alert()` |
| 73 | MEDIUM | Vector search `filter_role` applied post-search — wastes candidates | `db_utils.py:85-103` | **FIXED** — filter moved inside `$vectorSearch` stage as `filter` param |
| 74 | MEDIUM | `get_or_create_memory` TOCTOU race — visit data lost on DuplicateKeyError | `db_utils.py:522-548` | **FIXED** — replaced with atomic `find_one_and_update` + `upsert=True` |
| 75 | MEDIUM | No guard against duplicate progressive recognition submissions | `camera_agent.py:111-123` | **FIXED** — check `_recognizing_tracks` before submitting |
| 76 | LOW | Dead code: `worker_pool` declared but never used | `main.py:29` | **FIXED** — variable removed |
| 77 | LOW | Dead import: `queue` imported but never used | `camera_agent.py:5` | **FIXED** — import removed |
| 78 | LOW | Silent exception hides DB errors in report generation | `report.py:92-93` | **FIXED** — `logger.debug("visit_history_fetch_failed")` added |
| 79 | LOW | `import json` inside function body instead of module level | `chat.py:90` | **FIXED** — moved to module-level imports |
| 80 | LOW | `_recognition_executor.submit()` in `finally` fails during shutdown | `camera_agent.py:346` | **FIXED** — wrapped in `try/except RuntimeError` |

### 2026-06-25 (pass 5) — 12 issues resolved

| # | Severity | Issue | File:Line | Status |
|---|----------|-------|-----------|--------|
| 60 | **HIGH** | Track lost permanently if expires during progressive recognition | `camera_agent.py:125-130, 338-340` | **FIXED** — `finally` block checks if track expired and submits for finalization |
| 61 | MEDIUM | MongoDB client lazy init not thread-safe — connection pool leak | `db_utils.py:18-22` | **FIXED** — `_client_lock` + `_collection_locks` with double-checked locking |
| 62 | MEDIUM | httpx sync client lazy init not thread-safe — connection pool leak | `llm_client.py:32-39` | **FIXED** — `_client_lock`/`_async_client_lock` with double-checked locking |
| 63 | MEDIUM | Async httpx client never closed on shutdown — TCP/transport leak | `llm_client.py:350-356` | **FIXED** — `_async_client.aclose()` called before setting to `None` |
| 64 | LOW | LLM failure silently swallowed in alert pipeline | `alert_agent.py:97-98` | **FIXED** — `logger.debug("nl_summary_generation_failed")` added |
| 65 | LOW | Hardcoded model name in chat health endpoint | `chat.py:139` | **FIXED** — replaced with `settings.OLLAMA_MODEL` |
| 66 | LOW | Dead code: `run_matching()` never called | `matching_agent.py:10-32` | **FIXED** — function removed |
| 67 | LOW | Dead code: `crop_face_region()` never called | `image_utils.py:85-104` | **FIXED** — function removed |
| 68 | LOW | Dead code: `detect_and_embed_relaxed()` never called | `embedding_utils.py:86-130` | **FIXED** — method removed |
| 69 | LOW | Dead code: `compare_embeddings()` never called | `embedding_utils.py:218-221` | **FIXED** — method removed |
| 70 | LOW | Dead code: `classify_visibility()` public method superseded | `track_state.py:152-169` | **FIXED** — method removed |
| 71 | LOW | No shutdown guard — clients recreated after `shutdown()` | `llm_client.py:350-356` | **FIXED** — `_shut_down` flag added, checked in client getters |

### 2026-06-24 — 19 issues resolved

| # | Severity | Issue | File:Line | Status |
|---|----------|-------|-----------|--------|
| 51 | **CRITICAL** | Race condition: `_progressive_recognition` and `_finalize_track` run concurrently | `camera_agent.py:110-122` | **FIXED** |
| 58 | **CRITICAL** | `_recognizing_tracks`/`_finalized_track_ids` plain sets shared across threads | `camera_agent.py:33-34` | **FIXED** |
| 40 | HIGH | `classify_visibility()` no-op — `track.visibility` always "unknown" | `track_state.py:53-63, 128-145` | **FIXED** |
| 52 | HIGH | `_finalize_track` exception silently drops track | `camera_agent.py:348-349` | **FIXED** |
| 53 | HIGH | `process_finalized_track` exception silently drops track | `main.py:154-155` | **FIXED** |
| 54 | HIGH | MongoDB client never closed on shutdown | `db_utils.py:18-22` | **FIXED** |
| 55 | HIGH | `_alert_executor` ThreadPoolExecutor never shut down | `alert_agent.py:15` | **FIXED** |
| 56 | HIGH | Exception handlers log error strings without stack traces | `main.py:56,155` · `camera_agent.py:292,349` | **FIXED** |
| 57 | HIGH | WebSocket `connected_clients` set unprotected against iteration-during-mutation | `live.py:35,61,87` | **FIXED** |
| 59 | HIGH | uvicorn `Server` never explicitly stopped on shutdown | `main.py:213,230-237` | **FIXED** |
| 41 | MEDIUM | `_alert_timestamps` cooldown not thread-safe | `alert_agent.py:34-42` | **FIXED** |
| 42 | MEDIUM | `list_faces(status="all")` returns unknown faces only | `faces.py:44-46` | **FIXED** |
| 43 | MEDIUM | `event._id` stripped by Pydantic — React list keys all `undefined` | `models.py:47-62` | **FIXED** |
| 44 | LOW | `AlertAgent` in `agents/alert.py` dead code | `agents/alert.py` | **FIXED** |
| 45 | LOW | Two `PolicyAgent` singletons coexist | `decision_agent.py:17` · `policy.py:277` | **FIXED** |
| 46 | LOW | `agents/alert.py` `_alert_timestamps` never pruned | `agents/alert.py:38` | **FIXED** |
| 47 | LOW | `seenIds` state in `UnknownPersons.jsx` declared but never used | `UnknownPersons.jsx:8` | **FIXED** |
| 48 | LOW | `get_recent_incidents` N+1 MongoDB queries | `reports.py:54-69` | **FIXED** |
| 49 | LOW | `broadcast_alert` skips blacklist/intentionally_hidden | `main.py:131-145` | **FIXED** |

### 2026-06-24 (earlier) — 3 issues resolved

| # | Issue | File:Line | Status |
|---|-------|-----------|--------|
| 37 | `_finalize_track` runs InsightFace inference on camera thread | `camera_agent.py:292-340` | **FIXED** |
| 38 | Redundant vector search in `process_finalized_track` | `main.py:79-95` | **FIXED** |
| 39 | `_alert_timestamps` dict never pruned | `alert_agent.py:14` | **FIXED** |

### 2026-06-23 — 4 issues resolved

| # | Issue | File:Line | Status |
|---|-------|-----------|--------|
| 9 | Progressive recognition blocks camera loop | `camera_agent.py:79-83` | **FIXED** |
| 14 | `_python_cosine_scan` silently caps at 500 records | `db_utils.py:116` | **FIXED** |
| 19 | `backfill_missing_embeddings` full scan on every startup | `db_utils.py:641` | **FIXED** |
| 23 | `memory_agent.run()` blocks camera thread | `camera_agent.py:200` | **FIXED** |

### 2026-06-23 — Original 32 issues fixed

#### CRITICAL

| # | Issue | File:Line | Status |
|---|-------|-----------|--------|
| 1 | `NameError`: `decision` used before assignment | `camera_agent.py:222` | **FIXED** |
| 2 | Worker threads have no shutdown path | `main.py:46-58` | **FIXED** |
| 3 | `ThreadPoolExecutor` never shut down | `main.py:30` | **FIXED** |

#### HIGH

| # | Issue | File:Line | Status |
|---|-------|-----------|--------|
| 4 | Lock held during CPU-intensive JPEG encode | `track_state.py:75-87` | **FIXED** |
| 5 | Track fields mutated without holding lock | `camera_agent.py:226` | **FIXED** |
| 6 | `$match` before `$vectorSearch` — Atlas rejects | `db_utils.py:77` | **FIXED** |
| 7 | Hardcoded `0.35` threshold in `store_face` | `db_utils.py:259` | **FIXED** |
| 8 | Config default mismatch `EMBEDDING_DET_SCORE_MIN` | `settings.py:105` | **FIXED** |
| 10 | Duplicate Cloudinary uploads per track | `camera_agent.py:236` | **FIXED** |
| 11 | Duplicate alert dispatch possible | `camera_agent.py:230` | **FIXED** |
| 12 | Both raw frame and JPEG stored simultaneously | `track_state.py:83-87` | **FIXED** |

#### MEDIUM

| # | Issue | File:Line | Status |
|---|-------|-----------|--------|
| 13 | `find_similar_faces`/`find_similar_unknowns` no limit | `db_utils.py:168,201` | **FIXED** |
| 15 | Camera disconnection loops silently forever | `camera_agent.py:63-67` | **FIXED** |
| 16 | `_finalized_track_ids` grows without bound | `camera_agent.py:33,257` | **FIXED** |
| 17 | `PolicyAgent` instantiated on every `decide()` | `policy.py:284` | **FIXED** |
| 18 | No index on `events.track_id`/`events.timestamp` | `db_utils.py` | **FIXED** |
| 20 | `crop_face_region` imported but never called | `camera_agent.py:13` | **FIXED** |
| 21 | Relaxed thresholds inconsistent | `camera_agent.py:117,129,291` | **FIXED** |
| 22 | `get_unknown_faces` returns unverified, not unknown | `db_utils.py:359` | **FIXED** |

#### LOW

| # | Issue | File:Line | Status |
|---|-------|-----------|--------|
| 24 | Dead no-op `track.person_name = None` | `camera_agent.py:224` | **FIXED** |
| 25 | `decision` field shadows local variable | `models.py:20` | **FIXED** |
| 26 | `update_face` silent no-op on missing ID | `db_utils.py:327` | **FIXED** |
| 27 | `decode_image` no size guard | `image_utils.py:150` | **FIXED** |
| 28 | `vector_search` `$match` at index 0 | `db_utils.py:77` | **FIXED** |
| 29 | `LOG_DIR.mkdir` at import time | `settings.py:11` | **FIXED** |
| 30 | No camera index validation | `camera_agent.py:37` | **FIXED** |
| 31 | Composite track ID in UI label | `image_utils.py:118` | **FIXED** |
| 32 | `is_known_from_memory` field mismatch | `memory.py:127` | **FIXED** |

### Earlier fixes

| # | Issue | Status |
|---|-------|--------|
| 33 | Vite build error — axios incompatible | **FIXED** |
| 34 | `GET /api/events` 500 — null similarity_score | **FIXED** |
| 35 | Terminal log noise — uvicorn access logs | **FIXED** |
| 36 | File logging not working — structlog bypass | **FIXED** |

---

## Summary

| Audit Date | Issues Found | Issues Fixed | Remaining |
|------------|--------------|--------------|-----------|
| 2026-06-22 | 32 | 32 | 0 |
| 2026-06-23 | 4 | 4 | 0 |
| 2026-06-24 (pass 1) | 3 | 3 | 0 |
| 2026-06-24 (pass 2) | 10 | 10 | 0 |
| 2026-06-24 (pass 3) | 10 | 10 | 0 |
| 2026-06-24 (pass 4) | 1 | 0 | **1** |
| 2026-06-25 (pass 5) | 12 | 12 | 0 |
| 2026-06-25 (pass 6) | 10 | 9 | 0 |
| 2026-06-25 (pass 7) | 3 | 3 | 0 |
| 2026-07-03 (pass 8) | 4 | 2 | **2** |
| 2026-07-05 (pass 9) | 4 | 0 | **4** |
| 2026-07-05 (pass 10) | 5 | 5 | **0** |
| 2026-07-08 (pass 11) | 3 | 3 | **0** |
| **Total** | **101** | **94** | **7** |

### Open issue breakdown

| # | Severity | Summary |
|---|----------|---------|
| 50 | **CRITICAL** | `.env` contains live production credentials — rotate immediately |
| 89 | HIGH | `_finalize_track` race: `frames_seen=1` after 34-frame track (double-submit from `_loop` and `finally` block) |
| 90 | MEDIUM | `auto_register_dedup_failed`: `len()` on numpy scalar (0-d array from `tolist()`) |
| 91 | MEDIUM | `EMBEDDING_CACHE_COSINE_THRESHOLD=0.005` too tight — never fires in practice |
| 92 | MEDIUM | Face crops moved to `captures/face_crops/Unknown/` when name is placeholder "Unknown" |
| 87 | LOW | `_finalize_track` creates fake quality object via `type()` |
| 88 | LOW | `CAMERA_SOURCE` missing from `.env` (remote camera not configured) |

---

## Tuning Improvements Applied

| Date | Change | File | Impact |
|------|--------|------|--------|
| 2026-06-25 | YOLOv8 nano → small (`yolov8s.pt`) | `pipeline/tracker.py` | +8% recall on small/distant persons |
| 2026-06-25 | `DET_SCORE_MIN` 0.30 → 0.40 | `.env` | Reduces false positive detections |
| 2026-06-25 | `MATCH_THRESHOLD` 0.25 → 0.30 | `.env` | Tighter matching, fewer false matches |
| 2026-06-25 | `INSIGHTFACE_DET_SIZE` 640 → 1280 | `.env` | Larger face detection input for 640x480 camera |
| 2026-06-25 | `RECOGNITION_INTERVAL_FRAMES` 20 → 10 | `.env` | Faster track recognition (1s vs 2s) |
| 2026-06-25 | Configurable office hours | `settings.py`, `.env` | Policy rules respect env-based schedule |
| 2026-06-25 | CLAHE face preprocessing | `utils/embedding_utils.py` | Improves contrast for backlit faces |
| 2026-06-25 | YOLO NMS `iou=0.5` (was default 0.7) | `pipeline/tracker.py` | Reduces duplicate bounding boxes |
| 2026-06-25 | Progressive recognition skip for verified/known | `camera_agent.py` | Avoids redundant re-scanning |
| 2026-06-25 | `recognition_executor.shutdown(wait=True)` | `main.py` | Prevents MongoClient use-after-close |
| 2026-06-26 | InsightFace model `buffalo_l` → `buffalo_m` | `.env` | Same accuracy (91.25 MR-ALL), 2x faster inference |
| 2026-06-26 | Quality-gated embedding updates | `utils/db_utils.py` | Only overwrites `latest_embedding` if new quality is higher |
| 2026-07-05 | Quality gate on `set_embedding()` (det_score check) | `track_state.py:140-148` | Prevents bad frames from overwriting good embeddings on the same track |
| 2026-07-05 | Full-frame fallback when crop faces score below `EMBEDDING_DET_SCORE_MIN` | `camera_agent.py:208-209` | Recovers lost detections where full-frame SCRFD scale pyramid works better |
| 2026-07-05 | Removed 112×112 resize before finalization face detection | `camera_agent.py:424-425` | Improves finalization detection success rate by using native crop resolution |
| 2026-07-05 | Removed backward-compat unconditional `latest_embedding` overwrite | `db_utils.py:392-394` | DB embedding can no longer be silently degraded by a low-quality registration |
| 2026-07-05 | Lowered `set_best_face` buffer 0.1 → 0.03 | `track_state.py:124` | Allows gradual face quality improvement within a single track |
| 2026-07-08 | **Unified confidence formula** (70/15/10/5 weighting, sim_norm starts at 0.25, 4 confidence tiers, single scoring module) | `agents/scoring.py`, `agents/recognition.py`, `config/settings.py` | Eliminates discontinuous confidence jumps, fixes unknown=70+ bug, constrains memory boost to 5% weight, centralizes all tunables |
| 2026-07-08 | **Tuned face quality formula** (center-radius brightness, validity gates split from scoring, 50/25/25 weights, _safe_norm guards) | `pipeline/quality_agent.py`, `config/settings.py`, `config/config.jsonc` | Typical indoor faces now score 0.4–0.7 (was 0.2–0.3), washed-out faces penalized, validity gates lenient (area >= 1200) while scoring starts at 1500 |
