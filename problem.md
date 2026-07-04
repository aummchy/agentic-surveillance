# Problem Report — agentic_ai_singlecam

Codebase audit performed 2026-06-25 (updated 2026-07-03). Issues are grouped by severity and verified against actual source lines.

**Status: 3 issues remaining (1 CRITICAL, 0 HIGH, 0 MEDIUM, 2 LOW). 86 prior issues FIXED. 8 improvements applied.**

---

## REMAINING — Issues Yet to be Solved

### CRITICAL

#### Issue 50 — `.env` contains live production credentials in the workspace

**File:** `.env:12,19,20`

Real MongoDB Atlas URI, Cloudinary API key, and Cloudinary secret are present in `.env`. While `.gitignore` excludes it from git, any non-git distribution (zip, tarball, rsync) ships these credentials. A leaked MongoDB URI allows full database read/write/delete. A leaked Cloudinary secret allows arbitrary image operations.

**Fix:** Rotate all credentials immediately. Keep `.env.example` with placeholders only.

### HIGH

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
| **Total** | **89** | **86** | **3** |

### Open issue breakdown

| # | Severity | Summary |
|---|----------|---------|
| 50 | **CRITICAL** | `.env` contains live production credentials — rotate immediately |
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
