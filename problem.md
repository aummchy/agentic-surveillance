# Problem Report — agentic_ai_singlecam

Codebase audit performed 2026-06-24 (updated 2026-06-24). Issues are grouped by severity and verified against actual source lines.

**Status: 1 issue remaining (.env credentials). 58 prior issues FIXED.**

---

## REMAINING — Issues Yet to be Solved

### CRITICAL

#### Issue 50 — `.env` contains live production credentials in the workspace

**File:** `.env:12,19,20`

Real MongoDB Atlas URI, Cloudinary API key, and Cloudinary secret are present in `.env`. While `.gitignore` excludes it from git, any non-git distribution (zip, tarball, rsync) ships these credentials. A leaked MongoDB URI allows full database read/write/delete. A leaked Cloudinary secret allows arbitrary image operations.

**Fix:** Rotate all credentials immediately. Keep `.env.example` with placeholders only.

---

## SOLVED — Issues Fixed

### 2026-06-24 — 19 issues resolved

| # | Severity | Issue | File:Line | Status |
|---|----------|-------|-----------|--------|
| 51 | **CRITICAL** | Race condition: `_progressive_recognition` and `_finalize_track` run concurrently on same Track (no lock on shared sets) | `camera_agent.py:110-122` | **FIXED** — `_track_sets_lock` added, all access to `_recognizing_tracks`/`_finalized_track_ids` protected |
| 58 | **CRITICAL** | `_recognizing_tracks`/`_finalized_track_ids` plain sets shared across threads without synchronization | `camera_agent.py:33-34` | **FIXED** — same lock as #51 |
| 40 | HIGH | `classify_visibility()` no-op — `track.visibility` always "unknown", rules 6 & 7 never fire | `track_state.py:53-63, 128-145` | **FIXED** — `_classify_visibility_inplace()` called in `get_expired_tracks()` before removal from dict |
| 52 | HIGH | `_finalize_track` exception silently drops track — no event logged, no alert sent | `camera_agent.py:348-349` | **FIXED** — `on_track_finalized` moved to `finally` block, always called |
| 53 | HIGH | `process_finalized_track` exception silently drops track — no event logged | `main.py:154-155` | **FIXED** — fallback `_log_event` in exception handler with `"error"` status |
| 54 | HIGH | MongoDB client never closed on shutdown — connection pool leak | `db_utils.py:18-22` | **FIXED** — `close_client()` added and called during shutdown |
| 55 | HIGH | `_alert_executor` ThreadPoolExecutor never shut down | `alert_agent.py:15` | **FIXED** — `shutdown()` added and called during shutdown |
| 56 | HIGH | Exception handlers log error strings without stack traces | `main.py:56,155` · `camera_agent.py:292,349` | **FIXED** — `exc_info=True` added to all `logger.error()` calls |
| 57 | HIGH | WebSocket `connected_clients` set unprotected against iteration-during-mutation | `live.py:35,61,87` | **FIXED** — `list(connected_clients)` defensive copy before iteration |
| 59 | HIGH | uvicorn `Server` never explicitly stopped on shutdown | `main.py:213,230-237` | **FIXED** — `server.should_exit = True` set during shutdown |
| 41 | MEDIUM | `_alert_timestamps` cooldown not thread-safe — duplicate alerts possible | `alert_agent.py:34-42` | **FIXED** — `_alert_lock` threading.Lock wraps check-and-update |
| 42 | MEDIUM | `list_faces(status="all")` returns unknown faces only | `faces.py:44-46` | **FIXED** — explicit `elif status == "all"` branch with unfiltered query |
| 43 | MEDIUM | `event._id` stripped by Pydantic — React list keys all `undefined` | `models.py:47-62` | **FIXED** — `id: Optional[str] = Field(None, alias="_id")` added with `populate_by_name=True` |
| 44 | LOW | `AlertAgent` in `agents/alert.py` dead code — never wired in | `agents/alert.py` | **FIXED** — file deleted |
| 45 | LOW | Two `PolicyAgent` singletons coexist | `decision_agent.py:17` · `policy.py:277` | **FIXED** — duplicate `decide()` convenience function removed from `policy.py` |
| 46 | LOW | `agents/alert.py` `_alert_timestamps` never pruned | `agents/alert.py:38` | **FIXED** — file deleted with #44 |
| 47 | LOW | `seenIds` state in `UnknownPersons.jsx` declared but never used | `UnknownPersons.jsx:8` | **FIXED** — dead state removed |
| 48 | LOW | `get_recent_incidents` N+1 MongoDB queries | `reports.py:54-69` | **FIXED** — batch-fetch all visit histories in one `$in` query before loop |
| 49 | LOW | `broadcast_alert` skips blacklist/intentionally_hidden | `main.py:131-145` | **FIXED** — condition widened to `decision.should_alert` |

### 2026-06-24 (earlier) — 3 issues resolved

| # | Issue | File:Line | Status |
|---|-------|-----------|--------|
| 37 | `_finalize_track` runs InsightFace inference on camera thread | `camera_agent.py:292-340` | **FIXED** — offloaded to `_recognition_executor` thread pool |
| 38 | Redundant vector search in `process_finalized_track` | `main.py:79-95` | **FIXED** — reuses `track.pending_match_result` |
| 39 | `_alert_timestamps` dict never pruned | `alert_agent.py:14` | **FIXED** — `_prune_stale_alerts()` removes stale entries |

### 2026-06-23 — 4 issues resolved

| # | Issue | File:Line | Status |
|---|-------|-----------|--------|
| 9 | Progressive recognition blocks camera loop | `camera_agent.py:79-83` | **FIXED** — offloaded to `ThreadPoolExecutor` |
| 14 | `_python_cosine_scan` silently caps at 500 records | `db_utils.py:116` | **FIXED** — warning logged when truncated |
| 19 | `backfill_missing_embeddings` full scan on every startup | `db_utils.py:641` | **FIXED** — `_backfill_done` flag prevents re-runs |
| 23 | `memory_agent.run()` blocks camera thread | `camera_agent.py:200` | **FIXED** — runs on thread pool via `_progressive_recognition` |

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
| **Total** | **60** | **59** | **1** |

### Open issue breakdown

| # | Severity | Summary |
|---|----------|---------|
| 50 | **CRITICAL** | `.env` contains live production credentials — rotate immediately |
