# Problem Report — agentic_ai_singlecam

Codebase audit performed 2026-06-24 (updated 2026-06-24). Issues are grouped by severity and verified against actual source lines.

**Status: 10 issues remaining (new audit pass). 39 prior issues remain FIXED.**

---

## REMAINING — Issues Yet to be Solved

### HIGH

#### Issue 40 — `classify_visibility()` is a no-op: `track.visibility` is always "unknown"

**File:** `pipeline/track_state.py:53–63, 128–145` · `agents/camera_agent.py:298`

`get_expired_tracks()` deletes each track from `_tracks` *before* returning it:

```python
# track_state.py:53-63
def get_expired_tracks(self) -> list:
    with self._lock:
        ...
        for cid in to_remove:
            del self._tracks[cid]   # ← removed here
    return expired                   # ← objects returned but no longer in dict
```

Then `_finalize_track` immediately calls:

```python
# camera_agent.py:298
self.track_state.classify_visibility(track.track_id)
```

But `classify_visibility` does:

```python
# track_state.py:128-145
def classify_visibility(self, composite_id: str) -> str:
    with self._lock:
        track = self._tracks.get(composite_id)   # ← returns None (already deleted)
        if not track:
            return "unknown"                      # ← exits without mutating anything
```

**Effect:** `track.visibility` is permanently stuck at its dataclass default `"unknown"` for every finalized track. The two PolicyAgent rules that depend on it never fire:

- **Rule 6** (`visibility == "hidden"`) — person avoiding camera detection → never triggers, no `high` alert
- **Rule 7** (`visibility == "partial"`) — partial face view → never triggers (only `is_masked` path works)

**Fix:** Either call `classify_visibility` before removing from `_tracks` in `get_expired_tracks()`, or change `classify_visibility` to accept the track object directly and mutate it without a dict lookup.

---

### MEDIUM

#### Issue 41 — `_alert_timestamps` cooldown check is not thread-safe → duplicate alerts possible

**File:** `agents/alert_agent.py:34–42`

`should_send_alert()` is called from two different executor threads concurrently (the `_recognition_executor` in `camera_agent.py` and the `track_queue` worker in `main.py`). The read-check-write is not atomic:

```python
last = _alert_timestamps.get(key, 0)          # thread A reads last=0
if now - last < settings.ALERT_COOLDOWN_SECS:  # thread A: 60 > 0, passes
    return False
_alert_timestamps[key] = now                   # thread B also reads last=0, passes
return True                                    # both threads send the alert
```

CPython's GIL makes individual dict ops atomic but does not make this check-then-act sequence atomic.

**Fix:** Wrap the check and update in a `threading.Lock`.

---

#### Issue 42 — `list_faces(status="all")` silently returns only unknown faces

**File:** `dashboard/backend/routes/faces.py:44–46`

```python
else:                                                          # ← any status not "unknown"/"verified"
    result = await asyncio.to_thread(get_unknown_faces, ...)  # ← returns unknowns only
    return result
```

`?status=all` falls into the `else` branch and calls `get_unknown_faces()` which filters `role == "unknown"`. Callers expecting all faces receive only unknowns.

**Fix:** Add an explicit `elif status == "all"` branch (or `status is None`) that queries without a `role` filter, and raise `HTTP 400` for invalid values.

---

#### Issue 43 — `event._id` stripped by Pydantic → all React list keys are `undefined`

**File:** `dashboard/frontend/src/components/EventLog.jsx:127` · `dashboard/backend/models.py`

`get_events_with_faces()` (`db_utils.py:440`) converts `_id` to string, but `EventResponse` in `models.py` has no `_id` field. FastAPI's `response_model=EventsResponse` validation strips unknown fields, so `_id` never reaches the client.

`EventLog.jsx:127` then does:
```jsx
<div key={event._id} ...>   // event._id is undefined → all keys are undefined
```

React logs "Encountered two children with the same key (`undefined`)" and list reconciliation fails silently — wrong items may update or flash on rapid refreshes.

**Fix:** Either add `id: Optional[str] = Field(None, alias="_id")` to `EventResponse` (with `model_config = ConfigDict(populate_by_name=True)`), or use `event.track_id` as the key (which is unique per event).

---

### LOW

#### Issue 44 — `AlertAgent` in `agents/alert.py` is dead code — never wired in

**File:** `agents/alert.py`

`agents/alert.py` implements a full `AlertAgent(BaseAgent)` with context-aware channel selection (Phase 2.4 per AGENTS.md). It is never imported anywhere. Both `main.py` and `camera_agent.py` import directly from `agents/alert_agent.py`:

```python
from agents.alert_agent import dispatch   # main.py:15, camera_agent.py:267
```

`agents/alert.py` and `agents/alert_agent.py` duplicate alert logic with divergent implementations. This causes confusion about which is authoritative.

**Fix:** Either wire `AlertAgent.run()` into the pipeline and delete `alert_agent.py`, or delete `alert.py` and keep `alert_agent.py`.

---

#### Issue 45 — Two `PolicyAgent` singletons coexist at runtime

**File:** `agents/decision_agent.py:17` · `agents/policy.py:277`

Both files maintain a separate module-level singleton:

```python
# decision_agent.py:17
_policy_agent = None          # used by decision_agent.decide()

# policy.py:277
_policy_agent_instance = None # used by policy.decide()
```

`camera_agent.py` imports `from agents.decision_agent import decide` and `main.py` also imports `from agents.decision_agent import decide`, but `policy.py`'s own `decide()` convenience function creates yet another instance if called directly.

**Fix:** Delete the `decide()` convenience function from `policy.py` entirely (it duplicates `decision_agent.decide()`). Keep only `decision_agent._get_policy_agent()` as the one true singleton.

---

#### Issue 46 — `agents/alert.py` own `_alert_timestamps` never pruned

**File:** `agents/alert.py:38, 163–173`

`agents/alert.py` has its own `_alert_timestamps: Dict[str, float] = {}` with no pruning — the same unbounded-growth issue fixed in `alert_agent.py` (issue #39). Since `AlertAgent` is dead code (issue 44), this only matters if it gets wired in; the fix is the same as issue 44 (prune or remove).

---

#### Issue 47 — `seenIds` state in `UnknownPersons.jsx` declared but never used

**File:** `dashboard/frontend/src/components/UnknownPersons.jsx:8`

```jsx
const [seenIds, setSeenIds] = useState(new Set())  // declared
// setSeenIds is never called; seenIds is never read
```

Dead state. Wastes a `useState` slot and misleads readers about intended "new person" highlight logic (which is actually done via `isNewPerson()` using a 5-second timestamp check).

**Fix:** Remove the `seenIds` / `setSeenIds` declaration.

---

#### Issue 48 — `get_recent_incidents` makes N synchronous MongoDB calls inside one thread

**File:** `dashboard/backend/routes/reports.py:54–69`

```python
def _fetch():
    events = get_events_with_faces(limit=limit).get("events", [])
    for event in events:
        result = report_agent.run({...})  # ← calls get_visit_history() per event
        reports.append(result)
return await asyncio.to_thread(_fetch)
```

With `limit=100`, `_incident_report` calls `get_visit_history(person_id)` (one MongoDB query) per event — 100 sequential queries inside a single `asyncio.to_thread`. The endpoint will be slow for large limits and holds the thread for the entire duration.

**Fix:** Batch-fetch all needed `person_id` memory docs in one `find({"person_id": {"$in": person_ids}})` call before the loop (same pattern already used in `get_events_with_faces()` for face lookups).

---

#### Issue 49 — `broadcast_alert` skips blacklist and intentionally_hidden detections

**File:** `main.py:131–145`

```python
if decision.status in ("unknown", "masked_unknown") and not match_result.matched:
    ...
    asyncio.run_coroutine_threadsafe(broadcast_alert(alert_payload), loop)
```

`blacklist` (critical) and `intentionally_hidden` (high) alerts fire channel dispatch (`dispatch()`) but never trigger the WebSocket `broadcast_alert`. The dashboard only learns about them after the EventLog's 10-second polling interval.

**Fix:** Widen the condition to `decision.should_alert` (or specifically include the high-priority statuses) so any alert-worthy detection reaches the dashboard in real time.

---

## SOLVED — Issues Fixed

### 2026-06-24 — Final 3 issues resolved

| # | Issue | File:Line | Status |
|---|-------|-----------|--------|
| 37 | `_finalize_track` runs InsightFace inference on camera thread (fallback path) | `camera_agent.py:292-340` | **FIXED** — offloaded to `_recognition_executor` thread pool |
| 38 | Redundant vector search — `process_finalized_track` re-runs match even when `track.pending_match_result` exists | `main.py:79-95` | **FIXED** — reuses `track.pending_match_result` from progressive recognition |
| 39 | `_alert_timestamps` dict never pruned, grows indefinitely | `alert_agent.py:14` | **FIXED** — `_prune_stale_alerts()` removes entries older than 2x cooldown |

### 2026-06-23 — Previous 4 remaining issues resolved

| # | Issue | File:Line | Status |
|---|-------|-----------|--------|
| 9 | Progressive recognition blocks camera loop (sync InsightFace + MongoDB on camera thread) | `camera_agent.py:79-83` | **FIXED** — `_progressive_recognition` now submitted to `ThreadPoolExecutor` |
| 14 | `_python_cosine_scan` silently caps at 500 records with no log warning | `db_utils.py:116` | **FIXED** — `logger.warning("python_cosine_scan_truncated", ...)` fires when truncated |
| 19 | `backfill_missing_embeddings` full collection scan on every startup | `db_utils.py:641` | **FIXED** — `_backfill_done` flag prevents re-runs; `count_documents` early-exits |
| 23 | `memory_agent.run()` blocks camera thread (MongoDB query) | `camera_agent.py:200` | **FIXED** — runs inside `_progressive_recognition` which executes on thread pool |

### 2026-06-23 — Original 32 issues fixed

#### CRITICAL — Runtime Crashes / Guaranteed Failures

| # | Issue | File:Line | Status |
|---|-------|-----------|--------|
| 1 | `NameError`: `decision` used before assignment | `camera_agent.py:222` | **FIXED** — `decision = decide(...)` now assigned before use |
| 2 | Worker threads have no shutdown path | `main.py:46-58` | **FIXED** — `_shutdown_event` added, workers check `while not _shutdown_event.is_set()` |
| 3 | `ThreadPoolExecutor` never shut down | `main.py:30` | **FIXED** — `_encode_executor.shutdown(wait=False)` called during shutdown |

#### HIGH — Incorrect Behaviour / Silent Data Loss

| # | Issue | File:Line | Status |
|---|-------|-----------|--------|
| 4 | Lock held during CPU-intensive JPEG encode | `track_state.py:75-87` | **FIXED** — `cv2.imencode` moved outside `with self._lock` |
| 5 | Track fields mutated without holding the TrackState lock | `camera_agent.py:226`, `track_state.py` | **FIXED** — `set_person_name()` method added, all mutations go through locked methods |
| 6 | `$match` stage inserted before `$vectorSearch` — Atlas rejects it | `db_utils.py:77` | **FIXED** — `pipeline.insert(0, ...)` changed to `pipeline.append(...)` |
| 7 | Hardcoded `0.35` threshold in `store_face` ignores settings | `db_utils.py:259` | **FIXED** — changed to `settings.DEDUP_SIMILARITY_THRESHOLD` |
| 8 | `validate_config` and `settings.py` use different defaults for `EMBEDDING_DET_SCORE_MIN` | `settings.py:105` | **FIXED** — `validate_config` default changed to `0.40` to match module-level |
| 10 | Duplicate Cloudinary uploads per track | `camera_agent.py:236`, `main.py:66` | **FIXED** — `track.image_url` field added, both paths check before uploading |
| 11 | Duplicate alert dispatch possible | `camera_agent.py:230`, `main.py:122` | **FIXED** — both paths check `not track.alerted` before dispatching |
| 12 | Both raw frame and JPEG stored simultaneously | `track_state.py:83-87` | **FIXED** — `track.best_full_frame = None` set after upload, raw frame released |

#### MEDIUM — Logic Errors / Performance Problems

| # | Issue | File:Line | Status |
|---|-------|-----------|--------|
| 13 | `find_similar_faces` and `find_similar_unknowns` load entire collection with no limit | `db_utils.py:168, 201` | **FIXED** — both functions now have `.limit(500)` |
| 15 | Camera disconnection loops silently forever | `camera_agent.py:63-67` | **FIXED** — `consecutive_failures` counter added, reconnect after 30 failures |
| 16 | `_finalized_track_ids` grows without bound | `camera_agent.py:33, 257` | **FIXED** — set is pruned each frame to only keep active tracks |
| 17 | `PolicyAgent` instantiated on every `decide()` call | `policy.py:284` | **FIXED** — `PolicyAgent` is now a singleton via `_policy_agent_instance` global |
| 18 | No index on `events.track_id` or `events.timestamp` | `db_utils.py` | **FIXED** — indexes created on `track_id`, `timestamp`, and `status` |
| 20 | `crop_face_region` imported but never called | `camera_agent.py:13` | **FIXED** — dead import removed |
| 21 | Relaxed detection thresholds inconsistent between progressive and finalisation | `camera_agent.py:117, 129, 291` | **FIXED** — all hardcoded values replaced with `settings.DET_SCORE_RELAXED` |
| 22 | `get_unknown_faces` returns *unverified* faces, not *unknown* faces | `db_utils.py:359-374` | **FIXED** — query changed to `{"role": "unknown", "verified": {"$ne": True}}` |

#### LOW — Code Quality / Minor Issues

| # | Issue | File:Line | Status |
|---|-------|-----------|--------|
| 24 | `track.person_name` line 224 is a dead no-op | `camera_agent.py:224` | **FIXED** — line removed |
| 25 | `decision` field on `Track` dataclass shadows local variable name | `models.py:20`, `camera_agent.py:228` | **FIXED** — naming collision resolved |
| 26 | `update_face` silently does nothing if `person_id` does not exist | `db_utils.py:327` | **FIXED** — `update_one` result returned as `result.modified_count > 0` |
| 27 | `decode_image` has no size guard | `image_utils.py:150-152` | **FIXED** — `max_size_mb` parameter added (default 10MB) |
| 28 | `vector_search` inserts `$match` at index 0 to filter role | `db_utils.py:77` | **FIXED** — duplicate of #6, fixed together |
| 29 | `LOG_DIR.mkdir` runs at module import time | `settings.py:11` | **FIXED** — `mkdir()` moved inside `setup_file_logging()` |
| 30 | No validation that `CAMERA_INDEX` maps to a real device | `settings.py`, `camera_agent.py:37` | **FIXED** — pre-validation added: opens test capture, checks `isOpened()` |
| 31 | Composite track ID exposed in UI label | `image_utils.py:118` | **FIXED** — label now shows short numeric ID: `track.track_id.rsplit("_", 1)[-1]` |
| 32 | `is_known_from_memory` always `False` — field name mismatch | `memory.py:127` | **FIXED** — added `"known_visitor"` to the `is_known` check list |

### Earlier fixes

| # | Issue | Status |
|---|-------|--------|
| 33 | Vite build error — axios v1.18.0 incompatible with esbuild | **FIXED** — pinned to `axios@1.7.9` |
| 34 | `GET /api/events` 500 error — `similarity_score: float` rejected null values | **FIXED** — `Optional[float]` in `dashboard/backend/models.py` |
| 35 | Terminal log noise — uvicorn access logs spamming console | **FIXED** — set to `warning` + `access_log=False` |
| 36 | File logging not working — `structlog.PrintLoggerFactory` bypassed stdlib logging | **FIXED** — switched to `structlog.stdlib.LoggerFactory()` + `ProcessorFormatter` |

---

## Configuration Inconsistencies (resolved)

| Setting | `validate_config` default | Module-level default | Status |
|---|---|---|---|
| `EMBEDDING_DET_SCORE_MIN` | `0.40` | `0.40` | **ALIGNED** |
| `MATCH_THRESHOLD` | `0.35` (max check ≤ 0.45) | `0.25` | Different scale, documented |
| `DEDUP_SIMILARITY_THRESHOLD` | not validated | `0.40` | Low risk |

---

## Summary

| Audit Date | Issues Found | Issues Fixed | Remaining |
|------------|--------------|--------------|-----------|
| 2026-06-22 | 32 | 32 | 0 |
| 2026-06-23 | 4 | 4 | 0 |
| 2026-06-24 (pass 1) | 3 | 3 | 0 |
| 2026-06-24 (pass 2) | 10 | 0 | **10** |
| **Total** | **49** | **39** | **10** |

### Open issue breakdown

| # | Severity | Summary |
|---|----------|---------|
| 40 | HIGH | `classify_visibility()` no-op — `track.visibility` always "unknown", rules 6 & 7 never fire |
| 41 | MEDIUM | `_alert_timestamps` cooldown not thread-safe — duplicate alerts from concurrent threads |
| 42 | MEDIUM | `list_faces(status="all")` returns unknown faces only — `else` falls through to wrong query |
| 43 | MEDIUM | `event._id` stripped by Pydantic — React list keys all `undefined`, list reconciliation broken |
| 44 | LOW | `AlertAgent` in `agents/alert.py` dead code — never imported or wired in |
| 45 | LOW | Two `PolicyAgent` singletons — one in `decision_agent.py`, one in `policy.py` |
| 46 | LOW | `agents/alert.py` `_alert_timestamps` never pruned (related to #44 dead code) |
| 47 | LOW | `seenIds` state in `UnknownPersons.jsx` declared but never read or set |
| 48 | LOW | `get_recent_incidents` N+1 MongoDB queries — one `get_visit_history` call per event |
| 49 | LOW | `broadcast_alert` skips blacklist/intentionally_hidden — no real-time dashboard push |
