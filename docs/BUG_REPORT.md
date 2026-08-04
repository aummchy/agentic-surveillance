# Bug Report — agentic_ai_singlecam

> Generated: 2026-08-01  
> Updated: 2026-08-02  
> Scope: Full codebase static analysis across all Python source files.  
> Severity: 🔴 Critical · 🟠 High · 🟡 Medium · 🟢 Low / Style

---

## Summary Table

| # | File | Severity | Problem | Status |
|---|------|----------|---------|--------|
| 1 | `agents/camera_agent.py` | 🔴 Critical | `submit_time` variable assigned but result never used in `_loop` (dead write) | 🟢 Open |
| 2 | `agents/camera_agent.py` | 🟠 High | `result.quality.overall_score` accessed without null guard on success path | ✅ Fixed (explicit error + return) |
| 3 | `agents/decision_agent.py` | 🟠 High | `_policy_agent` singleton not thread-safe — two recognition workers race on first call | ✅ Fixed (threading.Lock) |
| 4 | `agents/track_processor.py` + `utils/db_utils.py` | 🟡 Medium | `datetime.utcnow()` deprecated since Python 3.12 | 🟢 Open |
| 5 | `agents/alert_agent.py` | 🟠 High | `dispatch()` returns `True` when alert is NOT dispatched — misleading API contract | ✅ Fixed (return False) |
| 6 | `agents/memory.py` | 🟡 Medium | `last_seen` from MongoDB may be a string; silently skips confidence boost | 🟢 Open |
| 7 | `pipeline/recognition_pipeline.py` | 🟠 High | Stale `pmr` (pending match result) reused via embedding cache — stale similarity/margin | ✅ Fixed (embedding cache removed) |
| 8 | `pipeline/track_state.py` | 🟠 High | TOCTOU race in `set_best_face` double-checked locking (threshold mismatch between checks) | ✅ Fixed (design confirmed correct, docs updated) |
| 9 | `utils/llm_client.py` | 🟡 Medium | `shutdown()` fire-and-forgets `aclose()` task — async HTTP client leaks on process exit | 🟢 Open |
| 10 | `utils/db_utils.py` | 🟡 Medium | `update_face` mean embedding computed from stale pre-push list; misleading comment | 🟢 Open |
| 11 | `utils/image_utils.py` | 🟡 Medium | `resolve_track_image_url` writes `track.image_url` without holding `track._lock` | 🟢 Open |
| 12 | `config/settings.py` | 🟡 Medium | `_get()` with `cast=list` returns raw strings when `default=[]` | 🟢 Open |
| 13 | `dashboard/backend/routes/live.py` | 🟢 Low | `FRAME_SKIP = 2` hardcoded — ignores `settings.FRAME_SKIP` | 🟢 Open |
| 14 | `dashboard/backend/routes/live.py` | 🟢 Low | f-string logger calls bypass structlog — lose structured context | 🟢 Open |
| 15 | `agents/policy.py` | 🟡 Medium | `track_lifetime` mixes `datetime.now().timestamp()` with `time.time()` — fragile | 🟢 Open |
| 16 | `pipeline/tracker.py` | 🟢 Low | `os.environ["OPENVINO_DEVICE"]` set on every frame — global write in hot path | 🟢 Open |
| 17 | `agents/camera_agent.py` | 🟠 High | `_finalized_track_ids` pruned to active tracks each frame — removes guard for expired tracks | ✅ Fixed (prune removed, comment added, cleanup after finalization) |
| 18 | `utils/db_utils.py` | 🟢 Low | Collection init uses unlocked outer check — fragile on free-threaded Python 3.13+ | 🟢 Open |
| 19 | `agents/camera_agent.py` | 🟠 High | Recognition thread pool too small (max_workers=2) — 30-60s recognition delay | ✅ Fixed (increased to 4) |
| 20 | `agents/camera_agent.py` | 🔴 Critical | ByteTrack ID reuse causes composite ID collision — new track blocked by old finalized ID | ✅ Fixed (cleanup after finalization) |
| 21 | `agents/camera_agent.py` | 🟠 High | No per-track HIGH alert cooldown — repeated alerts on every progressive recognition pass | ✅ Fixed (per-track cooldown added) |

---

## Detailed Bug Descriptions

---

### Bug #1 — `submit_time` dead write in `_loop`

**File:** `agents/camera_agent.py` — Line ~227  
**Severity:** 🔴 Critical (can hide real timing data; `UnboundLocalError` risk if guard changes)

**Problem:**
```python
# In CameraAgent._loop()
if settings.DEBUG_RECOGNITION:
    submit_time = self._timing.record_submit(track.track_id)  # ← assigned, never used in _loop
    pending = self._timing.submit_pending_count()
```

`submit_time` is a local variable in `_loop`. The timing value is already consumed inside `_progressive_recognition` via `self._timing.pop_submit()`. The assignment in `_loop` is a dead write that wastes a dict slot in `TimingCollector` and silently calls `record_submit` twice (once in `_loop`, once unused; real read is via `pop_submit` in the worker).

**Fix:**
```python
if settings.DEBUG_RECOGNITION:
    self._timing.record_submit(track.track_id)   # return value not needed here
    pending = self._timing.submit_pending_count()
    logger.debug("recognition_scheduled",
                 track_id=track.track_id,
                 pending=pending)
```

---

### Bug #2 — `result.quality.overall_score` unguarded on success path

**File:** `agents/camera_agent.py` — Line ~336  
**Severity:** 🟠 High (`AttributeError: 'NoneType' object has no attribute 'overall_score'`)

**Problem:**
```python
# Only runs when skip_reason is "success" — but quality is still checked to be safe
if result.face_crop is not None:
    self.track_state.set_best_face(
        track.track_id, result.face_crop,
        result.quality.overall_score,   # ← AttributeError if quality is ever None
        frame, result.face_ratio)
```

`recognition_pipeline.py` guarantees `quality` is set when `skip_reason == "success"`, but there is no assertion. Any future code path that sets `skip_reason=""` or a new value without setting `quality` will crash here.

**Fix:**
```python
if result.face_crop is not None and result.quality is not None:
    self.track_state.set_best_face(
        track.track_id, result.face_crop,
        result.quality.overall_score, frame, result.face_ratio)
```

---

### Bug #3 — `_policy_agent` singleton not thread-safe

**File:** `agents/decision_agent.py` — Lines 17–24  
**Severity:** 🟠 High (two recognition workers can create two `PolicyAgent` instances simultaneously)

**Problem:**
```python
_policy_agent = None

def _get_policy_agent() -> PolicyAgent:
    global _policy_agent
    if _policy_agent is None:           # ← unguarded read
        _policy_agent = PolicyAgent()   # ← race: two threads can both enter here
    return _policy_agent
```

**Fix:**
```python
import threading
_policy_agent = None
_policy_lock = threading.Lock()

def _get_policy_agent() -> PolicyAgent:
    global _policy_agent
    if _policy_agent is None:
        with _policy_lock:
            if _policy_agent is None:
                _policy_agent = PolicyAgent()
    return _policy_agent
```

---

### Bug #4 — `datetime.utcnow()` deprecated in Python 3.12+

**Files:** `agents/track_processor.py`, `utils/db_utils.py` (~12 occurrences)  
**Severity:** 🟡 Medium (`DeprecationWarning` now; will raise `AttributeError` in a future Python release)

**Problem:**
```python
from datetime import datetime
now = datetime.utcnow()   # DeprecationWarning in 3.12+
```

`datetime.utcnow()` returns a naïve datetime with no timezone info and is scheduled for removal.

**Fix:** Replace all occurrences with:
```python
from datetime import datetime, timezone
now = datetime.now(tz=timezone.utc)
```

Key locations:
- `agents/track_processor.py` lines ~194, ~229
- `utils/db_utils.py` — `store_face`, `update_face`, `log_event`, `get_or_create_memory`, `update_visit_memory` (~12 sites)

---

### Bug #5 — `alert_agent.dispatch()` returns `True` when nothing is dispatched

**File:** `agents/alert_agent.py` — Lines 59–64  
**Severity:** 🟠 High (misleading API contract; callers use return value to gate WebSocket broadcast)

**Problem:**
```python
def dispatch(track, decision, image_url=None) -> bool:
    if not decision.should_alert:
        return True    # ← True, but nothing was sent!

    if track.alerted:
        return True    # ← True, but nothing was sent!
```

`track_processor.py` does:
```python
alert_dispatched = dispatch(track, decision, image_url)
if decision.should_alert and alert_dispatched:
    # broadcast alert to WebSocket dashboard
```

The outer `decision.should_alert` guard saves this from actually broadcasting incorrectly, but the semantics are wrong. A future refactor could drop the outer guard and break things.

**Fix:**
```python
def dispatch(track, decision, image_url=None) -> bool:
    if not decision.should_alert:
        return False   # no alert needed

    if track.alerted:
        return False   # already alerted this track

    if not should_send_alert(track.track_id, decision.alert_level, decision.status):
        return False   # cooldown

    # ... send to channels ...
    return True
```

---

### Bug #6 — `last_seen` type ambiguity silently skips memory boost

**File:** `agents/memory.py` — Lines 98–101  
**Severity:** 🟡 Medium (first-time-back visitors may get zero confidence boost)

**Problem:**
```python
days_since_last = None
if last_seen:
    if isinstance(last_seen, datetime):
        days_since_last = (now - last_seen).days
    # If last_seen is a string (ISO format from MongoDB), days_since_last stays None
```

MongoDB can return dates as `datetime` objects (native BSON) or as strings depending on driver version and serialization. When `last_seen` is a string, `days_since_last` stays `None`, and all time-based boosts are skipped.

**Fix:**
```python
days_since_last = None
if last_seen:
    if isinstance(last_seen, datetime):
        days_since_last = (now - last_seen).days
    elif isinstance(last_seen, str):
        try:
            parsed = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
            days_since_last = (now - parsed.replace(tzinfo=None)).days
        except (ValueError, AttributeError):
            pass
```

---

### Bug #7 — Stale `pmr` propagated via embedding cache

**File:** `pipeline/recognition_pipeline.py` — Lines 274–284  
**Severity:** 🟠 High (stale `similarity_score`, `margin`, `candidate_count` in match result)

**Problem:**
```python
# _build_embedding
if had_cached and cached_emb is not None and pmr is not None:
    ...
    if cos_dist < EMBEDDING_CACHE_COSINE_THRESHOLD:   # 0.005 — nearly identical embeddings
        match = pmr          # ← stale MatchResult from previous recognition pass
        used_cached = True
```

`pmr` was set during an earlier recognition pass with a potentially lower-quality embedding. Reusing it means `match.second_best_similarity` and `match.margin` are stale, causing incorrect confidence scores. The intent was to avoid a MongoDB round-trip, but the match data it's skipping is precisely what drives the confidence formula.

**Fix:**
```python
# Always run the DB match. Cache can be used for embedding-level
# dedup but should not bypass the full match pipeline.
# Remove the match = pmr assignment and always call matching_fn:
if match is None:
    match = self._matching_fn(embedding_list, track_id=track.track_id)
```

Or if caching is critical for performance, at least add a freshness TTL (e.g., only cache for 1 recognition cycle):
```python
# In track, add: last_match_frame: int = 0
# Only reuse if last_match_frame >= current_frame - 1
```

---

### Bug #8 — TOCTOU race in `TrackState.set_best_face`

**File:** `pipeline/track_state.py` — Lines 199–225  
**Severity:** 🟠 High (concurrent workers can overwrite a better face with worse one in edge case)

**Problem:**
```python
def set_best_face(self, composite_id, face_crop, face_score, full_frame, face_ratio):
    with self._lock:
        track = self._tracks.get(composite_id)
        if not track:
            return
        with track._lock:
            if face_score <= track.best_face_score + 0.03:  # early-exit with +0.03 margin
                return

    # JPEG encode (expensive, 10-50ms) — outside lock, window opens here

    with self._lock:
        track = self._tracks.get(composite_id)
        if track:
            with track._lock:
                if face_score > track.best_face_score:   # strict check
                    ...
```

Thread A passes first check (score = 0.75, current = 0.70), Thread B also passes (score = 0.72, current = 0.70). Both encode JPEGs. B finishes first and sets score to 0.72. A arrives with 0.75 and passes the strict check, correctly updating. This is fine. **But** if scores are 0.71 and 0.74, and 0.74 finishes first (sets score to 0.74), then 0.71 arrives and fails strict check (0.71 < 0.74) — correctly skipped. The design is actually **correct** as implemented. The comment simply needs clarification.

**Fix (documentation only):**
```python
# Early-exit optimization: avoid expensive JPEG encode for a face
# score that is barely better. The +0.03 hysteresis prevents constant
# re-encoding for marginal improvements.
# The final strict re-check at write time handles the race window: if
# another worker wrote a better face during encoding, we correctly discard.
if face_score <= track.best_face_score + 0.03:
    return
```

---

### Bug #9 — `llm_client.shutdown()` leaks async HTTP client

**File:** `utils/llm_client.py` — Lines 387–401  
**Severity:** 🟡 Medium (`ResourceWarning: unclosed client session` logged at process exit)

**Problem:**
```python
else:
    loop.create_task(async_client.aclose())   # ← fire-and-forget; process exits before task runs
```

`loop.create_task()` schedules a coroutine to run on the event loop, but `shutdown()` returns immediately. The process exits before the task executes, leaving the `httpx.AsyncClient` open.

**Fix:**
```python
else:
    future = asyncio.run_coroutine_threadsafe(async_client.aclose(), loop)
    try:
        future.result(timeout=2.0)   # wait up to 2 seconds for clean close
    except Exception:
        pass
```

---

### Bug #10 — `update_face` mean embedding uses pre-push list; misleading comment

**File:** `utils/db_utils.py` — Lines 428–436  
**Severity:** 🟡 Medium (under concurrent writes, two threads compute mean from same base list, last writer wins)

**Problem:**
```python
# Recompute mean_embedding (query BEFORE push to avoid double-counting)
existing = collection.find_one({"person_id": person_id}, {"embeddings": 1})
if existing:
    all_embs = existing.get("embeddings", [])
    all_embs.append(embedding)          # appended in Python memory (not double-counting)
    all_embs = all_embs[-emb_cap:]
    update_ops["$set"]["mean_embedding"] = _compute_mean_embedding(all_embs)
```

The comment says "query BEFORE push to avoid double-counting" but then immediately `append(embedding)` — which IS counting the new embedding. The comment is backward. Under concurrent calls, two threads read the same `existing.embeddings`, both append the new embedding, both compute the same mean, and one $set overwrites the other (benign but wastes a write).

**Fix (comment correction):**
```python
# Read CURRENT embeddings before the $push, then include the new
# embedding in the Python-side list to compute the correct mean.
# The $push below durably stores it; the mean includes it pre-commit.
```

---

### Bug #11 — `resolve_track_image_url` writes `track.image_url` without lock

**File:** `utils/image_utils.py` — Lines 146–171  
**Severity:** 🟡 Medium (data race between camera thread reading track and worker thread writing it)

**Problem:**
```python
def resolve_track_image_url(track) -> str | None:
    if track.image_url:           # ← read without lock
        return track.image_url

    ...  # slow Cloudinary upload

    track.image_url = url         # ← write without lock
    return url
```

Called from worker threads in `track_processor.py` while the camera loop reads `track.image_url` via `draw_annotations`. This is a data race.

**Fix:**
```python
def resolve_track_image_url(track) -> str | None:
    with track._lock:
        if track.image_url:
            return track.image_url

    # Slow I/O outside lock
    jpeg_data = track.best_frame_jpeg or track.fallback_frame_jpeg
    ...
    url = upload_jpeg_to_cloudinary(jpeg_data)

    with track._lock:
        if not track.image_url:   # double-check: another worker may have uploaded first
            track.image_url = url
    return url
```

---

### Bug #12 — `_get()` returns raw strings when `default=[]`

**File:** `config/settings.py` — Lines 113–118  
**Severity:** 🟡 Medium (env override for any list setting with empty default produces string list)

**Problem:**
```python
def _get(env_key, config_key, default, cast=str):
    if cast is list:
        items = [v.strip() for v in val.split(",")]
        if items and default and len(default) > 0:   # ← skipped if default=[]
            elem_type = type(default[0])
            return [elem_type(v) for v in items]
        return items   # ← raw strings!
```

Any setting defined as `_get("KEY", "KEY", [], list)` will return a list of strings from `.env`, not typed elements.

**Fix:**
```python
def _get(env_key, config_key, default, cast=str, elem_type=None):
    if cast is list:
        items = [v.strip() for v in val.split(",")]
        et = elem_type or (type(default[0]) if default else str)
        try:
            return [et(v) for v in items]
        except (ValueError, TypeError):
            return items
    ...
```

Or ensure all list settings have a non-empty typed default:
```python
# Always pass a typed default for list settings:
SOME_LIST = _get("SOME_LIST", "SOME_LIST", ["default_str"], list)
```

---

### Bug #13 — `FRAME_SKIP` hardcoded in `live.py`

**File:** `dashboard/backend/routes/live.py` — Line 14  
**Severity:** 🟢 Low (config change in `.env`/`config.jsonc` has no effect on WebSocket broadcast rate)

**Problem:**
```python
FRAME_SKIP = 2  # broadcast every Nth frame to reduce load
```

This is duplicated from `settings.FRAME_SKIP`. Changing it in `.env` only affects `track_processor.py`.

**Fix:**
```python
from config import settings
FRAME_SKIP = settings.FRAME_SKIP
```

---

### Bug #14 — f-string logger calls bypass structlog in `live.py`

**File:** `dashboard/backend/routes/live.py` — Lines 109, 122  
**Severity:** 🟢 Low (inconsistency; structured fields lost from log pipeline)

**Problem:**
```python
logger.info(f"Client connected. Total clients: {len(connected_clients)}")
logger.error(f"WebSocket error: {e}")
```

**Fix:**
```python
logger.info("ws_client_connected", total=len(connected_clients))
logger.error("ws_error", error=str(e))
logger.info("ws_client_disconnected", total=len(connected_clients))
```

---

### Bug #15 — Mixed datetime sources in `policy.py` track lifetime

**File:** `agents/policy.py` — Line 144  
**Severity:** 🟡 Medium (works correctly today; breaks silently if someone changes one side to `datetime.utcnow()`)

**Problem:**
```python
track_lifetime = (datetime.now().timestamp() - track.first_seen) if track else 0
```

`datetime.now().timestamp()` returns a UTC epoch (same as `time.time()`), so this is correct. But the code mixes two different APIs to get the same value. If `datetime.utcnow()` were substituted, `.timestamp()` would be wrong (it would apply local timezone offset to a UTC datetime, producing a double-offset error).

**Fix:**
```python
import time
track_lifetime = (time.time() - track.first_seen) if track else 0
```

---

### Bug #16 — `os.environ` written on every frame in hot path

**File:** `pipeline/tracker.py` — Line 51  
**Severity:** 🟢 Low (performance waste; global mutation from camera thread)

**Problem:**
```python
def track_persons(frame, persist=True) -> list:
    model = get_model()
    os.environ["OPENVINO_DEVICE"] = settings.OPENVINO_DEVICE   # ← every frame, ~30x/sec
    ...
```

**Fix:** Move into `get_model()` (called once):
```python
def get_model() -> YOLO:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                os.environ["OPENVINO_DEVICE"] = settings.OPENVINO_DEVICE  # ← once
                _model = YOLO(settings.YOLO_MODEL)
    return _model
```

---

### Bug #17 — `_finalized_track_ids` pruned too aggressively

**File:** `agents/camera_agent.py` — Lines 246–248  
**Severity:** 🟠 High (expired tracks may be finalized twice if they briefly leave active detection)

**Problem:**
```python
all_tracks = self.track_state.get_all()
active_track_ids = {t.track_id for t in all_tracks}
with self._track_sets_lock:
    self._finalized_track_ids &= active_track_ids   # prune to only live tracks
```

Once a track expires and is removed from `TrackState`, its ID disappears from `active_track_ids`, so the prune removes it from `_finalized_track_ids`. If a stale reference to that track still exists, the `_finalized_track_ids` guard is gone. `Track.mark_finalized_once()` is the primary idempotency guard, but removing this secondary guard unnecessarily is fragile.

**Fix:** Remove the per-frame prune entirely. The set is bounded by the number of concurrent tracks (< 50 in practice) and doesn't need pruning:
```python
# Remove the pruning block entirely.
# _finalized_track_ids is bounded — at most one entry per concurrent track.
# mark_finalized_once() on Track is the real idempotency guard.
```

If memory is a concern, prune after a TTL instead of pruning to active tracks.

---

### Bug #18 — Collection init double-checked lock fragile on free-threaded Python

**File:** `utils/db_utils.py` — Lines 43–50  
**Severity:** 🟢 Low (safe on CPython with GIL; breaks on Python 3.13+ free-threaded build)

**Problem:**
```python
def get_faces_collection() -> Collection:
    global _faces_collection
    if _faces_collection is None:            # ← unguarded read
        with _collection_locks["faces"]:
            if _faces_collection is None:    # ← guarded re-check
                ...
```

On CPython the GIL makes the outer unguarded check safe. On free-threaded Python 3.13+ (`-X gil=0`), two threads can both see `None` and both enter the lock.

**Fix:** Remove the outer check — just always acquire the lock (it's uncontended after first call):
```python
def get_faces_collection() -> Collection:
    global _faces_collection
    with _collection_locks["faces"]:
        if _faces_collection is None:
            db = get_client()[settings.MONGODB_DATABASE]
            _faces_collection = db[settings.MONGODB_COLLECTION]
    return _faces_collection
```

---

## Quick-Fix Priority Order

| Priority | Bug # | Severity | Effort | Description | Status |
|----------|-------|----------|--------|-------------|--------|
| 1 | #3 | 🟠 High | Trivial | Add lock to `_policy_agent` singleton | ✅ Fixed |
| 2 | #5 | 🟠 High | Trivial | Fix `dispatch()` return values | ✅ Fixed |
| 3 | #17 | 🟠 High | Low | Remove `_finalized_track_ids` per-frame prune | ✅ Fixed |
| 4 | #20 | 🔴 Critical | Low | ByteTrack ID reuse causes composite ID collision | ✅ Fixed |
| 5 | #19 | 🟠 High | Trivial | Increase recognition thread pool to 4 workers | ✅ Fixed |
| 6 | #21 | 🟠 High | Low | Add per-track HIGH alert cooldown | ✅ Fixed |
| 7 | #16 | 🟢 Low | Trivial | Move `os.environ` write out of hot path | 🟢 Open |
| 8 | #13 | 🟢 Low | Trivial | Use `settings.FRAME_SKIP` in `live.py` | 🟢 Open |
| 9 | #9 | 🟡 Medium | Low | Fix async client leak in `shutdown()` | 🟢 Open |
| 10 | #11 | 🟡 Medium | Low | Add lock around `track.image_url` writes | 🟢 Open |
| 11 | #15 | 🟡 Medium | Trivial | Use `time.time()` instead of `datetime.now().timestamp()` | 🟢 Open |
| 12 | #4 | 🟡 Medium | Medium | Replace all `datetime.utcnow()` with `datetime.now(tz=timezone.utc)` | 🟢 Open |
| 13 | #6 | 🟡 Medium | Low | Parse ISO string `last_seen` in `memory.py` | 🟢 Open |
| 14 | #14 | 🟢 Low | Trivial | Use structured logging in `live.py` | 🟢 Open |
| 15 | #1 | 🔴 Critical | Trivial | Remove unused `submit_time` assignment in `_loop` | 🟢 Open |
| 16 | #18 | 🟢 Low | Trivial | Remove outer unlocked check in collection init | 🟢 Open |

---

## New Bugs (from 2026-08-02 log analysis)

---

### Bug #19 — Recognition thread pool too small

**File:** `agents/camera_agent.py` — Line 37  
**Severity:** 🟠 High (30-60s recognition delay)  
**Status:** ✅ Fixed (increased to 4)

**Problem:**
```python
self._recognition_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="recognition"
)
```

With 11+ concurrent tracks, the 2-worker pool saturates. Tracks queue for 30-60+ seconds before recognition, extending their lifetime and increasing concurrency (feedback loop).

**Log evidence:**
```
2026-08-02T17:18:19 RECOG trk=5 unknown UNKNOWN sim=0.209 dur=66s conf=18
```
Track 5 waited 66 seconds before its first recognition — the pool was saturated with tracks 2, 3, 6, 9.

**Fix:**
```python
self._recognition_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="recognition"
)
```

---

### Bug #20 — ByteTrack ID reuse causes composite ID collision

**File:** `agents/camera_agent.py` — Lines 31-34, 242-244  
**Severity:** 🔴 Critical (new track blocked from finalization)  
**Status:** ✅ Fixed (cleanup after finalization)

**Problem:**
Composite IDs use the format `{camera_id}_{session_epoch}_{byte_track_id}`. When ByteTrack reuses a numeric ID (e.g., after a track expires), the new track gets the same composite ID. `_finalized_track_ids` still contains the old ID, permanently blocking the new track from finalizing.

**Log evidence:**
```
# First lifecycle — same person, finalized correctly
2026-08-02T17:18:50 FINAL trk=2 Unknown KNOWN_VISITOR sim=0.880 conf=90

# Second lifecycle — different person, same ID, BLOCKED
2026-08-02T17:19:03 RECOG trk=2 unknown UNKNOWN sim=0.327 conf=28
```

**Fix:**
```python
def _finalize_track(self, track: Track):
    ...
    finally:
        if self.on_track_finalized:
            self.on_track_finalized(track)
        # Allow ByteTrack ID reuse — remove from set after finalization
        with self._track_sets_lock:
            self._finalized_track_ids.discard(track.track_id)
```

---

### Bug #21 — No per-track HIGH alert cooldown

**File:** `agents/camera_agent.py` — Lines 388-408  
**Severity:** 🟠 High (alert fatigue, 1 alert every 8.5s)  
**Status:** ✅ Fixed (per-track cooldown added)

**Problem:**
The progressive recognition path fires `should_alert=True` on every pass for UNKNOWN tracks. While `mark_alerted_once()` prevents duplicate dispatch during finalization, the POLICY log line shows `ALERT` every time, creating operator noise.

**Log evidence:**
```
2026-08-02T17:18:19 POLICY trk=5 UNKNOWN alert=high ALERT vis=0
2026-08-02T17:18:29 POLICY trk=5 UNKNOWN alert=high ALERT vis=0
2026-08-02T17:18:35 POLICY trk=5 UNKNOWN alert=high ALERT vis=0
2026-08-02T17:18:47 POLICY trk=5 UNKNOWN alert=high ALERT vis=0
2026-08-02T17:19:05 POLICY trk=5 UNKNOWN alert=high ALERT vis=0
```

5 HIGH alerts for the same track in 46 seconds.

**Fix:**
Added `last_alert_time` field to Track model. Progressive critical alerts now check cooldown:
```python
alert_cooldown_active = (
    result.decision.should_alert
    and result.decision.alert_level in ("high", "critical")
    and track.last_alert_time > 0
    and (time.time() - track.last_alert_time) < settings.ALERT_COOLDOWN_SECS
)
```

---

---

## Verification Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Static analysis
pip install pyflakes mypy
pyflakes agents/ pipeline/ utils/ config/
mypy agents/ pipeline/ utils/ config/ --ignore-missing-imports --no-error-summary
```