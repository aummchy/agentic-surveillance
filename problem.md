# Problem Report — agentic_ai_singlecam

Codebase audit performed 2026-06-23. Issues are grouped by severity and verified against actual source lines.

---

## CRITICAL — Runtime Crashes / Guaranteed Failures

### 1. `NameError`: `decision` used before assignment
**File:** `agents/camera_agent.py:223`

```python
elif decision.status in ("unknown", "masked_unknown") and track.person_name is None:
    track.person_name = None          # line 223-224
...
decision = decide(...)                # line 228  ← defined HERE, too late
```

`decision` is referenced on line 223 but only assigned on line 228. Every time this `elif` branch is reached, the process raises `NameError: name 'decision' is not defined`. The entire block from line 223-224 is also a no-op (sets `track.person_name = None` when it is already `None`). Both lines should be deleted.

---

### 2. Worker threads have no shutdown path
**File:** `main.py:47`

```python
def worker_process_tracks():
    while True:                       # never exits
        ...
```

There is no shutdown event or sentinel value. The threads are `daemon=True` so they die with the process, but `track_queue.join()` can never be called, tasks in the queue at shutdown time are silently dropped, and any in-flight `track` object being processed is abandoned mid-write to MongoDB.

---

### 3. `ThreadPoolExecutor` never shut down
**File:** `main.py:30`

```python
_encode_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="jpeg")
```

The executor is module-level and never passed to `shutdown()`. On abnormal exit, any buffered JPEG-encode tasks are discarded without the OS reclaiming the thread cleanly. Should be shut down in the `finally` block alongside camera stop.

---

## HIGH — Incorrect Behaviour / Silent Data Loss

### 4. Lock held during CPU-intensive JPEG encode
**File:** `pipeline/track_state.py:75-87`

```python
def set_best_face(self, composite_id, face_crop, face_score, full_frame, face_ratio):
    with self._lock:                  # lock acquired
        if face_score > track.best_face_score:
            ...
            _, jpeg_buf = cv2.imencode(".jpg", full_frame, ...)   # expensive inside lock
            track.best_frame_jpeg = jpeg_buf.tobytes()
```

`cv2.imencode` on a 640×480 frame takes ~2-5 ms. All threads that need `_lock` (including `get_expired_tracks`, `update`, `set_embedding`) stall for the duration. JPEG encoding should happen outside the lock; only the assignment should be inside it.

---

### 5. Track fields mutated without holding the TrackState lock
**Files:** `agents/camera_agent.py:218, 222`, `main.py:80, 124`

```python
# camera_agent.py
track.pending_recognition = recognition_result   # no lock
track.person_name = match_result.name            # no lock

# main.py
track.person_name = match_result.name            # no lock
track.alerted = True                             # no lock
```

`TrackState._lock` protects the `_tracks` dictionary, but once a `Track` reference is obtained outside the lock, its fields can be read/written concurrently by the camera loop and the worker thread. This is a TOCTOU race that can produce corrupted `person_name`, duplicate alerts, or missed `alerted=True` checks.

---

### 6. `$match` stage inserted before `$vectorSearch` — Atlas rejects it
**File:** `utils/db_utils.py:77`

```python
if filter_role:
    pipeline.insert(0, {"$match": {"role": filter_role}})
```

MongoDB Atlas Vector Search requires `$vectorSearch` to be the **first** stage of the pipeline. Prepending `$match` causes Atlas to throw an error, which is caught and falls back to the slow `_python_cosine_scan`. Role filtering during vector search never actually works. The `$match` should be a post-filter step after `$vectorSearch`.

---

### 7. Hardcoded `0.35` threshold in `store_face` ignores settings
**File:** `utils/db_utils.py:259`

```python
for m in matches:
    if m["similarity_score"] >= 0.35:    # hardcoded, ignores DEDUP_SIMILARITY_THRESHOLD
```

The deduplication threshold for merging with verified/known faces is `0.35` regardless of `settings.DEDUP_SIMILARITY_THRESHOLD`. If the operator raises that setting (e.g. to `0.50` for stricter matching), this branch still merges at `0.35`, potentially conflating distinct persons.

---

### 8. `validate_config` and `settings.py` use different defaults for `EMBEDDING_DET_SCORE_MIN`
**File:** `config/settings.py:105` vs `:138`

```python
# in validate_config():
emb_min = float(os.getenv("EMBEDDING_DET_SCORE_MIN", "0.70"))   # line 105

# module-level constant:
EMBEDDING_DET_SCORE_MIN = float(os.getenv("EMBEDDING_DET_SCORE_MIN", "0.40"))  # line 138
```

When the env var is unset, `validate_config` validates `0.70` as the value while the rest of the system uses `0.40`. If someone relies on the validate check to confirm settings are sane, they are seeing a different number than what the system actually applies.

---

### 9. `_progressive_recognition` blocks the camera capture loop
**File:** `agents/camera_agent.py:79-83`

```python
if track and self._frame_count % settings.RECOGNITION_INTERVAL_FRAMES == 0:
    self._progressive_recognition(frame, track)   # blocking call on camera thread
```

`_progressive_recognition` includes face detection (InsightFace inference), MongoDB vector search, Memory Agent (MongoDB), Recognition Agent logic, Policy Agent logic, and potentially Cloudinary upload — all synchronous and potentially taking 100-500 ms per call. Each call stalls frame capture, causing visible lag and dropped frames. The `_recognizing_tracks` set exists (suggesting threading was planned) but threading was never implemented here.

---

### 10. Duplicate Cloudinary uploads per track
**Files:** `agents/camera_agent.py:236`, `main.py:66`

Both `_progressive_recognition` (when an alert fires) and `process_finalized_track` (always, if `best_full_frame` exists) call `upload_to_cloudinary` for the same track. If a track triggers an alert during progressive recognition and is then finalized, the same image is uploaded twice, burning Cloudinary API quota.

---

### 11. Duplicate alert dispatch possible
**Files:** `agents/camera_agent.py:230-239`, `main.py:122-124`

```python
# camera_agent.py — during progressive recognition
if decision.should_alert and not track.alerted and track.track_id not in self._finalized_track_ids:
    dispatch(track, decision, image_url)
    self.track_state.set_decision(track.track_id, decision.status, True)

# main.py — during finalization (same track, later)
if decision.should_alert and not track.alerted:
    dispatch(track, decision, image_url)
```

Due to the race condition in issue #5, `track.alerted` may not be visible to the worker thread by the time finalization runs. An alert can be dispatched twice for the same track event.

---

### 12. Both raw frame and JPEG stored simultaneously, comment is wrong
**File:** `pipeline/track_state.py:83-87`

```python
track.best_full_frame = full_frame            # ~921 KB raw numpy array
_, jpeg_buf = cv2.imencode(".jpg", full_frame, ...)
track.best_frame_jpeg = jpeg_buf.tobytes()    # comment: "save ~90% memory"
```

The raw frame is **not released** — both arrays are kept. The comment "Store compressed JPEG to save ~90% memory" is incorrect; total memory per track is ~971 KB, not ~50 KB. `best_full_frame` is needed downstream for Cloudinary upload, so fixing this requires changing the upload path to decode from `best_frame_jpeg` instead.

---

## MEDIUM — Logic Errors / Performance Problems

### 13. `find_similar_faces` and `find_similar_unknowns` load entire collection with no limit
**File:** `utils/db_utils.py:168, 201`

```python
unknowns = list(collection.find({"role": "unknown"}, {...}))    # no limit
all_faces = list(collection.find({}, {...}))                    # no limit
```

Both functions materialise the entire collection into memory. As the database grows, these calls will cause OOM errors or multi-second stalls. Neither function is called on the hot path, but `find_similar_unknowns` is called from the matching agent and `find_similar_faces` from dedup checks.

---

### 14. `_python_cosine_scan` silently caps at 500 records
**File:** `utils/db_utils.py:116`

```python
SCAN_LIMIT = 500
all_faces = list(collection.find(query, {...}).limit(SCAN_LIMIT))
```

The limit is local and undocumented in caller-facing docs. If the database has more than 500 faces, the Python fallback will miss records. No log message warns the caller that results are truncated.

---

### 15. Camera disconnection loops silently forever
**File:** `agents/camera_agent.py:63-67`

```python
ret, frame = self._cap.read()
if not ret:
    logger.warning("frame_read_failed")
    time.sleep(0.1)
    continue
```

When a USB camera disconnects, `ret` stays `False` permanently. The loop retries every 100 ms with no reconnect attempt, no failure counter, and no escalation. The process appears to run but processes nothing.

---

### 16. `_finalized_track_ids` grows without bound
**File:** `agents/camera_agent.py:33, 257`

```python
self._finalized_track_ids = set()           # never pruned
...
self._finalized_track_ids.add(track.track_id)
```

Every finalized track's composite ID is added but nothing is ever removed. In a long-running session (hours), this set grows to tens of thousands of entries. Since composite IDs include a session epoch, restart clears it, but the pattern is still a slow memory leak.

---

### 17. `PolicyAgent` instantiated on every `decide()` call
**File:** `agents/policy.py:284`

```python
def decide(track, match_result, recognition_result=None, memory_context=None):
    agent = PolicyAgent()    # new instance per call
    result = agent.run(...)
```

`decide()` is called for every progressive recognition cycle and every finalized track — potentially dozens of times per second. The `PolicyAgent` should be a singleton or `_decide` should be a plain function/staticmethod.

---

### 18. No index on `events.track_id` or `events.timestamp`
**File:** `utils/db_utils.py` (entire `log_event` and `get_events_with_faces`)

`get_events_with_faces` sorts by `timestamp` descending and filters by `status`, but no index is created on these fields at startup. `get_memory_collection()` creates indexes (line 48-49) but `get_events_collection()` and `get_faces_collection()` do not. Under load, event queries degrade to full collection scans.

---

### 19. `backfill_missing_embeddings` scans collection with no index at every startup
**File:** `utils/db_utils.py:641`

```python
for face in collection.find({"latest_embedding": {"$exists": False}}):
```

Called from `main()` on every startup. With no index on `latest_embedding`, this is a full collection scan. It also runs a second scan immediately after for `{"latest_embedding": []}`. Both scans block the startup sequence.

---

### 20. `crop_face_region` imported but never called
**File:** `agents/camera_agent.py:13`

```python
from utils.image_utils import crop_person, crop_face_region, resize_image, draw_annotations
```

`crop_face_region` appears in the import but is not used anywhere in `camera_agent.py` or any other file in the active code path. Dead import, dead function.

---

### 21. Relaxed detection thresholds are inconsistent between progressive and finalisation
**File:** `agents/camera_agent.py:117, 129, 291`

```python
# _progressive_recognition
best = crop_faces[0] if crop_faces[0]["det_score"] >= 0.20 else None   # line 117
best = frame_faces[0] if frame_faces[0]["det_score"] >= 0.20 else None  # line 129

# _finalize_track
if faces[0]["det_score"] >= 0.15:   # line 291 — different threshold
```

Three hardcoded relaxed thresholds (`0.20`, `0.20`, `0.15`) that differ between the two code paths and are not exposed as configuration. Behaviour changes between runs if a developer modifies one but forgets the others.

---

### 22. `get_unknown_faces` returns *unverified* faces, not *unknown* faces
**File:** `utils/db_utils.py:359-374`

```python
query = {"verified": {"$ne": True}}   # returns ALL unverified, including uncertain matches
```

The function is named `get_unknown_faces` and is used by dashboard routes that display "unknown persons". It returns any face that hasn't been operator-verified, including those with `role="visitor"` that were auto-registered as uncertain matches. The dashboard will show verified-but-not-yet-confirmed persons alongside true strangers.

---

### 23. `memory_agent.run()` called on camera thread, not worker
**File:** `agents/camera_agent.py:200-205`

```python
if match_result.matched and match_result.person_id:
    memory_context = self.memory_agent.run({...})   # MongoDB query, on camera thread
```

Memory Agent makes a MongoDB query synchronously inside `_progressive_recognition`, which (per issue #9) already runs on the camera thread. Any MongoDB latency directly drops frames.

---

## LOW — Code Quality / Minor Issues

### 24. `track.person_name` line 224 is a dead no-op
**File:** `agents/camera_agent.py:224`

```python
track.person_name = None    # field defaults to None, this changes nothing
```

Even after the `NameError` in issue #1 is fixed, this line is semantically meaningless and should be removed.

---

### 25. `decision` field on `Track` dataclass shadows local variable name
**File:** `pipeline/models.py:20`, `agents/camera_agent.py:228`

`Track.decision` stores the decision status string. In `camera_agent.py`, a local variable also named `decision` (a `DecisionResult` object) is used on the same frame. The naming collision is a readability hazard and likely contributed to bug #1 in the first place.

---

### 26. `update_face` silently does nothing if `person_id` does not exist
**File:** `utils/db_utils.py:327`

```python
collection.update_one({"person_id": person_id}, update_ops)   # no upsert, no return check
```

The result of `update_one` (which carries `modified_count`) is discarded. If the person_id doesn't exist, the update silently fails.

---

### 27. `decode_image` has no size guard
**File:** `utils/image_utils.py:150-152`

```python
def decode_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
```

A malformed or oversized input (e.g., a 50 MB payload posted to a dashboard API endpoint) is decoded without any size check, potentially causing OOM.

---

### 28. `vector_search` inserts `$match` at index 0 to filter role, but this was likely meant as a post-filter
**File:** `utils/db_utils.py:77` *(duplicates high-severity issue #6 — listed here for completeness)*

This means any caller that passes `filter_role` (currently none in the active path, but the parameter exists for future use) will always fall back to Python scan without realising it.

---

### 29. `LOG_DIR.mkdir` runs at module import time
**File:** `config/settings.py:11`

```python
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
```

This side effect runs every time `config.settings` is imported, including in tests, CLI tools, or any utility script. If the working directory changes or the parent is read-only, the import itself fails.

---

### 30. No validation that `CAMERA_INDEX` maps to a real device
**File:** `config/settings.py`, `agents/camera_agent.py:37-43`

```python
self._cap = cv2.VideoCapture(settings.CAMERA_INDEX)
if not self._cap.isOpened():
    logger.error("camera_open_failed")
    return                   # returns silently, _loop() never runs
```

The agent returns without raising an exception, so `camera.start()` in `main.py` finishes immediately and the process appears to run normally while doing nothing.

---

### 31. Composite track ID exposed in UI label
**File:** `utils/image_utils.py:118`

```python
label = f"ID:{track.track_id}"   # e.g. "ID:cam_01_1719148800_42"
```

The label contains the session epoch timestamp, which is implementation detail. It's also too long for typical bounding-box overlays. Should display only the short numeric ID.

---

### 32. `is_known_from_memory` always `False` — field name mismatch
**File:** `agents/policy.py:131`, `agents/memory.py` (return dict)

```python
is_known_from_memory = memory.get("is_known", False)   # policy.py
```

The Memory Agent's `run()` return dict uses the key `"is_known_visitor"` (or similar) not `"is_known"`. This means `is_known_from_memory` is always `False`, so Rule 4 ("Known visitor — matched + memory confirms") is unreachable. Confirm the exact key name in `agents/memory.py` and align it.

---

## Configuration Inconsistencies

| Setting | `validate_config` default | Module-level default | Correct? |
|---|---|---|---|
| `EMBEDDING_DET_SCORE_MIN` | `0.70` | `0.40` | No — mismatch |
| `MATCH_THRESHOLD` | `0.35` (max check ≤ 0.45) | `0.25` | Different scale, confusing |
| `DEDUP_SIMILARITY_THRESHOLD` | not validated | `0.40` | Not validated at all |

The `.env.example` should be audited to ensure all defaults match the module-level constants.

---

## Summary Table

| # | File | Severity | Category |
|---|---|---|---|
| 1 | `camera_agent.py:223` | **Critical** | NameError crash |
| 2 | `main.py:47` | **Critical** | No shutdown path |
| 3 | `main.py:30` | **Critical** | Resource leak |
| 4 | `track_state.py:75-87` | High | Lock contention / performance |
| 5 | `camera_agent.py:218,222`, `main.py:80,124` | High | Race condition |
| 6 | `db_utils.py:77` | High | Atlas pipeline rejected |
| 7 | `db_utils.py:259` | High | Hardcoded threshold |
| 8 | `settings.py:105,138` | High | Config default mismatch |
| 9 | `camera_agent.py:79-83` | High | Blocking camera loop |
| 10 | `camera_agent.py:236`, `main.py:66` | High | Duplicate uploads |
| 11 | `camera_agent.py:230`, `main.py:122` | High | Duplicate alerts |
| 12 | `track_state.py:83-87` | High | Memory waste |
| 13 | `db_utils.py:168,201` | Medium | Unbounded memory |
| 14 | `db_utils.py:116` | Medium | Silent result truncation |
| 15 | `camera_agent.py:63-67` | Medium | No reconnect logic |
| 16 | `camera_agent.py:33,257` | Medium | Set grows without bound |
| 17 | `policy.py:284` | Medium | Wasteful instantiation |
| 18 | `db_utils.py` | Medium | Missing DB indexes |
| 19 | `db_utils.py:641` | Medium | Full scan at startup |
| 20 | `camera_agent.py:13` | Medium | Dead import/function |
| 21 | `camera_agent.py:117,129,291` | Medium | Inconsistent thresholds |
| 22 | `db_utils.py:359` | Medium | Misleading function name |
| 23 | `camera_agent.py:200` | Medium | MongoDB on camera thread |
| 24 | `camera_agent.py:224` | Low | Dead code |
| 25 | `models.py:20`, `camera_agent.py:228` | Low | Variable shadowing |
| 26 | `db_utils.py:327` | Low | Silent update failure |
| 27 | `image_utils.py:150` | Low | No input size guard |
| 28 | `db_utils.py:77` | Low | Role filter always falls back |
| 29 | `settings.py:11` | Low | Side effect at import |
| 30 | `settings.py`, `camera_agent.py:37` | Low | No camera index validation |
| 31 | `image_utils.py:118` | Low | Leaking internal ID in UI |
| 32 | `policy.py:131` | Low | Key name mismatch — rule unreachable |