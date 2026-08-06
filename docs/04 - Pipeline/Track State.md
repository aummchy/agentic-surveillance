# Track State

> Thread-safe dictionary managing all active tracks. Handles track creation, updates, expiry, visibility classification, and in-flight recognition tracking.

**File**: `pipeline/track_state.py` (313 lines)

## Class: `TrackState`

### Internal state
```python
_tracks: Dict[str, Track]       # composite_id → Track
_lock: threading.Lock            # protects _tracks
_session_epoch: int              # int(time.time()) at startup
_in_flight: Dict[str, int]      # composite_id → active recognition count
```

## Core operations

### `update(camera_id, track_id, box) → Optional[Track]`

Called every frame for every detected person.

```
1. composite_id = make_composite_id(camera_id, track_id)
2. Under _lock:
   a. IF composite_id exists:
      - Update last_seen, person_box, total_frames_seen
      - Return existing track
   b. ELSE:
      - Create new Track with first_seen=now, last_seen=now
      - Add to _tracks dict
      - Return new track
```

### `get_expired_tracks() → list`

Called every frame. Returns tracks that have timed out.

```
1. Under _lock:
   For each track:
   - IF is_expired(TRACK_TIMEOUT_SECS=15.0) OR is_max_lifetime_exceeded(MAX_TRACK_SECS=300):
     - IF already reported (expired_reported=True): skip
     - Classify visibility (before removal)
     - Mark as expired_reported=True
     - Add to expired list
     - IF no in-flight recognition: add to removal list
2. Remove tracks with no in-flight recognition
3. Return expired list
```

### `begin_recognition(composite_id)` / `end_recognition(composite_id)`

Tracks how many recognition tasks are active for each track:

```
begin: _in_flight[composite_id] += 1
end:   _in_flight[composite_id] -= 1
       IF count reaches 0 AND track expired:
           → remove from _tracks (cleanup)
```

This prevents removing a track while a recognition thread is still writing to it.

## Visibility classification

**File**: `pipeline/track_state.py:140-151`

Classified when track expires (before removal):

```
IF max_face_ratio >= VISIBLE_FACE_RATIO (0.025):
    visibility = "visible"

ELIF max_face_ratio >= PARTIAL_FACE_RATIO (0.010):
    visibility = "partial"

ELIF is_masked OR face_detected_once:
    visibility = "partial"           — mask or distant detectable face

ELIF NOT face_detected_once AND total_frames_seen >= MIN_TRACK_FRAMES (15):
    visibility = "hidden"            — long track, never saw face

ELSE:
    visibility = "unknown"           — short track, no face detected
```

## Track setters (thread-safe)

All setters follow the pattern: acquire `_lock`, find track, acquire `track._lock`, update field.

| Setter | What it updates |
|--------|----------------|
| `set_best_face()` | face crop, quality score, full frame, JPEG, face ratio |
| `set_embedding()` | embedding, mask status, det score (quality-gated) |
| `set_decision()` | decision status string |
| `set_person_name()` | person name from match |
| `set_cached_embedding()` | cached embedding for cache optimization |
| `set_pending_match_result()` | match result from progressive recognition |
| `set_pending_memory_context()` | memory context from progressive recognition |
| `set_pending_recognition_data()` | recognition result |
| `set_recognition_snapshot()` | quality + status at last recognition |
| `update_face_visibility()` | face detection count, max face ratio |
| `ensure_fallback_frame()` | first-frame JPEG (guarantees every track gets a photo) |

## `ensure_fallback_frame()`

Double-checked locking pattern:
```
1. Quick check under lock: if fallback_frame_jpeg is not None → return
2. Encode JPEG outside lock (expensive)
3. Re-check under lock: if still None → write
```

Guarantees every track gets at least one photo, independent of face quality.

## See also
- [[Data Models]] — Track dataclass definition
- [[Camera Agent]] — updates and reads track state
- [[Thread Architecture]] — locking mechanisms
- [[Track Processor]] — reads track snapshots at finalization
