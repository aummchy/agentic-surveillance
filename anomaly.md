# ANOMALY.md — System Failure Analysis

Deep analysis of why the surveillance system is breaking. Every issue is mapped to its effect on the pipeline.

---

## TABLE OF CONTENTS

1. [Pipeline Flow Overview](#1-pipeline-flow-overview)
2. [Critical: Face Detection & Embedding Failures](#2-critical-face-detection--embedding-failures)
3. [Critical: Matching & Recognition Failures](#3-critical-matching--recognition-failures)
4. [Critical: Dashboard Sections Not Loading](#4-critical-dashboard-sections-not-loading)
5. [Critical: Configuration Mismatches](#5-critical-configuration-mismatches)
6. [Thread Safety & Race Conditions](#6-thread-safety--race-conditions)
7. [Data Flow Breaks](#7-data-flow-breaks)
8. [Alert System Issues](#8-alert-system-issues)
9. [Dashboard Frontend Issues](#9-dashboard-frontend-issues)
10. [Dead Code & Misleading Logic](#10-dead-code--misleading-logic)

---

## 1. PIPELINE FLOW OVERVIEW

The system pipeline is:

```
Camera Frame
  → YOLOv8 Detection + ByteTrack Tracking (pipeline/tracker.py)
  → Track State Accumulation (pipeline/track_state.py)
  → Progressive Face Recognition (agents/camera_agent.py)
      → Crop person region
      → InsightFace detection + embedding (utils/embedding_utils.py)
      → Quality scoring (pipeline/quality_agent.py)
  → Track Finalization (when track expires)
      → MongoDB Vector Search (utils/db_utils.py)
      → Recognition Decision (agents/recognition.py)
      → Policy Decision (agents/policy.py)
      → Alert Dispatch (agents/alert_agent.py)
      → Visit Memory Update (agents/memory.py)
      → MongoDB Event Log
      → WebSocket Broadcast to Dashboard
```

**Each link in this chain has failures documented below.**

---

## 2. CRITICAL: FACE DETECTION & EMBEDDING FAILURES

### 2a. Resolution Mismatch: Camera vs InsightFace

| Setting | Value | Source |
|---------|-------|--------|
| `FRAME_WIDTH` | `640` | `.env` line 92 |
| `FRAME_HEIGHT` | `480` | `.env` line 93 |
| `INSIGHTFACE_DET_SIZE` | `1280` | `settings.py` line 216 |

**What happens:** The camera captures at 640x480 but InsightFace's detection model expects 1280px input. When a 640x480 frame is sent to InsightFace with `det_size=(1280,1280)`, the model upscales it internally, introducing interpolation artifacts that degrade small-face detection. Faces that would be detected at native 1280x720 are missed or produce low-quality embeddings at 640x480.

**Effect:** Fewer faces detected → more "unknown" classifications → system appears broken.

### 2b. Embedding Detection Score Threshold Mismatch

| Setting | Value | Source |
|---------|-------|--------|
| `EMBEDDING_DET_SCORE_MIN` | `0.30` | `.env` line 49 |
| `EMBEDDING_DET_SCORE_MIN` (default) | `0.40` | `settings.py` line 229 |
| `EMBEDDING_DET_SCORE_MIN` (config) | `0.40` | `config.jsonc` line 36 |

**What happens:** The `.env` overrides to `0.30`, allowing embeddings from very low-confidence detections. These low-quality embeddings produce unreliable vector matches — either false positives (wrong person matched) or false negatives (genuine person not matched).

### 2c. Face Crop Fallback Logic Bug

**File:** `agents/camera_agent.py` line 178

```python
frame_faces = [] if crop_faces else app.detect_faces_raw(frame, ...)
```

**What happens:** If `crop_faces` is non-empty (face detected in person crop), `frame_faces` is set to `[]`. Later on line 195, if `best` is None (no face in crop met the threshold), the code checks `if not best and frame_faces:` — but `frame_faces` is empty, so the full-frame fallback is skipped entirely.

**In practice:** Since `detect_faces_raw` on line 177 uses `min_score=DET_SCORE_RELAXED` (0.20), all faces in `crop_faces` already passed 0.20. The `best` assignment on line 190 always succeeds if `crop_faces` is non-empty. So this is a dead code path, not a runtime bug — but it means the fallback never triggers when it logically should.

### 2d. CLAHE Applied Per-Call (Performance)

**File:** `utils/embedding_utils.py` lines 36-52

CLAHE (contrast-limited adaptive histogram equalization) is applied to every face detection call. With 2 recognition threads, this doubles the compute time per face. Under heavy load (many concurrent tracks), this creates a bottleneck where recognition threads fall behind the camera frame rate.

---

## 3. CRITICAL: MATCHING & RECOGNITION FAILURES

### 3a. MATCH_THRESHOLD Too Low

| Setting | Value | Source |
|---------|-------|--------|
| `MATCH_THRESHOLD` | `0.25` | `.env` line 31 |
| `MATCH_THRESHOLD` (default) | `0.45` | `settings.py` line 255 |
| Recommended range | `0.40-0.60` | `config.jsonc` line 14 |

**What happens:** ArcFace embeddings are 512-dimensional unit vectors. A cosine similarity of 0.25 means the angle between embeddings is ~75° — very different. At this threshold:
- **False positives:** Different people with similar-looking faces (same hair color, similar lighting) get matched as the same person
- **False negatives still happen:** Genuine same-person matches under poor lighting may score below 0.25

**Effect:** The system produces wrong match results — matching strangers while rejecting acquaintances.

### 3b. Vector Search Score Conversion

**File:** `utils/db_utils.py` lines 111, 195

Atlas `$vectorSearch` returns a score in range [0, 2] where:
```
atlas_score = (1 + raw_cosine) / 2
```

The code converts back:
```python
raw_cosine = (atlas_score * 2) - 1
```

Then compares against `MATCH_THRESHOLD` using `compare_similarity()`. **This conversion is correct**, but the threshold of 0.25 means `compare_similarity` returns True for any `raw_cosine >= 0.25`, which is a very permissive match.

### 3c. Recognition Decision Logic

**File:** `agents/recognition.py` lines 97-167

The recognition agent has three cases:

1. **Similarity >= 0.90:** High confidence (70-95%), classified as "known"
2. **Similarity >= MATCH_THRESHOLD (0.25):** Moderate confidence, classified as "known" or "uncertain"
3. **Similarity >= MATCH_THRESHOLD * 0.8 (0.20):** Borderline, classified as "uncertain"
4. **Below all thresholds:** Classified as "unknown"

**Problem:** With MATCH_THRESHOLD=0.25, the "known" classification triggers at very low similarity. Case 2 covers the range [0.25, 0.90] — a huge range where very different people get classified as "known" with confidence scores from ~0% to 70%.

### 3d. Policy Agent Threshold Mismatch

**File:** `agents/policy.py` line 211

```python
if match_result.similarity >= settings.MATCH_THRESHOLD:
```

This uses the same 0.25 threshold, so even borderline matches trigger the "known_visitor" classification path.

---

## 4. CRITICAL: DASHBOARD SECTIONS NOT LOADING

### 4a. Frontend Image URLs Broken

**Files:** `App.jsx:21`, `EventLog.jsx:79`, `UnknownPersons.jsx:53`, `VerifiedPersons.jsx:65`

All components use:
```javascript
const getImageUrl = (url) => {
  if (!url) return '/placeholder.jpg';
  if (url.startsWith('http')) return url;
  return `http://localhost:8000/${url}`;
};
```

**Problems:**
1. **Hardcoded `localhost:8000`** — If accessed from another machine, all images break
2. **Vite proxy doesn't cover `/captures`** — Only `/api` and `/ws` are proxied, so direct image URLs fail
3. **`VerifyModal.jsx:36` doesn't call `getImageUrl()`** — Raw relative paths are used, breaking images in the verify modal

### 4b. Stats Endpoint Uses `fetch()` Without Error Checking

**File:** `App.jsx:29-36`

```javascript
const res = await fetch('/api/events/stats');
const data = await res.json();
setStats(data);
```

**Problem:** `fetch()` does NOT throw on HTTP errors (400, 500). A server error returns a JSON error object like `{detail: "Server shutting down"}`, which gets set as `stats` state. The dashboard then tries to render `stats.total_events` which is `undefined`, causing blank/broken sections.

### 4c. Silent WebSocket Error Swallowing

**File:** `App.jsx:72`

```javascript
} catch (e) {}
```

**Problem:** Any malformed WebSocket message (corrupted base64, network error, JSON parse failure) is silently swallowed. The user sees the live feed freeze with no error indication.

### 4d. UnknownPersons "New Person" Flash Timing

**File:** `UnknownPersons.jsx:71-75`

```javascript
const isNewPerson = (person) => {
  const created = new Date(person.created_at);
  const now = new Date();
  return (now - created) < 5000;
};
```

**Problem:** The "new person" red border only shows for 5 seconds after creation. If the user isn't watching the exact moment a person appears, they never see the indicator. This makes the "unknown persons" section appear to not update.

### 4e. Chat Panel Has No Conversation Context

**Files:** `ChatPanel.jsx`, `dashboard/backend/routes/chat.py:108`

Each chat request sends only the current message — no conversation history. Multi-turn queries like "show me events" followed by "filter by critical" don't work because the second message has no context.

### 4f. Event Log Missing Status Filters

**File:** `EventLog.jsx:105-124`

Tabs only offer "All", "Unknown", "Verified". There's no way to filter for `masked_unknown`, `blacklist`, or `intentionally_hidden` — all lumped under "Unknown".

---

## 5. CRITICAL: CONFIGURATION MISMATCHES

### 5a. Three-Way Settings Conflict

The system has three configuration layers: `.env` → `config.jsonc` → hardcoded defaults. The `.env` takes highest priority, but it often conflicts with the other two:

| Setting | `.env` | `config.jsonc` | `settings.py` default |
|---------|--------|----------------|----------------------|
| `FRAME_WIDTH` | `640` | `1280` | `1280` |
| `FRAME_HEIGHT` | `480` | `720` | `720` |
| `MATCH_THRESHOLD` | `0.25` | `0.45` | `0.45` |
| `PERSON_CONF_THRESHOLD` | `0.5` | `0.40` | `0.40` |
| `EMBEDDING_DET_SCORE_MIN` | `0.30` | `0.40` | `0.40` |

**Effect:** The `.env` values override everything, but developers editing `config.jsonc` think their changes take effect — they don't. The three-layer system creates confusion about what's actually running.

### 5b. Validation Uses Different Defaults Than Runtime

**File:** `config/settings.py` line 177

```python
det_min = _get("EMBEDDING_DET_SCORE_MIN", float, 0.40)
```

The validation function defaults to `0.40`, but the actual setting defaults to `0.40` while `.env` overrides to `0.30`. The validation passes because `0.30 <= 0.40`, but the effective value is lower than what the codebase was designed for.

---

## 6. THREAD SAFETY & RACE CONDITIONS

### 6a. TOCTOU in `update_visit_memory`

**File:** `utils/db_utils.py` lines 591-648

```python
memory = get_or_create_memory(person_id)  # find_one_and_update (upsert)
# ... modify memory in Python ...
collection.update_one({"person_id": person_id}, {"$set": ...})
```

**What happens:** Between `get_or_create_memory` and `update_one`, another thread can modify the same person's visit record. Two concurrent visits for the same person will overwrite each other's visit count and history.

**Effect:** Visit counts are lost, memory context is wrong, confidence boosts are incorrect.

### 6b. Track State Race During Recognition

**File:** `agents/camera_agent.py` lines 140-152, `pipeline/track_state.py` lines 89-107

The camera loop calls `get_expired_tracks()` which removes tracks from the shared `_tracks` dict. Simultaneously, recognition threads hold references to those track objects and call `track_state.update_face_visibility()` and `track_state.set_best_face()`.

**What happens:** After a track is expired and removed, recognition thread calls `track_state.set_best_face(composite_id, ...)` — the track is no longer in `_tracks`, so the call silently does nothing. The face quality data is lost.

**Effect:** Tracks that expire during recognition lose their best face data, so the finalization step has a lower-quality embedding to work with.

### 6c. `set_best_face` Double Lock Pattern

**File:** `pipeline/track_state.py` lines 89-107

```python
with self._lock:
    track = self._tracks.get(composite_id)
    if not track or face_score <= track.best_face_score + 0.1:
        return
# JPEG encoding happens OUTSIDE the lock
_, jpeg_buf = cv2.imencode(...)
with self._lock:
    track = self._tracks.get(composite_id)
    if track and face_score > track.best_face_score:
        # update
```

**What happens:** The track could be removed between the two lock acquisitions. The JPEG is encoded unnecessarily. This is a performance waste, not a correctness bug — the second lock check correctly handles the removed track case.

---

## 7. DATA FLOW BREAKS

### 7a. Auto-Registration Bloats Face Database

**File:** `main.py` lines 140-165

```python
if decision.status in ("unknown", "masked_unknown"):
    store_face(embedding=embedding, role="unknown", tags=["auto_registered"])
```

**What happens:** Every single unknown person (including masked people, people passing by quickly, etc.) gets a new face record in MongoDB. Over time, the `faces` collection fills with thousands of duplicate "unknown" entries.

**Effect:**
- Vector search gets slower as the collection grows (more candidates to scan)
- The dashboard's "Unknown Persons" section shows hundreds of entries
- Dedup logic (`DEDUP_SIMILARITY_THRESHOLD=0.40`) only catches similar-looking unknowns, not the same person from different angles

### 7b. Embedding History Capped at 10

**File:** `utils/db_utils.py` line 392

```python
"$slice": -10
```

**What happens:** Each face record keeps only the last 10 embeddings. If a person visits 20 times, only the last 10 embeddings are retained. The `mean_embedding` is computed from these 10, not all 20.

**Effect:** Older embedding patterns are lost. If lighting conditions were better in earlier visits, those higher-quality embeddings are discarded.

### 7c. Visit Memory Race Loses History

(See Section 6a above)

### 7d. `broadcast_event()` Is Dead Code

**File:** `dashboard/backend/routes/live.py` lines 52-75

The function `broadcast_event()` is defined but never called anywhere in the codebase. Events are only delivered via REST polling (every 10 seconds), not real-time push.

**Effect:** The dashboard doesn't show events in real-time. Users must wait up to 10 seconds for the event log to refresh.

---

## 8. ALERT SYSTEM ISSUES

### 8a. Global Dedup Suppresses Critical Alerts

**File:** `agents/alert_agent.py` lines 41-54

Unverified/unknown alerts use a single global key `"global:unverified"`. Only one unverified alert can be sent per cooldown period.

**What happens:** If a routine unknown person triggers an alert, then a genuinely suspicious person (blacklisted, after-hours) appears within the cooldown window, the critical alert is suppressed.

### 8b. Console Alerts Use `print()` Not Logger

**File:** `agents/alert_agent.py` line 132

```python
print(f"\n{'='*60}\n...")
```

**What happens:** Console alerts bypass structlog and the rotating file handler. They don't appear in `logs/surveillance.log`, making it impossible to correlate console output with file-based debugging.

### 8c. Alert Payload Name Can Be None

**File:** `agents/policy.py` lines 236-285

Rules 6-9 (hidden, masked, after-hours, default unknown) create `DecisionResult` without setting `person_id` or `name`. The alert payload includes `decision.name` which will be `None`.

**Effect:** Alert notifications show "None" as the person's name instead of a meaningful label.

---

## 9. DASHBOARD FRONTEND ISSUES

### 9a. No Shared API Client

Each component constructs its own HTTP calls:
- `App.jsx` uses `fetch()` (raw)
- All other components use `axios`

**Problems:**
- No centralized error handling
- No base URL configuration (hardcoded `localhost:8000` in `getImageUrl()`)
- `fetch()` doesn't throw on HTTP errors, so `App.jsx` silently gets error objects as stats

### 9b. `unknowns` State Duplicated

- `App.jsx:12` has `unknowns` state set via `onUnknownsLoaded` callback but **never read**
- `UnknownPersons.jsx:5` has its own local `unknowns` state

**Effect:** Dead state in App.jsx. The callback pattern is unnecessary.

### 9c. WebSocket Ping Timer Leak

**File:** `App.jsx:53-56`

On reconnect, `connectWebSocket()` creates a new WebSocket and sets a ping interval on `ws._pingInterval`. But it doesn't clear the old WebSocket's ping timer. The old timer fires and tries to send on a closed socket.

### 9d. Notification Popup Timer Not Cleared

**File:** `App.jsx:70`

The 8-second auto-dismiss timer for `activeAlert` is not cleared when a new alert replaces it. If two alerts fire within 8 seconds, the second alert gets dismissed by the first alert's timer.

### 9e. No Error Boundaries

**File:** `main.jsx`

If any component throws during render (e.g., `person.created_at` is undefined, causing `new Date(undefined).toISOString()` to fail), the entire React tree unmounts with a white screen. There are no error boundaries to catch and display errors gracefully.

### 9f. `fetchStats` Silent Failure

**File:** `App.jsx:29-36`

```javascript
const res = await fetch('/api/events/stats');
const data = await res.json();
setStats(data);
```

If the server returns a 500 error, `data` becomes `{detail: "Internal Server Error"}`. `setStats(data)` succeeds. Then `stats.total_events` is `undefined`, causing the stats section to render blank values.

### 9g. Chat Panel Message List Grows Unbounded

**File:** `ChatPanel.jsx:5-7`

Messages accumulate in state without a cap. Long sessions slow React rendering.

### 9h. `lucide-react` Dependency Unused

**File:** `package.json:14`

Lists `"lucide-react": "^0.294.0"` but no component imports icons from it. All icons are Unicode characters. This is wasted dependency weight.

---

## 10. DEAD CODE & MISLEADING LOGIC

### 10a. Dead Functions

| Function | File | Line | Status |
|----------|------|------|--------|
| `detect_and_embed()` | `pipeline/face.py` | 6 | Never called from main pipeline |
| `embed_only()` | `pipeline/face.py` | 10 | Never called from main pipeline |
| `embed_only()` | `utils/embedding_utils.py` | 106 | Never called from main pipeline |
| `decode_image()` | `utils/image_utils.py` | 169 | Never called from main pipeline |
| `broadcast_event()` | `dashboard/backend/routes/live.py` | 52 | Never called from anywhere |

### 10b. Dead Variables

| Variable | File | Line | Issue |
|----------|------|------|-------|
| `worker_pool` | `main.py` | 237 | Declared in `global` but never assigned |
| `_broadcast_frame_counter` | `main.py` | 34, 45 | Incremented but never read |
| `unknowns` | `App.jsx` | 12 | Set via callback but never consumed |

### 10c. Misleading Variable Names

| Variable | File | Line | Issue |
|----------|------|------|-------|
| `MASK_CONFIDENCE_PENALITY` | `config/settings.py` | 264 | Typo: should be "PENALTY" |
| `frame_faces` | `camera_agent.py` | 178 | Named as if it's always set, but only set in one branch |

### 10d. `asyncio.set_event_loop()` Misleading

**File:** `main.py:274`

Sets the event loop on the main thread, but the loop actually runs in a daemon thread (uvicorn). The `set_event_loop` call is misleading for debugging.

### 10e. `agent.md` Referenced But Missing

**File:** `AGENTS.md`

References `agent.md` as "the build specification" but the file does not exist in the repository.

### 10f. Stale `yolov8n.pt` at Repository Root

**File:** `yolov8n.pt` (6MB)

Old YOLOv8 nano model remains at root even though `.env` points to `models/yolov8s.pt`. Tracked by git, adds 6MB of dead weight.

---

## ROOT CAUSE SUMMARY

The system is breaking because of a **cascading failure chain**:

1. **Camera resolution (640x480) is too low** for InsightFace's detection model (expects 1280px) → faces are missed or produce poor embeddings

2. **MATCH_THRESHOLD=0.25 is too low** → wrong people get matched as "known", genuine matches get rejected

3. **Auto-registration creates thousands of duplicate unknown entries** → vector search slows down, dashboard floods with unknowns

4. **Dashboard fetches don't check HTTP status** → server errors silently become UI state, sections render blank

5. **WebSocket errors are silently swallowed** → live feed freezes with no error indication

6. **Configuration conflicts** between `.env`, `config.jsonc`, and `settings.py` → developers can't predict what values are actually active

7. **Thread races in visit memory** → visit history data is lost, confidence boosts are wrong

8. **No real-time event push** → dashboard appears to not update

**The threshold changes alone won't fix this.** The fundamental issues are: camera resolution mismatch, auto-registration bloat, broken error handling in the dashboard, and configuration confusion.

---

## RECOMMENDED FIX ORDER

1. **Fix camera resolution** — Set `FRAME_WIDTH=1280` and `FRAME_HEIGHT=720` in `.env` to match InsightFace's `det_size`
2. **Raise MATCH_THRESHOLD** — Set to `0.40` in `.env` (within ArcFace recommended range)
3. **Fix dashboard error handling** — Add `res.ok` checks in `App.jsx`, add WebSocket error logging
4. **Fix image URL resolution** — Use Vite proxy for `/captures`, fix `VerifyModal.jsx`
5. **Fix auto-registration** — Add dedup before `store_face`, or remove auto-registration for unknowns
6. **Fix configuration clarity** — Remove `.env` overrides that conflict with `config.jsonc`, document the three-layer system
7. **Fix thread safety** — Make `update_visit_memory` atomic using MongoDB `$inc` operations
8. **Wire up `broadcast_event()`** — Enable real-time event push to dashboard
9. **Add error boundaries** — Prevent full React tree unmount on component errors
10. **Clean up dead code** — Remove unused functions, variables, and dependencies
