# Agentic AI Surveillance System

## Abstract

An AI-powered real-time surveillance system that uses a multi-agent architecture for autonomous person detection, tracking, facial recognition, and intelligent decision-making. The system combines YOLOv8 for object detection, ByteTrack for multi-object tracking, InsightFace (ArcFace) for 512-dimensional face embeddings, and MongoDB Atlas Vector Search for similarity matching. Six specialized agents — Memory, Recognition, Policy, Alert, Report, and Camera — collaborate in a pipeline to produce context-aware security decisions. A local LLM (Gemma 3 4B / Qwen 3.5 4B via Ollama) provides natural language summaries and a conversational dashboard interface. The system runs on Python 3.11 with a FastAPI + React frontend, achieving real-time performance through thread-pooled progressive recognition and I/O decoupled from the camera loop.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Agent Pipeline](#agent-pipeline)
3. [Detection and Tracking](#detection-and-tracking)
4. [Face Detection and Embedding](#face-detection-and-embedding)
5. [Face Quality Scoring](#face-quality-scoring)
6. [Vector Search and Matching](#vector-search-and-matching)
7. [Recognition Agent — Confidence Computation](#recognition-agent--confidence-computation)
8. [Memory Agent — Visit History](#memory-agent--visit-history)
9. [Policy Agent — Decision Rules](#policy-agent--decision-rules)
10. [Alert Agent — Dispatch and Deduplication](#alert-agent--dispatch-and-deduplication)
11. [Report Agent](#report-agent)
12. [Mask Detection Heuristic](#mask-detection-heuristic)
13. [Visibility Classification](#visibility-classification)
14. [Track Lifecycle](#track-lifecycle)
15. [Threading Model](#threading-model)
16. [Data Models](#data-models)
17. [MongoDB Schema](#mongodb-schema)
18. [All Configuration Parameters](#all-configuration-parameters)
19. [Mathematical Formulas](#mathematical-formulas)
20. [Dashboard and API](#dashboard-and-api)
21. [Project Structure](#project-structure)
22. [Quick Start](#quick-start)
23. [Troubleshooting](#troubleshooting)

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        MAIN CAMERA LOOP                              │
│                  (agents/camera_agent.py)                            │
│                                                                      │
│  Camera ──► YOLOv8 Detection ──► ByteTrack Tracking ──► Frame Loop   │
│                                                                      │
│  Every 10 frames per track:                                         │
│     └──► Progressive Recognition (background thread pool, 2 workers)│
│                                                                      │
│  On track expiry:                                                    │
│     └──► Finalize Track ──► Track Queue ──► Worker Pool (2 threads) │
└──────────────────────────────────────────────────────────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────────┐                    ┌─────────────────────────┐
│  Progressive Recogn. │                    │  Finalized Track Worker  │
│  (2 background thds) │                    │  (2 background threads)  │
│                      │                    │                          │
│  MatchingAgent       │                    │  MatchingAgent            │
│  MemoryAgent         │                    │  MemoryAgent              │
│  RecognitionAgent    │                    │  RecognitionAgent         │
│  PolicyAgent         │                    │  PolicyAgent              │
│  AlertAgent (CRITICAL│                    │  store_face()             │
│   only)              │                    │  record_visit()           │
└─────────────────────┘                    │  AlertAgent (all levels) │
                                            │  log_event()              │
                                            └─────────────────────────┘
                                                     │
                                                     ▼
                                          ┌─────────────────────┐
                                          │  MongoDB Atlas        │
                                          │  ├── faces            │
                                          │  ├── events           │
                                          │  └── visit_memory     │
                                          └─────────────────────┘
                                                     │
                                                     ▼
                                          ┌─────────────────────┐
                                          │  FastAPI (port 8000) │
                                          │  REST + WebSocket     │
                                          └─────────────────────┘
                                                     │
                                                     ▼
                                          ┌─────────────────────┐
                                          │  React Dashboard      │
                                          │  (port 5173)          │
                                          │  ├── Live Feed (WS)   │
                                          │  ├── Event Log        │
                                          │  ├── Person Cards     │
                                          │  └── AI Chat Panel    │
                                          └─────────────────────┘
```

### Component Communication Flow

```
CameraAgent (main thread)
    │
    ├── every 10 frames ──► Progressive Recognition [background thread]
    │       │
    │       ├── MatchingAgent.vector_search(embedding)
    │       │       └── Atlas Vector Search or Python cosine fallback
    │       ├── MemoryAgent.run(person_id, camera_id)
    │       │       └── MongoDB visit_memory lookup
    │       ├── RecognitionAgent.run(similarity, quality, memory, ...)
    │       │       └── Multi-signal confidence → status
    │       ├── PolicyAgent.run(recognition, memory, visibility, ...)
    │       │       └── Priority-ordered rules → decision
    │       ├── AlertAgent.dispatch() [CRITICAL only]
    │       └── Store results on Track object
    │
    ├── on track expiry ──► _finalize_track() [background]
    │       └── attempt final embedding ──► on_track_finalized callback
    │               └── track_queue.put(track)
    │
    └── Worker Pool (2 threads, consuming track_queue)
            └── process_finalized_track()
                    ├── Reuse or re-run MatchingAgent
                    ├── Reuse or re-run RecognitionAgent
                    ├── Reuse or re-run MemoryAgent
                    ├── PolicyAgent.run() → final decision
                    ├── store_face() → MongoDB (if should_register)
                    ├── MemoryAgent.record_visit()
                    ├── AlertAgent.dispatch() → all channels
                    ├── log_event() → MongoDB events
                    └── broadcast_alert() → WebSocket
```

---

## Agent Pipeline

### Agent Definitions

| Agent | File | Purpose | Input | Output |
|-------|------|---------|-------|--------|
| **CameraAgent** | `agents/camera_agent.py` | Camera loop, detection, tracking, orchestration | Camera frames | Tracks, annotated frames |
| **MatchingAgent** | `agents/matching_agent.py` | Embedding-to-database similarity search | 512-dim embedding | MatchResult |
| **MemoryAgent** | `agents/memory.py` | Visit history tracking and pattern analysis | person_id, camera_id | visit_count, confidence_boost |
| **RecognitionAgent** | `agents/recognition.py` | Multi-signal identity classification | similarity, quality, memory, mask, duration | status, confidence |
| **PolicyAgent** | `agents/policy.py` | Business rule evaluation | recognition result + all context | DecisionResult |
| **AlertAgent** | `agents/alert_agent.py` | Alert dispatch and deduplication | DecisionResult | channels, priority |
| **ReportAgent** | `agents/report.py` | Human-readable report generation | Event data | title, summary, recommendation |

### Agent Execution Order

```
Person Detected (YOLO + ByteTrack)
    │
    ▼
┌──────────────────┐
│  MatchingAgent   │  Vector search: find similar faces in DB
└────────┬─────────┘
         │ MatchResult {person_id, similarity, tags, verified, ...}
         ▼
┌──────────────────┐
│  MemoryAgent     │  Lookup visit history, compute confidence boost
└────────┬─────────┘
         │ MemoryContext {visit_count, is_known, confidence_boost, ...}
         ▼
┌──────────────────┐
│ RecognitionAgent │  Multi-signal classification: known/unknown/uncertain
└────────┬─────────┘
         │ RecognitionResult {status, confidence, similarity, ...}
         ▼
┌──────────────────┐
│  PolicyAgent     │  Apply 9 priority-ordered rules
└────────┬─────────┘
         │ DecisionResult {status, alert_level, should_alert, ...}
         ▼
┌──────────────────┐
│   AlertAgent     │  Cooldown dedup → dispatch to channels
└──────────────────┘
```

---

## Detection and Tracking

### YOLOv8 Person Detection

- **Model:** YOLOv8 small (`yolov8s.pt`) — ~22MB, improved small-object detection
- **Class filter:** Class 0 (person only)
- **Confidence threshold:** `PERSON_CONF_THRESHOLD = 0.5`
- **Device:** Configurable (`cpu` or GPU index)

```python
results = model.track(
    frame,
    persist=True,              # ByteTrack state persists across frames
    tracker="bytetrack.yaml",  # Ultralytics built-in ByteTrack
    classes=[0],               # Person only
    conf=PERSON_CONF_THRESHOLD,
    device=YOLO_DEVICE,
    verbose=False
)
```

### ByteTrack Multi-Object Tracking

- **Configuration:** Ultralytics built-in `bytetrack.yaml`
- **State persistence:** `persist=True` maintains track IDs across frames
- **Track ID format:** `{camera_id}_{session_epoch}_{byte_track_id}`
  - `session_epoch` = process start time (ensures uniqueness across camera restarts)

### Track Expiry Criteria

A track is considered expired (person left frame) when:

| Condition | Default | Description |
|-----------|---------|-------------|
| `(now - last_seen) > TRACK_TIMEOUT_SECS` | 8.0 seconds | No detection for 8 seconds |
| `(now - first_seen) > MAX_TRACK_SECS` | 300 seconds (5 min) | Maximum track lifetime |

---

## Face Detection and Embedding

### InsightFace Singleton

- **Model:** `buffalo_m` (SCRFD detection + ArcFace embedding)
- **Detection input size:** 1280×1280
- **Execution provider:** `CPUExecutionProvider` (configurable)
- **Embedding dimension:** 512 (L2-normalized ArcFace)
- **Thread-safe:** Double-checked locking singleton, loaded once per process

### Detection Methods

| Method | Detection Score Threshold | Use Case |
|--------|--------------------------|----------|
| `detect_and_embed(image)` | `DET_SCORE_MIN = 0.40` | Standard face detection |
| `detect_and_embed_relaxed(image)` | `DET_SCORE_RELAXED = 0.20` | Fallback for distant faces |
| `embed_only(image, det_score)` | `EMBEDDING_DET_SCORE_MIN = 0.40` | Embedding when face already detected |
| `detect_faces_raw(image, min_score)` | Configurable | Returns all faces above threshold |

### Progressive Recognition Flow

Every 10 frames per track, recognition runs in a background thread:

```
1. Crop person from frame using person bounding box
2. Run InsightFace on CROPPED person:
   a. Try DET_SCORE_MIN (0.40)
   b. Try DET_SCORE_RELAXED (0.20)
3. If no face in crop, try FULL FRAME:
   a. Try DET_SCORE_MIN (0.40)
   b. Try DET_SCORE_RELAXED (0.20)
4. If no face found → return
5. Compute face_ratio = face_area / person_area
6. If det_score < EMBEDDING_DET_SCORE_MIN (0.40) → return
7. Extract face crop from full frame coordinates
8. Compute quality score (blur + brightness + area)
9. Update track's best face if quality improved
10. Store embedding and mask status
11. Run Matching Agent → Recognition Agent → Policy Agent
12. Only dispatch CRITICAL alerts during progressive (others deferred to finalization)
```

### Similarity Computation

```python
# L2-normalize both embeddings
emb1_norm = emb1 / (norm(emb1) + 1e-6)
emb2_norm = emb2 / (norm(emb2) + 1e-6)

# Cosine similarity = dot product of normalized vectors
similarity = dot(emb1_norm, emb2_norm)
```

### Atlas Vector Search Score Conversion

MongoDB Atlas Vector Search returns a score in [0, 1]:

```
atlas_score = (1 + cosine_similarity) / 2
```

Reverse conversion to raw cosine:

```
raw_cosine = (atlas_score × 2) − 1
```

---

## Face Quality Scoring

### Quality Metrics

| Metric | Raw Value | Normalization | Weight |
|--------|-----------|---------------|--------|
| **Blur** | Laplacian variance of face crop | `min(variance / 1000, 1.0)` | 60% |
| **Brightness** | Mean V-channel in HSV | `brightness / 255.0` | 25% |
| **Area** | Face pixel area (h × w) | `min(area / 10000, 1.0)` | 15% |

### Quality Score Formula

```
overall_score = blur_norm × 0.60 + brightness_norm × 0.25 + area_norm × 0.15
```

### Validity Criteria (all must pass)

| Criterion | Threshold |
|-----------|-----------|
| Blur (Laplacian variance) | ≥ 30 |
| Brightness (V-channel mean) | 30–240 |
| Face area (pixels) | ≥ 900 |

### Best Face Selection

- Only updates if new quality score > current best + 0.1 (hysteresis to prevent flickering)
- JPEG encoding at quality=60 performed outside the lock (expensive operation)

---

## Vector Search and Matching

### Atlas Vector Search Pipeline

```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "latest_embedding",
            "queryVector": embedding,        # 512-dim float list
            "numCandidates": limit * 10,      # 50 candidates for limit=5
            "limit": limit                    # 5
        }
    },
    {
        "$addFields": {
            "score": {"$meta": "vectorSearchScore"}
        }
    }
]
```

### Score Conversion

```
raw_cosine = (atlas_vectorSearchScore × 2) − 1
```

### Match Filtering

Only results where `raw_cosine >= MATCH_THRESHOLD` (default 0.30, max 0.45) are returned.

### Python Fallback

When Atlas Vector Search is unavailable:

1. Fetch up to **500** documents from `faces` collection
2. L2-normalize query and stored embeddings
3. Compute dot product for each pair
4. Sort descending, return top `limit` above threshold
5. Logs warning about truncation if `total_count > 500`

### Deduplication During Storage

- Vector search with `limit=3`
- If any match ≥ `DEDUP_SIMILARITY_THRESHOLD` (0.40): update existing document instead of inserting new
- Prevents duplicate person records for the same individual

### Quality-Gated Embedding Updates

The system protects `latest_embedding` (the vector used for Atlas Vector Search) from being overwritten by low-quality detections:

```python
# In update_face():
if quality_score is not None:
    existing = collection.find_one({"person_id": person_id}, {"latest_embedding_quality": 1})
    current_quality = (existing or {}).get("latest_embedding_quality", 0.0)
    if quality_score > current_quality:
        update_ops["$set"]["latest_embedding"] = embedding
        update_ops["$set"]["latest_embedding_quality"] = quality_score
else:
    update_ops["$set"]["latest_embedding"] = embedding  # backward compat
```

**Why this matters:**
- Without quality gating, every new detection overwrites `latest_embedding`, even if the face is blurry/backlit
- Vector search only queries `latest_embedding` (not the `embeddings` array)
- A single bad overwrite can tank similarity scores for all subsequent matches
- Quality gating ensures the best-quality embedding persists

**Impact:** Similarity scores improve from ~69% to 85%+ as high-quality embeddings are preserved.

---

## Recognition Agent — Confidence Computation

### Classification Cases

| Case | Condition | Status |
|------|-----------|--------|
| Very high similarity | `similarity >= 0.90` | `"known"` |
| Above threshold | `similarity >= MATCH_THRESHOLD (0.30)` | `"known"` if confidence ≥ 70, else `"uncertain"` |
| Below threshold but good quality | `quality >= 0.8 AND similarity >= 0.24` | `"uncertain"` |
| Low similarity | Default | `"unknown"` |

### Confidence Formula

```
sim_score = min(60, (similarity − MATCH_THRESHOLD) / (1.0 − MATCH_THRESHOLD) × 60)
quality_score = face_quality × 25
duration_score = min(15, track_duration / 10)
memory_boost = clamp(−10, +20, boost)   // from Memory Agent

confidence = sim_score + quality_score + duration_score + memory_boost

if is_masked:
    confidence ×= 0.85   // 15% penalty for masked faces

confidence = clamp(0, 100, confidence)
```

### Confidence Component Ranges

| Component | Range | Description |
|-----------|-------|-------------|
| `sim_score` | 0–60 | Linear interpolation from threshold to 1.0 |
| `quality_score` | 0–25 | Face quality composite score |
| `duration_score` | 0–15 | 1 point per 10 seconds tracked |
| `memory_boost` | −10 to +20 | Visit history adjustment |
| Mask penalty | ×0.85 | Applied when face is masked |

---

## Memory Agent — Visit History

### Visit Memory Document

```python
{
    "person_id": str,              # Unique identifier
    "visit_count": int,            # Total visits observed
    "first_seen": datetime,        # First detection time
    "last_seen": datetime,         # Most recent detection
    "last_camera": str,            # Last camera ID
    "last_status": str,            # Last recognition status
    "typical_hours": list[int],    # Last 20 visit hours (0-23)
    "typical_cameras": list[str],  # Last 5 camera IDs
    "avg_similarity": float,       # Rolling average of last 10 scores
    "similarity_history": list[float],  # Last 10 similarity scores
    "status_history": list[dict]   # Last 10 {status, timestamp}
}
```

### Confidence Boost Formula

Returns range: **[−10, +20]**

```
boost = 0

# Returning visitor bonus
if visit_count > 0:
    boost += min(10, visit_count × 2)    # +2 per visit, max +10

# Recency bonus
if days_since_last <= 7:   boost += 5
elif days_since_last <= 30: boost += 2

# Consistency bonus
if avg_similarity > 0.8: boost += 3
elif avg_similarity > 0.6: boost += 1

# Pattern bonus
if is_typical_time:   boost += 2   # current hour within 2h of top 3 hours
if is_typical_camera: boost += 1   # same camera as usual

# Penalty for similarity drop
if avg_similarity > 0 AND current_similarity < avg_similarity × 0.7:
    boost -= 5

return clamp(−10, 20, boost)
```

### Typical Time Check

```python
# Get top 3 most common hours from history
common_hours = sorted(hour_counts, key=count, reverse=True)[:3]
# Is current hour within 2 hours of any common hour?
is_typical_time = any(abs(current_hour − h) <= 2 for h in common_hours)
```

---

## Policy Agent — Decision Rules

### Priority-Ordered Rules (first match wins)

| Priority | Rule | Condition | Status | Alert Level | Alert? | Register? |
|----------|------|-----------|--------|-------------|--------|-----------|
| 1 | **Blacklist** | `"blacklist" in tags` | `blacklist` | `critical` | Yes | No |
| 2 | **Authorized** | `"authorized" in tags` | `authorized` | `none` | No | No |
| 3 | **Verified** | `verified=True` in DB | `verified` | `none` | No | No |
| 4 | **Known Visitor (memory)** | `matched=True AND is_known_from_memory=True` | `known_visitor` | `low` | No | No |
| 5a | **High-confidence match** | `similarity >= 0.85 OR confidence >= 80` | `known_visitor` | `low` | No | No |
| 5b | **Matched** | `similarity >= MATCH_THRESHOLD` | `known_visitor` | `low` | No | No |
| 5c | **Uncertain match** | `matched=True` (low similarity) | `uncertain` | `low` | No | Yes |
| 6 | **Intentionally Hidden** | `visibility == "hidden"` | `intentionally_hidden` | `high` | Yes | No |
| 7 | **Masked/Partial** | `is_masked OR visibility == "partial"` | `masked_unknown` | `medium`/`high` | Yes | Yes |
| 8 | **After-Hours Unknown** | Not office hours (configurable) or not weekday | `unknown` | `high` | Yes | Yes |
| 9 | **Default Unknown** | During office hours | `unknown` | `medium` | Yes | Yes |

### Time Context

- `is_office_hours = OFFICE_HOURS_START <= current_hour <= OFFICE_HOURS_END` (configurable, default 9–17)
- `is_weekday = now.weekday() in OFFICE_DAYS` (configurable, default Monday–Friday)

### Masked/Partial Escalation

- If `track_lifetime > LOITER_SECS (30s)`: `alert_level = "high"`, reason = "Masked unknown person loitering"
- Else: `alert_level = "medium"`, reason = "Unknown person with partial visibility or mask"

---

## Alert Agent — Dispatch and Deduplication

### Deduplication Logic

```python
_UNVERIFIED_STATUSES = {"unknown", "masked_unknown", "uncertain", "intentionally_hidden"}

def should_send_alert(track_id, alert_level, status):
    if status in _UNVERIFIED_STATUSES:
        key = "global:unverified"    # Global dedup for ALL unverified
    else:
        key = f"{track_id}:{alert_level}"  # Per-track dedup

    if now - last_alert_time[key] < ALERT_COOLDOWN_SECS (60s):
        return False
    last_alert_time[key] = now
    return True
```

### Alert Channels

| Channel | Implementation | Threading |
|---------|---------------|-----------|
| **Console** | Formatted print with level prefix | Synchronous |
| **Email** | SMTP with TLS, MIME multipart | Async (thread pool) |
| **SMS** | Twilio REST client | Async (thread pool) |
| **Webhook** | HTTP POST JSON, 10s timeout | Async (thread pool) |

### Stale Pruning

- Prunes alert timestamps older than `2 × ALERT_COOLDOWN_SECS` (120s)
- Runs on every `should_send_alert` call

### Alert Level Prefixes

| Level | Console Prefix |
|-------|---------------|
| `critical` | `!!!` |
| `high` | `!!` |
| `medium` | `!` |
| `low` | `*` |

---

## Report Agent

### Report Types

| Type | Trigger | Content |
|------|---------|---------|
| **Incident** | Single alert event | Title, summary, recommendation, severity |
| **Summary** | Daily/weekly request | Aggregate stats, peak hours, LLM executive summary |
| **Person** | Per-person query | Visit history, pattern analysis, avg similarity |
| **Stats** | Dashboard load | Real-time counts from MongoDB |

### Recommendation Templates

| Status | Recommendation |
|--------|---------------|
| `blacklist` | "IMMEDIATE ACTION REQUIRED. Contact security team." |
| `masked_unknown` | "Manual verification recommended. Monitor closely." |
| `intentionally_hidden` | "Investigate. Person may be attempting to avoid detection." |
| `unknown` | "Manual verification recommended if in restricted area." |
| `uncertain` | "Low confidence match. Verify identity if important." |
| + `visit_count > 3` | Appends "Returning visitor — check if pattern is normal." |

---

## Mask Detection Heuristic

### Algorithm

Uses 5 facial landmarks from InsightFace/SCRFD:

```
landmarks[0], landmarks[1] = left eye, right eye
landmarks[2] = nose tip
landmarks[3], landmarks[4] = left mouth corner, right mouth corner
```

### Steps

```
1. nose_tip = landmarks[2]
2. mouth_center = (landmarks[3] + landmarks[4]) / 2
3. eye_midpoint = (landmarks[0] + landmarks[1]) / 2
4. lower_face_height = abs(mouth_center.y − nose_tip.y)
5. upper_face_height = abs(eye_midpoint.y − nose_tip.y)
6. ratio = lower_face_height / upper_face_height
7. If ratio < 0.3  →  masked = True
```

**Interpretation:** A masked face compresses the lower-face region relative to the upper face, producing a ratio below 0.3.

**Limitations:** Geometric heuristic, not a classifier. Works best for front-facing faces with stable landmarks.

---

## Visibility Classification

### Face-to-Person Ratio

```
face_ratio = face_area / person_area
           = (fx2 − fx1) × (fy2 − fy1) / (px2 − px1) × (py2 − py1)
```

### Classification Rules

| Condition | Visibility |
|-----------|------------|
| `max_face_ratio >= 0.025` | `"visible"` |
| `max_face_ratio >= 0.010` | `"partial"` |
| `is_masked OR face_detected_once` (but ratio < 0.010) | `"partial"` |
| `NOT face_detected_once AND total_frames >= 30` | `"hidden"` |
| Default | `"unknown"` |

---

## Track Lifecycle

### Composite Track ID

```
{camera_id}_{session_epoch}_{byte_track_id}
```

Example: `cam_01_1719340800_42`

- `camera_id`: Logical camera identifier (default `cam_01`)
- `session_epoch`: Process start time (epoch seconds) — ensures uniqueness across restarts
- `byte_track_id`: Integer ID from ByteTrack tracker

### Track State Machine

```
NEW ──► ACTIVE ──► EXPIRED ──► FINALIZED ──► STORED
  │        │          │            │             │
  │        │          │            │             └── MongoDB face document
  │        │          │            └── Worker processes track
  │        │          └── track_timeout OR max_lifetime
  │        └── Updated every frame with detection
  └── Created on first detection
```

### Progressive Recognition Check

Before running progressive recognition on a track:
- Skip if already verified or known (high-confidence match)
- Skip if already matched with `similarity > 0.85` (high-confidence match)
- Skip if already matched with `similarity > 0.80` (during progressive)
- Only runs every `RECOGNITION_INTERVAL_FRAMES` (10) frames

---

## Threading Model

### Thread Pool

| Thread | Purpose | Pool Size |
|--------|---------|-----------|
| Main thread | Camera loop (capture + tracking + annotation) | 1 |
| `recognition-N` | Progressive recognition workers | 2 |
| `jpeg-N` | JPEG encoding + WebSocket broadcast | 2 |
| `alert-N` | Async alert dispatch (email/SMS/webhook + LLM) | 2 |
| `worker-N` | Track finalization + DB operations | 2 |
| `uvicorn` | FastAPI HTTP server | 1 |

### Thread Safety Mechanisms

| Mechanism | Location | Protects |
|-----------|----------|----------|
| `threading.Lock` | `TrackState` | Track dictionary mutations |
| `threading.Lock` | `_track_sets_lock` | `_recognizing_tracks`, `_finalized_track_ids` |
| `threading.Lock` | `InsightFaceSingleton` | Model loading (double-checked locking) |
| `queue.Queue` | `track_queue` | Track finalization handoff |
| Thread pool | `ThreadPoolExecutor` | Bounded concurrent work |

### I/O Decoupling

- Camera loop **never blocks** on MongoDB, Cloudinary, alerts, or LLM
- All I/O operations run via `queue.Queue` + worker threads
- Pre-encoded JPEG bytes stored on Track to avoid re-encoding

---

## Data Models

### Track

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `track_id` | `str` | — | Composite ID: `{camera_id}_{session_epoch}_{byte_track_id}` |
| `first_seen` | `float` | — | Epoch timestamp of first detection |
| `last_seen` | `float` | — | Epoch timestamp of most recent detection |
| `person_box` | `tuple` | — | Bounding box `(x1, y1, x2, y2)` |
| `best_face_crop` | `ndarray` | `None` | Best quality face image |
| `best_face_score` | `float` | `0.0` | Quality score of best face |
| `best_full_frame` | `ndarray` | `None` | Full frame with best face |
| `best_frame_jpeg` | `bytes` | `None` | Pre-encoded JPEG (~50KB) |
| `is_masked` | `bool` | `False` | Mask detected on this track |
| `embedding` | `list` | `None` | ArcFace 512-dim embedding |
| `decision` | `str` | `None` | Final decision string |
| `person_name` | `str` | `None` | Name from match result |
| `alerted` | `bool` | `False` | Whether alert was dispatched |
| `image_url` | `str` | `None` | Cloudinary URL after upload |
| `last_recognition_frame` | `int` | `0` | Frame number of last recognition run |
| `total_frames_seen` | `int` | `0` | Count of frames this track appeared in |
| `frames_with_detectable_face` | `int` | `0` | Frames where a face was detected |
| `face_detected_once` | `bool` | `False` | Whether a face was ever seen |
| `max_face_ratio` | `float` | `0.0` | Largest face-to-person area ratio |
| `best_face_ratio` | `float` | `0.0` | Ratio at best face quality |
| `visibility` | `str` | `"unknown"` | Classification: visible/partial/hidden/unknown |
| `pending_embedding` | `list` | `None` | In-progress embedding |
| `pending_match` | `dict` | `None` | In-progress match data |
| `pending_recognition` | `dict` | `None` | Recognition Agent output |
| `pending_match_result` | `MatchResult` | `None` | Full match result |
| `pending_memory_context` | `dict` | `None` | Cached memory context |
| `decision` | `DecisionResult` | `None` | Final decision result (set by worker) |

### MatchResult

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `person_id` | `str` | `None` | Matched person ID |
| `name` | `str` | `None` | Person name |
| `role` | `str` | `None` | Role (visitor/unknown/authorized) |
| `tags` | `list` | `[]` | Tags (blacklist, authorized, etc.) |
| `similarity_score` | `float` | `0.0` | Raw cosine similarity |
| `image_url` | `str` | `None` | Reference image URL |
| `matched` | `bool` | `False` | Whether a match was found |
| `verified` | `bool` | `False` | Whether person is operator-verified |
| `alert_level` | `str` | `"low"` | Stored alert level |

### RecognitionResult

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `status` | `str` | `"unknown"` | known/unknown/uncertain | Classification |
| `confidence` | `float` | `0.0` | 0–100 | Confidence percentage |
| `similarity` | `float` | `0.0` | −1 to 1 | Raw cosine similarity |
| `face_quality` | `float` | `0.0` | 0–1 | Quality composite |
| `is_masked` | `bool` | `False` | — | Mask detected |
| `track_duration` | `float` | `0.0` | seconds | Track lifetime |
| `reason` | `str` | `""` | — | Human-readable explanation |

### DecisionResult

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | `str` | `"unknown"` | Final status |
| `alert_level` | `str` | `"none"` | Alert severity |
| `person_id` | `str` | `None` | Matched person ID |
| `name` | `str` | `None` | Person name |
| `reason` | `str` | `""` | Decision rationale |
| `should_alert` | `bool` | `False` | Whether to dispatch alert |
| `should_register` | `bool` | `False` | Whether to store in DB |
| `nl_summary` | `str` | `""` | Natural language summary |

### QualityResult

| Field | Type | Description |
|-------|------|-------------|
| `blur_score` | `float` | Laplacian variance |
| `brightness` | `float` | Mean V-channel in HSV |
| `face_area` | `int` | Pixels (h × w) |
| `is_valid` | `bool` | All criteria pass |
| `overall_score` | `float` | Weighted composite 0–1 |

---

## MongoDB Schema

### Database: `surveillance`

#### `faces` Collection

```python
{
    "person_id": str,                    # Unique identifier
    "name": str,                         # Display name
    "role": str,                         # "visitor" | "unknown" | "authorized"
    "embeddings": list,                  # Array of embeddings (last 10 via $slice)
    "latest_embedding": list,            # 512-dim vector for search
    "latest_embedding_quality": float,   # Quality score (0-1) of latest embedding
    "embedding_model": "arcface",        # Model used
    "images": [                          # Reference images
        {
            "id": uuid,
            "url": str,
            "captured_at": datetime
        }
    ],
    "source": {
        "camera_id": str,
        "captured_at": datetime
    },
    "quality_scores": dict,              # Best quality metrics
    "tags": list[str],                   # "auto_registered", "blacklist", "authorized", "verified"
    "verified": bool,                    # Operator-verified
    "verified_at": datetime,
    "verified_by": str,
    "alert_level": str,                  # "low" | "medium" | "high" | "critical"
    "created_at": datetime,
    "updated_at": datetime
}
```

**Indexes:**
- `vector_index`: Atlas Vector Search on `latest_embedding` (512 dims, cosine similarity)

#### `events` Collection

```python
{
    "track_id": str,
    "camera_id": str,
    "timestamp": datetime,
    "status": str,
    "alert_level": str,
    "person_id": str,
    "name": str,
    "is_masked": bool,
    "similarity_score": float,
    "image_url": str,
    "reason": str,
    "alerted": bool
}
```

**Indexes:** `track_id`, `timestamp`, `status`

#### `visit_memory` Collection

```python
{
    "person_id": str,                    # Unique index
    "visit_count": int,
    "first_seen": datetime,
    "last_seen": datetime,
    "last_camera": str,
    "last_status": str,
    "typical_hours": list[int],          # Last 20 visit hours
    "typical_cameras": list[str],        # Last 5 camera IDs
    "avg_similarity": float,             # Rolling avg of last 10
    "similarity_history": list[float],   # Last 10 scores
    "status_history": list[dict]         # Last 10 {status, timestamp}
}
```

**Indexes:** `person_id` (unique), `last_seen`

---

## All Configuration Parameters

### Required

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `MONGODB_URI` | — | str | **Required.** MongoDB Atlas connection string |

### Camera

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `CAMERA_INDEX` | `0` | int | cv2 VideoCapture device index |
| `FRAME_WIDTH` | `640` | int | Capture width (pixels) |
| `FRAME_HEIGHT` | `480` | int | Capture height (pixels) |
| `CAMERA_ID` | `"cam_01"` | str | Logical camera identifier |

### YOLO Detection

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `YOLO_MODEL` | `"models/yolov8s.pt"` | str | YOLOv8 model path (n/s/m variants) |
| `YOLO_DEVICE` | `"cpu"` | str | Compute device (`cpu` or GPU index) |
| `PERSON_CONF_THRESHOLD` | `0.5` | float | YOLO confidence for person detection |

### Tracking

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `TRACK_TIMEOUT_SECS` | `8.0` | float | Seconds of no detection before track expires |
| `MAX_TRACK_SECS` | `300` | float | Maximum track lifetime (seconds) |

### Face Detection

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `INSIGHTFACE_MODEL` | `"buffalo_m"` | str | InsightFace model (buffalo_m/l/s) |
| `INSIGHTFACE_DET_SIZE` | `1280` | int | Detection input size |
| `INSIGHTFACE_PROVIDER` | `"CPUExecutionProvider"` | str | ONNX execution provider |
| `DET_SCORE_MIN` | `0.40` | float | Standard face detection score threshold |
| `DET_SCORE_RELAXED` | `0.20` | float | Relaxed face detection threshold (fallback) |
| `EMBEDDING_DET_SCORE_MIN` | `0.40` | float | Minimum score for embedding generation |

### Recognition

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `RECOGNITION_INTERVAL_FRAMES` | `10` | int | Run face recognition every N frames per track |
| `LOITER_SECS` | `30` | float | Seconds before masked unknown escalates |
| `MIN_TRACK_FRAMES` | `30` | int | Min frames before hidden classification triggers |
| `OFFICE_HOURS_START` | `9` | int | Office hours start (hour 0-23) |
| `OFFICE_HOURS_END` | `17` | int | Office hours end (hour 0-23) |
| `OFFICE_DAYS` | `"0,1,2,3,4"` | str | Comma-separated weekday numbers (0=Mon, 6=Sun) |

### Face Quality

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `QUALITY_BLUR_MAX` | `1000` | float | Max Laplacian variance for blur normalization |
| `QUALITY_AREA_MAX` | `10000` | float | Max face area for area normalization |
| `VISIBLE_FACE_RATIO` | `0.025` | float | Face-to-person area ratio for "visible" |
| `PARTIAL_FACE_RATIO` | `0.010` | float | Face-to-person area ratio for "partial" |

### Matching

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `MATCH_THRESHOLD` | `0.30` | float | Cosine similarity threshold for match (max 0.45) |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.40` | float | Threshold for deduplication when storing faces |

### Alerting

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `ALERT_CHANNELS` | `["console"]` | list | Channels: console, email, sms, webhook |
| `ALERT_COOLDOWN_SECS` | `60` | float | Min seconds between same alerts |
| `ALERT_WEBHOOK_URL` | `""` | str | Webhook POST URL |
| `SMTP_HOST` | `""` | str | SMTP server host |
| `SMTP_PORT` | `587` | int | SMTP server port |
| `SMTP_USER` | `""` | str | SMTP username |
| `SMTP_PASS` | `""` | str | SMTP password |
| `ALERT_EMAIL_TO` | `""` | str | Recipient email address |
| `TWILIO_ACCOUNT_SID` | `""` | str | Twilio account SID |
| `TWILIO_AUTH_TOKEN` | `""` | str | Twilio auth token |
| `TWILIO_FROM` | `""` | str | Twilio sender number |
| `ALERT_SMS_TO` | `""` | str | SMS recipient number |

### Local LLM (Ollama)

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `OLLAMA_URL` | `"http://localhost:11434"` | str | Ollama server URL |
| `OLLAMA_MODEL` | `"gemma3:4b"` | str | Model name (swap to `qwen3.5:4b` for better reasoning) |
| `OLLAMA_TIMEOUT` | `30` | int | Request timeout (seconds) |

### MongoDB

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `MONGODB_DATABASE` | `"surveillance"` | str | Database name |
| `MONGODB_COLLECTION` | `"faces"` | str | Face embeddings collection |
| `MONGODB_EVENTS_COLLECTION` | `"events"` | str | Event logs collection |

### Cloudinary (Optional)

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `CLOUDINARY_CLOUD_NAME` | `""` | str | Cloudinary cloud name |
| `CLOUDINARY_API_KEY` | `""` | str | Cloudinary API key |
| `CLOUDINARY_API_SECRET` | `""` | str | Cloudinary API secret |

### Logging

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `LOG_FORMAT` | `"console"` | str | Console renderer or `"json"` |

### Validation Rules

1. `MONGODB_URI` is **required** (exits if missing)
2. `MATCH_THRESHOLD` must be ≤ 0.45
3. `TRACK_TIMEOUT_SECS` must be > 0
4. `MAX_TRACK_SECS` must be > 0
5. `DET_SCORE_MIN` must be in (0, 1)
6. `DET_SCORE_RELAXED` must be in (0, 1) and ≤ `DET_SCORE_MIN`
7. `EMBEDDING_DET_SCORE_MIN` must be in (0, 1)
8. `ALERT_CHANNELS` entries must be in {console, email, sms, webhook}

---

## Mathematical Formulas

### Cosine Similarity

```
similarity = dot(L2(emb1), L2(emb2))

where L2(x) = x / (||x|| + 1e-6)
```

### Atlas Score Conversion

```
raw_cosine = (atlas_vectorSearchScore × 2) − 1
```

### Face Quality Composite

```
Q = blur_norm × 0.60 + brightness_norm × 0.25 + area_norm × 0.15

where:
  blur_norm     = min(Laplacian_var / 1000, 1.0)
  brightness_norm = V_channel_mean / 255.0
  area_norm     = min(face_area / 10000, 1.0)
```

### Recognition Confidence

```
C = sim_score + quality_score + duration_score + memory_boost

where:
  sim_score      = min(60, (sim − threshold) / (1 − threshold) × 60)
  quality_score  = face_quality × 25
  duration_score = min(15, track_duration / 10)
  memory_boost   = clamp(−10, +20, boost)

if is_masked: C ×= 0.85
C = clamp(0, 100, C)
```

### Memory Confidence Boost

```
boost = 0
boost += min(10, visit_count × 2)      # returning visitor
boost += recency_bonus                   # +5 if ≤7d, +2 if ≤30d
boost += consistency_bonus               # +3 if avg>0.8, +1 if avg>0.6
boost += pattern_bonus                   # +2 typical time, +1 typical camera
boost -= similarity_drop_penalty         # −5 if current < 70% of avg
boost = clamp(−10, +20, boost)
```

### Mask Detection

```
ratio = |mouth_center.y − nose_tip.y| / |eye_midpoint.y − nose_tip.y|
masked = ratio < 0.3
```

### Face-to-Person Ratio

```
face_ratio = face_area / person_area
           = (fx2−fx1)(fy2−fy1) / (px2−px1)(py2−py1)
```

### Alert Deduplication

```
key = "global:unverified"              if status ∈ {unknown, masked_unknown, uncertain, intentionally_hidden}
key = f"{track_id}:{alert_level}"      otherwise

send = (now − last_alert_time[key]) >= ALERT_COOLDOWN_SECS
```

---

## Dashboard and API

### Frontend (React + Vite)

- **Live Feed:** Real-time camera via WebSocket with color-coded bounding boxes
- **Event Log:** Paginated events with color-coded status badges
- **Unknown Persons:** Grid of detected person cards
- **Verify Modal:** Assign names to unknown persons
- **AI Chat:** Natural language queries about surveillance data
- **NL Summaries:** Human-readable alert summaries

### Overlay Colors

| State | Border Color | Label |
|-------|-------------|-------|
| No decision yet / unknown | Red | `UNVERIFIED` |
| Masked person (any state) | Red | Appends `MASK` |
| Intentionally hidden | Red | `HIDDEN` |
| Blacklisted | Red | `BLACKLIST` |
| Known visitor | Yellow | `KNOWN VISITOR` |
| Verified / authorized | Green | `Verified` or `AUTHORIZED` |

### REST API Endpoints

#### Faces

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/faces` | List faces (`?status=unknown\|verified\|all`) |
| GET | `/api/faces/{person_id}` | Get single face |
| POST | `/api/faces/{person_id}/verify` | Verify person |
| PUT | `/api/faces/{person_id}` | Update face |
| DELETE | `/api/faces/{person_id}` | Delete face |

#### Events

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/events` | List events |
| GET | `/api/events/stats` | Dashboard statistics |
| GET | `/api/events/unknown` | Unknown events |
| GET | `/api/events/alerts` | High/critical alerts |

#### Reports

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/reports/stats` | Dashboard statistics |
| GET | `/api/reports/summary?period=daily` | Daily/weekly summary |
| GET | `/api/reports/person/{person_id}` | Person visit history |
| GET | `/api/reports/incidents` | Recent incidents |

#### Chat

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Ask questions about surveillance data |
| GET | `/api/chat/health` | Check LLM status |

#### WebSocket

| Protocol | Path | Description |
|----------|------|-------------|
| WS | `/ws/live` | Live camera feed + alerts |

### WebSocket Message Types

```json
{"type": "frame",  "data": "<base64 JPEG>"}
{"type": "alert",  "data": {"status": "...", "alert_level": "...", "nl_summary": "..."}}
{"type": "event",  "data": {"track_id": "...", "status": "...", "timestamp": "..."}}
```

---

## Project Structure

```
surveillance-system/
├── main.py                          # Entry point, wires everything
├── .env                             # Your config (not in git)
├── .env.example                     # Config template
├── requirements.txt                 # Python dependencies
│
├── agents/                          # Intelligent Agents
│   ├── base.py                      # BaseAgent ABC
│   ├── camera_agent.py              # Camera capture + ByteTrack + progressive recognition
│   ├── matching_agent.py            # Embedding + MongoDB vector search
│   ├── decision_agent.py            # Delegates to PolicyAgent
│   ├── recognition.py               # Multi-signal identity classification
│   ├── memory.py                    # Visit history tracking
│   ├── policy.py                    # Business rule evaluation
│   ├── alert.py                     # Alert dispatch (console/email/sms/webhook)
│   ├── report.py                    # Report generation
│   └── alert_agent.py               # Legacy alert dispatch
│
├── pipeline/                        # CV Pipeline
│   ├── models.py                    # Dataclasses (Track, MatchResult, DecisionResult, etc.)
│   ├── tracker.py                   # YOLOv8 + ByteTrack (single model instance, NMS iou=0.5)
│   ├── track_state.py               # Track lifecycle with threading.Lock
│   ├── face.py                      # SCRFD detection + ArcFace embedding + mask heuristic
│   └── quality_agent.py             # Image quality scoring
│
├── utils/                           # Utilities
│   ├── db_utils.py                  # MongoDB CRUD + vector search + quality-gated embedding updates
│   ├── embedding_utils.py           # InsightFace singleton (buffalo_m, CLAHE preprocessing)
│   ├── llm_client.py                # Ollama HTTP client (generate, chat, NL summaries)
│   └── image_utils.py               # Image processing, crop, save, upload
│
├── config/
│   └── settings.py                  # Configuration loader + validate_config()
│
├── models/                          # Model weights (gitignored)
│   └── yolov8s.pt                   # YOLOv8 small model
│
├── logs/                            # System logs (gitignored)
│   └── surveillance.log             # Rotating: 5MB × 5 backups
│
└── dashboard/
    ├── backend/
    │   ├── main.py                  # FastAPI app (REST + WebSocket)
    │   ├── models.py                # Pydantic response models
    │   └── routes/
    │       ├── faces.py             # Face CRUD endpoints
    │       ├── events.py            # Event endpoints
    │       ├── reports.py           # Report endpoints
    │       ├── live.py              # WebSocket live feed
    │       └── chat.py              # AI chat endpoint
    └── frontend/
        ├── package.json
        ├── vite.config.js
        └── src/
            ├── App.jsx              # Main React app
            ├── index.css            # Bright, clean styling
            └── components/
                ├── LiveFeed.jsx     # WebSocket camera feed
                ├── EventLog.jsx     # Event log
                ├── UnknownPersons.jsx # Person cards
                ├── VerifyModal.jsx  # Verification dialog
                └── ChatPanel.jsx    # Conversational AI chat
```

---

## Quick Start

### Prerequisites

- Python 3.11 (InsightFace/onnxruntime wheels unreliable on other versions)
- MongoDB Atlas cluster with `surveillance` database
- Atlas Vector Search index named `vector_index` on `faces.latest_embedding` (512 dims, cosine)
- Camera device at `CAMERA_INDEX=0` (or adjust in `.env`)
- Ollama with a model installed (e.g., `ollama pull gemma3:4b` or `ollama pull qwen3.5:4b`)

### Installation

```bash
git clone <your-repo-url>
cd surveillance-system

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt

cd dashboard/frontend
npm install
cd ../..

cp .env.example .env
# Edit .env — set MONGODB_URI at minimum
```

### MongoDB Atlas Setup

1. Create free cluster at [mongodb.com](https://www.mongodb.com)
2. Create database: `surveillance`
3. Create collections: `faces`, `events`, `visit_memory`
4. Create Vector Search Index:
   - Name: `vector_index`
   - Collection: `faces`
   - Path: `latest_embedding`
   - Dimensions: `512`
   - Similarity: `cosine`
5. Whitelist your IP

### Run

```bash
# Terminal 1: Start surveillance
python main.py

# Terminal 2: Start dashboard
cd dashboard/frontend
npm run dev
```

Dashboard: http://localhost:5173 | API: http://localhost:8000

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `MONGODB_URI is required` | Set `MONGODB_URI` in `.env` |
| Camera window black | Change `CAMERA_INDEX` in `.env` |
| No face embeddings | Lower `DET_SCORE_MIN` to `0.20` |
| Low similarity scores (~69%) | Quality-gated embeddings now prevent low-quality overwrites. Delete old embeddings and re-embed with `buffalo_m` model for best results |
| Dashboard shows nothing | Ensure FastAPI running on port 8000 |
| Vite build error (`env/data.js`) | Run `npm install axios@1.7.9` — axios 1.7.10+ breaks Vite's esbuild |
| `GET /api/events` returns 500 | `similarity_score` is null — ensure `Optional[float]` in `dashboard/backend/models.py` |
| Terminal too noisy | Console shows `INFO+`; full debug logs go to `logs/surveillance.log` |
| `FutureWarning` from insightface | Harmless — `estimate` deprecated in InsightFace 0.26 |
| Slow performance | Use GPU: set `YOLO_DEVICE=0` |
| No local camera window | System streams via WebSocket — open `http://localhost:5173` |
| Camera reconnect loops | Auto-reconnects after 30 consecutive failures (~3s). Check USB connection |
| Track shows UNVERIFIED briefly | Normal — recognition runs every 10 frames (~2s). Set `RECOGNITION_INTERVAL_FRAMES=5` for even faster first recognition |
| LLM not responding | Check Ollama is running: `ollama serve`. Verify model: `ollama list`. Falls back to templates if unavailable |
| Chat returns static responses | LLM offline — check `GET /api/chat/health` |
| Swap LLM model | Change `OLLAMA_MODEL` in `.env` (e.g., `qwen3.5:4b`) |

---

## LLM Integration Details

### Models Tested

| Model | Size | VRAM | MMLU-Pro | Notes |
|-------|------|------|----------|-------|
| Gemma 3 4B | ~2.7GB (Q4_K_M) | ~4GB | ~43% | Default |
| Qwen 3.5 4B | ~2.7GB (Q4_K_M) | ~4GB | 79.1% | Better reasoning |

### LLM Parameters

| Parameter | Generate | Chat | NL Summary | Incident Report | Executive Summary |
|-----------|----------|------|------------|-----------------|-------------------|
| `temperature` | 0.3 | 0.2 | 0.2 | 0.3 | 0.3 |
| `max_tokens` | 512 | 1024 | 150 | 250 | 300 |

### LLM Usage

- **NL Alert Summaries:** 1–2 sentence factual descriptions of alerts
- **Executive Summaries:** 3–5 sentence pattern analysis for daily/weekly reports
- **Conversational Dashboard:** Natural language queries about surveillance data
- **Fallback:** Template-based strings when LLM unavailable

---

## License

[Add your license here]
