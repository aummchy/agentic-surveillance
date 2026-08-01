# Data Flow

> Per-frame step-by-step walkthrough of what happens from camera capture to final decision.

## Phase 1: Frame Capture (every frame)

**File**: `agents/camera_agent.py:118-155`

```
1. ret, frame = cap.read()                    ← OpenCV VideoCapture
2. frame = cv2.resize(frame, (1280, 720))     ← Fixed resolution
3. tracks = track_persons(frame)               ← YOLOv8 + ByteTrack
```

For each detected person `t` in `tracks`:
```
4. composite_id = f"{camera_id}_{session_epoch}_{byte_track_id}"
   Example: "cam_01_1782040060_3"
5. track = track_state.update(camera_id, t["track_id"], t["box"])
   → Creates new Track or updates existing (last_seen, person_box)
```

## Phase 2: Progressive Recognition (every 20 frames)

**File**: `agents/camera_agent.py:168-234` → `pipeline/recognition_pipeline.py:78-150`

For each active track, every `RECOGNITION_INTERVAL_FRAMES` (20) frames:

```
6. Check skip conditions:
   - Already resolved (verified/known_visitor/authorized)?
   - High confidence match (>0.85)?
   - Quality didn't improve by ≥0.10?
   → If any: skip, continue to next track

7. Submit to ThreadPoolExecutor(max_workers=2):
   _progressive_recognition(frame.copy(), track)
```

### Inside progressive recognition

**File**: `pipeline/recognition_pipeline.py:78-150`

```
8. FACE DETECTION (_detect_face):
   a. Crop person bounding box from frame
   b. Run InsightFace SCRFD on person crop (min_score=0.20)
   c. If no embedding-grade face in crop → fallback to full-frame detection
   d. Pick best face by det_score
   e. Gate: det_score must be ≥ EMBEDDING_DET_SCORE_MIN (0.40)
   f. Extract face crop + compute face_area/person_area ratio

9. QUALITY ASSESSMENT (_assess_quality):
   a. Laplacian variance (blur)
   b. Mean HSV V-channel (brightness)
   c. Face area in pixels
   d. Validity gates: blur ≥ 40, brightness in [35,255], area ≥ 1200
   e. Weighted score: 50% blur + 25% brightness + 25% area
   f. If NOT valid → RETURN (skip embedding, skip search)

10. EMBEDDING + MATCHING (_build_embedding):
    a. Extract 512-dim ArcFace embedding
    b. Check embedding cache: if cosine distance from last < 0.005 → reuse match
    c. Otherwise: MongoDB Atlas Vector Search
    d. Atlas returns top-5 matches with vectorSearchScore
    e. Convert: raw_cosine = (atlas_score × 2) - 1
    f. Match if raw_cosine ≥ MATCH_THRESHOLD (0.45)
    g. Compute margin = top1 - top2

11. MEMORY LOOKUP (_lookup_memory):
    a. Query visit_memory collection for this person_id
    b. Return visit_count, typical_hours, last_seen, is_known, confidence_boost

12. RECOGNITION SCORING (_run_recognition → agents/scoring.py):
    a. Normalize 5 components to [0,1]
    b. Weighted sum: 0.65×sim + 0.15×quality + 0.10×track + 0.05×memory + 0.05×margin
    c. Apply mask penalty: ×0.85 if masked
    d. confidence = 1 + 99 × adjusted  →  [1..100]
    e. Status: matched+conf≥70 → "known", conf≥55 → "uncertain", else "unknown"

13. POLICY DECISION (_run_policy → agents/policy.py):
    a. Evaluate 9 rules in priority order (first match wins)
    b. Return DecisionResult(status, alert_level, should_alert, should_register)

14. Cache results on Track object:
    - track.pending_match_result
    - track.pending_recognition
    - track.pending_memory_context
    - track.decision = status
    - track.confidence (never downgrades)
```

## Phase 3: Track Expiry + Finalization

**File**: `agents/camera_agent.py:236-241` → `agents/track_processor.py:64-256`

```
15. Track expires when:
    - No detection for >3 seconds (TRACK_TIMEOUT_SECS)
    - Or track alive >300 seconds (MAX_TRACK_SECS)

16. Classify visibility BEFORE removal:
    - face_ratio ≥ 0.025 → "visible"
    - face_ratio ≥ 0.010 → "partial"
    - masked or face_detected_once → "partial"
    - never saw face + ≥15 frames → "hidden"
    - else → "unknown"

17. Submit _finalize_track(track) to executor

18. Inside finalization:
    a. Retry embedding if still None (finalizer.py)
    b. Enqueue to track_queue

19. Worker thread picks up track (worker_process_tracks):
    a. Take snapshot of track state
    b. Reuse match_result from progressive recognition if available
    c. Reuse recognition_result if match was reused
    d. Run final decide()
    e. Auto-register unknowns to MongoDB (with dedup)
    f. Record visit in memory
    g. Dispatch alert if should_alert
    h. Broadcast event to dashboard via WebSocket
    i. log_event() to events collection
```

## Phase 4: Dashboard Broadcasting (every frame)

**File**: `agents/track_processor.py:44-62`

```
20. Every FRAME_SKIP (2) frames:
    a. Resize to 640×360
    b. JPEG encode (quality=65)
    c. Submit to executor → broadcast_frame() via WebSocket
```

## See also
- [[Thread Architecture]] — which thread does what
- [[Recognition Pipeline]] — detailed pipeline walkthrough
- [[Confidence Scoring]] — the weighted formula
- [[Policy Rules]] — the 9-rule decision tree
