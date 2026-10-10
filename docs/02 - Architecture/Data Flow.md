# Data Flow

> Per-frame step-by-step walkthrough of what happens from camera capture to final decision.
> Verified against code on 2026-10-03 (Phase 2 audit). Function names are used instead
> of line numbers — line numbers rot.

## Phase 1: Frame Capture (every frame)

**Files**: `agents/camera_agent.py` → `pipeline/tracker.py` → `pipeline/track_state.py`

```
1. ret, frame = cap.read()                          ← OpenCV VideoCapture
2. frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))   ← configurable,
                                              default 1920×1080
3. tracks = track_persons(frame)                    ← YOLOv8 + ByteTrack
```

For each detected person `t` in `tracks`:

```
4. composite_id = f"{camera_id}_{session_epoch}_{byte_track_id}_{generation}"
   (make_composite_id; example: "cam_01_1782040060_3_0")
   The generation suffix handles ByteTrack ID reuse within a session.
5. track = track_state.update(camera_id, t["track_id"], t["box"])
   → Creates new Track or updates existing (last_seen, person_box)
```

Then `_process_tracks` deduplicates overlapping boxes (IoU ≥
`OVERLAP_IOU_THRESHOLD` skips scheduling) and, for each surviving track,
either schedules a progressive recognition pass or, when the track has
expired, schedules finalization.

## Phase 2: Progressive Recognition (every 20 frames)

**Files**: `agents/camera_agent.py` → `agents/recognition_worker.py` → `pipeline/recognition_pipeline.py`

For each active track, every `RECOGNITION_INTERVAL_FRAMES` (20) frames:

```
6. Check skip conditions (should_skip_recognition, agents/recognition_throttle.py) — first match wins:
   - "resolved"          track.decision is already a RESOLVED status
                         (KNOWN_VISITOR / VERIFIED / AUTHORIZED)
   - "high_confidence"   a prior pass matched above HIGH_CONFIDENCE_SIMILARITY (0.85)
   - "rescan_interval"   UNKNOWN/UNCERTAIN but the backoff has not elapsed
                         (RESCAN_INTERVAL_SECS=3, budget MAX_RESCAN_ATTEMPTS=3;
                          allowing a rescan spends one attempt as a side effect)
   - "quality_throttle"  face quality did not improve by ≥ MIN_QUALITY_IMPROVEMENT (0.10)
   → If any: skip, continue to next track

   RecognitionPipeline.run() also re-checks high_confidence internally
   (reads pending_match_result under the track lock) before doing any work.

7. Submit to ThreadPoolExecutor(max_workers=RECOGNITION_MAX_WORKERS=4):
   progressive_recognition(frame.copy(), track)     [agents/recognition_worker.py]
```

### Inside progressive recognition (`RecognitionPipeline.run`)

```
8. FACE DETECTION (_detect_face):
   a. Crop person bounding box from frame
   b. Run InsightFace SCRFD on person crop (min_score=DET_SCORE_RELAXED=0.20)
   c. If no embedding-grade face in crop → full-frame detection, but only
      when ENABLE_FULL_FRAME_FALLBACK is true (currently false, so this
      path does not run)
   d. Pick best face by det_score (SCRFD results arrive sorted descending,
      so index 0 is the best)
   e. Gate: det_score must be ≥ EMBEDDING_DET_SCORE_MIN (0.40) — a face
      good enough to log is not good enough to embed
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
    b. MongoDB Atlas Vector Search — fresh query each pass, no embedding cache
    c. Atlas returns top-5 matches with vectorSearchScore
    d. Convert: raw_cosine = (atlas_score × 2) - 1
    e. Match if raw_cosine ≥ MATCH_THRESHOLD (0.45)
    f. Compute margin = top1 - top2

11. MEMORY LOOKUP (_lookup_memory → agents/memory.py):
    a. Query/create the visit_memory document for this person_id
    b. Return visit_count, typical_hours, last_seen, avg_similarity,
       is_known, confidence_boost (plus typical-time/camera flags)

12. RECOGNITION SCORING (_run_recognition → agents/scoring.py):
    a. Normalize 5 components to [0,1]
    b. Weighted sum: 0.65×sim + 0.15×quality + 0.10×track + 0.05×memory + 0.05×margin
    c. Mask penalty: base × (1 - MASK_PENALTY_MAX(0.15) × mask_norm),
       i.e. up to ×0.85 for a fully masked face
    d. confidence = 1 + 99 × clip(adjusted, 0, 1)  →  [1..100]
    e. Status (confidence_status): matched+conf≥70 → KNOWN;
       matched+conf≥55 → UNCERTAIN; matched+conf<55 → UNKNOWN;
       unmatched+conf≥55 → UNCERTAIN; else UNKNOWN

13. POLICY DECISION (_run_policy → agents/policy.py decide()):
    a. Evaluate 9 rules in priority order (first match wins)
    b. Return DecisionResult(status, alert_level, should_alert, should_register)

14. Write back (handle_pipeline_result, agents/recognition_worker.py — the ONLY write-back site), in order:
    - update_face_visibility
    - set_best_face (quality-hysteresis: needs +0.03 improvement) [+ debug crop]
    - set_embedding (only if det_score beats existing by ≥ 0.05)
    - set_person_name / set_pending_match_result   (match)
    - set_pending_recognition_data                 (recognition)
    - set_pending_memory_context                   (memory)
    - handle_decision_and_alert → set_decision,
      update_confidence_if_higher, set_recognition_snapshot,
      critical-alert dispatch
    Each write goes through a TrackState setter that applies its own
    upgrade rule, so a later weaker pass cannot overwrite a stronger one.
```

## Phase 3: Track Expiry + Finalization

**Files**: `agents/camera_agent.py` → `agents/track_finalization.py` → `agents/finalizer.py` → `agents/track_processor.py`

```
15. Track expires when:
    - No detection for > TRACK_TIMEOUT_SECS (15), or
    - alive > MAX_TRACK_SECS (300)

16. Classify visibility BEFORE removal (_classify_visibility_inplace):
    - max_face_ratio ≥ VISIBLE_FACE_RATIO (0.025)   → visible
    - max_face_ratio ≥ PARTIAL_FACE_RATIO (0.010)   → partial
    - masked or face_detected_once                   → partial
    - never saw a face and total_frames_seen ≥ 15    → hidden
    - else                                           → unknown

17. Submit TrackFinalizer.finalize_track(track) to the recognition executor — two routes
    can reach it (finalize_expired when the track is not being
    recognized, cleanup_after_recognition when a worker finishes after the
    track vanished); Track.mark_finalized_once() is the third and final
    duplicate guard.

18. Inside TrackFinalizer.finalize_track:
    a. retry_embedding if still None (agents/finalizer.py, crop first;
       full-frame fallback again gated by ENABLE_FULL_FRAME_FALLBACK)
    b. on_track_finalized → main.py enqueues the track onto track_queue

19. Worker thread picks up track (worker_process_tracks → TrackProcessor.process):
    a. Take snapshot of track state (TrackSnapshot), release best_full_frame
    b. No embedding? log an UNKNOWN event and stop
    c. Reuse pending_match_result if present, else fresh vector search
    d. Reuse recognition + memory only if the match was also reused
       (fresh_match rule in _run_recognition_and_memory)
    e. Run final decide()
    f. Auto-register unknowns to MongoDB (quality gate + deduplicate_identity)
    g. Record visit in memory
    h. Dispatch alert if should_alert (mark_alerted_once gate)
    i. Broadcast alert if dispatched
    j. log_event() to the events collection
    k. Broadcast event to dashboard via WebSocket
```

## Phase 4: Dashboard Broadcasting (every frame)

**File**: `agents/track_processor.py` → `handle_frame`

```
20. Every FRAME_SKIP (2) frames:
    a. Resize to BROADCAST_WIDTH × BROADCAST_HEIGHT (960×540)
    b. JPEG encode (JPEG_QUALITY_BROADCAST = 85)
    c. Submit to the JPEG executor → broadcast_frame() via WebSocket
       (skipped entirely if the asyncio loop is no longer running)
```

## See also

- [[Thread Architecture]] — which thread does what
- [[Data Models]] — the structures these stages pass around
- [[Confidence Scoring]] — the weighted formula
- [[Policy Rules]] — the 9-rule decision tree
