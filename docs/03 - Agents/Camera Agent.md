# Camera Agent

> The main camera loop. Captures frames, runs YOLO detection + ByteTrack tracking, schedules progressive recognition, handles track expiry, and broadcasts annotated frames.

**File**: `agents/camera_agent.py` (452 lines)

## Role in system

```
CameraAgent._loop()  ← THIS IS THE MAIN THREAD
    │
    ├─ YOLO + ByteTrack detection (every frame)
    ├─ Progressive recognition scheduling (every 20 frames)
    ├─ Track expiry detection → submit finalization
    └─ Frame annotation + WebSocket broadcast
```

## Class: `CameraAgent`

### Constructor (`__init__`)
```
on_track_finalized: callback when track expires
on_frame_annotated: callback for annotated frames
recognition_pipeline: RecognitionPipeline instance (or default)
```

Internal state:
- `track_state: TrackState` — thread-safe track dictionary
- `_recognizing_tracks: set` — tracks currently being recognized (prevents duplicates)
- `_finalized_track_ids: set` — tracks already submitted for finalization
- `_recognition_executor: ThreadPoolExecutor(max_workers=4)` — runs recognition off main thread
- `_timing: TimingCollector` — performance metrics

### `start()` (line 86)
```
1. Open camera (VideoCapture with configured backend)
2. Set frame dimensions (FRAME_WIDTH × FRAME_HEIGHT)
3. Log camera_started event
4. Enter _loop() (blocks until Ctrl+C)
5. On exit: stop()
```

### `_loop()` (line 118) — THE MAIN LOOP

Every iteration:

```
1. ret, frame = cap.read()
   - If fail: increment consecutive_failures
   - If 10 failures: try reconnect (release + re-open)
   - If file source: break on EOF

2. frame = cv2.resize(frame, (1280, 720))

3. tracks = track_persons(frame)  → YOLO + ByteTrack

4. For each track t:
   a. composite_id = make_composite_id(camera_id, bt_track_id)
   b. track = track_state.update(camera_id, bt_track_id, box)
   c. Every RECOGNITION_INTERVAL_FRAMES (20) frames:
      - Skip if already resolved or high-confidence
      - Skip if quality didn't improve by ≥0.10
      - Submit _progressive_recognition(frame.copy(), track) to executor

5. expired = track_state.get_expired_tracks()
   For each expired track:
   - If not recognizing and not already finalized:
     → submit _finalize_track(track) to executor

6. annotated = draw_annotations(frame, all_tracks)
7. on_frame_annotated(annotated)  → JPEG encode + WebSocket broadcast
8. Log performance stats every 1 second
```

### `_progressive_recognition()` (line 302) — runs in executor

```
1. Add track_id to _recognizing_tracks
2. track_state.begin_recognition(track_id)
3. track_state.ensure_fallback_frame(track_id, frame)  ← guarantee photo

4. result = self._pipeline.run(frame, track)  → RecognitionPipeline
   Returns PipelineResult with: decision, match, recognition, memory, embedding, quality, face_crop

5. Handle skip reasons:
   - "high_confidence" → return (nothing to do)
   - "no_face", "low_quality", "embedding_failed" → store best face if valid, return

6. Store results on track:
   - set_best_face() — face crop + quality
   - set_embedding() — embedding + mask status
   - set_cached_embedding() — for cache optimization
   - set_pending_match_result() — match result
   - set_pending_recognition_data() — recognition result
   - set_pending_memory_context() — memory context

7. Handle progressive alerts:
   - CRITICAL alerts → dispatch immediately (don't wait for finalization)
   - Other alerts → defer to finalization
   - Update confidence (never downgrades)

8. Finally:
   - end_recognition(track_id) → decrement in-flight counter
   - Remove from _recognizing_tracks
   - If track was removed during recognition → submit finalization
```

### `_finalize_track()` (line 444) — runs in executor

```
1. retry_embedding(track, set_embedding)  → finalizer.py
   If track has no embedding, try one last InsightFace detection
2. on_track_finalized(track)  → enqueues to track_queue
```

### `_open_capture()` (static, line 53)
```
- If CAMERA_SOURCE is set: use it (RTSP URL or file path)
- Else: use CAMERA_INDEX (integer device index)
- If CAMERA_BACKEND is set: use specific backend (dshow, msmf)
- For network streams: set read timeout to 1000ms
```

## Key design decisions

1. **Recognition runs off main thread** — `_recognition_executor` has 4 workers, so 4 tracks can be recognized concurrently
2. **Quality throttle** — skips recognition if face quality didn't improve by ≥0.10 (avoids wasted CPU)
3. **Embedding cache** — if new embedding is nearly identical to last (cosine <0.005), skip Atlas search
4. **Critical alerts fire early** — blacklist alerts dispatch during progressive recognition, not deferred to finalization
5. **Fallback frame** — every track gets at least one photo (even if face quality is bad)

## See also
- [[Recognition Pipeline]] — the pipeline that runs inside progressive recognition
- [[Track State]] — thread-safe track dictionary
- [[Track Processor]] — consumes finalized tracks
- [[Data Flow]] — per-frame walkthrough
