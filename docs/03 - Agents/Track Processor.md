# Track Processor

> Orchestrates track finalization: runs final matching, recognition, memory, policy decision, stores faces, records visits, dispatches alerts, and broadcasts to dashboard.

**File**: `agents/track_processor.py` (282 lines)

## Role in system

```
TrackQueue → worker_process_tracks() → TrackProcessor.process(track)
                                            │
                                            ├─ match (if not cached)
                                            ├─ recognize (if not cached)
                                            ├─ decide (policy)
                                            ├─ store_face (MongoDB)
                                            ├─ record_visit (memory)
                                            ├─ dispatch (alerts)
                                            └─ broadcast (WebSocket)
```

## Class: `TrackProcessor`

### Constructor
```
loop: asyncio event loop (for WebSocket broadcasts)
memory_agent: MemoryAgent instance (or default)
```

Internal state:
- `_encode_executor: ThreadPoolExecutor(max_workers=2)` — JPEG encode + WebSocket broadcast
- `_frame_counter: int` — counts frames for FRAME_SKIP

### `handle_frame(frame)` (line 44)
Called by `CameraAgent.on_frame_annotated`. Handles WebSocket frame broadcasting:
```
1. frame_counter += 1
2. If counter % FRAME_SKIP != 0 → return (skip this frame)
3. Submit to executor:
   a. preview = cv2.resize(frame, (640, 360))
   b. JPEG encode (quality=90)
   c. broadcast_frame(jpeg_bytes) via WebSocket
```

### `process(track)` (line 64) — THE CORE METHOD

Called by worker threads from `worker_process_tracks()`.

```
1. snap = track.snapshot()  ← thread-safe copy of all track state
2. image_url = resolve_track_image_url(track)  ← Cloudinary or local

3. IF snap.embedding is None:
   → log_event("unknown", "none", alerted=False)
   → return (nothing to match)

4. MATCHING:
   If snap.pending_match_result exists (from progressive recognition):
       → reuse it
   Else:
       → run_matching_from_embedding(snap.embedding)

5. RECOGNITION:
   If snap.pending_recognition exists AND match was reused:
       → reuse it
   Else:
       → RecognitionAgent.run({...})

6. MEMORY:
   If snap.pending_memory_context exists AND match was reused:
       → reuse it
   Else:
       → MemoryAgent.run({...})

7. DECISION:
   decision = decide(track, match_result, recognition_result, memory_context)

8. AUTO-REGISTRATION:
   IF decision.should_register:
       a. If status is "unknown" or "masked_unknown":
          - find_similar_unknowns(embedding) for dedup
          - If similar found → update_face() (merge)
          - Else → store_face() (new record)
       b. Else → store_face() with name/role from match

9. VISIT RECORDING:
   IF match_result.matched:
       memory_agent.record_visit(person_id, camera_id, status, similarity)

10. ALERT DISPATCH:
    IF decision.should_alert AND track.mark_alerted_once():
        → dispatch(track, decision, image_url)
        → broadcast_alert(alert_payload) via WebSocket

11. EVENT LOGGING:
    → log_event(track_id, status, alert_level, ...)
    → broadcast_event(event_payload) via WebSocket

12. LOG track_finalized with full details:
    status, confidence, similarity, top2, margin, frames, face, visit_count
```

## Progressive caching pattern

The key optimization: results from progressive recognition (stored on Track) are reused at finalization to avoid redundant DB calls.

```
Progressive recognition (every 20 frames):
  track.pending_match_result = match_result
  track.pending_recognition = recognition_result
  track.pending_memory_context = memory_context

Finalization (track expires):
  snap = track.snapshot()  ← captures all pending results
  IF snap.pending_match_result → reuse (skip Atlas query)
  IF snap.pending_recognition → reuse (skip recognition scoring)
  IF snap.pending_memory_context → reuse (skip memory lookup)
```

## See also
- [[Camera Agent]] — schedules progressive recognition
- [[Matching Agent]] — vector search
- [[Recognition Agent]] — confidence scoring
- [[Policy Agent]] — decision rules
- [[Memory Agent]] — visit history
- [[Alert Agent]] — alert dispatch
