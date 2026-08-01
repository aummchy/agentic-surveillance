# Finalizer

> Last-chance embedding generation when a track expires without an embedding. Tries one final InsightFace detection before giving up.

**File**: `agents/finalizer.py` (82 lines)

## Function: `retry_embedding(track, set_embedding)`

```python
def retry_embedding(track: Track, set_embedding):
```

### When it runs

Called by `CameraAgent._finalize_track()` when a track expires (person leaves frame) but has `track.embedding is None`.

### Processing

```
1. IF track.embedding is not None → return (nothing to do)

2. Try detect_faces_raw(track.best_face_crop, min_score=DET_SCORE_RELAXED=0.20)
   → Uses the best face crop saved during progressive recognition

3. IF no faces in crop → try detect_faces_raw(track.best_full_frame, min_score=DET_SCORE_RELAXED=0.20)
   → Falls back to the full frame

4. Pick best face with two-tier priority:
   a. First pass: any face with det_score >= EMBEDDING_DET_SCORE_MIN (0.40)
   b. Second pass: any face with det_score >= DET_SCORE_RELAXED (0.20)

5. IF found:
   a. embedding = best["embedding"].tolist()
   b. Try set_embedding(track_id, embedding, is_masked, det_score)
      - If accepted → done
      - If track was removed → write directly to track object (bypass TrackState)
      - If rejected (quality too low) → skip

6. IF not found:
   → log "no_embedding_after_retries" with visibility + frame count
```

## Why this exists

During progressive recognition, the camera loop runs face detection every 20 frames. But if the person was only visible briefly or the face was at a bad angle, the track might expire with no embedding. This final retry gives one last chance using the best available data (best face crop or full frame) with relaxed thresholds.

## Quality gate in set_embedding

```python
# In TrackState.set_embedding():
if track.embedding is None or det_score > track.embedding_det_score + 0.05:
    track.embedding = embedding
    return (True, "ok")
return (False, "rejected_quality")
```

Even at finalization, a low-quality embedding won't overwrite a better one.

## See also
- [[Camera Agent]] — calls _finalize_track() which calls retry_embedding()
- [[Face Detection & Embedding]] — the detection pipeline
- [[Track State]] — set_embedding() quality gate
