# Recognition Pipeline

> Orchestrates the full face-recognition flow: detect face → assess quality → generate embedding → search database → look up memory → compute confidence → make policy decision.

**File**: `pipeline/recognition_pipeline.py` (352 lines)

## Class: `RecognitionPipeline`

### Constructor
```python
RecognitionPipeline(
    matching_fn=run_matching_from_embedding,
    recognition_agent=RecognitionAgent(),
    memory_agent=MemoryAgent(),
    decide_fn=decide
)
```

All dependencies are injectable (for testing).

### Method: `run(frame, track) → PipelineResult`

```python
def run(self, frame: np.ndarray, track: Track) -> PipelineResult:
```

## Processing pipeline

```
1. HIGH CONFIDENCE SKIP:
   If track already has pending_match_result with similarity > 0.85:
       → return PipelineResult(skip_reason="high_confidence")

2. FACE DETECTION (_detect_face):
   → FaceDet or None

3. QUALITY ASSESSMENT (_assess_quality):
   → QualityResult (is_valid, overall_score)

4. If NOT valid:
   → return PipelineResult(skip_reason="low_quality")

5. EMBEDDING + MATCHING (_build_embedding):
   → embedding list + MatchResult (or None)

6. MEMORY LOOKUP (_lookup_memory):
   → memory context dict

7. RECOGNITION (_run_recognition):
   → recognition result dict (status, confidence, reason)

8. POLICY (_run_policy):
   → DecisionResult

9. Return PipelineResult with all results
```

## Step 2: Face Detection (`_detect_face`)

```
Input: frame (full), track (with person_box)

1. Crop person bounding box from frame
2. Detect faces in person crop (min_score=DET_SCORE_RELAXED=0.20)
3. Check if any face has det_score >= EMBEDDING_DET_SCORE_MIN (0.40)
4. IF no embedding-grade face in crop:
   → Fallback: detect faces in full frame
5. Pick best face by det_score
6. Gate: det_score must be >= EMBEDDING_DET_SCORE_MIN (0.40)
7. Compute face_ratio = face_area / person_area
8. Extract face crop from frame using frame coordinates
```

Two-stage detection ensures we find faces even when the person crop is tight.

## Step 3: Quality Assessment (`_assess_quality`)

```
1. compute_quality(face_crop)  → pipeline/quality_agent.py
2. Returns QualityResult:
   - blur_score: Laplacian variance
   - brightness: mean HSV V-channel
   - face_area: height × width
   - is_valid: passes all validity gates
   - overall_score: weighted composite [0,1]
```

See [[Quality Assessment]] for full details.

## Step 5: Embedding + Matching (`_build_embedding`)

```
1. Extract 512-dim ArcFace embedding from best face
2. Check embedding cache:
   - If track has cached_embedding AND pending_match_result:
     - Compute cosine distance between new and cached
     - If distance < 0.005 → reuse previous match (skip Atlas query)
3. If cache miss:
   - matching_fn(embedding) → vector_search → MatchResult
4. Update track.cached_embedding = new_embedding
```

## Step 6: Memory Lookup (`_lookup_memory`)

```
1. If no match or no person_id → return {}
2. If similarity > HIGH_CONFIDENCE_SIMILARITY (0.85):
   → skip memory (high confidence, no boost needed)
3. memory_agent.run({person_id, camera_id, similarity, status})
```

## Step 7: Recognition (`_run_recognition`)

```
recognition_agent.run({
    similarity, is_masked, face_quality, track_duration,
    memory_context, top2, margin, name, track_id
})
→ {status, confidence, reason}
```

See [[Recognition Agent]] for full details.

## Step 8: Policy (`_run_policy`)

```
decide(track, match, recognition, memory)
→ DecisionResult(status, alert_level, should_alert, should_register)
```

See [[Policy Agent]] for full details.

## PipelineResult

```python
PipelineResult(
    decision=DecisionResult(...),
    match=MatchResult(...),
    recognition={"status": "known", "confidence": 79, ...},
    memory={"visit_count": 5, "confidence_boost": 12, ...},
    embedding=[0.013, -0.082, ...],
    quality=QualityResult(...),
    face_crop=numpy_array,
    face_bbox=(x1, y1, x2, y2),
    face_ratio=0.15,
    det_score=0.87,
    is_masked=False,
    metrics=RecognitionMetrics(...),
    skip_reason="success"
)
```

## Skip reasons

| skip_reason | When | What happens |
|-------------|------|-------------|
| `high_confidence` | Existing match > 0.85 | Nothing stored, early return |
| `no_face` | No face detected anywhere | Nothing stored |
| `low_quality` | Face fails validity gates | Best face stored if valid, no embedding |
| `embedding_failed` | Embedding generation failed | Quality stored, no embedding |
| `success` | Full pipeline completed | Everything stored |

## See also
- [[Face Detection & Embedding]] — InsightFace details
- [[Quality Assessment]] — quality scoring
- [[Vector Search & Matching]] — Atlas search
- [[Confidence Scoring]] — recognition formula
- [[Policy Rules]] — decision rules
