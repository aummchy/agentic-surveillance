# Face Detection & Embedding

> InsightFace SCRFD detects faces, ArcFace generates 512-dimensional embeddings, and a geometric heuristic detects masks.

**File**: `utils/embedding_utils.py` (133 lines)

## InsightFace Singleton

**Class**: `InsightFaceSingleton`

Loaded once as a process-wide singleton. Thread-safe initialization with `_lock`. Inference serialized with `_inference_lock` (InsightFace is not thread-safe).

```python
class InsightFaceSingleton:
    _instance = None
    _lock = threading.Lock()
    _inference_lock = threading.Lock()

    def __init__(self):
        # One-time initialization:
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(1280, 1280))
```

## CLAHE contrast enhancement

Applied to every image BEFORE face detection:

```
1. Convert BGR → LAB color space
2. Extract L channel
3. Compute l_std = stddev(L)
4. IF l_std >= 40.0: skip CLAHE (contrast already adequate)
5. ELSE: apply CLAHE with:
   - clipLimit = CLAHE_CLIP_LIMIT (default 2.0)
   - tileGridSize = CLAHE_TILE_SIZE × CLAHE_TILE_SIZE (default 8×8)
6. Convert LAB → BGR
```

Skips CLAHE when contrast is already good — saves processing time on well-lit frames.

## Face detection: `detect_faces_raw(image, min_score)`

```python
def detect_faces_raw(self, image: np.ndarray, min_score: float = 0.0) -> list:
```

### Processing
```
1. Apply CLAHE to image
2. faces = app.get(image)  ← SCRFD face detection (under inference lock)
3. Filter: keep faces where det_score >= min_score
4. For each face, extract:
   - det_score: float (detection confidence)
   - embedding: 512-dim L2-normalized vector (ArcFace)
   - bbox: (x1, y1, x2, y2) in pixel coords
   - is_masked: bool (geometric mask heuristic)
5. Sort by det_score descending
6. Return list of dicts
```

### Output format
```python
[
    {
        "det_score": 0.87,
        "embedding": array([0.013, -0.082, ...]),  # 512 floats, L2-normalized
        "bbox": (120, 45, 280, 210),
        "is_masked": False
    },
    ...
]
```

## Mask detection heuristic

**File**: `utils/embedding_utils.py:79-94`

```python
def _detect_mask_geometric(self, landmarks):
    # landmarks from SCRFD: [left_eye, right_eye, nose, left_mouth, right_mouth]
    nose_tip = landmarks[2]
    mouth_center = (landmarks[3] + landmarks[4]) / 2
    upper_face = (landmarks[0] + landmarks[1]) / 2

    lower_face_height = abs(mouth_center.y - nose_tip.y)
    upper_face_height = abs(upper_face.y - nose_tip.y)

    ratio = lower_face_height / upper_face_height
    return ratio < MASK_RATIO_THRESHOLD  # default 0.3
```

**No classifier.** Just geometric: if the lower face (mouth area) appears compressed relative to the upper face, the person is likely masked.

## Atlas score conversion

```python
def atlas_score_to_cosine(atlas_score):
    return (atlas_score * 2) - 1
```

```python
def compare_similarity(raw_cosine, threshold=None):
    if threshold is None:
        threshold = settings.MATCH_THRESHOLD  # 0.45
    return raw_cosine >= threshold
```

## Two-stage detection in Recognition Pipeline

**File**: `pipeline/recognition_pipeline.py:152-238`

```
Stage 1: Detect in person_crop (tighter, more focused)
    crop_faces = detect_faces_raw(person_crop, min_score=DET_SCORE_RELAXED=0.20)

Stage 2: If no embedding-grade face in crop, detect in full frame
    crop_has_embedding_quality = any(f["det_score"] >= EMBEDDING_DET_SCORE_MIN for f in crop_faces)
    IF NOT crop_has_embedding_quality:
        frame_faces = detect_faces_raw(frame, min_score=DET_SCORE_RELAXITY=0.20)

Best face selection:
    1. Try crop_faces[0] (highest det_score)
    2. Else try frame_faces[0]
    3. Convert bbox to frame coordinates if detected in crop

Gate: best["det_score"] must be >= EMBEDDING_DET_SCORE_MIN (0.40)
```

## Quality-gated embedding update

In `TrackState.set_embedding()`:
```
IF track.embedding is None OR det_score > track.embedding_det_score + 0.05:
    → overwrite embedding
ELSE:
    → reject (quality too low)
```

In `db_utils.update_face()`:
```
IF new quality_score > stored latest_embedding_quality:
    → overwrite latest_embedding
ELSE:
    → skip (don't downgrade)
```

## See also
- [[Quality Assessment]] — face quality scoring
- [[Vector Search & Matching]] — how embeddings are searched
- [[Recognition Pipeline]] — orchestrates detection → embedding → search
- [[All Thresholds]] — detection-related thresholds
