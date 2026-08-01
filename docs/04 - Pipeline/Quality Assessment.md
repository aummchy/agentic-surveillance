# Quality Assessment

> Two-tier face quality system: validity gates reject unusable faces, then a weighted quality score ranks usable faces for embedding comparison.

**File**: `pipeline/quality_agent.py` (74 lines)

## Function: `compute_quality(face_crop)`

```python
def compute_quality(face_crop: np.ndarray) -> QualityResult:
```

### Output: `QualityResult`

```python
QualityResult(
    blur_score=200.0,       # Laplacian variance
    brightness=145.0,       # Mean HSV V-channel
    face_area=3000,         # height × width in pixels
    is_valid=True,          # passes all validity gates
    overall_score=0.63      # weighted composite [0,1]
)
```

## Tier 1: Validity Gates

All three must pass for `is_valid = True`:

| Gate | Formula | Threshold | What it catches |
|------|---------|-----------|----------------|
| Blur | `Laplacian(gray).var()` | ≥ 40 | Blurry, motion-blurred faces |
| Brightness | `mean(HSV_V)` | in [35, 255] | Too dark (night) or overexposed |
| Area | `height × width` | ≥ 1200 px² | Tiny faces (too far from camera) |

**If any gate fails**: `is_valid = False`, face is rejected. No embedding generated, no MongoDB search.

## Tier 2: Quality Score

Weighted composite for faces that pass validity gates:

### Raw measurements

| Metric | Function | Range |
|--------|----------|-------|
| Blur | `cv2.Laplacian(gray, cv2.CV_64F).var()` | 0 → ∞ (higher = sharper) |
| Brightness | `np.mean(hsv[:, :, 2])` | 0 → 255 |
| Area | `h × w` of face crop | 0 → ∞ |

### Normalization

**Blur** — shifted linear:
```
blur_norm = min(max(blur_raw - 40, 0) / (350 - 40), 1.0)
```
- At 40 (min gate) → 0.0
- At 350 → 1.0

**Brightness** — center-radius model:
```
bright_norm = 1 - min(|brightness_raw - 145| / 110, 1.0)
```
- Peak at 145 (well-lit) → 1.0
- At 35 or 255 → 0.0
- Model: brightness far from center = lower score

**Area** — shifted linear:
```
area_norm = min(max(face_area - 1500, 0) / (10000 - 1500), 1.0)
```
- At 1500 (min useful) → 0.0
- At 10000 → 1.0

### Weighted composite

```
overall_score = 0.50 × blur_norm + 0.25 × bright_norm + 0.25 × area_norm
```

Blur is weighted highest (50%) because it most affects embedding quality.

### Score examples

| Condition | Blur | Bright | Area | `overall_score` | Label |
|-----------|:----:|:------:|:----:|:---------------:|:-----:|
| Barely valid | 40 | 35 | 1200 | 0.05 | low |
| Typical | 200 | 120 | 3000 | 0.37 | low |
| Good | 500 | 150 | 5400 | 0.63 | medium |
| Excellent | 1000 | 200 | 10000 | 0.90 | high |

### Quality level labels

| `overall_score` | Label |
|:---------------:|-------|
| ≥ 0.8 | "high face quality" |
| ≥ 0.5 and < 0.8 | "medium face quality" |
| < 0.5 | "low face quality" |

## Best face selection hysteresis

**File**: `pipeline/track_state.py:199-225`

```
set_best_face(composite_id, face_crop, face_score, full_frame, face_ratio):
    1. Quick exit: if face_score <= track.best_face_score + 0.03 → return
       (requires >3% improvement to replace)
    2. Encode full_frame to JPEG (QUALITY_STORE=85) — outside lock
    3. Under lock: if face_score > track.best_face_score:
       - Replace best_face_crop, best_face_score, best_full_frame,
         best_face_ratio, best_frame_jpeg
```

The +0.03 buffer prevents oscillation between similar-quality faces.

## Quality-gated recognition skip

**File**: `agents/camera_agent.py:196-200`

```
IF quality.is_valid:
    → set_best_face() — store best face crop
    → Continue to embedding generation and search
ELSE:
    → RETURN (exits _progressive_recognition early)
    → try/finally ensures end_recognition() still runs (track cleanup)
```

This prevents:
- Storing embeddings from blurry/dark/small faces
- Searching MongoDB with low-quality embeddings
- Incorrectly matching blurry faces to known persons

## See also
- [[Confidence Scoring]] — how quality feeds into confidence (15% weight)
- [[Face Detection & Embedding]] — the detection pipeline that calls compute_quality()
- [[All Thresholds]] — every quality-related threshold
