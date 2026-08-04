# Embedding Utils

> InsightFace singleton, CLAHE contrast enhancement, face detection, mask heuristic, and similarity utilities.

**File**: `utils/embedding_utils.py` (135 lines)

## Classes

### `InsightFaceSingleton`

Thread-safe singleton wrapping InsightFace's `FaceAnalysis`. Loaded once at startup, never reloaded per-frame.

**Key methods:**

| Method | Purpose |
|--------|---------|
| `__init__()` | Load InsightFace model (`buffalo_l`) once. Redirects stdout during loading. |
| `_apply_clahe(image)` | CLAHE contrast enhancement. Skips if L-channel std ≥ 40 (already good contrast). Skips for crops < 200px. |
| `_detect_mask_geometric(landmarks)` | Heuristic: lower_face/upper_face ratio < 0.3 → masked. No classifier. |
| `detect_faces_raw(image, min_score)` | Run SCRFD detection + ArcFace embedding. Returns list of dicts sorted by det_score. |

**Configuration:**
- `INSIGHTFACE_MODEL`: Model pack (default: `buffalo_l`)
- `INSIGHTFACE_PROVIDER`: ONNX Runtime provider (default: `CPUExecutionProvider`)
- `INSIGHTFACE_DET_SIZE`: Detection input size (default: `640`)
- `CLAHE_CLIP_LIMIT`: CLAHE clip limit (default: `2.0`)
- `CLAHE_TILE_SIZE`: CLAHE tile grid (default: `8`)
- `MASK_RATIO_THRESHOLD`: Mask detection ratio (default: `0.3`)

**Thread safety:**
- `_inference_lock`: Ensures only one InsightFace inference runs at a time (GIL-unsafe native code).
- `_lock`: Class-level lock for singleton initialization.

## Standalone functions

| Function | Purpose |
|----------|---------|
| `get_insightface()` | Returns singleton `InsightFaceSingleton` instance |
| `compare_similarity(raw_cosine, threshold)` | Returns `True` if `raw_cosine >= threshold` (default: `MATCH_THRESHOLD`) |
| `atlas_score_to_cosine(atlas_score)` | Converts Atlas `vectorSearchScore` to raw cosine: `(score * 2) - 1` |

## CLAHE behavior

1. Convert BGR → LAB, extract L channel
2. If L-channel std ≥ 40 → skip CLAHE (already good contrast)
3. Otherwise apply CLAHE with `clipLimit=CLAHE_CLIP_LIMIT`, `tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE)`
4. Convert back to BGR

For grayscale images, same logic but no color conversion needed.

CLAHE is skipped for tiny crops (< 200px) because the 8×8 tile grid destroys facial features on small images.

## See also
- [[Face Detection & Embedding]] — full walkthrough of detection + embedding pipeline
- [[Quality Assessment]] — quality scoring
- [[Vector Search & Matching]] — how embeddings are used
