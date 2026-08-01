# Face Quality Scoring

> Two-tier system: validity gates reject unusable faces, weighted scoring ranks usable ones.

**File**: `pipeline/quality_agent.py`

## Tier 1: Validity Gates

| Gate | Formula | Threshold | Rejects |
|------|---------|-----------|---------|
| Blur | `Laplacian(gray).var()` | ≥ 40 | Blurry, motion-blurred |
| Brightness | `mean(HSV_V)` | [35, 255] | Too dark, overexposed |
| Area | `h × w` | ≥ 1200 px² | Tiny faces |

**All three must pass** for `is_valid = True`.

## Tier 2: Weighted Score

### Normalization

```
blur_norm   = min(max(blur_raw - 40, 0) / (350 - 40), 1.0)
bright_norm = 1 - min(|brightness_raw - 145| / 110, 1.0)
area_norm   = min(max(face_area - 1500, 0) / (10000 - 1500), 1.0)
```

### Composite

```
overall_score = 0.50 × blur_norm + 0.25 × bright_norm + 0.25 × area_norm
```

### Examples

| Blur | Bright | Area | Score | Label |
|:----:|:------:|:----:|:-----:|:-----:|
| 40 | 35 | 1200 | 0.05 | low |
| 200 | 120 | 3000 | 0.37 | low |
| 500 | 150 | 5400 | 0.63 | medium |
| 1000 | 200 | 10000 | 0.90 | high |

## Impact on confidence

Quality contributes 15% to the [[Confidence Scoring]] formula:
```
quality_norm × 0.15 → 0 to 0.15 of final confidence
```

## See also
- [[Confidence Scoring]] — how quality feeds in
- [[Face Detection & Embedding]] — where quality is assessed
- [[All Thresholds]] — quality-related thresholds
