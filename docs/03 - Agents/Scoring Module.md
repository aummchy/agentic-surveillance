# Scoring Module

> Pure scoring functions for recognition confidence. All tunables come from `config/settings.py` — no hardcoded values.

**File**: `agents/scoring.py` (258 lines)

## Functions

### `compute_confidence(raw_cosine, face_quality, track_seconds, memory_boost, is_masked, margin, track_id)`

The main entry point. Returns confidence score 1-100.

```
1. Normalize all 5 components to [0,1]
2. Weighted sum: base = Σ(weight_i × normalized_i)
3. Apply mask penalty: adjusted = base × (1 - 0.15 × mask_norm)
4. Scale: confidence = 1 + 99 × clip(adjusted, 0, 1)
5. Log calculation to logs/calculation.log (if ENABLE_CALC_LOG)
6. Return confidence (int)
```

### `is_match(raw_cosine)`

Binary identity gate:
```
raw_cosine >= MATCH_THRESHOLD (0.45)  →  True
raw_cosine <  MATCH_THRESHOLD (0.45)  →  False
```

### `confidence_status(confidence, matched)`

Maps confidence + match gate → status string:
```
matched + conf >= 70  → "known"
matched + conf >= 55  → "uncertain"
matched + conf <  55  → "unknown"
!matched + conf >= 55 → "uncertain"
!matched + conf <  55 → "unknown"
```

## Normalization functions

### `normalize_cosine(raw_cosine)`
```
sim_norm = clip((raw_cosine - SIM_NORM_MIN) / (SIM_NORM_MAX - SIM_NORM_MIN), 0, 1)
         = clip((raw_cosine - 0.25) / (0.80 - 0.25), 0, 1)
```
- At 0.25 → 0.0 (low similarity)
- At 0.80 → 1.0 (high similarity)
- Below 0.25 → clipped to 0
- Above 0.80 → clipped to 1

### `normalize_quality(face_quality)`
```
If None or ≤ 0 → return DEFAULT_FACE_QUALITY (0.50)
Else → clip(face_quality, 0, 1)
```

### `normalize_track_duration(track_seconds)`
```
track_norm = min(track_seconds / TRACK_SATURATION_SECS, 1.0)
           = min(track_seconds / 1.5, 1.0)
```
- 0s → 0.0, 1.5s+ → 1.0 (saturated)

### `normalize_memory(memory_boost)`
```
memory_norm = clip(memory_boost, 0, MEMORY_NORM_MAX) / MEMORY_NORM_MAX
            = clip(memory_boost, 0, 20) / 20
```
- Boost of 0 → 0.0, boost of 20 → 1.0

### `normalize_margin(margin)`
```
If None → return 0.5 (neutral)
Else → clip(margin / MARGIN_NORM_MAX, 0, 1)
      = clip(margin / 0.30, 0, 1)
```

### `normalize_mask(is_masked)`
```
1.0 if is_masked else 0.0
```

## Weight breakdown

| Component | Weight | Normalization | Effective range |
|-----------|:------:|:-------------:|:---------------:|
| Similarity | 0.65 | [0.25..0.80] → [0..1] | 0–0.65 |
| Face Quality | 0.15 | [0..1] direct | 0–0.15 |
| Track Duration | 0.10 | secs / 1.5, saturated | 0–0.10 |
| Memory Boost | 0.05 | [0..20] → [0..1] | 0–0.05 |
| Margin | 0.05 | [0..0.30] → [0..1] | 0–0.05 |
| **Base total** | **1.00** | | **0–1.00** |
| Mask penalty | ×(1 - 0.15×mask) | | ×1.0 or ×0.85 |

## Calculation log

When `ENABLE_CALC_LOG=True`, every confidence computation is logged to `logs/calculation.log` with a full tabular breakdown:

```
═══════════════════════════════════════════════════════════════════════
                      CONFIDENCE SCORING FORMULA
═══════════════════════════════════════════════════════════════════════
  base = 0.65×sim + 0.15×quality + 0.10×track + 0.05×memory + 0.05×margin
  adjusted = base × (1 - 0.15×mask)
  confidence = int(round(1 + 99 × clip(adjusted, 0, 1)))
═══════════════════════════════════════════════════════════════════════
───────────────────────────────────────────────────────────────────────
  track=cam_01_...  │  2026-07-09 10:30:15
───────────────────────────────────────────────────────────────────────
  Component              Raw   Normalized              Weighted
  ──────────────────── ────── ──────────── ────────────────────────────
  Similarity           0.704      0.825   0.65×0.825 = 0.536
  Face Quality         0.512      0.512   0.15×0.512 = 0.077
  Track Duration       10.5s      1.000   0.10×1.000 = 0.100
  Memory Boost         18.0       0.900   0.05×0.900 = 0.045
  Margin               0.202      0.675   0.05×0.675 = 0.034
  ──────────────────── ────── ──────────── ────────────────────────────
  BASE = 0.536 + 0.077 + 0.100 + 0.045 + 0.034 = 0.792
  MASK = 0.0 (unmasked) → no penalty
  ADJUSTED = 0.792 × 1.0000 = 0.792
  ──────────────────────────────────────────────────────────────────────
  CONFIDENCE = 1 + 99 × 0.792 = 79
  STATUS = known  (matched=true, threshold=0.45)
═══════════════════════════════════════════════════════════════════════
```

## See also
- [[Confidence Scoring]] — human-readable formula explanation
- [[Recognition Agent]] — uses these functions
- [[All Thresholds]] — every threshold used
