# Confidence Scoring

> The weighted normalization formula that converts 5 signals into a single confidence score (1-100).

## The formula

```
base = 0.65×sim_norm + 0.15×quality_norm + 0.10×track_norm + 0.05×memory_norm + 0.05×margin_norm
adjusted = base × (1 - 0.15×mask_norm)
confidence = int(round(1 + 99 × clip(adjusted, 0, 1)))
```

**File**: `agents/scoring.py:181-233`

## Components

### 1. Similarity (65% weight)

```
sim_norm = clip((raw_cosine - 0.25) / (0.80 - 0.25), 0, 1)
```

| Raw cosine | Normalized |
|:----------:|:----------:|
| 0.25 | 0.000 |
| 0.45 | 0.364 |
| 0.60 | 0.636 |
| 0.70 | 0.818 |
| 0.80 | 1.000 |

Similarity dominates (65%) because it's the strongest identity signal.

### 2. Face Quality (15% weight)

```
quality_norm = clip(face_quality, 0, 1)
```
- None or 0 → fallback 0.50 (neutral)
- Direct pass-through since quality is already [0,1]

### 3. Track Duration (10% weight)

```
track_norm = min(track_seconds / 1.5, 1.0)
```

| Duration | Normalized |
|:--------:|:----------:|
| 0s | 0.000 |
| 0.5s | 0.333 |
| 1.0s | 0.667 |
| 1.5s+ | 1.000 (saturated) |

Longer tracks give more time for face detection → higher confidence.

### 4. Memory Boost (5% weight)

```
memory_norm = clip(memory_boost, 0, 20) / 20
```

| Boost | Normalized |
|:-----:|:----------:|
| 0 | 0.000 |
| 10 | 0.500 |
| 20 | 1.000 |

### 5. Margin (5% weight)

```
margin_norm = clip(margin / 0.30, 0, 1)   # None → 0.50
```

| Margin | Normalized |
|:------:|:----------:|
| 0.00 | 0.000 |
| 0.15 | 0.500 |
| 0.30+ | 1.000 |

### Mask penalty

```
adjusted = base × (1 - 0.15 × mask_norm)
```
- Unmasked: ×1.0 (no penalty)
- Masked: ×0.85 (15% penalty)

## Weight breakdown

| Component | Weight | Norm range | Effective range |
|-----------|:------:|:----------:|:---------------:|
| Similarity | 0.65 | [0..1] | 0–0.65 |
| Face Quality | 0.15 | [0..1] | 0–0.15 |
| Track Duration | 0.10 | [0..1] | 0–0.10 |
| Memory Boost | 0.05 | [0..1] | 0–0.05 |
| Margin | 0.05 | [0..1] | 0–0.05 |
| **Base total** | **1.00** | | **0–1.00** |
| Mask penalty | ×(1-0.15×mask) | | ×1.0 or ×0.85 |

## Status mapping

```
matched=True  + confidence >= 70  → "known"
matched=True  + confidence >= 55  → "uncertain"
matched=True  + confidence <  55  → "unknown"
matched=False + confidence >= 55  → "uncertain"
matched=False + confidence <  55  → "unknown"
```

Where `matched = raw_cosine >= 0.45` (MATCH_THRESHOLD).

## Worked example

```
Input:
  raw_cosine    = 0.704
  face_quality  = 0.512
  track_seconds = 10.5s
  memory_boost  = 18.0
  margin        = 0.202
  is_masked     = False

Step 1 — Normalize:
  sim_norm     = (0.704 - 0.25) / 0.55 = 0.825
  quality_norm = 0.512
  track_norm   = min(10.5 / 1.5, 1) = 1.000
  memory_norm  = 18.0 / 20 = 0.900
  margin_norm  = 0.202 / 0.30 = 0.675
  mask_norm    = 0.0

Step 2 — Weighted sum:
  base = 0.65×0.825 + 0.15×0.512 + 0.10×1.000 + 0.05×0.900 + 0.05×0.675
       = 0.536 + 0.077 + 0.100 + 0.045 + 0.034
       = 0.792

Step 3 — Mask penalty:
  adjusted = 0.792 × (1 - 0.15×0) = 0.792

Step 4 — Scale:
  confidence = 1 + 99 × 0.792 = 79.4 → 79

Step 5 — Status:
  matched = 0.704 >= 0.45 → True
  confidence = 79 >= 70 → "known"
```

## Key behaviors

1. **Max-confidence gate**: Never downgrades. If new confidence < existing, update is skipped.
2. **Quality-gated throttle**: Skips if quality didn't improve by ≥0.10.
3. **Critical alerts always update**: Even if confidence would downgrade.

## See also
- [[Scoring Module]] — implementation code
- [[Recognition Agent]] — calls compute_confidence()
- [[All Thresholds]] — every threshold used
