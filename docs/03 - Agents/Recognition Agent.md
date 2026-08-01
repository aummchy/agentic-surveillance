# Recognition Agent

> Makes structured recognition decisions based on multiple signals: face similarity, quality, track duration, memory context, and margin. Produces a confidence score (1-100) and status (known/uncertain/unknown).

**File**: `agents/recognition.py` (175 lines)

## Role in system

```
Matching Agent → similarity_score (raw cosine)
Memory Agent   → confidence_boost (-10 to +20)
                    │
                    ▼
            Recognition Agent
            (agents/recognition.py + agents/scoring.py)
                    │
                    ▼
            status + confidence (1-100)
                    │
                    ▼
            Policy Agent → decision
```

## Class: `RecognitionAgent`

### Input to `run()`
```python
{
    "similarity": 0.723,           # raw cosine from ArcFace
    "is_masked": False,
    "face_quality": 0.65,          # 0-1 from quality scoring
    "track_duration": 12.5,        # seconds
    "memory_context": {            # from Memory Agent
        "visit_count": 5,
        "confidence_boost": 12,
        "is_typical_time": True
    },
    "top2": 0.499,                 # second-best similarity
    "margin": 0.224,               # top1 - top2
    "name": "John Doe",            # for display
    "track_id": "cam_01_..."
}
```

### Processing

```
1. Extract inputs
2. Check is_match(similarity)  → True if raw_cosine >= 0.45
3. compute_confidence(raw_cosine, quality, track_seconds, memory_boost, is_masked, margin)
4. confidence_status(confidence, matched)  → "known" / "uncertain" / "unknown"
5. Build reason string
6. Return RecognitionResult.to_dict()
```

### Output
```python
{
    "status": "known",
    "confidence": 79,
    "similarity": 0.723,
    "face_quality": 0.65,
    "is_masked": False,
    "track_duration": 12.5,
    "reason": "Decision: known. raw cosine=0.723, above match threshold 0.45. good face quality. returning visitor (5 previous visits)."
}
```

## Confidence computation

**File**: `agents/scoring.py:181-233`

The confidence formula uses **weighted normalization** — 5 components normalized to [0,1], combined with weights, then scaled to [1-100]:

```
base = 0.65×sim_norm + 0.15×quality_norm + 0.10×track_norm + 0.05×memory_norm + 0.05×margin_norm
adjusted = base × (1 - 0.15×mask_norm)
confidence = int(round(1 + 99 × clip(adjusted, 0, 1)))
```

### Normalization functions

| Component | Function | Input → Output | Notes |
|-----------|----------|---------------|-------|
| `sim_norm` | `normalize_cosine(raw)` | raw_cosine [0.25..0.80] → [0..1] | Shifted: 0.25=0, 0.80=1 |
| `quality_norm` | `normalize_quality(q)` | quality [0..1] → [0..1] | None/0 → fallback 0.50 |
| `track_norm` | `normalize_track_duration(secs)` | seconds → [0..1] | Saturates at 1.5s |
| `memory_norm` | `normalize_memory(boost)` | boost [0..20] → [0..1] | Clamped to [0,20] |
| `margin_norm` | `normalize_margin(margin)` | margin [0..0.30] → [0..1] | None → 0.50 |
| `mask_norm` | `normalize_mask(masked)` | bool → 0.0 or 1.0 | 1.0 if masked |

### Status mapping

```
matched=True  + confidence >= 70  → "known"       (CONFIDENCE_KNOWN_MIN)
matched=True  + confidence >= 55  → "uncertain"    (CONFIDENCE_UNCERTAIN_MIN)
matched=True  + confidence <  55  → "unknown"
matched=False + confidence >= 55  → "uncertain"
matched=False + confidence <  55  → "unknown"
```

### Key behaviors

1. **Max-confidence gate**: Recognition never downgrades. If new confidence < existing track.confidence, update is skipped. Critical alerts always update.

2. **Quality-gated throttle**: Skips recognition if face quality didn't improve by ≥0.10 (`MIN_QUALITY_IMPROVEMENT`) since last recognition.

3. **Mask penalty**: Masked faces get confidence multiplied by 0.85 (15% penalty).

## Worked example

```
Input:
  raw_cosine    = 0.704
  face_quality  = 0.512
  track_seconds = 10.5s
  memory_boost  = 18.0
  margin        = 0.202
  is_masked     = False

Normalization:
  sim_norm     = (0.704 - 0.25) / 0.55 = 0.825
  quality_norm = 0.512
  track_norm   = min(10.5 / 1.5, 1) = 1.000
  memory_norm  = 18.0 / 20 = 0.900
  margin_norm  = 0.202 / 0.30 = 0.675
  mask_norm    = 0.0

Weighted sum:
  base = 0.65×0.825 + 0.15×0.512 + 0.10×1.000 + 0.05×0.900 + 0.05×0.675
       = 0.536 + 0.077 + 0.100 + 0.045 + 0.034 = 0.792

  adjusted = 0.792 × (1 - 0) = 0.792
  confidence = 1 + 99 × 0.792 = 79
  status = "known" (matched=True, 0.704 >= 0.45, confidence=79 >= 70)
```

## See also
- [[Confidence Scoring]] — detailed formula breakdown
- [[Scoring Module]] — normalization functions
- [[Policy Agent]] — uses confidence for final decision
- [[Memory Agent]] — provides confidence_boost
