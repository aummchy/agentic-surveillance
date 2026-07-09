# Recognition Pipeline Formulas

Complete reference for the face quality → matching → memory → recognition → policy decision pipeline.

---

## 1. Face Quality Score

**File:** `pipeline/quality_agent.py` · `pipeline/models.py:48-53`

### Raw measurements

| Metric | Function | Code |
|--------|----------|------|
| Blur | Laplacian variance of grayscale crop | `cv2.Laplacian(gray, cv2.CV_64F).var()` |
| Brightness | Mean HSV Value channel | `np.mean(hsv[:, :, 2])` |
| Area | Height × Width of face crop | `h * w` |

### Normalization

Shifted so a face at the minimum validity threshold scores near zero, and the full range is usable:

```
blur_norm    = min(max(blur_raw - QUALITY_BLUR_MIN, 0) / (QUALITY_BLUR_MAX - QUALITY_BLUR_MIN), 1.0)
bright_norm  = 1 - min(|brightness_raw - QUALITY_BRIGHTNESS_CENTER| / QUALITY_BRIGHTNESS_RADIUS, 1.0)
area_norm    = min(max(face_area - QUALITY_FACE_AREA_MIN, 0) / (QUALITY_AREA_MAX - QUALITY_FACE_AREA_MIN), 1.0)
```

| Metric | Norm at min threshold | Norm at max |
|--------|:--------------------:|:-----------:|
| Blur (40 → 350) | `(40-40)/310 = 0` | `1.0` |
| Brightness (center=145, radius=110) | `1-|V-145|/110` | `1.0` at V=145 |
| Area (1500 → 10000) | `(1500-1500)/8500 = 0` | `1.0` |

### Validity check vs. overall_score

- **`is_valid`** = whether the face is usable (passes minimum thresholds: blur ≥ 40, brightness 35–255, area ≥ 1200).
- **`overall_score`** = how good the face is relative to the full range. A barely-valid face scores ~0.03 ("low"). An excellent face scores 0.8+ ("high").

### Examples

| Condition | Blur | Bright | Area | `overall_score` | Label |
|-----------|:----:|:------:|:----:|:---------------:|:-----:|
| Barely valid | 40 | 35 | 1200 | `0×0.50 + 0.182×0.25 + 0×0.25 = 0.05` | low |
| Typical | 200 | 120 | 3000 | `0.516×0.50 + 0.227×0.25 + 0.176×0.25 = 0.37` | low |
| Good | 500 | 150 | 5400 | `1.0×0.50 + 0.055×0.25 + 0.459×0.25 = 0.63` | medium |
| Excellent | 1000 | 200 | 10000 | `1.0×0.50 + 0.591×0.25 + 1.0×0.25 = 0.90` | high |

### Validity gate

`is_valid = True` only when **all three** pass:

| Check | Threshold |
|-------|-----------|
| `blur_valid` | `blur_raw >= 40` |
| `bright_valid` | `35 <= brightness_raw <= 255` |
| `area_valid` | `face_area >= 1200` px² |

### Quality level labels (used in recognition reason string)

| `overall_score` | Label |
|:---------------:|-------|
| `>= 0.8` | "high face quality" |
| `>= 0.5` and `< 0.8` | "medium face quality" |
| `< 0.5` | **"low face quality"** |

### Live thresholds (from `config/config.jsonc`)

| Setting | Value | Purpose |
|---------|-------|---------|
| `QUALITY_BLUR_MIN` | 40 | Min Laplacian variance for scoring normalization |
| `QUALITY_BLUR_MAX` | 350 | Normalization cap for blur |
| `QUALITY_BRIGHTNESS_CENTER` | 145 | Center of brightness model (peak score) |
| `QUALITY_BRIGHTNESS_RADIUS` | 110 | Radius of brightness model |
| `QUALITY_FACE_AREA_MIN` | 1500 | Min face area for scoring normalization |
| `QUALITY_AREA_MAX` | 10000 | Normalization cap for area |
| `QUALITY_WEIGHT_BLUR` | 0.50 | Weight of blur in score |
| `QUALITY_WEIGHT_BRIGHT` | 0.25 | Weight of brightness |
| `QUALITY_WEIGHT_AREA` | 0.25 | Weight of area |
| `QUALITY_VALID_BLUR_MIN` | 40 | Validity gate: min blur |
| `QUALITY_VALID_BRIGHTNESS_MIN` | 35 | Validity gate: min brightness |
| `QUALITY_VALID_BRIGHTNESS_MAX` | 255 | Validity gate: max brightness |
| `QUALITY_VALID_FACE_AREA_MIN` | 1200 | Validity gate: min face area |

---

## 2. Vector Search (Matching)

**Files:** `utils/db_utils.py` · `agents/matching_agent.py`

### Atlas $vectorSearch

```
collection     = faces
index          = vector_index
path           = latest_embedding
dimensions     = 512
similarity     = cosine
numCandidates  = 150
limit          = 5
```

### Score conversion

Atlas returns `vectorSearchScore` in range `[0, 1]` where:
```
vectorSearchScore = (1 + raw_cosine) / 2
```

To recover raw cosine similarity:
```
raw_cosine = (atlas_score × 2) - 1
```

The `similarity_score` in `MatchResult` is the **raw cosine** (range `[-1, 1]`, but 0 to 1 for non-opposite embeddings). Only the best match (highest score) is used.

### Match threshold

```
MATCH_THRESHOLD = 0.45
```

If `similarity_score >= 0.45`, the match is considered a hit. Otherwise it's a miss.

### MatchResult fields

```python
MatchResult(
    person_id,          # MongoDB _id of matched face
    name,               # Person's display name
    role,               # "visitor", "authorized", etc.
    tags,               # ["auto_registered", "verified", "blacklist", "authorized"]
    similarity_score,   # Raw cosine similarity (0.0 to 1.0)
    matched,            # True if similarity >= MATCH_THRESHOLD
    verified,           # True if "verified" in tags
    alert_level,        # "low", "medium", "high", "critical"
)
```

---

## 3. Memory Agent — Confidence Boost

**File:** `agents/memory.py:149-184`

### Boost formula

```
boost = 0

# Returning visitor bonus
boost += min(10, visit_count × 2)          # +2 per visit, max +10

# Recency bonus
if days_since_last <= 7:   boost += 5
elif days_since_last <= 30: boost += 2

# Consistency bonus
if avg_similarity > 0.8:   boost += 3
elif avg_similarity > 0.6: boost += 1

# Pattern bonuses
if is_typical_time:  boost += 2
if is_typical_camera: boost += 1

# Penalty: current match much worse than usual
if current_similarity < avg_similarity × 0.7:  boost -= 5

return clamp(boost, -10, +20)
```

### Range

| Component | Min | Max |
|-----------|-----|-----|
| Visit count | 0 | +10 |
| Recency | 0 | +5 |
| Consistency | 0 | +3 |
| Typical time | 0 | +2 |
| Typical camera | 0 | +1 |
| Penalty | -5 | 0 |
| **Total (clamped)** | **-10** | **+20** |

### Typical time check

```python
current_hour = now.hour
common_hours = top 3 most frequent visit hours
is_typical_time = any(abs(current_hour - h) <= 2 for h in common_hours)
```

---

## 4. Recognition Agent — Confidence & Status

**File:** `agents/recognition.py` · `agents/scoring.py`

### Decision flow

```
Face detected → vector search → match_result (similarity, top2, margin)
                                       │
                                       ▼
                              Memory Agent → memory_boost (-10 to +20)
                                       │
                                       ▼
                          compute_confidence(raw_cosine, quality, ...)
                                       │
                                       ▼
                          confidence_status(confidence, matched)
                                       │
                                       ▼
                              status = "known" | "uncertain" | "unknown"
```

### Confidence formula (weighted normalization)

```
base = 0.65×sim_norm + 0.15×quality_norm + 0.10×track_norm + 0.05×memory_norm + 0.05×margin_norm
adjusted = base × (1 - 0.15×mask_norm)
confidence = int(round(1 + 99 × clip(adjusted, 0, 1)))
```

**File:** `agents/scoring.py:compute_confidence()`

### Normalization functions

| Component | Function | Input | Output | Notes |
|-----------|----------|-------|--------|-------|
| `sim_norm` | `normalize_cosine(raw)` | raw cosine `[-1, 1]` | `[0, 1]` | `(raw - 0.25) / (0.80 - 0.25)`, clipped |
| `quality_norm` | `normalize_quality(q)` | `float [0, 1]` or `None` | `[0, 1]` | `None`/`0` → fallback `0.50` |
| `track_norm` | `normalize_track_duration(secs)` | seconds | `[0, 1]` | `min(secs / 1.5, 1.0)` — saturates at 1.5s |
| `memory_norm` | `normalize_memory(boost)` | boost `[-10, +20]` | `[0, 1]` | `clip(boost, 0, 20) / 20` |
| `margin_norm` | `normalize_margin(margin)` | margin `[0, 1]` | `[0, 1]` | `clip(margin / 0.30, 1.0)`, `None` → `0.50` |
| `mask_norm` | `normalize_mask(masked)` | bool | `0.0` or `1.0` | `1.0` if masked |

### Weight breakdown

| Component | Weight | Normalization | Effective range |
|-----------|:------:|:-------------:|:---------------:|
| Similarity | 0.65 | `[0.25..0.80]` → `[0..1]` | 0–0.65 |
| Face Quality | 0.15 | `[0..1]` direct | 0–0.15 |
| Track Duration | 0.10 | `secs / 1.5` saturated | 0–0.10 |
| Memory Boost | 0.05 | `[0..20]` → `[0..1]` | 0–0.05 |
| Margin | 0.05 | `[0..0.30]` → `[0..1]` | 0–0.05 |
| **Base total** | **1.00** | | **0–1.00** |
| Mask penalty | ×(1 - 0.15×mask) | | ×1.0 or ×0.85 |

### Thresholds

| Threshold | Value | Effect |
|-----------|-------|--------|
| `MATCH_THRESHOLD` | 0.45 | `raw_cosine >= 0.45` → `matched=True` |
| `CONFIDENCE_KNOWN_MIN` | 70 | `matched + conf >= 70` → `"known"` |
| `CONFIDENCE_UNCERTAIN_MIN` | 55 | `conf >= 55` → `"uncertain"` else `"unknown"` |

### Status determination

```python
def confidence_status(confidence, matched):
    if matched:
        if confidence >= 70:  return "known"
        if confidence >= 55:  return "uncertain"
        return "unknown"
    if confidence >= 55:  return "uncertain"
    return "unknown"
```

### Worked example

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
       = 0.536 + 0.077 + 0.100 + 0.045 + 0.034
       = 0.792

  adjusted = 0.792 × (1 - 0.15×0) = 0.792
  confidence = 1 + 99 × 0.792 = 79
  status = "known"  (matched=True, 0.704 >= 0.45, confidence=79 >= 70)
```

### Match threshold gate

```
raw_cosine >= 0.45  →  matched = True
raw_cosine <  0.45  →  matched = False
```

Even if `matched=False`, confidence can still reach `"uncertain"` (>= 55) from quality + track + memory signals.

---

## 5. Policy Agent — Final Decision

**File:** `agents/policy.py`

### Rule priority (highest wins)

```
1. BLACKLIST   → status="blacklist",     alert="critical", should_alert=True
2. AUTHORIZED  → status="authorized",    alert="none",     should_alert=False
3. VERIFIED    → status="verified",      alert="none",     should_alert=False
4. KNOWN       → status="known_visitor", alert="low",      should_alert=False
5. MATCHED     → depends on similarity/confidence:
     - sim >= 0.85 or conf >= 80  → known_visitor
     - sim >= 0.45                → known_visitor
     - else                       → uncertain + register
6. HIDDEN      → status="intentionally_hidden", alert="high"
7. MASKED      → status="masked_unknown", alert depends on loitering
8. AFTER-HOURS → status="unknown", alert="high"
9. OFFICE-HOURS→ status="unknown", alert="medium"
```

### Policy decision fields

```python
DecisionResult(
    status,          # "verified" | "known_visitor" | "blacklist" | "unknown" | etc.
    alert_level,     # "none" | "low" | "medium" | "high" | "critical"
    should_alert,    # True = dispatch notification
    should_register, # True = store face in DB as new person
    reason,          # Human-readable explanation
)
```

---

## 6. Threshold Reference (all from `config/config.jsonc`)

### Matching

| Setting | Value | Effect |
|---------|-------|--------|
| `MATCH_THRESHOLD` | 0.45 | Min cosine similarity for a match |
| `DEDUP_SIMILARITY_THRESHOLD` | 0.40 | Min similarity to merge auto-registrations |

### Recognition

| Setting | Value | Effect |
|---------|-------|--------|
| `VERY_HIGH_SIMILARITY` | 0.90 | Definite known (Case 1) |
| `HIGH_CONFIDENCE_SIMILARITY` | 0.85 | Skip re-recognition in camera loop |
| `KNOWN_VISITOR_SIMILARITY` | 0.85 | Policy auto-escalates to known_visitor |
| `KNOWN_VISITOR_CONFIDENCE` | 80 | Policy auto-escalates if confidence >= 80 |
| `BORDERLINE_FACE_QUALITY` | 0.8 | Threshold for "high face quality" label |
| `MASK_CONFIDENCE_PENALTY` | 0.85 | Confidence multiplier for masked faces |

### Policy

| Setting | Value | Effect |
|---------|-------|--------|
| `OFFICE_HOURS_START` | 9 | After-hours alert if before this hour |
| `OFFICE_HOURS_END` | 17 | After-hours alert if after this hour |
| `OFFICE_DAYS` | Mon–Fri | Weekday check for after-hours |
| `LOITER_SECS` | 30 | Masked person loitering threshold |

### Face Detection

| Setting | Value | Effect |
|---------|-------|--------|
| `DET_SCORE_MIN` | 0.40 | Standard face detection threshold |
| `DET_SCORE_RELAXED` | 0.20 | Permissive fallback threshold |
| `EMBEDDING_DET_SCORE_MIN` | 0.40 | Min detection score to generate embedding |
| `RECOGNITION_INTERVAL_FRAMES` | 10 | Run recognition every N frames |

---

## 7. Pipeline Flow Diagram

```
Camera frame
    │
    ▼
YOLO person detection (track_persons)
    │
    ▼
ByteTrack tracking (track_state)
    │
    ▼
Progressive recognition (every 10 frames)
    │
    ├── Face detection (SCRFD via InsightFace)
    │       │
    │       ▼
    ├── Quality gate (compute_quality)
    │       │
    │       ▼
    ├── ArcFace embedding (512-dim)
    │       │
    │       ▼
    ├── Embedding cache check (cosine distance < 0.005? → reuse last match)
    │       │
    │       ▼
    ├── Atlas $vectorSearch → MatchResult
    │       │
    │       ▼
    ├── Memory Agent → confidence_boost (-10 to +20)
    │       │
    │       ▼
    ├── Recognition Agent → status + confidence (0-100)
    │       │
    │       ▼
    └── Policy Agent → DecisionResult
            │
            ▼
    ● CRITICAL alert? → dispatch immediately
    ● Otherwise → deferred to finalization
```


          Camera frame
               │
               ▼
     CLAHE contrast enhancement
               │
               ▼
     SCRFD face detection → det_score=0.87, bbox, landmarks
               │
               ▼
     Face alignment (affine transform via landmarks)
               │
               ▼
     ArcFace ResNet-100 → 512-dim embedding
               │
               ▼
     L2-normalize → ‖embedding‖ = 1
               │
               ▼
     Atlas $vectorSearch (HNSW index, cosine)
               │
               ▼
     (1 + dot_product) / 2  → Atlas score = 0.8615
               │
               ▼
     atlas_score_to_cosine → raw_cosine = (0.8615×2)-1 = 0.723
               │
               ▼
     compare_similarity(0.723 >= 0.45) → MatchResult(matched=True, similarity=0.723)
Want me to document this full flow in FORMULAS.md as a new section?