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
bright_norm  = brightness_raw / 255.0
area_norm    = min(max(face_area - QUALITY_FACE_AREA_MIN, 0) / (QUALITY_AREA_MAX - QUALITY_FACE_AREA_MIN), 1.0)
```

| Metric | Norm at min threshold | Norm at max |
|--------|:--------------------:|:-----------:|
| Blur (15 → 1000) | `(15-15)/985 = 0` | `1.0` |
| Brightness (0 → 255) | `0/255 = 0` | `1.0` |
| Area (800 → 10000) | `(800-800)/9200 = 0` | `1.0` |

### Validity check vs. overall_score

- **`is_valid`** = whether the face is usable (passes minimum thresholds: blur ≥ 15, brightness 30–240, area ≥ 800).
- **`overall_score`** = how good the face is relative to the full range. A barely-valid face scores ~0.03 ("low"). An excellent face scores 0.8+ ("high").

### Examples

| Condition | Blur | Bright | Area | `overall_score` | Label |
|-----------|:----:|:------:|:----:|:---------------:|:-----:|
| Barely valid | 15 | 30 | 800 | `0×0.60 + 0.118×0.25 + 0×0.15 = 0.03` | low |
| Typical | 200 | 120 | 3000 | `0.188×0.60 + 0.471×0.25 + 0.239×0.15 = 0.27` | low |
| Good | 500 | 150 | 5400 | `0.492×0.60 + 0.588×0.25 + 0.500×0.15 = 0.52` | medium |
| Excellent | 1000 | 200 | 10000 | `1.0×0.60 + 0.784×0.25 + 1.0×0.15 = 0.95` | high |

### Validity gate

`is_valid = True` only when **all three** pass:

| Check | Threshold |
|-------|-----------|
| `blur_valid` | `blur_raw >= 15` |
| `bright_valid` | `30 <= brightness_raw <= 240` |
| `area_valid` | `face_area >= 800` px² (~28×28) |

### Quality level labels (used in recognition reason string)

| `overall_score` | Label |
|:---------------:|-------|
| `>= 0.8` | "high face quality" |
| `>= 0.5` and `< 0.8` | "medium face quality" |
| `< 0.5` | **"low face quality"** |

### Live thresholds (from `config/config.jsonc`)

| Setting | Value | Purpose |
|---------|-------|---------|
| `QUALITY_BLUR_MIN` | 15 | Min Laplacian variance for valid face |
| `QUALITY_BRIGHTNESS_MIN` | 30 | Min HSV-V brightness |
| `QUALITY_BRIGHTNESS_MAX` | 240 | Max HSV-V brightness |
| `QUALITY_FACE_AREA_MIN` | 800 | Min face area (px²) |
| `QUALITY_BLUR_MAX` | 1000 | Normalization cap for blur |
| `QUALITY_AREA_MAX` | 10000 | Normalization cap for area |
| `QUALITY_WEIGHT_BLUR` | 0.60 | Weight of blur in score |
| `QUALITY_WEIGHT_BRIGHT` | 0.25 | Weight of brightness |
| `QUALITY_WEIGHT_AREA` | 0.15 | Weight of area |

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

**File:** `agents/recognition.py`

### Decision tree (4 cases)

```
                    ┌─────────────────────────────┐
                    │     similarity >= 0.90?      │
                    │     (VERY_HIGH_SIMILARITY)   │
                    └──────────┬──────────────────┘
                               │ YES
                    ┌──────────▼──────────┐
                    │  Case 1: "known"    │
                    │  confidence =       │
                    │  70 + (sim-0.9)×250 │
                    │  + memory_boost     │
                    │  capped at 95       │
                    └─────────────────────┘
                               │ NO
                    ┌──────────▼──────────────────┐
                    │     similarity >= 0.45?      │
                    │     (MATCH_THRESHOLD)        │
                    └──────────┬──────────────────┘
                               │ YES
                    ┌──────────▼──────────────────┐
                    │  Case 2: _compute_confidence │
                    │                              │
                    │  if confidence >= 70: known  │
                    │  else:             uncertain │
                    └──────────────────────────────┘
                               │ NO
                    ┌──────────▼──────────────────┐
                    │ face_quality >= 0.8 AND     │
                    │ similarity >= 0.36 (0.45×0.8)│
                    └──────────┬──────────────────┘
                               │ YES                      │ NO
                    ┌──────────▼──────────┐   ┌───────────▼───────────┐
                    │  Case 3: "uncertain"│   │  Case 4: "unknown"    │
                    │  confidence =       │   │  confidence =         │
                    │  40 + sim×30 + boost│   │  max(60, 100-sim×100) │
                    └─────────────────────┘   └───────────────────────┘
```

### Case 1: Very high similarity (`similarity >= 0.90`)

```
status     = "known"
confidence = min(95, 70 + (similarity - 0.90) × 250 + memory_boost)
```

Example: `sim=0.92, boost=10` → `70 + 0.02×250 + 10 = 85`

### Case 2: Good similarity (`0.45 <= similarity < 0.90`) — Main formula

```python
def _compute_confidence(similarity, face_quality, track_duration, is_masked, memory_boost):
    sim_score    = min(60, (similarity - 0.45) / (1.0 - 0.45) × 60)     # 0-60 points
    quality_score = face_quality × 25                                     # 0-25 points
    duration_score = min(15, track_duration / 10)                        # 0-15 points
    confidence = sim_score + quality_score + duration_score + memory_boost

    if is_masked:
        confidence *= 0.85  # MASK_CONFIDENCE_PENALTY

    return clamp(confidence, 0, 100)
```

#### Component breakdown

| Component | Range | Derived from |
|-----------|:-----:|--------------|
| `sim_score` | 0–60 | Linear map of `[0.45, 1.0]` → `[0, 60]` |
| `quality_score` | 0–25 | `face_quality` × 25 |
| `duration_score` | 0–15 | `min(15, track_duration / 10)` = saturates at 150s |
| `memory_boost` | -10–+20 | From Memory Agent |
| **Subtotal** | **-10–+120** | before mask penalty |
| *Mask penalty* | ×0.85 | if `is_masked=True` |
| **Final** | **clamped 0–100** | |

#### Thresholds

| Rule | Value |
|------|-------|
| `confidence >= 70` | status = `"known"` |
| `confidence < 70` | status = `"uncertain"` |

#### Worked example (your log)

```
Input:
  similarity    = 0.754
  face_quality  = 0.37 (low, < 0.5)   ← after normalization fix, same raw values
  track_duration = 2s (approx)          would produce similar face_quality
  memory_boost  = 18.0
  is_masked     = False

Calculation:
  sim_score      = min(60, (0.754 - 0.45) / 0.55 × 60) = min(60, 33.16) = 33.16
  quality_score  = 0.37 × 25    = 9.25
  duration_score = min(15, 2/10) = min(15, 0.2) = 0.2
  confidence     = 33.16 + 9.25 + 0.2 + 18.0 = 60.61

Result:
  confidence     = 60.61
  status         = "uncertain"  (confidence < 70)
  reason         = "Borderline match — needs verification. similarity=75.40% above threshold. low face quality."
```

### Case 3: Below threshold, good face quality (`similarity < 0.45` but `>= 0.36` and `face_quality >= 0.8`)

```
status     = "uncertain"
confidence = 40 + similarity × 30 + memory_boost
```

### Case 4: Low similarity (`similarity < 0.36`)

```
status     = "unknown"
confidence = max(60, 100 - similarity × 100)
```

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
