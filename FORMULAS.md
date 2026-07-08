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

### Design principle

Validity gates and scoring normalization are **separate concerns**:
- **Validity gates** control whether `set_best_face()` stores the crop on the track (boolean pass/fail).
- **Scoring normalization** controls the continuous `overall_score` that feeds into recognition confidence.

They do not need identical thresholds. The validity gate is slightly lenient; the scoring normalization penalizes small/weak faces continuously.

### Validity gates

`is_valid = True` only when **all three** pass:

| Check | Threshold | Purpose |
|-------|-----------|---------|
| `blur_valid` | `blur_raw >= 40` | Reject too-blurry crops |
| `bright_valid` | `35 <= brightness_raw <= 255` | Reject unusably dark crops |
| `area_valid` | `face_area >= 1200` px² (~35×35) | Reject tiny crops |

Faces that fail these gates are not stored via `set_best_face()`. The track's `best_face_score` stays 0.0, and recognition receives `None` → defaults to `0.50`.

### Scoring normalization

```python
blur_norm   = clip((blur_raw - 40)   / (350 - 40),     0, 1)
bright_norm = 1 - min(abs(brightness_raw - 145) / 110, 1)
area_norm   = clip((face_area - 1500) / (10000 - 1500), 0, 1)

overall_score = blur_norm × 0.50 + bright_norm × 0.25 + area_norm × 0.25
```

| Metric | Range | Norm at min | Norm at max | Weight |
|--------|:-----:|:-----------:|:-----------:|:------:|
| Blur | 40 → 350 | `(40-40)/310 = 0` | `1.0` | 50% |
| Brightness | 35 ↔ 255 (center=145, radius=110) | `1 - 110/110 = 0` | `1.0` | 25% |
| Area | 1500 → 10000 | `(1500-1500)/8500 = 0` | `1.0` | 25% |

### Examples

| Condition | Blur | Bright | Area | `overall_score` | Label |
|-----------|:----:|:------:|:----:|:---------------:|:-----:|
| Weak | 50 | 60 | 1500 | `0.03×0.50 + 0.23×0.25 + 0×0.25 = 0.07` | low |
| Typical indoor | 200 | 120 | 3000 | `0.52×0.50 + 0.77×0.25 + 0.18×0.25 = 0.48` | usable |
| Good | 350 | 145 | 5400 | `1.0×0.50 + 1.0×0.25 + 0.46×0.25 = 0.86` | good |
| Excellent | 500 | 145 | 10000 | `1.0×0.50 + 1.0×0.25 + 1.0×0.25 = 1.00` | good |

### Quality level labels (used in recognition reason string)

| `overall_score` | Label |
|:---------------:|-------|
| `>= 0.55` | "good face quality" |
| `>= 0.25` and `< 0.55` | "usable face quality" |
| `< 0.25` | **"low face quality"** |

### Live thresholds (from `config/config.jsonc`)

#### Validity gates

| Setting | Value | Purpose |
|---------|:-----:|---------|
| `QUALITY_VALID_BLUR_MIN` | 40 | Min Laplacian variance for valid face |
| `QUALITY_VALID_BRIGHTNESS_MIN` | 35 | Min HSV-V brightness |
| `QUALITY_VALID_BRIGHTNESS_MAX` | 255 | Max HSV-V brightness |
| `QUALITY_VALID_FACE_AREA_MIN` | 1200 | Min face area (px²) |

#### Scoring normalization

| Setting | Value | Purpose |
|---------|:-----:|---------|
| `QUALITY_BLUR_MIN` | 40 | Blur normalization floor |
| `QUALITY_BLUR_MAX` | 350 | Blur normalization cap |
| `QUALITY_BRIGHTNESS_CENTER` | 145 | Brightness ideal center point |
| `QUALITY_BRIGHTNESS_RADIUS` | 110 | Brightness falloff radius |
| `QUALITY_FACE_AREA_MIN` | 1500 | Area normalization floor |
| `QUALITY_AREA_MAX` | 10000 | Area normalization cap |
| `QUALITY_WEIGHT_BLUR` | 0.50 | Weight of blur in score |
| `QUALITY_WEIGHT_BRIGHT` | 0.25 | Weight of brightness |
| `QUALITY_WEIGHT_AREA` | 0.25 | Weight of area |

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

**Files:** `agents/scoring.py` (formulas), `agents/recognition.py` (orchestration), `config/settings.py` (tunables)

### Architecture

```
  similarity threshold → matched (yes/no)     ← identity gate
  unified formula → confidence (1–100)         ← scoring
  confidence tiers + match gate → status      ← decision
```

### Unified confidence formula

```python
# From agents/scoring.py (all tunables from config/settings.py)

sim_norm     = clip((raw_cosine - 0.25) / 0.55, 0, 1)
quality_norm = 0.5 if missing else clip(face_quality, 0, 1)
track_norm   = clip(track_seconds / 1.5, 0, 1)
memory_norm  = clip(memory_boost, 0, 20) / 20
mask_norm    = 1.0 if masked else 0.0

base     = 0.70*sim + 0.15*quality + 0.10*track + 0.05*memory
adjusted = base * (1.0 - 0.15 * mask_norm)
confidence = round(1 + 99 * clip(adjusted, 0, 1))
```

### Status logic

```
matched = similarity >= MATCH_THRESHOLD (0.45)

if matched (sim >= 0.45):
    confidence >= 70 → "known"
    confidence >= 55 → "uncertain"
    else            → "unknown"

if not matched (sim < 0.45):
    confidence >= 55 → "uncertain"
    else            → "unknown"
```

### Component weights

| Component | Weight | Range (normalized) | Max contribution |
|-----------|:------:|:------------------:|:----------------:|
| Cosine similarity | 70% | [0, 1] | 70% of base |
| Face quality | 15% | [0, 1] (0.5 default if missing) | 15% of base |
| Track duration | 10% | [0, 1] (saturates at 1.5s) | 10% of base |
| Memory boost | 5% | [0, 1] (memory_boost 0→0, 20→1) | 5% of base |
| Mask penalty | — | 0 or 1 (boolean) | up to −15% of base |

### Confidence scale

| Tier | Range | Meaning |
|------|:-----:|---------|
| Very confident | 85–100 | Auto‑mark attendance |
| Likely correct | 70–84 | Auto‑mark if stable across frames |
| Borderline | 55–69 | Show as uncertain, do not auto‑mark |
| Low confidence | 1–54 | No mark, no action |

### Worked example (your log)

```
Input:
  raw_cosine    = 0.7314
  face_quality  = 0.0 (missing → defaults to 0.5)
  track_duration = 0.7s
  memory_boost  = 5.0
  is_masked     = False

Normalization:
  sim_norm     = (0.7314 - 0.25) / 0.55 = 0.875
  quality_norm = 0.5          (missing quality)
  track_norm   = 0.7 / 1.5   = 0.467
  memory_norm  = 5.0 / 20.0  = 0.25
  mask_norm    = 0.0

Base:
  base = 0.70(0.875) + 0.15(0.5) + 0.10(0.467) + 0.05(0.25)
       = 0.6125 + 0.075 + 0.0467 + 0.0125
       = 0.7467

Adjusted: 0.7467 * 1.0 = 0.7467
Confidence = round(1 + 99 × 0.7467) = 75

Result:
  status     = "known"     (matched + confidence >= 70)
  confidence = 75
  reason     = "Decision: known. raw cosine=0.731, above match threshold 0.45. face quality unavailable."
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
