# All Thresholds

> Every threshold, limit, and boundary value in the system with its purpose and effect.

## Detection & Tracking

| Threshold | Default | Purpose | File |
|-----------|---------|---------|------|
| `PERSON_CONF_THRESHOLD` | 0.40 | YOLO person detection confidence minimum | settings.py |
| `DET_SCORE_MIN` | 0.40 | Standard face detection threshold | settings.py |
| `DET_SCORE_RELAXED` | 0.20 | Relaxed face detection fallback (must be ≤ DET_SCORE_MIN) | settings.py |
| `EMBEDDING_DET_SCORE_MIN` | 0.40 | Minimum det_score to generate embedding | settings.py |
| `TRACK_TIMEOUT_SECS` | 15.0 | Person gone this long = track ends | settings.py |
| `MAX_TRACK_SECS` | 300 | Force-finalize after this many seconds | settings.py |
| `RECOGNITION_INTERVAL_FRAMES` | 20 | Run progressive recognition every N frames | settings.py |
| `MIN_TRACK_FRAMES` | 15 | Min frames before "hidden" classification | settings.py |

## Matching

| Threshold | Default | Purpose | File |
|-----------|---------|---------|------|
| `MATCH_THRESHOLD` | 0.45 | Raw cosine for face match (max recommended) | settings.py |
| `DEDUP_SIMILARITY_THRESHOLD` | 0.5 | Merge auto-registrations above this similarity | settings.py |

## Recognition

| Threshold | Default | Purpose | File |
|-----------|---------|---------|------|
| `HIGH_CONFIDENCE_SIMILARITY` | 0.85 | Skip re-recognition in camera loop | settings.py |
| `KNOWN_VISITOR_SIMILARITY` | 0.85 | Policy auto-escalates to known_visitor | settings.py |
| `KNOWN_VISITOR_CONFIDENCE` | 80 | Policy auto-escalates if confidence >= 80 | settings.py |
| `CONFIDENCE_KNOWN_MIN` | 70 | matched + conf >= 70 → "known" | settings.py |
| `CONFIDENCE_UNCERTAIN_MIN` | 55 | conf >= 55 → "uncertain" | settings.py |
| `MIN_QUALITY_IMPROVEMENT` | 0.10 | Min quality improvement to trigger re-recognition | settings.py |
| `RESCAN_INTERVAL_SECS` | 3 | Time between rescan attempts for unknowns | settings.py |
| `MAX_RESCAN_ATTEMPTS` | 3 | Max rescan attempts per track | settings.py |

## Quality gates

| Threshold | Default | Purpose | File |
|-----------|---------|---------|------|
| `QUALITY_VALID_BLUR_MIN` | 40 | Validity gate: min Laplacian variance | settings.py |
| `QUALITY_VALID_BRIGHTNESS_MIN` | 35 | Validity gate: min HSV-V brightness | settings.py |
| `QUALITY_VALID_BRIGHTNESS_MAX` | 255 | Validity gate: max HSV-V brightness | settings.py |
| `QUALITY_VALID_FACE_AREA_MIN` | 1200 | Validity gate: min face area px² | settings.py |

## Quality scoring normalization

| Threshold | Default | Purpose | File |
|-----------|---------|---------|------|
| `QUALITY_BLUR_MIN` | 40 | Blur normalization floor | settings.py |
| `QUALITY_BLUR_MAX` | 350 | Blur normalization cap | settings.py |
| `QUALITY_BRIGHTNESS_CENTER` | 145 | Brightness model center (peak score) | settings.py |
| `QUALITY_BRIGHTNESS_RADIUS` | 110 | Brightness model radius | settings.py |
| `QUALITY_FACE_AREA_MIN` | 1500 | Area normalization floor | settings.py |
| `QUALITY_AREA_MAX` | 10000 | Area normalization cap | settings.py |
| `QUALITY_WEIGHT_BLUR` | 0.50 | Blur weight in overall score | settings.py |
| `QUALITY_WEIGHT_BRIGHT` | 0.25 | Brightness weight | settings.py |
| `QUALITY_WEIGHT_AREA` | 0.25 | Area weight | settings.py |
| `REGISTRATION_QUALITY_MIN` | 0.6 | Min quality for auto-registration | settings.py |

## Confidence formula normalization

| Threshold | Default | Purpose | File |
|-----------|---------|---------|------|
| `SIM_NORM_MIN` | 0.25 | Similarity normalization floor | settings.py |
| `SIM_NORM_MAX` | 0.80 | Similarity normalization cap | settings.py |
| `TRACK_SATURATION_SECS` | 1.5 | Track duration normalization cap | settings.py |
| `MEMORY_NORM_MAX` | 20.0 | Memory boost normalization cap | settings.py |
| `MARGIN_NORM_MAX` | 0.30 | Margin normalization cap | settings.py |
| `MASK_PENALTY_MAX` | 0.15 | Max mask confidence penalty | settings.py |
| `DEFAULT_FACE_QUALITY` | 0.50 | Fallback quality when unavailable | settings.py |

## Confidence formula weights

| Weight | Default | Component |
|--------|---------|-----------|
| `WEIGHT_SIMILARITY` | 0.65 | Face similarity |
| `WEIGHT_QUALITY` | 0.15 | Face quality |
| `WEIGHT_TRACK` | 0.10 | Track duration |
| `WEIGHT_MEMORY` | 0.05 | Memory boost |
| `WEIGHT_MARGIN` | 0.05 | Match margin |

## Policy

| Threshold | Default | Purpose | File |
|-----------|---------|---------|------|
| `OFFICE_HOURS_START` | 9 | After-hours alert if before this hour | settings.py |
| `OFFICE_HOURS_END` | 17 | After-hours alert if after this hour | settings.py |
| `OFFICE_DAYS` | [0,1,2,3,4] | Mon–Fri for after-hours check | settings.py |
| `LOITER_SECS` | 30 | Masked person loitering threshold | settings.py |
| `MASK_RATIO_THRESHOLD` | 0.30 | Geometric mask detection ratio | settings.py |

## Visibility

| Threshold | Default | Purpose | File |
|-----------|---------|---------|------|
| `VISIBLE_FACE_RATIO` | 0.025 | face_area/person_area ≥ this = "visible" | settings.py |
| `PARTIAL_FACE_RATIO` | 0.010 | face_area/person_area ≥ this = "partial" | settings.py |

## Alerting

| Threshold | Default | Purpose | File |
|-----------|---------|---------|------|
| `ALERT_COOLDOWN_SECS` | 60 | Min seconds between same-type alerts | settings.py |

## Vector search

| Threshold | Default | Purpose | File |
|-----------|---------|---------|------|
| `VECTOR_SEARCH_CANDIDATES` | 150 | Atlas HNSW candidate pool | settings.py |
| `SCAN_LIMIT` | 500 | Max docs for Python cosine fallback | settings.py |

## Embedding history

| Threshold | Default | Purpose | File |
|-----------|---------|---------|------|
| `EMBEDDING_HISTORY_CAP` | 25 | Max past embeddings per person | settings.py |

## JPEG compression

| Threshold | Default | Purpose | File |
|-----------|---------|---------|------|
| `JPEG_QUALITY_STORE` | 85 | Quality for stored images | settings.py |
| `JPEG_QUALITY_BROADCAST` | 90 | Quality for WebSocket broadcast | settings.py |

## CLAHE

| Threshold | Default | Purpose | File |
|-----------|---------|---------|------|
| `CLAHE_CLIP_LIMIT` | 2.0 | CLAHE contrast clip limit | settings.py |
| `CLAHE_TILE_SIZE` | 8 | CLAHE tile grid size | settings.py |

## See also
- [[All Config Settings]] — every configuration option
- [[Confidence Scoring]] — how thresholds affect confidence
- [[Policy Rules]] — how policy thresholds trigger decisions
