# All Config Settings

> Every configuration option in the system, where it's defined, and what it does.

## Configuration sources

1. **`.env`** — Secrets only (API keys, URIs, passwords)
2. **`config/config.jsonc`** — Tunables (thresholds, weights, intervals)
3. **`config/settings.py`** — Hardcoded defaults

Priority: `.env` > `config.jsonc` > defaults

## All settings

### Secrets (env-only)

| Setting | Default | Purpose |
|---------|---------|---------|
| `MONGODB_URI` | (required) | Atlas connection string |
| `MONGODB_DATABASE` | `surveillance` | Database name |
| `MONGODB_COLLECTION` | `faces` | Faces collection |
| `MONGODB_EVENTS_COLLECTION` | `events` | Events collection |
| `CLOUDINARY_CLOUD_NAME` | `""` | Cloudinary cloud |
| `CLOUDINARY_API_KEY` | `""` | Cloudinary key |
| `CLOUDINARY_API_SECRET` | `""` | Cloudinary secret |
| `ALERT_WEBHOOK_URL` | `""` | Webhook URL |
| `SMTP_HOST` | `""` | SMTP host |
| `SMTP_PORT` | `587` | SMTP port |
| `SMTP_USER` | `""` | SMTP user |
| `SMTP_PASS` | `""` | SMTP password |
| `ALERT_EMAIL_TO` | `""` | Email recipient |
| `TWILIO_ACCOUNT_SID` | `""` | Twilio SID |
| `TWILIO_AUTH_TOKEN` | `""` | Twilio token |
| `TWILIO_FROM` | `""` | Twilio sender |
| `ALERT_SMS_TO` | `""` | SMS recipient |

### Pipeline / Model

| Setting | Default | Purpose |
|---------|---------|---------|
| `YOLO_MODEL` | `models/yolov8s_openvino_model/` | YOLO model path |
| `YOLO_DEVICE` | `cpu` | YOLO device |
| `INSIGHTFACE_MODEL` | `buffalo_l` | InsightFace model pack |
| `INSIGHTFACE_DET_SIZE` | `640` | Face detection input size |
| `INSIGHTFACE_PROVIDER` | `CPUExecutionProvider` | ONNX provider |
| `OPENVINO_DEVICE` | `GPU` | OpenVINO accelerator |

### Camera

| Setting | Default | Purpose |
|---------|---------|---------|
| `CAMERA_SOURCE` | `videos/low_4.mp4` | RTSP URL or file path |
| `CAMERA_INDEX` | `0` | Webcam device index |
| `CAMERA_ID` | `cam_01` | Camera identifier |
| `CAMERA_BACKEND` | `""` | Video backend |
| `FRAME_WIDTH` | `1920` | Capture width |
| `FRAME_HEIGHT` | `1080` | Capture height |
| `FRAME_SKIP` | `2` | Skip every Nth frame |

### Detection & Tracking

| Setting | Default | Purpose |
|---------|---------|---------|
| `PERSON_CONF_THRESHOLD` | 0.40 | YOLO confidence |
| `TRACK_TIMEOUT_SECS` | 15.0 | Track expiry |
| `MAX_TRACK_SECS` | 300 | Max track lifetime |
| `DET_SCORE_MIN` | 0.40 | Standard face detection |
| `DET_SCORE_RELAXED` | 0.20 | Relaxed face detection |
| `EMBEDDING_DET_SCORE_MIN` | 0.40 | Min for embedding generation |
| `ENABLE_FULL_FRAME_FALLBACK` | False | Full-frame SCRFD scan fallback |

### Recognition

| Setting | Default | Purpose |
|---------|---------|---------|
| `RECOGNITION_INTERVAL_FRAMES` | 20 | Progressive recognition interval |
| `MIN_QUALITY_IMPROVEMENT` | 0.10 | Quality throttle delta |
| `RESCAN_INTERVAL_SECS` | 3 | Rescan interval |
| `MAX_RESCAN_ATTEMPTS` | 3 | Max rescans |

### Quality

| Setting | Default | Purpose |
|---------|---------|---------|
| `QUALITY_VALID_BLUR_MIN` | 40 | Validity gate |
| `QUALITY_VALID_BRIGHTNESS_MIN` | 35 | Validity gate |
| `QUALITY_VALID_BRIGHTNESS_MAX` | 255 | Validity gate |
| `QUALITY_VALID_FACE_AREA_MIN` | 1200 | Validity gate |
| `QUALITY_BLUR_MIN` | 40 | Scoring normalization |
| `QUALITY_BLUR_MAX` | 350 | Scoring normalization |
| `QUALITY_BRIGHTNESS_CENTER` | 145 | Scoring normalization |
| `QUALITY_BRIGHTNESS_RADIUS` | 110 | Scoring normalization |
| `QUALITY_FACE_AREA_MIN` | 1500 | Scoring normalization |
| `QUALITY_AREA_MAX` | 10000 | Scoring normalization |
| `QUALITY_WEIGHT_BLUR` | 0.50 | Score weight |
| `QUALITY_WEIGHT_BRIGHT` | 0.25 | Score weight |
| `QUALITY_WEIGHT_AREA` | 0.25 | Score weight |
| `REGISTRATION_QUALITY_MIN` | 0.6 | Auto-registration gate |

### Matching

| Setting | Default | Purpose |
|---------|---------|---------|
| `MATCH_THRESHOLD` | 0.45 | Match gate (max 0.45) |
| `DEDUP_SIMILARITY_THRESHOLD` | 0.5 | Dedup gate |

### Confidence

| Setting | Default | Purpose |
|---------|---------|---------|
| `WEIGHT_SIMILARITY` | 0.65 | Component weight |
| `WEIGHT_QUALITY` | 0.15 | Component weight |
| `WEIGHT_TRACK` | 0.10 | Component weight |
| `WEIGHT_MEMORY` | 0.05 | Component weight |
| `WEIGHT_MARGIN` | 0.05 | Component weight |
| `SIM_NORM_MIN` | 0.25 | Normalization floor |
| `SIM_NORM_MAX` | 0.80 | Normalization cap |
| `TRACK_SATURATION_SECS` | 1.5 | Track norm cap |
| `MEMORY_NORM_MAX` | 20.0 | Memory norm cap |
| `MARGIN_NORM_MAX` | 0.30 | Margin norm cap |
| `MASK_PENALTY_MAX` | 0.15 | Mask penalty |
| `DEFAULT_FACE_QUALITY` | 0.50 | Fallback quality |
| `CONFIDENCE_KNOWN_MIN` | 70 | "known" threshold |
| `CONFIDENCE_UNCERTAIN_MIN` | 55 | "uncertain" threshold |

### Recognition thresholds

| Setting | Default | Purpose |
|---------|---------|---------|
| `HIGH_CONFIDENCE_SIMILARITY` | 0.85 | Skip re-recognition |
| `KNOWN_VISITOR_SIMILARITY` | 0.85 | Policy escalation |
| `KNOWN_VISITOR_CONFIDENCE` | 80 | Policy escalation |

### Policy

| Setting | Default | Purpose |
|---------|---------|---------|
| `OFFICE_HOURS_START` | 9 | Office start |
| `OFFICE_HOURS_END` | 17 | Office end |
| `OFFICE_DAYS` | [0,1,2,3,4] | Mon–Fri |
| `LOITER_SECS` | 30 | Loitering threshold |
| `MASK_RATIO_THRESHOLD` | 0.30 | Mask detection ratio |

### Visit Dedup

| Setting | Default | Purpose |
|---------|---------|---------|
| `MIN_VISIT_GAP_SECS` | 60 | Suppress visit if same person recorded within this window |

### Visibility

| Setting | Default | Purpose |
|---------|---------|---------|
| `VISIBLE_FACE_RATIO` | 0.025 | "visible" threshold |
| `PARTIAL_FACE_RATIO` | 0.010 | "partial" threshold |

### Alerting

| Setting | Default | Purpose |
|---------|---------|---------|
| `ALERT_CHANNELS` | ["console","webhook"] | Alert channels |
| `ALERT_COOLDOWN_SECS` | 60 | Dedup cooldown |

### Vector search

| Setting | Default | Purpose |
|---------|---------|---------|
| `VECTOR_SEARCH_CANDIDATES` | 150 | Atlas candidate pool |
| `SCAN_LIMIT` | 500 | Python fallback limit |
| `EMBEDDING_HISTORY_CAP` | 25 | Embedding FIFO cap |

### JPEG

| Setting | Default | Purpose |
|---------|---------|---------|
| `JPEG_QUALITY_STORE` | 85 | Stored image quality |
| `JPEG_QUALITY_BROADCAST` | 90 | Broadcast quality |

### CLAHE

| Setting | Default | Purpose |
|---------|---------|---------|
| `CLAHE_CLIP_LIMIT` | 2.0 | Contrast clip |
| `CLAHE_TILE_SIZE` | 8 | Tile grid size |

### LLM

| Setting | Default | Purpose |
|---------|---------|---------|
| `LLM_ENABLED` | False | Enable Ollama LLM features |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `gemma3:4b` | Model name |
| `OLLAMA_TIMEOUT` | 30 | Request timeout |

### Debug

| Setting | Default | Purpose |
|---------|---------|---------|
| `DEBUG_RECOGNITION` | True | Verbose recognition logs |
| `DEBUG_FACE_CROPS` | True | Save face crops |
| `DEBUG_DUPLICATE_BOXES` | True | Duplicate detection logs |
| `PERFORMANCE_STATS` | False | FPS + timing logs |
| `ENABLE_CALC_LOG` | True | Confidence calculation log |
| `CALC_LOG_MAX_SIZE_MB` | 10 | Calc log max size |

## See also
- [[All Thresholds]] — threshold details
- [[Configuration System]] — how settings are loaded
- [[Environment Variables]] — env-only settings
