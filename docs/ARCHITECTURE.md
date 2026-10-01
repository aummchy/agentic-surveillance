# ARCHITECTURE.md — agentic_ai_singlecam

## Tech Stack & How It Connects

| Component | What we use | How |
|-----------|------------|-----|
| **Detection** | YOLOv8 (`yolov8s.pt` or OpenVINO IR) via Ultralytics | Singleton model in `pipeline/tracker.py:track_persons()`. OpenVINO IR export for GPU: `yolo export model=yolov8s.pt format=openvino half=True`, then set `YOLO_MODEL=yolov8s_openvino_model/`, `YOLO_DEVICE=intel:GPU`. ~8× speedup (128ms→16ms) on Arc iGPU. |
| **Tracking** | ByteTrack (custom `config/bytetrack_surveillance.yaml`) | Cross-frame person ID 0,1,2... per session. Converted to composite IDs `{cam_id}_{epoch}_{track_id}_{generation}` in `pipeline/track_state.py`. Tuned for fixed-camera surveillance: `track_high_thresh=0.45`, `track_buffer=60`, `new_track_thresh=0.50`. |
| **Face detection** | InsightFace SCRFD (`buffalo_l` in `.env`) | Loaded once as singleton in `utils/embedding_utils.py:InsightFaceSingleton`. Two-stage: crop person first, full frame fallback when crop has no faces ≥ `EMBEDDING_DET_SCORE_MIN`. CLAHE applied before detection. No redundant `DET_SCORE_MIN` tier. |
| **Face embedding** | InsightFace ArcFace (512-d normed) | Generated from best face detected. Quality-gated: only overwrites `track.embedding` if `det_score` exceeds existing by ≥ 0.05. Used for vector search (fresh Atlas query every recognition pass — no embedding cache). |
| **Face quality** | Two-tier quality system (`pipeline/quality_agent.py`) | Validity gates reject unusable faces (blur < 40, brightness outside 35-255, area < 1200px²). Weighted composite score [0,1] (50% blur, 25% brightness, 25% area). |
| **Face ratio** | `pipeline/quality_agent.py:compute_face_ratio()` | Simple ratio: face_area / person_bbox_area. Used for visibility classification (visible/partial/hidden). |
| **Mask detection** | Geometric heuristic (landmark nose/mouth ratio) | `InsightFaceSingleton._detect_mask_geometric()` in `utils/embedding_utils.py` — no classifier. Ratio < 0.3 = masked. |
| **Vector search** | MongoDB Atlas `$vectorSearch` (index `vector_index`, 512d cosine) | `utils/db_search.py:vector_search()` (re-exported via `db_utils.py`). Atlas score converted: `raw_cosine = (score*2)-1`. Fallback to Python numpy cosine scan on Atlas failure. |
| **MongoDB** | Atlas with 3 collections | `faces` (person DB + embeddings + vector index), `events` (track log), `visit_memory` (visit history). Thread-safe lazy singleton client. |
| **Image storage** | Cloudinary + local fallback | `utils/image_utils.py:upload_to_cloudinary()`. Pre-encoded JPEG bytes also accepted (`upload_jpeg_to_cloudinary`). Falls to `captures/{track_id}.jpg`. |
| **LLM** | Ollama (Gemma 3 4B / Qwen 3.5 4B) via `httpx` | `utils/llm_client.py` — connection-pooled HTTP client. `generate_nl_summary()` for alert descriptions, `chat_completion()` for dashboard chat. 3 retries, cached health check (10s). Graceful fallback to templates. |
| **Recognition logic** | Custom `RecognitionAgent` (`agents/recognition.py`) + `agents/scoring.py` | Weighted normalization: 0.65×sim + 0.15×quality + 0.10×track + 0.05×memory + 0.05×margin. Mask penalty ×0.85. Max-confidence gate prevents downgrades. |
| **Memory** | Custom `MemoryAgent` (`agents/memory.py`) | Visit count, typical hours, confidence boost. Stored in MongoDB `visit_memory` collection. Boosts recognition confidence +10 for returning visitors. |
| **Policy** | Custom `PolicyAgent` (`agents/policy.py`) | 9-rule priority tree: blacklist > authorized > verified > known_visitor > hidden > masked > after-hours > office-hours unknown. Also considers loitering time. |
| **Alerts** | Console/SMTP/Twilio/Webhook | `agents/alert_agent.py:dispatch()`. LLM summary generated async. Per-alert-level cooldown dedup (60s). Blacklist dispatched immediately; all others deferred to track finalization. |
| **Dashboard API** | FastAPI (port 8000) | Runs in separate thread inside `main.py`. CORS to localhost:5173. Routes: faces CRUD, events, reports, chat, WebSocket. |
| **Live feed** | WebSocket (`/ws/live`) via `broadcast_frame()` | JPEG encoded at quality 90, base64, JSON. Every 2nd frame. 1MB cap. 1s send timeout per client. |
| **Chat** | `POST /api/chat` + Ollama | Intent routing via keywords → fetches relevant data (stats/events/unknowns) → LLM prompted with data. Falls back to raw summary. |
| **Config** | `.env` (secrets) + `config.jsonc` (tunables) | `config/settings.py` resolves: env var > jsonc > hardcoded default. JSONC supports comments via custom parser. `validate_config()` on startup. |
| **Logging** | structlog (3-tier) | Terminal (CompactTerminalRenderer, INFO+), JSON file `logs/surveillance.jsonl` (5MB × 5 rotating, DEBUG+), debug file `logs/surveillance.debug.log` (10MB × 3 rotating, DEBUG+). Noisy pymongo/insightface silenced to WARNING. |
| **Status system** | `config/status.py:Status(IntEnum)` | Centralized enum: UNKNOWN=1, UNCERTAIN=2, KNOWN=3, KNOWN_VISITOR=4, VERIFIED=5, AUTHORIZED=6, BLACKLIST=7, MASKED_UNKNOWN=8, HIDDEN=9. Higher = more trusted. `is_known = status >= 3`. |
| **Base agent** | `agents/base.py:BaseAgent` | Abstract base class for all agents. |
| **Track processor** | `agents/track_processor.py` | Track finalization + dashboard broadcasting. |
| **Finalizer** | `agents/finalizer.py` | Final embedding retry on track expiry. |
| **Timing** | `agents/timing.py` | Thread-safe timing diagnostics collector. |
| **Report** | `agents/report.py` | Incident/stats/summary reports via LLM. |
| **Matching agent** | `agents/matching_agent.py` | Embedding + MongoDB vector search orchestration. |
| **DB client** | `utils/db_client.py` | MongoDB connection singleton. |
| **DB faces** | `utils/db_faces.py` | Face CRUD, deduplication, embedding history. |
| **DB events** | `utils/db_events.py` | Event logging + stats queries. |
| **DB memory** | `utils/db_memory.py` | Visit memory CRUD. |
| **DB search** | `utils/db_search.py` | Vector search (Atlas + Python fallback) + backfill. |
| **Pipeline models** | `pipeline/models.py` | Track, MatchResult, DecisionResult, QualityResult dataclasses. |
| **Recognition pipeline** | `pipeline/recognition_pipeline.py` | Orchestrates detect → quality → embed → match → decide. |

## Architecture Diagram

```
main.py
├── # Thread: CameraAgent._loop()
│   ├── YOLOv8 detect persons per frame
│   ├── TrackState.update() (thread-safe, composite IDs with generation)
│   ├── Every 20 frames: ThreadPool → progressive_recognition()
│   │   ├── InsightFace detection + embedding (quality-gated overwrite)
│   │   ├── Quality gate: skip embedding if face invalid (blur/brightness/area)
│   │   ├── MongoDB vector search (MatchingAgent)
│   │   ├── MemoryAgent lookup
│   │   ├── RecognitionAgent (multi-factor)
│   │   └── PolicyAgent (9 rules) → cache on Track
│   ├── get_expired_tracks() → Queue
│   └── broadcast_frame() via WebSocket
│
├── # 2 Workers: consume Queue → process_finalized_track()
│   ├── Upload image to Cloudinary
│   ├── Matching (if not cached)
│   ├── Recognition + Memory (if not cached)
│   ├── PolicyAgent → DecisionResult
│   ├── store_face() in MongoDB
│   ├── record_visit() in MongoDB
│   └── dispatch() alerts (console sync, email/sms/webhook async)
│
├── # Thread: Uvicorn (FastAPI port 8000)
│   ├── REST: faces, events, reports, chat
│   └── WebSocket: /ws/live (frame + event + alert broadcast)
│
└── SHUTDOWN: drain queue → close executors → close MongoDB → close httpx
```

## Connection Patterns

- **Camera loop never blocks** — all I/O via `queue.Queue` + `ThreadPoolExecutor`
- **Thread safety** — `TrackState._lock` for all track mutations; JPEG encoding done outside lock
- **Progressive caching** — results from progressive recognition stored on `Track` object, reused at finalization to avoid redundant DB calls
- **OpenVINO GPU** — YOLO OpenVINO IR models use `device=intel:GPU` format. Ultralytics' backend parses `intel:` prefix to extract the OpenVINO device while setting PyTorch device to `cpu`. InsightFace stays on CPU (OpenVINO EP not used due to DLL compatibility issues).
- **Quality-gated embedding** — per-track: only overwrites if `det_score` exceeds existing by ≥ 0.05. Per-DB: only overwrites `latest_embedding` if new `quality_score` > stored quality (no backward-compat unconditional overwrite)
- **Quality-gated recognition skip** — `camera_agent.py:291-296`: if face quality fails validity gates (blur < 40, brightness outside 35-255, area < 1200px²), `else: return` prevents embedding generation, MongoDB search, and identity assignment. Prevents blurry faces from being incorrectly matched to known persons.
- **Best face hysteresis** — quality score buffer of +0.03 in `track_state.py:202` requires >3% improvement to replace best face
- **Atlas vector search fallback** — if Atlas `$vectorSearch` fails/timeouts, falls to `_python_cosine_scan()` scanning up to 500 docs
- **LLM optional** — system runs without Ollama; NL summaries fall back to template strings
- **Composite track IDs** — Format: `{camera_id}_{session_epoch}_{byte_track_id}_{generation}` — unique across camera restarts and ByteTrack ID reuse
- **Status is numeric** — Use `Status.X` from `config/status.py` — never raw strings. `is_known = status >= 3`. Higher number = more trusted.
- **Atlas score conversion** — Atlas `vectorSearchScore = (1+cosine)/2`. Always convert: `raw_cosine = (atlas_score * 2) - 1` before comparing against `MATCH_THRESHOLD`.

## MongoDB Collections

| Collection | Atlas Search Index | Key Fields |
|-----------|-------------------|------------|
| `faces` | `vector_index` (latest_embedding, 512d cosine) | person_id, name, role, tags, verified, alert_level, images[], embeddings[], latest_embedding, mean_embedding, latest_embedding_quality, embedding_model, person_crop_url, source, quality_scores, verified_at, verified_by, created_at, updated_at |
| `events` | None | track_id, camera_id, timestamp, status, alert_level, person_id, similarity_score, name, is_masked, image_url, person_crop_url, reason, alerted |
| `visit_memory` | None | person_id (unique), visit_count, typical_hours[], typical_cameras[], avg_similarity, first_seen, last_seen, last_camera, last_status, best_status, similarity_history[], status_history[], created_at, updated_at |

## Key Thresholds

| Setting | Default | Note |
|---------|---------|------|
| MATCH_THRESHOLD | 0.45 | Max recommended. Converts via `(atlas_score*2)-1`. |
| HIGH_CONFIDENCE_SIMILARITY | 0.85 | Skips memory + recognition, immediate known. |
| EMBEDDING_DET_SCORE_MIN | 0.40 | Minimum score to generate embedding. |
| DET_SCORE_RELAXED | 0.20 | Entry gate for face detection (single tier, `DET_SCORE_MIN` removed). |
| QUALITY_BLUR_MIN | 40 | Minimum Laplacian variance for scoring normalization. |
| QUALITY_FACE_AREA_MIN | 1500 | Min face area for scoring normalization. |
| TRACK_TIMEOUT_SECS | 15.0 | Person gone for 15s = track ends (configurable in config.jsonc). |
| RECOGNITION_INTERVAL_FRAMES | 20 | Run recognition every N frames |
| ALERT_COOLDOWN_SECS | 60 | Per-level dedup window. |
| CONFIDENCE_KNOWN_MIN | 70 | Minimum confidence for "known" status |
| CONFIDENCE_UNCERTAIN_MIN | 55 | Minimum confidence for "uncertain" status |
| KNOWN_VISITOR_SIMILARITY | 0.85 | Auto-escalate to known_visitor |
| AUTO_REGISTERED_SIMILARITY | 0.65 | Self-match threshold |
| VERIFIED_SIMILARITY_THRESHOLD | 0.75 | Verified person threshold |
| MASK_RATIO_THRESHOLD | 0.3 | Mask detection geometric heuristic |
| LOITER_SECS | 30 | Loitering detection timeout |
| OVERLAP_IOU_THRESHOLD | 0.50 | Track overlap dedup |
| MIN_VISIT_GAP_SECS | 60 | Visit dedup window |
| WEIGHT_SIMILARITY | 0.65 | Confidence formula weight (similarity) |
| WEIGHT_QUALITY | 0.15 | Confidence formula weight (quality) |
| WEIGHT_TRACK | 0.10 | Confidence formula weight (track duration) |
| WEIGHT_MEMORY | 0.05 | Confidence formula weight (visit memory) |
| WEIGHT_MARGIN | 0.05 | Confidence formula weight (margin) |
