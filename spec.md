# spec.md — agentic_ai_singlecam

## Tech Stack & How It Connects

| Component | What we use | How |
|-----------|------------|-----|
| **Detection** | YOLOv8 (`yolov8s.pt`) via Ultralytics | Singleton model in `pipeline/tracker.py:track_persons()`. Returns person boxes every frame. |
| **Tracking** | ByteTrack (Ultralytics built-in `bytetrack.yaml`) | Cross-frame person ID 0,1,2... per session. Converted to composite IDs `{cam_id}_{epoch}_{track_id}` in `pipeline/track_state.py`. |
| **Face detection** | InsightFace SCRFD (`buffalo_l` in `.env`) | Loaded once as singleton in `utils/embedding_utils.py:InsightFaceSingleton`. Two-stage: crop person first, full frame fallback when crop has no faces ≥ `EMBEDDING_DET_SCORE_MIN`. CLAHE applied before detection. No redundant `DET_SCORE_MIN` tier. |
| **Face embedding** | InsightFace ArcFace (512-d normed) | Generated from best face detected. Quality-gated: only overwrites `track.embedding` if `det_score` exceeds existing by ≥ 0.05. Used for vector search. Embedding cache skips Atlas search if cosine distance < 0.005 from last searched embedding. |
| **Mask detection** | Geometric heuristic (landmark nose/mouth ratio) | `_detect_mask_geometric()` — no classifier. Ratio < 0.3 = masked. |
| **Vector search** | MongoDB Atlas `$vectorSearch` (index `vector_index`, 512d cosine) | `utils/db_utils.py:vector_search()`. Atlas score converted: `raw_cosine = (score*2)-1`. Fallback to Python numpy cosine scan on Atlas failure. |
| **MongoDB** | Atlas with 3 collections | `faces` (person DB + embeddings + vector index), `events` (track log), `visit_memory` (visit history). Thread-safe lazy singleton client. |
| **Image storage** | Cloudinary + local fallback | `utils/image_utils.py:upload_to_cloudinary()`. Pre-encoded JPEG bytes also accepted (`upload_jpeg_to_cloudinary`). Falls to `captures/{track_id}.jpg`. |
| **LLM** | Ollama (Gemma 3 4B / Qwen 3.5 4B) via `httpx` | `utils/llm_client.py` — connection-pooled HTTP client. `generate_nl_summary()` for alert descriptions, `chat_completion()` for dashboard chat. 3 retries, cached health check (10s). Graceful fallback to templates. |
| **Recognition logic** | Custom `RecognitionAgent` (`agents/recognition.py`) | Multi-factor: similarity (0-60) + face quality (0-25) + track duration (0-15) + memory boost (-10/+20). Mask penalty ×0.85. Replaces simple threshold check. |
| **Memory** | Custom `MemoryAgent` (`agents/memory.py`) | Visit count, typical hours, confidence boost. Stored in MongoDB `visit_memory` collection. Boosts recognition confidence +10 for returning visitors. |
| **Policy** | Custom `PolicyAgent` (`agents/policy.py`) | 9-rule priority tree: blacklist > authorized > verified > known_visitor > hidden > masked > after-hours > office-hours unknown. Also considers loitering time. |
| **Alerts** | Console/SMTP/Twilio/Webhook | `agents/alert_agent.py:dispatch()`. LLM summary generated async. Per-alert-level cooldown dedup (60s). Blacklist dispatched immediately; all others deferred to track finalization. |
| **Dashboard API** | FastAPI (port 8000) | Runs in separate thread inside `main.py`. CORS to localhost:5173. Routes: faces CRUD, events, reports, chat, WebSocket. |
| **Live feed** | WebSocket (`/ws/live`) via `broadcast_frame()` | JPEG encoded at quality 65, base64, JSON. Every 2nd frame. 1MB cap. 1s send timeout per client. |
| **Chat** | `POST /api/chat` + Ollama | Intent routing via keywords → fetches relevant data (stats/events/unknowns) → LLM prompted with data. Falls back to raw summary. |
| **Config** | `.env` (secrets) + `config.jsonc` (tunables) | `config/settings.py` resolves: env var > jsonc > hardcoded default. JSONC supports comments via custom parser. `validate_config()` on startup. |
| **Logging** | structlog | Console INFO+, file DEBUG to `logs/surveillance.log` (5MB × 5 rotating). Noisy pymongo silenced to WARNING. |

## Architecture Diagram

```
main.py
├── # Thread: CameraAgent._loop()
│   ├── YOLOv8 detect persons per frame
│   ├── TrackState.update() (thread-safe, composite IDs)
│   ├── Every 10 frames: ThreadPool → progressive_recognition()
│   │   ├── InsightFace detection + embedding (quality-gated overwrite)
│   │   ├── Embedding cache check (cosine dist < 0.005 → skip Atlas)
│   │   ├── MongoDB vector search (MatchingAgent) [if cache miss]
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
- **Embedding cache** — if new embedding's cosine distance from last searched < 0.005, skip Atlas round-trip and reuse prior `MatchResult`
- **Quality-gated embedding** — per-track: only overwrites if `det_score` exceeds existing by ≥ 0.05. Per-DB: only overwrites `latest_embedding` if new `quality_score` > stored quality (no backward-compat unconditional overwrite)
- **Best face hysteresis** — quality score buffer reduced to +0.03 (was +0.10) for gradual improvement
- **Atlas vector search fallback** — if Atlas `$vectorSearch` fails/timeouts, falls to `_python_cosine_scan()` scanning up to 500 docs
- **LLM optional** — system runs without Ollama; NL summaries fall back to template strings

## MongoDB Collections

| Collection | Atlas Search Index | Key Fields |
|-----------|-------------------|------------|
| `faces` | `vector_index` (latest_embedding, 512d cosine) | person_id, name, role, tags, verified, alert_level, images[], embeddings[], latest_embedding |
| `events` | None | track_id, camera_id, timestamp, status, alert_level, person_id, similarity_score |
| `visit_memory` | None | person_id (unique), visit_count, typical_hours[], typical_cameras[], avg_similarity |

## Key Thresholds

| Setting | Default | Note |
|---------|---------|------|
| MATCH_THRESHOLD | 0.45 | Max recommended. Converts via `(atlas_score*2)-1`. |
| VERY_HIGH_SIMILARITY | 0.90 | Skips memory + recognition, immediate known. |
| EMBEDDING_DET_SCORE_MIN | 0.40 | Minimum score to generate embedding. |
| DET_SCORE_RELAXED | 0.20 | Entry gate for face detection (single tier, `DET_SCORE_MIN` removed). |
| EMBEDDING_CACHE_COSINE_THRESHOLD | 0.005 | Skip Atlas search if cosine dist from last searched < threshold. |
| QUALITY_BLUR_MIN | 15 | Minimum Laplacian variance (relaxed from 30). |
| QUALITY_FACE_AREA_MIN | 800 | Min face area ~28×28 px (relaxed from 40×40). |
| TRACK_TIMEOUT_SECS | 3.0 | Person gone for 3s = track ends. |
| ALERT_COOLDOWN_SECS | 60 | Per-level dedup window. |
