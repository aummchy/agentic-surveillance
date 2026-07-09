# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the full system (starts FastAPI on :8000 automatically)
python main.py

# Frontend dev server (proxies /api and /ws to :8000)
cd dashboard/frontend && npm run dev      # http://localhost:5173

# Install
pip install -r requirements.txt
cd dashboard/frontend && npm install
```

No test suite, linter, or CI pipeline exists. Run tests with: `python -m pytest tests/ -v`

**Python version:** 3.11 required — InsightFace/onnxruntime wheels are unreliable on other versions.

## Architecture

```
main.py
├── CameraAgent         — capture loop, ByteTrack, progressive recognition (every 10 frames)
├── track_queue         — Queue decouples camera from blocking I/O (2 worker threads)
├── uvicorn/FastAPI     — started in a daemon thread sharing the same asyncio event loop
├── Ollama LLM          — Gemma 3 4B / Qwen 3.5 4B for NL summaries + chat
└── On track finalized:
    MatchingAgent → MemoryAgent → RecognitionAgent → PolicyAgent → AlertAgent → DB/Cloudinary
```

### Agent pipeline (per track)

| Agent | File | Role |
|-------|------|------|
| **Recognition** | `agents/recognition.py` | Combines similarity + face quality + track duration → status/confidence |
| **Memory** | `agents/memory.py` | Looks up visit history in `visit_memory` collection → confidence boost |
| **Policy** | `agents/policy.py` | Applies prioritized rules → status, alert_level, should_alert, should_register |
| **Alert** | `agents/alert_agent.py` | Dispatches to console/email/SMS/webhook with cooldown |
| **Matching** | `agents/matching_agent.py` | MongoDB Atlas Vector Search → cosine similarity match |
| **LLM** | `utils/llm_client.py` | Ollama HTTP client for NL summaries, reports, and chat |

`decision_agent.py` is a thin wrapper that delegates to `PolicyAgent` (kept for backward compatibility).

### Policy rules (priority order)
1. `blacklist` tag → critical alert
2. `authorized` tag → no alert
3. `verified=True` → alert based on stored level
4. matched + memory confirms → known_visitor, low
5. matched, confidence ≥ 80 → known_visitor
6. matched, confidence < 80 → uncertain
7. visibility == "hidden" → high alert (person avoided camera)
8. masked or partial visibility → medium/high (escalates after `LOITER_SECS`)
9. after-hours unknown → high
10. unknown during office hours → medium

### Key data flow details

- **Track IDs** are composite: `{camera_id}_{session_epoch}_{byte_track_id}` — unique across restarts.
- **Progressive recognition** runs every `RECOGNITION_INTERVAL_FRAMES` (default 10) frames. Results (`pending_match_result`, `pending_recognition`) are stored on the `Track` object and reused at finalization to avoid double work. Quality-gated: skips if face quality didn't improve by ≥ 0.10. Max-confidence gate: recognition never downgrades, only upgrades confidence.
- **`TrackState`** is the single source of truth for live tracks, protected by `threading.Lock`. All mutations go through its setter methods.
- **InsightFace** is a double-checked locking singleton in `utils/embedding_utils.py`. Never instantiate it directly; always call `get_insightface()`.
- **YOLO/ByteTrack** model is loaded once at import time in `pipeline/tracker.py`.

### Atlas vector search threshold conversion

Atlas returns `vectorSearchScore` in `[0, 1]` (cosine mapped as `(1 + cosine) / 2`). Always convert back before comparing:

```python
raw_cosine = (atlas_score * 2) - 1   # in atlas_score_to_cosine()
compare_similarity(raw_cosine)         # compares against settings.MATCH_THRESHOLD
```

Keep `MATCH_THRESHOLD` ≤ 0.45 (default 0.45). Higher values reject genuine same-person matches under indoor lighting. If Atlas Vector Search is unavailable, `vector_search()` falls back to a Python cosine scan (capped at 500 docs).

### Dashboard

- FastAPI (`dashboard/backend/main.py`) serves REST + WebSocket. Started inside `main.py` via uvicorn in a daemon thread.
- Vite (`dashboard/frontend/vite.config.js`) proxies `/api` → `:8000` and `/ws` → `:8000` so the frontend only talks to `:5173`.
- Live feed and alerts flow over WebSocket `/ws/live`. Frames are JPEG-encoded at 65% quality, base64-wrapped in JSON.
- `broadcast_frame` / `broadcast_alert` in `routes/live.py` must be called with `asyncio.run_coroutine_threadsafe(...)` from any non-async thread.

## Critical gotchas

- **`axios` is pinned to `1.7.9`** in `dashboard/frontend/package.json`. Versions ≥ 1.7.10 break Vite's esbuild. Do not upgrade.
- **MongoDB collections needed:** `faces`, `events`, `visit_memory`. Atlas Vector Search index named `vector_index` on `faces.latest_embedding` (512 dims, cosine).
- **`MONGODB_URI` is the only required env var.** Everything else has defaults in `config/settings.py`.
- **No local OpenCV window.** The camera feed is streamed via WebSocket only — open the browser to see it.
- **Mask detection** uses a geometric heuristic: `lower_face_height / upper_face_height < 0.3` (5-point landmarks). Not a classifier — works best on front-facing faces.
- **`FutureWarning` from insightface** (`estimate` deprecated in 0.26) is harmless.
- **OpenVINO GPU acceleration:** After YOLO export (`yolo export model=yolov8s.pt format=openvino half=True`), set `YOLO_MODEL=models/yolov8s_openvino_model/` and `YOLO_DEVICE=intel:GPU` in `.env`. The `intel:` prefix is required — bare `GPU` fails in Ultralytics. Expect ~8× speedup on Arc iGPU (128ms → 16ms). InsightFace stays on CPU (OpenVINO EP has DLL issues on Windows).
- Logs split: console shows `INFO+` with ANSI colors, full debug (face quality scores, detection scores, recognition details) goes to `logs/surveillance.jsonl` (5 MB × 5 rotating backups) and `logs/surveillance.debug.log` (10 MB × 3).
- **InsightFace/YOLO print noise suppressed:** stdout captured via `contextlib.redirect_stdout`, ultralytics logger set to CRITICAL after import.
- **Blank lines suppressed:** `_BlankFilter` drops events not in `TERMINAL_ALLOWLIST`.
- **Mobile camera:** Install IP Webcam app on your phone, start server, and set `CAMERA_SOURCE=http://<phone-ip>:8080/video` in `.env`.
- **LLM (Ollama) must be running** for chat and NL summaries. System falls back to template strings if unavailable. Swap model via `OLLAMA_MODEL` env var.
- **LLM runs off critical path.** `dispatch()` calls LLM only after alert decision is made. Report generation runs on demand. Camera loop never blocks on LLM.

## MongoDB document schemas

**`faces` collection** key fields: `person_id`, `name`, `role`, `tags` (array, includes `authorized`/`blacklist`/`verified`), `latest_embedding` (list[float], 512-dim ArcFace), `embeddings` (last 10), `images`, `verified` (bool), `alert_level`.

**`events` collection** key fields: `track_id`, `camera_id`, `timestamp`, `status`, `alert_level`, `person_id`, `similarity_score`, `image_url`, `alerted`.

**`visit_memory` collection** key fields: `person_id` (unique index), `visit_count`, `last_seen`, `avg_similarity`, `typical_hours`, `typical_cameras`, `last_status`.

## Configuration reference (`.env`)

| Variable | Default | Notes |
|----------|---------|-------|
| `MONGODB_URI` | — | **Required** |
| `MATCH_THRESHOLD` | `0.45` | Max 0.45 |
| `TRACK_TIMEOUT_SECS` | `3.0` | Seconds before track expires |
| `RECOGNITION_INTERVAL_FRAMES` | `10` | Frames between recognition runs |
| `DET_SCORE_MIN` | `0.40` | Face detection threshold |
| `ALERT_CHANNELS` | `console` | `console,email,sms,webhook` |
| `CAMERA_SOURCE` | `""` | HTTP/RTSP URL for mobile camera (e.g. `http://192.168.x.x:8080/video`). Overrides `CAMERA_INDEX`. |
| `CAMERA_INDEX` | `0` | Webcam device index (ignored if `CAMERA_SOURCE` is set) |
| `YOLO_DEVICE` | `cpu` | Set `0` for CUDA GPU, `intel:GPU` for OpenVINO (Arc iGPU) |
| `INSIGHTFACE_PROVIDER` | `CPUExecutionProvider` | Set `CUDAExecutionProvider` for GPU |
| `CLOUDINARY_*` | — | Optional image archival |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma3:4b` | Swap to `qwen3.5:4b` for better reasoning |
| `OLLAMA_TIMEOUT` | `30` | LLM request timeout (seconds) |