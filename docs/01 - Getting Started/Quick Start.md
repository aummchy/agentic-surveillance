# Quick Start

## Two-terminal setup

### Terminal 1 — Surveillance pipeline
```bash
python main.py
```
This starts the camera, YOLO detection, face recognition, alert engine, and FastAPI server (port 8000) all in one process.

### Terminal 2 — Dashboard frontend
```bash
cd dashboard/frontend
npm install
npm run dev
```
Opens the React dashboard at http://localhost:5173.

## What happens on startup

1. `config/settings.py` loads `.env` (secrets) + `config/config.jsonc` (tunables)
2. `validate_config()` checks MONGODB_URI, thresholds, channels
3. Background threads pre-warm YOLO and InsightFace models (avoids 1-5s cold load on first frame)
4. Background thread checks MongoDB Atlas Vector Search index
5. 2 worker threads start (consume finalized tracks from a queue)
6. FastAPI server starts on port 8000 (REST + WebSocket)
7. Camera opens, main loop begins
8. On `Ctrl+C`: drains queue, closes executors, shuts down MongoDB + LLM

## Minimal `.env`

```
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
```

Everything else has defaults. See [[Environment Variables]] for full list.

## Verifying it works

- Terminal shows colored one-liner events: `CAM started`, `MATCH`, `RECOG`, `FINAL`
- Open http://localhost:5173 for live video feed + events
- Open http://localhost:8000/docs for FastAPI Swagger UI

## Common first-run issues

| Issue | Fix |
|-------|-----|
| `camera_open_failed` | Check `CAMERA_INDEX=0` or set `CAMERA_SOURCE` in `.env` |
| `atlas_search_index_missing` | Create Atlas Vector Search index named `vector_index` on `faces.latest_embedding` |
| `llm_unavailable` | Ollama not running — alerts use template strings instead (system still works) |
| MSMF frame drops (error -1072875772) | Set `CAMERA_BACKEND=dshow` in `.env` |
