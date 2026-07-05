# AGENTS.md — agentic_ai_singlecam

## What this is

AI-powered surveillance system: YOLOv8 person detection → ByteTrack tracking → InsightFace face recognition → autonomous decision engine → alerts + **local LLM** for NL summaries and conversational dashboard. Python 3.11, MongoDB Atlas vector search, FastAPI + React dashboard, Ollama (Gemma 3 4B / Qwen 3.5 4B).

## Quick start

```bash
# Terminal 1 — surveillance pipeline
python main.py

# Terminal 2 — dashboard
cd dashboard/frontend
npm install
npm run dev
```

Dashboard: http://localhost:5173. API: http://localhost:8000.

## Prerequisites

- Python 3.11 (InsightFace/onnxruntime wheels unreliable on other versions)
- MongoDB Atlas cluster with `surveillance` database, `faces`/`events`/`visit_memory` collections
- Atlas Vector Search index named `vector_index` on `faces.latest_embedding` (512 dims, cosine)
- `.env` with at minimum `MONGODB_URI` (copy from `.env.example`)
- Camera device at `CAMERA_INDEX=0` or `CAMERA_SOURCE=http://<phone-ip>:8080/video` for mobile (adjust in `.env`)
- For Intel Arc iGPU acceleration: `pip install openvino` and export YOLO (optional — CPU works too)

## Commands

| Task | Command |
|------|---------|
| Run surveillance | `python main.py` |
| Run dashboard API | FastAPI starts automatically on port 8000 inside `main.py` |
| Run dashboard frontend | `cd dashboard/frontend && npm run dev` |
| Install Python deps | `pip install -r requirements.txt` |
| Install frontend deps | `cd dashboard/frontend && npm install` |
| Export YOLO to OpenVINO IR | `yolo export model=models/yolov8s.pt format=openvino half=True` |

No test suite, linter, or CI pipeline exists in this repo.

## Architecture

```
main.py (entry point, wires everything)
├── agents/camera_agent.py    — capture loop + ByteTrack + progressive recognition
├── agents/matching_agent.py  — embedding + MongoDB vector search
├── agents/decision_agent.py  — delegates to PolicyAgent
├── agents/memory.py          — visit history tracking
├── agents/alert_agent.py     — alert dispatch (console/email/sms/webhook)
├── pipeline/tracker.py       — YOLOv8 + ByteTrack (single model instance)
├── pipeline/face.py          — SCRFD detection + ArcFace embedding + mask heuristic
├── pipeline/models.py        — Track, MatchResult, DecisionResult, etc.
├── pipeline/track_state.py   — per-track accumulation with threading.Lock
├── utils/db_utils.py         — MongoDB CRUD + vector search + Python fallback
├── utils/embedding_utils.py  — InsightFace singleton (load once, never per-frame)
├── utils/llm_client.py       — Ollama HTTP client (generate, chat, NL summaries)
├── utils/image_utils.py      — crop, save, upload to Cloudinary
├── config/settings.py        — loads .env, validate_config()
└── dashboard/
    ├── backend/main.py       — FastAPI app (REST + WebSocket)
    ├── backend/routes/chat.py — POST /api/chat, GET /api/chat/health
    └── frontend/             — React + Vite
```

## Critical patterns

- **Load models once.** YOLO in `tracker.py`, InsightFace as singleton in `embedding_utils.py`. Never reload in per-frame loops.
- **Thread safety.** `TrackState` uses `threading.Lock`. Camera thread and worker pool (2 threads) run concurrently.
- **I/O decoupled from camera.** MongoDB, Cloudinary, alerts run via `queue.Queue` + workers. Camera loop must never block.
- **Threshold conversion.** Atlas `vectorSearchScore = (1+cosine)/2`. Always convert back: `raw_cosine = (atlas_score * 2) - 1` before comparing against `MATCH_THRESHOLD`. Use `compare_similarity()` in `db_utils.py`.
- **Match threshold.** Keep `MATCH_THRESHOLD` ≤ 0.45. Default is 0.25. Higher rejects genuine same-person matches under indoor lighting.
- **One decision per track.** Recognition + decision runs once when track ends (or progressively every 20 frames). Not per-frame.
- **Composite track IDs.** Format: `{camera_id}_{session_epoch}_{byte_track_id}` — unique across camera restarts.

## Known issues (0 remaining)

All 32 issues from the problem report have been fixed.

## OpenVINO GPU acceleration

After installing `openvino` and exporting YOLO to OpenVINO IR (`yolo export model=models/yolov8s.pt format=openvino half=True`):

1. Set `YOLO_MODEL=models/yolov8s_openvino_model/` in `.env`
2. Set `YOLO_DEVICE=intel:GPU` in `.env` (the `intel:` prefix is required — bare `GPU` fails)
3. InsightFace stays on `CPUExecutionProvider` (OpenVINO EP has known DLL compatibility issues on Windows)

Benchmark on Arc 130T iGPU: YOLOv8s 128ms CPU → 15.6ms GPU (8.2× speedup).

## Gotchas

- `axios` pinned to `1.7.9` in `dashboard/frontend/` — versions ≥1.7.10 break Vite's esbuild
- Console shows INFO+ only; full debug logs go to `logs/surveillance.log` (5MB × 5 backups)
- `FutureWarning` from insightface is harmless (deprecated `estimate` in 0.26)
- Camera streams via WebSocket, not a local OpenCV window — open browser to see feed
- Mask detection uses geometric landmark heuristic (lower_face/upper_face ratio < 0.3), not a classifier
- Blacklisted persons always trigger critical alert even if also authorized (tag priority: blacklist > authorized > known_visitor)
- LLM (Ollama) must be running for chat and NL summaries — system falls back to template strings if unavailable
- Swap LLM model via single env var: `OLLAMA_MODEL=qwen3.5:4b` in `.env` (Qwen 3.5 4B has better reasoning: MMLU-Pro 79.1% vs ~43%)
- Both Gemma 3 4B and Qwen 3.5 4B fit in 4GB VRAM at Q4_K_M quantization (~2.7GB weights)
