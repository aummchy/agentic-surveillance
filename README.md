# Agentic AI Surveillance System

AI-powered real-time surveillance: YOLOv8 person detection, ByteTrack tracking, InsightFace face recognition, autonomous decision engine, alerts, and local LLM for NL summaries.

## How to Run

### One-time setup

```powershell
# 1) Secrets — .env is gitignored, so it is NOT in the repo. Create it:
copy .env.example .env
#    then edit .env — at minimum MONGODB_URI (MongoDB Atlas connection string)

# 2) Python environment (Python 3.11 venv, per Prerequisites)
py -3.11 -m venv venv
venv\Scripts\pip install -r requirements.txt
#    (POSIX: python3.11 -m venv venv && venv/bin/pip install -r requirements.txt)

# 3) Frontend deps
cd dashboard\frontend
npm install
cd ..\..
```

> Note: this repo has **no committed `venv/`** — create it as above, or install
> deps into a global Python. On Linux/macOS use `venv/bin/python` wherever
> `venv\Scripts\python.exe` appears below.

### Run

```powershell
# Terminal 1 — surveillance pipeline (+ FastAPI on :8000, starts automatically)
venv\Scripts\python.exe main.py        # or: python main.py (global install)

# Terminal 2 — dashboard
cd dashboard\frontend
npm run dev
```

Dashboard: http://localhost:5173 | API: http://localhost:8000

### Run on a video file (instead of a live camera)

Set `CAMERA_SOURCE` in `.env` — it overrides `CAMERA_INDEX`:

```powershell
CAMERA_SOURCE=D:/videos/sample.mp4     # forward slashes; relative paths work too, e.g. videos/low_4.mp4
```

- Works for local files, `rtsp://…` and `http://…/video` streams alike.
- End-of-file: logs `video_complete` and the pipeline shuts down (no looping).
- Back to webcam: remove/comment `CAMERA_SOURCE` → falls back to `CAMERA_INDEX=0`.
- `CAMERA_BACKEND` (e.g. `dshow`) applies only to webcam indexes, not files/URLs.
- Alternative location: `CAMERA_SOURCE` in `config/config.jsonc` (the `.env` value wins if both are set).

### Verify install

```powershell
venv\Scripts\python.exe -m pytest tests/ -v
```

## Prerequisites

- Python 3.11 (InsightFace/onnxruntime wheels unreliable on other versions)
- MongoDB Atlas cluster with `surveillance` database, `faces`/`events`/`visit_memory` collections
- Atlas Vector Search index named `vector_index` on `faces.latest_embedding` (512 dims, cosine)
- `.env` with at minimum `MONGODB_URI` — **gitignored, never shipped in the repo**; create it from `.env.example` (see How to Run)
- Camera device at `CAMERA_INDEX=0` or `CAMERA_SOURCE=http://<phone-ip>:8080/video` for mobile (adjust in `.env`)
- For Intel Arc iGPU acceleration: `pip install openvino` and export YOLO (optional — CPU works too)

## Tech Stack

High-level view — full detail lives in [CURRENT_ARCHITECTURE.md Appendix A](docs/CURRENT_ARCHITECTURE.md) and the [architecture index §3](docs/02%20-%20Architecture/index.md).

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| Detection / tracking | YOLOv8 (`ultralytics`) + ByteTrack; optional OpenVINO or CUDA acceleration |
| Face recognition | InsightFace `buffalo_l` — SCRFD detection + ArcFace 512-d embeddings; OpenCV for image operations |
| Database | MongoDB Atlas — vector search (`vector_index`, 512-d cosine), 3 collections; Cloudinary image archive |
| Backend | FastAPI + Uvicorn — REST + WebSocket on port 8000; PyMongo; structlog (3-tier logging) |
| Frontend | React 18 + Vite 5 + axios — dashboard on port 5173 |
| LLM | Ollama local — Gemma 3 4B / Qwen 3.5 4B via `httpx` (optional; falls back to template strings) |
| Alerts | Email (SMTP) · Twilio SMS · webhook |
| Config & tests | `.env` secrets + `config/config.jsonc` tunables · pytest (97 tests) |

## Commands

| Task | Command |
|------|---------|
| Create Python 3.11 venv | `py -3.11 -m venv venv` |
| Run surveillance | `venv\Scripts\python.exe main.py` (or `python main.py`) |
| Run dashboard API | FastAPI starts automatically on port 8000 inside `main.py` |
| Run dashboard frontend | `cd dashboard/frontend && npm run dev` |
| Install Python deps | `venv\Scripts\pip install -r requirements.txt` (or `pip install -r requirements.txt`) |
| Install frontend deps | `cd dashboard/frontend && npm install` |
| Export YOLO to OpenVINO IR | `yolo export model=models/yolov8s.pt format=openvino half=True` |
| Run tests | `venv\Scripts\python.exe -m pytest tests/ -v` (or `python -m pytest tests/ -v`) |
| Run specific test | `python -m pytest tests/test_recognition.py -v` |

## Documentation

| Document | Description |
|----------|-------------|
| [AGENTS.md](AGENTS.md) | AI agent instructions, architecture, coding rules, do/don'ts |
| [plan.md](plan.md) | Phased roadmap — what is done, what is next, gates |
| [docs/CURRENT_ARCHITECTURE.md](docs/CURRENT_ARCHITECTURE.md) | As-built architecture: components, threads, tech stack, thresholds |
| [docs/00 - Home.md](docs/00%20-%20Home.md) | Vault map-of-content: links to all remaining docs |

## Project Structure

```
main.py                    # Entry point
AGENTS.md                  # AI agent context (architecture, rules, do/don'ts)
plan.md                    # Phased roadmap with gates
agents/                    # Business logic (camera, matching, recognition, policy, alerts)
pipeline/                  # CV pipeline (YOLO, ByteTrack, face detection, quality)
utils/                     # Infrastructure (MongoDB, LLM, image processing)
config/                    # Settings loader + config.jsonc
dashboard/
  backend/                 # FastAPI REST + WebSocket
  frontend/                # React + Vite SPA
tests/                     # pytest test suite (84 tests)
scripts/                   # Debug/dev utilities
docs/                      # Documentation (CURRENT_ARCHITECTURE + focused vault: 00-Home → 11-Issues)
models/                    # ML weights (gitignored)
logs/                      # Runtime logs (gitignored)
captures/                  # Face captures (gitignored)
```

## Configuration

**Priority chain:** `.env` (secrets) > `config/config.jsonc` (tunables) > hardcoded defaults

Edit `config/config.jsonc` for detection, quality, and recognition settings. Keep secrets in `.env` only.

## License

Private project.
