# Agentic AI Surveillance System

AI-powered real-time surveillance: YOLOv8 person detection, ByteTrack tracking, InsightFace face recognition, autonomous decision engine, alerts, and local LLM for NL summaries.

## Quick Start

```bash
# Terminal 1 — surveillance pipeline
python main.py

# Terminal 2 — dashboard
cd dashboard/frontend
npm install
npm run dev
```

Dashboard: http://localhost:5173 | API: http://localhost:8000

## Prerequisites

- Python 3.11
- MongoDB Atlas cluster with vector search index
- `.env` with `MONGODB_URI` (copy from `.env.example`)
- Camera at `CAMERA_INDEX=0` or `CAMERA_SOURCE=http://<phone-ip>:8080/video`

## Commands

| Task | Command |
|------|---------|
| Run surveillance | `python main.py` |
| Run dashboard frontend | `cd dashboard/frontend && npm run dev` |
| Install Python deps | `pip install -r requirements.txt` |
| Install frontend deps | `cd dashboard/frontend && npm install` |
| Run tests | `python -m pytest tests/ -v` |

## Documentation

| Document | Description |
|----------|-------------|
| [AGENTS.md](AGENTS.md) | AI agent instructions, architecture, coding rules, do/don'ts |
| [TOOLS.md](TOOLS.md) | Developer tools — repomix, ctags, ast-grep, commands |
| [SYSTEM_INDEX.md](SYSTEM_INDEX.md) | Complete system index — repo layout, execution flow, config map |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Tech stack, components, connection patterns, MongoDB schema |
| [docs/00 - Home.md](docs/00%20-%20Home.md) | Vault map-of-content: links to all architecture, agent, pipeline, and reference notes |

## Project Structure

```
main.py                    # Entry point
AGENTS.md                  # AI agent context (architecture, rules, do/don'ts)
TOOLS.md                   # Developer tools (repomix, ctags, ast-grep)
SYSTEM_INDEX.md            # Complete system index
agents/                    # Business logic (camera, matching, recognition, policy, alerts)
pipeline/                  # CV pipeline (YOLO, ByteTrack, face detection, quality)
utils/                     # Infrastructure (MongoDB, LLM, image processing)
config/                    # Settings loader + config.jsonc
dashboard/
  backend/                 # FastAPI REST + WebSocket
  frontend/                # React + Vite SPA
tests/                     # pytest test suite (76 tests)
scripts/                   # Debug/dev utilities
docs/                      # Documentation (ARCHITECTURE + structured vault: 00-Home → 10-Problems)
models/                    # ML weights (gitignored)
logs/                      # Runtime logs (gitignored)
captures/                  # Face captures (gitignored)
```

## Configuration

**Priority chain:** `.env` (secrets) > `config/config.jsonc` (tunables) > hardcoded defaults

Edit `config/config.jsonc` for detection, quality, and recognition settings. Keep secrets in `.env` only.

## License

Private project.
