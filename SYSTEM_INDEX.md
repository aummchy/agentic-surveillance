# SYSTEM_INDEX.md

Quick-lookup navigation map for AI agents. One sentence per file. See `AGENTS.md` for architecture, `docs/ARCHITECTURE.md` for tech stack, `docs/FORMULAS.md` for formulas.

---

## Repository Layout

```
├── main.py                          # Entry point: wires agents, starts camera + FastAPI + workers
├── AGENTS.md                        # AI agent context: architecture, coding rules, do/don'ts
├── TOOLS.md                         # Developer tools: repomix, ctags, ast-grep, commands
├── config/
│   ├── settings.py                  # Loads .env + config.jsonc, resolves 80+ settings, 3-tier logging
│   ├── config.jsonc                 # Tunable parameters (thresholds, weights, sizes, schedules)
│   └── bytetrack_surveillance.yaml  # ByteTrack tracker tuning for fixed-camera indoor use
├── agents/
│   ├── camera_agent.py              # Camera loop: YOLO+ByteTrack, progressive recognition, face crops
│   ├── matching_agent.py            # Vector search via MongoDB Atlas $vectorSearch
│   ├── recognition.py               # Multi-signal identity classification (calls scoring.py)
│   ├── scoring.py                   # Weighted confidence formula + calculation log writer
│   ├── policy.py                    # 9-rule priority business logic (blacklist/authorized/after-hours)
│   ├── memory.py                    # Visit history tracking + contextual confidence boost
│   ├── alert_agent.py               # Alert dispatch: console/SMTP/Twilio/Webhook with cooldown dedup
│   ├── report.py                    # Human-readable incident/stats/summary reports via LLM
│   ├── decision_agent.py            # Backward-compat shim delegating to PolicyAgent
│   └── base.py                      # Abstract BaseAgent class
├── pipeline/
│   ├── tracker.py                   # YOLOv8 singleton + ByteTrack track_persons() generator
│   ├── track_state.py               # Thread-safe per-track state, composite ID generation, expiry
│   ├── quality_agent.py             # Face quality scoring: validity gates + weighted composite [0,1]
│   ├── face.py                      # compute_face_ratio() — face area / person bbox
│   └── models.py                    # Data classes: Track, MatchResult, DecisionResult, QualityResult
├── utils/
│   ├── db_utils.py                  # MongoDB CRUD, Atlas vector search, Python cosine fallback
│   ├── embedding_utils.py           # InsightFace singleton (SCRFD+ArcFace), compare_similarity()
│   ├── image_utils.py               # Crop, resize, save, Cloudinary upload, blur/brightness scoring
│   └── llm_client.py                # Ollama HTTP client: generate, chat, health check, retry
├── dashboard/
│   ├── backend/
│   │   ├── main.py                  # FastAPI app: CORS, routers, static /captures mount
│   │   ├── models.py                # Pydantic response/request models
│   │   └── routes/                  # faces, events, reports, chat, live (WebSocket)
│   └── frontend/src/
│       ├── App.jsx                  # Root: WebSocket, stats polling, view routing, alert state
│       ├── components/              # LiveFeed, UnknownPersons, VerifiedPersons, EventLog, ChatPanel, VerifyModal, ErrorBoundary
│       └── utils/api.js             # Axios instance with base URL + error interceptor
├── tests/                           # 3 test files: recognition, embedding_history, thread_safety
├── docs/                            # ARCHITECTURE, FORMULAS, ISSUES, CHANGELOG, PAPER, TERMINAL_OUTPUT
├── models/                          # YOLOv8s weights + OpenVINO IR (gitignored)
├── scripts/                         # Debug/utility scripts (gitignored)
├── captures/                        # Face crop storage (gitignored)
└── logs/                            # surveillance.jsonl, debug.log, calculation.log (gitignored)
```

---

## Execution Flow

```
python main.py
  ├── validate_config() + log_formula_header()
  ├── Start 2 worker threads: process_finalized_track()
  ├── Start FastAPI (port 8000) in daemon thread
  ├── Background: startup checks + prewarm YOLO + InsightFace
  └── CameraAgent.start() [BLOCKS]
        └── Per frame:
              ├── YOLOv8 detect → ByteTrack → TrackState.update()
              ├── SCRFD face detect → quality score → ArcFace embed
              ├── Progressive recognition every N frames
              ├── on_frame_annotated → JPEG → WebSocket broadcast
              └── on_track_finalized → track_queue.put()

Worker: vector_search → MemoryAgent → RecognitionAgent → PolicyAgent → store/visit/alert/log
```

---

## Config Map

| File | Controls | Override |
|------|----------|----------|
| `.env` | Secrets only: MONGODB_URI, Cloudinary, SMTP, Twilio | Highest |
| `config/config.jsonc` | All tunables: thresholds, weights, sizes, schedules | Middle |
| `config/settings.py` | Loads both, resolves via `_get(env_key, config_key, default)` | Resolver |
| `config/bytetrack_surveillance.yaml` | ByteTrack tracker params | ByteTrack reads directly |

Priority: `.env` > `config.jsonc` > hardcoded defaults in `settings.py`

---

## Dashboard Routes

| Method | Endpoint | Purpose |
|--------|----------|---------|
| WS | `/ws/live` | Live frame + alerts + events push |
| GET | `/api/events/stats` | Dashboard statistics |
| GET | `/api/events` | Event listing with status filters |
| GET | `/api/faces` | Unknown/verified person listing |
| POST | `/api/faces/{id}/verify` | Name an unknown person |
| DELETE | `/api/faces/{id}` | Delete face record |
| GET | `/api/reports/stats` | Report statistics |
| GET | `/api/reports/summary` | Daily/weekly summary |
| POST | `/api/chat` | LLM conversational query |
| GET | `/api/chat/health` | LLM availability check |
| * | `/captures/*` | Face crop images (static) |

---

## Where to Modify

| Task | File(s) |
|------|---------|
| **Change match threshold** | `config/config.jsonc` (`MATCH_THRESHOLD`) |
| **Change camera resolution** | `config/config.jsonc` (`FRAME_WIDTH`, `FRAME_HEIGHT`) |
| **Change YOLO confidence** | `config/config.jsonc` (`PERSON_CONF_THRESHOLD`) |
| **Change face quality gates** | `config/config.jsonc` (`QUALITY_VALID_*`) |
| **Change quality scoring weights** | `config/config.jsonc` (`QUALITY_WEIGHT_*`) |
| **Change confidence formula** | `agents/scoring.py` (5-tuple in `compute_confidence`) |
| **Change recognition thresholds** | `agents/recognition.py` (similarity bands) |
| **Change policy rules** | `agents/policy.py` (9-rule priority tree) |
| **Add new alert channel** | `agents/alert_agent.py` + `config/config.jsonc` (`ALERT_CHANNELS`) |
| **Change LLM model** | `.env` (`OLLAMA_MODEL`) |
| **Change LLM prompts** | `utils/llm_client.py` |
| **Change vector search** | `utils/db_utils.py` (`vector_search()`) |
| **Fix face quality** | `pipeline/quality_agent.py` |
| **Fix confidence calc** | `agents/scoring.py` + `logs/calculation.log` |
| **Fix MongoDB ops** | `utils/db_utils.py` |
| **Fix camera loop** | `agents/camera_agent.py` |
| **Fix threading** | `agents/camera_agent.py` + `pipeline/track_state.py` |
| **Add API endpoint** | `dashboard/backend/routes/` + register in `main.py` |
| **Add dashboard panel** | `dashboard/frontend/src/components/` + import in `App.jsx` |
| **Add test** | `tests/` (run `python -m pytest tests/ -v`) |
