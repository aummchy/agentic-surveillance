# SYSTEM_INDEX.md

Quick-lookup navigation map for AI agents. One sentence per file. See `AGENTS.md` for architecture, `LOCATIONS.md` for quick navigation, `TOOLS.md` for developer tools, `docs/ARCHITECTURE.md` for tech stack, `docs/FORMULAS.md` for formulas.

---

## Repository Layout

```
├── main.py                          # Entry point: wires agents, starts camera + FastAPI + workers
├── AGENTS.md                        # AI agent context: architecture, coding rules, do/don'ts
├── LOCATIONS.md                     # Quick navigation: 20-line file-to-feature map
├── TOOLS.md                         # Developer tools: repomix, ctags, ast-grep, commands
├── config/                          # Configuration system
│   ├── settings.py                  # Loads .env + config.jsonc, resolves 80+ settings, 3-tier logging
│   ├── config.jsonc                 # Tunable parameters (thresholds, weights, sizes, schedules)
│   └── bytetrack_surveillance.yaml  # ByteTrack tracker tuning for fixed-camera indoor use
├── agents/                          # Decision pipeline agents
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
├── pipeline/                        # Computer vision pipeline
│   ├── tracker.py                   # YOLOv8 singleton + ByteTrack track_persons() generator
│   ├── track_state.py               # Thread-safe per-track state, composite ID generation, expiry
│   ├── quality_agent.py             # Face quality scoring: validity gates + weighted composite [0,1]
│   ├── face.py                      # compute_face_ratio() — face area / person bbox
│   └── models.py                    # Data classes: Track, MatchResult, DecisionResult, QualityResult
├── utils/                           # Shared utilities
│   ├── db_utils.py                  # MongoDB CRUD, Atlas vector search, Python cosine fallback
│   ├── embedding_utils.py           # InsightFace singleton (SCRFD+ArcFace), compare_similarity()
│   ├── image_utils.py               # Crop, resize, save, Cloudinary upload, blur/brightness scoring
│   └── llm_client.py                # Ollama HTTP client: generate, chat, health check, retry
├── dashboard/
│   ├── backend/
│   │   ├── main.py                  # FastAPI app: CORS, routers, static /captures mount
│   │   ├── models.py                # Pydantic response/request models
│   │   └── routes/
│   │       ├── live.py              # /ws/live — WebSocket: live frame + alert + event push
│   │       ├── faces.py             # /api/faces — CRUD for unknown/verified persons
│   │       ├── events.py            # /api/events — event listing, stats, filtering
│   │       ├── reports.py           # /api/reports — stats, summary, incident reports via LLM
│   │       └── chat.py              # /api/chat — conversational LLM endpoint
│   └── frontend/src/
│       ├── App.jsx                  # Root: WebSocket, stats polling, view routing, alert state
│       ├── components/
│       │   ├── LiveFeed.jsx         # Live camera frame display + alert overlay
│       │   ├── UnknownPersons.jsx   # Unknown faces panel with verify button
│       │   ├── VerifiedPersons.jsx  # Verified persons panel with delete
│       │   ├── EventLog.jsx         # Event history with status tabs + real-time prepend
│       │   ├── ChatPanel.jsx        # LLM chat interface
│       │   ├── VerifyModal.jsx      # Modal for naming unknown person
│       │   └── ErrorBoundary.jsx    # React error boundary
│       └── utils/api.js             # Axios instance with base URL + error interceptor
├── tests/
│   ├── test_recognition.py          # Recognition threshold + confidence formula regression
│   ├── test_embedding_history.py    # Embedding history cap + double-counting fix
│   └── test_thread_safety.py        # Atomic visit-memory + in-flight track deferral
├── docs/
│   ├── ARCHITECTURE.md              # Tech stack reference
│   ├── FORMULAS.md                  # End-to-end data flow with exact formulas
│   ├── ISSUES.md                    # Active bugs + remediation plan (7 remaining)
│   ├── CHANGELOG.md                 # Completed fixes, 94 issues resolved
│   ├── PAPER.md                     # Academic paper draft
│   └── TERMINAL_OUTPUT.md           # Terminal logging event format reference
├── models/
│   ├── yolov8s.pt                   # YOLOv8s weights (~22MB, gitignored)
│   └── yolov8s_openvino_model/      # OpenVINO IR export for Intel iGPU acceleration
├── scripts/                         # Debug/utility scripts (gitignored)
├── captures/face_crops/             # Runtime face crop storage (gitignored)
└── logs/                            # Runtime logs: surveillance.jsonl, debug.log, calculation.log
```

---

## Execution Flow

```
python main.py
  ├── validate_config()
  ├── log_formula_header() → logs/calculation.log
  ├── Background: _check_llm_background()
  ├── Start 2 worker threads: process_finalized_track()
  ├── Start FastAPI (port 8000) in daemon thread
  ├── Background: _run_startup_checks() + _prewarm_yolo() + _prewarm_insightface()
  └── CameraAgent.start() [BLOCKS]
        └── Per frame:
              ├── YOLOv8 detect → ByteTrack associate → TrackState.update()
              ├── SCRFD face detect → quality score → ArcFace embed
              ├── Progressive recognition every N frames
              ├── on_frame_annotated → JPEG → WebSocket broadcast
              └── on_track_finalized → track_queue.put()

Worker (process_finalized_track):
  track_queue.get()
    ├── vector_search() → MatchResult
    ├── MemoryAgent.run() → MemoryContext
    ├── RecognitionAgent.run() → recognition_result
    ├── PolicyAgent.run() → DecisionResult
    ├── store_face() if registering
    ├── record_visit() if matched
    ├── dispatch() if alerting
    └── log_event() + broadcast_event()
```

---

## Module Dependencies (who imports whom)

```
main.py → config.settings, agents.*, utils.db_utils, utils.image_utils, utils.llm_client,
          pipeline.models, pipeline.tracker, utils.embedding_utils, dashboard.backend.routes.live

camera_agent.py → config.settings, pipeline.tracker, pipeline.track_state, pipeline.quality_agent,
                   pipeline.face, pipeline.models, utils.image_utils, utils.embedding_utils,
                   agents.recognition, agents.policy, agents.memory

matching_agent.py → pipeline.models, utils.db_utils

recognition.py → agents.base, agents.scoring, config.settings

scoring.py → config.settings

policy.py → config.settings, pipeline.models

memory.py → config.settings, utils.db_utils

alert_agent.py → config.settings, pipeline.models, utils.llm_client

db_utils.py → config.settings, utils.embedding_utils, pymongo

embedding_utils.py → config.settings, insightface

llm_client.py → config.settings, httpx

dashboard/backend/main.py → dashboard.backend.routes.*
routes/chat.py → utils.llm_client, config.settings, utils.db_utils
routes/events.py → dashboard.backend.models, utils.db_utils
routes/faces.py → dashboard.backend.models, utils.db_utils
routes/live.py → (stdlib only: asyncio, json, base64)
routes/reports.py → agents.report
```

---

## Config Map

| File | Controls | Override priority |
|------|----------|-------------------|
| `.env` | Secrets only: MONGODB_URI, Cloudinary, SMTP, Twilio, CAMERA_SOURCE | Highest (env vars) |
| `config/config.jsonc` | All tunables: thresholds, weights, sizes, schedules, channels | Middle |
| `config/settings.py` | Loads both, resolves via `_get(env_key, config_key, default)` | N/A (resolver) |
| `config/bytetrack_surveillance.yaml` | ByteTrack tracker params (track_thresh, buffer, match_thresh) | ByteTrack reads directly |

Priority chain: `.env` > `config.jsonc` > hardcoded defaults in `settings.py`

---

## Dashboard Routes

| Method | Endpoint | Handler | Purpose |
|--------|----------|---------|---------|
| WS | `/ws/live` | `routes/live.py` | Live frame + alerts + events push |
| GET | `/api/events/stats` | `routes/events.py` | Dashboard statistics |
| GET | `/api/events` | `routes/events.py` | Event listing with status filters |
| GET | `/api/faces` | `routes/faces.py` | Unknown/verified person listing |
| POST | `/api/faces/{id}/verify` | `routes/faces.py` | Name an unknown person |
| PUT | `/api/faces/{id}` | `routes/faces.py` | Update face metadata |
| DELETE | `/api/faces/{id}` | `routes/faces.py` | Delete face record |
| GET | `/api/reports/stats` | `routes/reports.py` | Report statistics |
| GET | `/api/reports/summary` | `routes/reports.py` | Daily/weekly summary |
| POST | `/api/chat` | `routes/chat.py` | LLM conversational query |
| GET | `/api/chat/health` | `routes/chat.py` | LLM availability check |
| * | `/captures/*` | static mount | Face crop images |

---

## Where to Modify

| Task | File(s) |
|------|---------|
| **Change match threshold** | `config/config.jsonc` (`MATCH_THRESHOLD`) — NOT `.env` |
| **Change camera resolution** | `config/config.jsonc` (`FRAME_WIDTH`, `FRAME_HEIGHT`) |
| **Change YOLO confidence** | `config/config.jsonc` (`PERSON_CONF_THRESHOLD`) + `config/bytetrack_surveillance.yaml` (`track_high_thresh`) |
| **Change face quality gates** | `config/config.jsonc` (`QUALITY_VALID_*`) |
| **Change quality scoring weights** | `config/config.jsonc` (`QUALITY_WEIGHT_*`) |
| **Change confidence formula weights** | `agents/scoring.py` (hardcoded 5-tuple at top of `compute_confidence`) |
| **Change recognition decision bands** | `agents/recognition.py` (lines 97-167, similarity thresholds) |
| **Change policy rules** | `agents/policy.py` (`PolicyAgent.run()`, 9-rule priority tree) |
| **Add new alert channel** | `agents/alert_agent.py` (`dispatch()` method) + `config/config.jsonc` (`ALERT_CHANNELS`) |
| **Change alert cooldown** | `config/config.jsonc` (`ALERT_COOLDOWN_SECS`) |
| **Change office hours** | `config/config.jsonc` (`OFFICE_HOURS_START/END`, `OFFICE_DAYS`) |
| **Change vector search params** | `config/config.jsonc` (`VECTOR_SEARCH_*`) + `utils/db_utils.py` (`vector_search()`) |
| **Change LLM model** | `.env` (`OLLAMA_MODEL`) |
| **Change LLM prompts** | `utils/llm_client.py` (system prompts in `generate_nl_summary`, `chat_completion`) |
| **Change face crop storage** | `utils/image_utils.py` (`save_image`, `upload_to_cloudinary`) |
| **Change embedding pipeline** | `utils/embedding_utils.py` (InsightFace singleton) + `pipeline/quality_agent.py` |
| **Change track state lifecycle** | `pipeline/track_state.py` (`TrackState`, `get_expired_tracks`) |
| **Change ByteTrack behavior** | `config/bytetrack_surveillance.yaml` |
| **Add new API endpoint** | `dashboard/backend/routes/` (new file) + register in `dashboard/backend/main.py` |
| **Add new dashboard panel** | `dashboard/frontend/src/components/` (new file) + import in `App.jsx` |
| **Change WebSocket protocol** | `dashboard/backend/routes/live.py` + `App.jsx` (WS handler) |
| **Fix face quality scoring** | `pipeline/quality_agent.py` |
| **Fix confidence calculation** | `agents/scoring.py` + `logs/calculation.log` |
| **Fix MongoDB operations** | `utils/db_utils.py` |
| **Fix camera loop** | `agents/camera_agent.py` |
| **Fix blurry face identity corruption** | `agents/camera_agent.py:306-331` — quality gate skip (`else: return` prevents embedding storage from low-quality faces) |
| **Fix threading/race conditions** | `agents/camera_agent.py` (track finalization) + `pipeline/track_state.py` (locks) |
| **Add new test** | `tests/` (follow existing `test_*.py` pattern, run `python -m pytest tests/ -v`) |
| **Change logging format** | `config/settings.py` (CompactTerminalRenderer, formatters) |
| **Change terminal event display** | `config/settings.py` (`TERMINAL_ALLOWLIST`) |
