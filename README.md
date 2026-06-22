# Surveillance System — Agentic AI

AI-powered real-time person detection, tracking, and facial recognition with **5 intelligent agents** working together.

---

## What This System Does

```
Camera → Detect Person → Track → Recognize → Decide → Alert → Report
                          ↓         ↓          ↓        ↓        ↓
                       ByteTrack  Memory    Policy    Context  Summary
                                  Agent     Agent     Agent    Agent
```

**Key Features:**
- Real-time person detection (YOLOv8) and tracking (ByteTrack)
- Face recognition with memory of past visits
- Intelligent decisions based on multiple factors
- Context-aware alerts (not just "unknown → alert")
- Live dashboard with React + WebSocket
- Structured logging with structlog + rotating file logs

---

## Quick Start (5 minutes)

### 1. Clone and install

```bash
git clone <your-repo-url>
cd surveillance-system

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Install frontend

```bash
cd dashboard/frontend
npm install
cd ../..
```

### 3. Configure environment

```bash
# Copy the example config
cp .env.example .env

# Edit .env and add your MongoDB URI (REQUIRED)
# All other settings have sensible defaults
```

**Minimum required configuration:**
```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
```

### 4. Set up MongoDB Atlas

1. Create free cluster at [mongodb.com](https://www.mongodb.com)
2. Create database: `surveillance`
3. Create collections: `faces`, `events`, `visit_memory`
4. Create Vector Search Index:
   - Name: `face_vector_index`
   - Collection: `faces`
   - Path: `latest_embedding`
   - Dimensions: `512`
   - Similarity: `cosine`
5. Whitelist your IP

### 5. Run

```bash
# Terminal 1: Start surveillance
python main.py

# Terminal 2: Start dashboard
cd dashboard/frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## How It Works — The 5 Agents

### Agent Pipeline

```
Person Detected
    │
    ▼
┌─────────────────┐
│  Memory Agent   │ ← Checks visit history
│  (Phase 2.2)    │   "Have I seen this person before?"
└────────┬────────┘
         │ memory_context
         ▼
┌─────────────────┐
│ Recognition     │ ← Analyzes face + memory
│ Agent (2.1)     │   "Is this person known?"
└────────┬────────┘
         │ recognition_result
         ▼
┌─────────────────┐
│ Policy Agent    │ ← Applies all rules
│ (Phase 2.3)     │   "What should we do?"
└────────┬────────┘
         │ decision
         ▼
┌─────────────────┐
│ Alert Agent     │ ← Chooses response
│ (Phase 2.4)     │   "How should we alert?"
└────────┬────────┘
         │ alert
         ▼
┌─────────────────┐
│ Report Agent    │ ← Generates summary
│ (Phase 2.5)     │   "What happened?"
└─────────────────┘
```

### Agent Details

| Agent | Input | Output | Purpose |
|-------|-------|--------|---------|
| **Memory** | person_id, camera_id | visit_count, confidence_boost | Track visit history |
| **Recognition** | similarity, quality, memory | status, confidence | Decide if person is known |
| **Policy** | all context | status, alert_level, actions | Apply business rules |
| **Alert** | status, alert_level | channels, priority | Choose alert method |
| **Report** | all data | title, summary, recommendation | Generate human-readable report |

---

## Decision Logic

The Policy Agent considers these rules (in priority order):

| Priority | Rule | Alert Level | Action |
|----------|------|-------------|--------|
| 1 | Blacklisted person | `critical` | Alert immediately, all channels |
| 2 | Authorized person | `none` | No alert |
| 3 | Verified visitor | varies | Alert based on stored level |
| 4 | Known visitor (memory) | `low` | Log only |
| 5 | Uncertain match | `low` | Log, may register |
| 6 | Person hiding face | `high` | Alert |
| 7 | Masked unknown | `medium`/`high` | Alert, register |
| 8 | Unknown after hours | `high` | Alert |
| 9 | Unknown (default) | `medium` | Alert, register |

---

## Configuration Reference

### Required

| Variable | Description |
|----------|-------------|
| `MONGODB_URI` | MongoDB Atlas connection string |

### Face Matching

| Variable | Default | Description |
|----------|---------|-------------|
| `MATCH_THRESHOLD` | `0.25` | Cosine similarity threshold (0-1, max 0.45) |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.40` | Threshold for merging duplicate unknowns |

### Detection

| Variable | Default | Description |
|----------|---------|-------------|
| `PERSON_CONF_THRESHOLD` | `0.5` | YOLO confidence for person detection |
| `TRACK_TIMEOUT_SECS` | `2.0` | Seconds before track expires |
| `MAX_TRACK_SECS` | `300` | Maximum track lifetime (5 min) |
| `DET_SCORE_MIN` | `0.50` | Minimum face detection score |
| `EMBEDDING_DET_SCORE_MIN` | `0.40` | Minimum score for embedding |

### Recognition

| Variable | Default | Description |
|----------|---------|-------------|
| `RECOGNITION_INTERVAL_FRAMES` | `30` | Run recognition every N frames |
| `LOITER_SECS` | `30` | Seconds before masked unknown escalates |
| `MIN_TRACK_FRAMES` | `30` | Min frames before hidden classification triggers |

### Alerting

| Variable | Default | Description |
|----------|---------|-------------|
| `ALERT_CHANNELS` | `console` | Channels: console, email, sms, webhook |
| `ALERT_COOLDOWN_SECS` | `60` | Min seconds between same alerts |

### Camera

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_INDEX` | `0` | Webcam device index |
| `FRAME_WIDTH` | `640` | Capture width |
| `FRAME_HEIGHT` | `480` | Capture height |
| `CAMERA_ID` | `cam_01` | Logical camera identifier |

---

## Project Structure

```
surveillance-system/
├── main.py                          # Entry point
├── .env                             # Your config (not in git)
├── .env.example                     # Config template
├── requirements.txt                 # Python dependencies
│
├── models/                          # Model weights (gitignored)
│   ├── .gitkeep
│   └── yolov8n.pt                   # YOLOv8 nano model
│
├── logs/                            # System logs (gitignored)
│   ├── .gitkeep
│   └── surveillance.log             # Rotating file: 5MB × 5 backups
│
├── agents/                          # Intelligent Agents
│   ├── base.py                      # BaseAgent ABC
│   ├── recognition.py               # Phase 2.1: Recognition decisions
│   ├── memory.py                    # Phase 2.2: Visit history
│   ├── policy.py                    # Phase 2.3: Business rules
│   ├── alert.py                     # Phase 2.4: Alert dispatch
│   ├── report.py                    # Phase 2.5: Report generation
│   ├── camera_agent.py              # Camera capture + tracking
│   ├── matching_agent.py            # Vector search matching
│   ├── decision_agent.py            # Delegates to PolicyAgent
│   └── alert_agent.py               # Legacy alert dispatch
│
├── pipeline/                        # CV Pipeline
│   ├── models.py                    # Dataclasses (Track, etc.)
│   ├── detector.py                  # YOLOv8 detection
│   ├── tracker.py                   # ByteTrack tracking
│   ├── track_state.py               # Track lifecycle
│   ├── face.py                      # Face detection/embedding
│   └── quality_agent.py             # Image quality scoring
│
├── utils/                           # Utilities
│   ├── db_utils.py                  # MongoDB + memory CRUD
│   ├── embedding_utils.py           # InsightFace singleton
│   └── image_utils.py               # Image processing
│
├── config/
│   └── settings.py                  # Configuration loader
│
└── dashboard/
    ├── backend/
    │   ├── main.py                  # FastAPI app
    │   ├── models.py                # Pydantic models
    │   └── routes/
    │       ├── faces.py             # Face CRUD endpoints
    │       ├── events.py            # Event endpoints
    │       ├── reports.py           # Report endpoints
    │       └── live.py              # WebSocket live feed
    └── frontend/
        ├── package.json
        ├── vite.config.js
        └── src/
            ├── App.jsx              # Main React app
            ├── index.css            # Bright, clean styling
            └── components/
                ├── LiveFeed.jsx     # WebSocket camera feed
                ├── EventLog.jsx     # Event log
                ├── UnknownPersons.jsx # Person cards
                └── VerifyModal.jsx  # Verification dialog
```

---

## API Endpoints

### Faces

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/faces` | List faces (`?status=unknown\|verified\|all`) |
| GET | `/api/faces/{person_id}` | Get single face |
| POST | `/api/faces/{person_id}/verify` | Verify person |
| PUT | `/api/faces/{person_id}` | Update face |
| DELETE | `/api/faces/{person_id}` | Delete face |

### Events

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/events` | List events |
| GET | `/api/events/stats` | Dashboard statistics |
| GET | `/api/events/unknown` | Unknown events |
| GET | `/api/events/alerts` | High/critical alerts |

### Reports

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/reports/stats` | Dashboard statistics |
| GET | `/api/reports/summary?period=daily` | Daily/weekly summary |
| GET | `/api/reports/person/{person_id}` | Person visit history |
| GET | `/api/reports/incidents` | Recent incidents |

### WebSocket

| Protocol | Path | Description |
|----------|------|-------------|
| WS | `/ws/live` | Live camera feed + alerts |

---

## Dashboard

The React dashboard provides:

- **Live Feed** — Real-time camera via WebSocket
- **Event Log** — Paginated events with filtering
- **Unknown Persons** — Grid of detected person cards
- **Verify Modal** — Assign names to unknown persons
- **Bright, clean UI** — White background, blue accents, high contrast

---

## Logging

Logs are written to **two destinations** via structlog + standard logging:

| Destination | Level | What you see |
|-------------|-------|-------------|
| **Console** (stdout) | `INFO` and above | `[info]` messages, alerts, errors — clean and minimal |
| **`logs/surveillance.log`** | `DEBUG` (all) | Full system log including `[debug]` frames, face quality scores, recognition details |

The file log uses Python's `RotatingFileHandler` — max **5 MB** per file, **5 backups** kept.

- `[debug]` messages only appear in the log file, not on console
- `logs/surveillance.log` also captures stdlib logs from third-party libraries (MongoDB driver, uvicorn)
- To see debug output on console, adjust the console handler level in `config/settings.py`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `MONGODB_URI is required` | Set `MONGODB_URI` in `.env` |
| Camera window black | Change `CAMERA_INDEX` in `.env` |
| No face embeddings | Lower `DET_SCORE_MIN` to `0.20` |
| Dashboard shows nothing | Ensure FastAPI running on port 8000 |
| Vite build error (`env/data.js`) | Run `npm install axios@1.7.9` — axios 1.7.10+ is incompatible with Vite's esbuild |
| `GET /api/events` returns 500 | `similarity_score` is null in MongoDB — ensure `Optional[float]` in `dashboard/backend/models.py` |
| Terminal too noisy | Console shows `INFO+`; full debug logs go to `logs/surveillance.log` |
| `FutureWarning` from insightface | Harmless — `estimate` deprecated in InsightFace 0.26, will be removed in 2.2. Safe to ignore |
| Slow performance | Use GPU: set `YOLO_DEVICE=0` |
| No local camera window | The system streams via WebSocket — open `http://localhost:5173` in your browser to see the feed |

---

## License

[Add your license here]
