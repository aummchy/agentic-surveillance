# Agentic AI Visitor Surveillance System — Build Specification

> **Purpose of this document.** This is a complete, self-contained build
> specification. An AI agent (or developer) should be able to build the entire
> system **from scratch** using only this file. It defines the full target
> architecture, the tech stack, data contracts, directory layout, and
> acceptance criteria.
>
> Read this top to bottom before writing any code. Build phase by phase. Do not
> skip the data contracts — every agent communicates through them.

---

## 1. What we are building

An **always-on, multi-agent video surveillance system** that:

1. Watches a camera feed continuously.
2. Detects **people** (not just faces) with YOLOv8.
3. Tracks each person across frames with **ByteTrack** (stable per-person ID).
4. Crops the face region and recognizes identity with **SCRFD + ArcFace**.
5. Stores face embeddings in **MongoDB Atlas** and searches them with vector
   search.
6. Makes an **autonomous decision** (authorized / known visitor / unknown /
   **masked unknown → HIGH ALERT**) without requiring an operator in the loop.
7. Raises **alerts** through a configurable alerting system.
8. Archives best-face images in **Cloudinary**.
9. Surfaces everything on a **web dashboard** with an audit log.

It is an academic / internship-grade project, not a hardened production
deployment — but it should be architecturally honest (real tracker, real
decision engine, real alerts), not a demo held together with `time.sleep`.

### Target pipeline (authoritative)

```
Camera
  ↓
YOLOv8n            (person detection)
  ↓
ByteTrack          (multi-object tracking → stable track_id per person)
  ↓
Face Crop          (extract face region from each tracked person box)
  ↓
SCRFD / InsightFace Detection   (precise face box + landmarks + mask check)
  ↓
Face Visibility Analyzer        (face_area / person_area ratio, track-level metrics)
  ↓
ArcFace Embedding  (512-dim L2-normalized vector)
  ↓
MongoDB Atlas Vector Search     (cosine similarity match)
  ↓
Decision Engine    (authorized / known / unknown / masked-unknown → HIGH ALERT)
  ↓
Cloudinary         (image archival)
  ↓
Dashboard          (live view, visitor log, alerts, audit trail)
```

### Why this architecture (key design decisions)

- **Person-first, not face-first.** Detecting people with YOLO and tracking them
  means we keep a stable identity for someone even when their face is turned
  away, occluded, or masked. Face detection runs *inside* each tracked person
  box. **The tracker provides stable identity across frames.**
- **One decision per track, not per frame.** We accumulate the best face for a
  track_id over its lifetime, recognize once, decide once, alert once. No
  duplicate records, no flicker.
- **Autonomous decision engine.** Recognition + policy run without blocking on
  operator `input()`. Operators review *after the fact* on the dashboard.
- **Mask handling is a first-class signal**, not an afterthought — see §7.

---

## 2. Tech stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | **Python 3.11** | Pin to 3.11 (insightface/onnxruntime wheels are reliable here) |
| Video / image | **OpenCV** (`opencv-python`) | Camera capture, drawing, image IO |
| Person detection | **Ultralytics YOLOv8** (`ultralytics`) | `yolov8n.pt` (nano) for CPU; person class only |
| Tracking | **ByteTrack** | Built into Ultralytics: `model.track(..., tracker="bytetrack.yaml", persist=True)` |
| Face detection | **InsightFace SCRFD** | Bundled in the `buffalo_l` model pack |
| Face embedding | **InsightFace ArcFace** (`buffalo_l`) | 512-dim, L2-normalized, ~99.8% LFW |
| Mask detection | **Lightweight classifier** | See §7 — landmark heuristic OR small CNN |
| Vector DB | **MongoDB Atlas** (`pymongo`) | Atlas Vector Search index on `embedding` |
| Image storage | **Cloudinary** (`cloudinary`) | Best-face archival, returns secure URL |
| Orchestration | **LangGraph** (`langgraph`) — optional but recommended | Stateful multi-agent coordination; can start with plain Python callbacks |
| Config | **python-dotenv** | All secrets/config in `.env` |
| Numerics | **numpy** | Embedding math |
| Backend API | **FastAPI** + **uvicorn** | Dashboard backend / REST + WebSocket |
| Frontend | **React** (Vite) OR **Streamlit** | Streamlit for speed; React for a real UI |
| Alerting | **smtplib / Twilio / webhook** | Pluggable; see §8 |
| Runtime | **onnxruntime** | InsightFace inference backend (CPU by default) |
| Local LLM | **Ollama** + **Gemma 3 4B** / **Qwen 3.5 4B** | NL summaries, report generation, conversational chat |
| HTTP client | **httpx** | Async Ollama API calls |

> **GPU note.** Everything runs on CPU. If a CUDA GPU is available, set
> InsightFace providers to `CUDAExecutionProvider` and YOLO `device=0` — the
> code should read the device from config, never hardcode it.

### `requirements.txt` (target — all phases)

```txt
# Core CV
opencv-python>=4.8.0
numpy>=1.24.0

# Detection + tracking
ultralytics>=8.1.0          # YOLOv8 + ByteTrack

# Face recognition
insightface>=0.7.3          # SCRFD detection + ArcFace embedding
onnxruntime>=1.16.0         # CPU inference (use onnxruntime-gpu if CUDA)

# Storage
pymongo>=4.6.0              # MongoDB Atlas + vector search
cloudinary>=1.36.0          # Image archival

# Orchestration
langgraph>=0.1.0            # Optional: stateful agent graph
langchain>=0.1.0

# Backend / dashboard
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
# streamlit>=1.30.0         # Alternative simple dashboard

# Alerting
twilio>=8.0.0              # Optional SMS
requests>=2.31.0          # Webhook alerts

# LLM
httpx>=0.27.0             # Ollama HTTP client

# Config
python-dotenv>=1.0.0
```

---

## 3. Repository layout (target)

```
surveillance-system/
├── agent.md                      # ← this file
├── README.md
├── requirements.txt
├── .env.example                  # template — NEVER commit real .env
├── .gitignore                    # must ignore .env, captures/, models/, *.pt
├── main.py                       # entry point, wires the pipeline
│
├── config/
│   └── settings.py               # loads .env, exposes constants, validate_config()
│
├── agents/
│   ├── camera_agent.py           # capture + tracking loop
│   ├── quality_agent.py          # best-face selection
│   ├── matching_agent.py         # embedding + DB search
│   ├── decision_agent.py         # autonomous authorize/alert policy
│   └── alert_agent.py            # dispatch alerts
│
├── pipeline/
│   ├── detector.py               # YOLOv8 person detection wrapper
│   ├── tracker.py                # ByteTrack wrapper → (track_id, box)
│   ├── face.py                   # SCRFD detect + ArcFace embed + mask check
│   ├── visibility_analyzer.py    # face_area / person_area ratio, track-level visibility
│   ├── track_state.py            # per-track accumulator (best face, decision)
│   └── decision_engine.py        # visibility + identity → alert decision
│
├── utils/
│   ├── image_utils.py            # blur/brightness/crop/resize/save/draw
│   ├── embedding_utils.py        # InsightFace singleton, embed, compare
│   ├── llm_client.py             # Ollama HTTP client (Gemma/Qwen)
│   └── db_utils.py               # MongoDB CRUD + vector search + audit log
│
├── dashboard/
│   ├── backend/                  # FastAPI app (REST + WebSocket live feed)
│   │   └── routes/
│   │       └── chat.py           # POST /api/chat, GET /api/chat/health
│   └── frontend/                 # React (Vite) or Streamlit app
│       └── src/components/
│           └── ChatPanel.jsx     # Conversational AI chat UI
│
├── models/                       # downloaded weights (gitignored)
│   ├── yolov8n.pt
│   └── (insightface buffalo_l auto-downloads to ~/.insightface)
│
└── captures/                     # runtime output (gitignored)
    ├── raw_frames/
    ├── best_faces/
    ├── known_faces/
    └── unknown_faces/
```

---

## 4. Data contracts (build these first — every agent depends on them)

These are the interfaces between agents. Lock them down before implementing.

### 4.1 Track object (in-memory, per person)

```python
@dataclass
class Track:
    track_id: str                  # composite: f"{camera_id}_{session_epoch}_{byte_track_id}"
    first_seen: float              # epoch seconds
    last_seen: float
    person_box: tuple              # (x1, y1, x2, y2) latest
    best_face_crop: np.ndarray | None   # face crop with best face_ratio (for ArcFace)
    best_face_score: float         # quality score of best_face_crop
    best_full_frame: np.ndarray | None  # full frame for the best face
    is_masked: bool                # latest mask determination
    embedding: list | None         # 512-dim, set after recognition
    decision: str | None           # see DecisionResult.status
    alerted: bool                  # has an alert already fired for this track
    # Progressive recognition state
    last_recognition_frame: int    # frame number of last recognition attempt
    pending_embedding: list | None # partial embedding from progressive recognition
    pending_match: dict | None     # partial match result from progressive recognition
    # Face visibility (track-level)
    total_frames_seen: int         # total frames this track appeared
    frames_with_detectable_face: int  # frames where SCRFD detected a face (analytics)
    face_detected_once: bool       # True if face was ever detected (decision logic)
    max_face_ratio: float          # max(face_area / person_area) across all frames
    best_face_ratio: float         # face_ratio of the best face detected
    visibility: str                # "visible" | "partial" | "hidden" | "unknown"
    max_track_secs: float = 300.0  # force finalization after this many seconds
```

### 4.2 Quality result

```python
{
    "blur_score": float,           # Laplacian variance (higher = sharper)
    "brightness": float,           # mean HSV-V, 0–255
    "face_area": int,              # bbox pixel area
    "is_valid": bool,              # passed all thresholds
    "overall_score": float         # normalized (0-1): blur_norm*0.60 + bright_norm*0.25 + area_norm*0.15
}
```

### 4.3 Embedding result

```python
{
    "embedding": np.ndarray | None,  # 512-dim L2-normalized
    "face_detected": bool,
    "detection_score": float,        # SCRFD det_score (threshold: DET_SCORE_MIN=0.50)
    "embedding_score": float,        # quality gate (threshold: EMBEDDING_DET_SCORE_MIN=0.70)
    "bbox": tuple | None,            # (x1, y1, x2, y2)
    "is_masked": bool,               # mask present on this face
    "error": str | None
}
```

### 4.4 Match result (from DB search)

```python
[
    {
        "person_id": str,
        "name": str,
        "role": str,                 # employee | visitor | contractor | unknown
        "tags": list[str],           # e.g. ["authorized", "vip", "blacklist"]
        "similarity_score": float,   # cosine, 0–1 in Atlas score space
        "image_url": str
    },
    ...
]
```

### 4.5 Decision result (the heart of autonomy)

```python
{
    "status": str,        # "authorized" | "known_visitor" | "unknown"
                          #   | "masked_unknown" | "intentionally_hidden" | "blacklist"
    "alert_level": str,   # "none" | "low" | "medium" | "high" | "critical"
    "person_id": str | None,
    "name": str | None,
    "reason": str,        # human-readable explanation for the audit log
    "should_alert": bool,
    "should_register": bool
}
```

### 4.6 MongoDB documents

**`faces` collection** (one per enrolled identity):

```python
{
    "person_id": "visitor_20260615_120000_0001",
    "name": "John Doe",
    "role": "visitor",                 # employee|visitor|contractor|unknown
    "embeddings": [[...], [...]],     # multiple 512-dim lists (ArcFace) over time
    "latest_embedding": [...],        # most recent embedding (fast search field)
    "embedding_model": "arcface",
    "images": [                       # multiple images over time
        {"url": "https://res.cloudinary.com/...", "captured_at": <datetime>}
    ],
    "source": {"camera_id": "cam_01", "captured_at": <datetime>},
    "quality_scores": {...},
    "tags": ["authorized"],            # authorized | blacklist | vip | known ...
    "created_at": <datetime>,
    "updated_at": <datetime>
}
```

**`events` collection** (audit log — one per decision):

```python
{
    "_id": <ObjectId>,                # use _id as event identifier (no separate event_id)
    "track_id": "cam_01_1718467200_42",  # composite track key
    "camera_id": "cam_01",
    "timestamp": <datetime>,
    "status": "masked_unknown",        # mirrors DecisionResult.status
    "alert_level": "high",
    "person_id": null,                 # null if unknown
    "name": null,
    "is_masked": true,
    "similarity_score": 0.32,         # actual best match score (even if below threshold)
    "image_url": "https://res.cloudinary.com/...",
    "reason": "Unknown person wearing mask — identity cannot be verified",
    "alerted": true
}
```

**Auto-registration dedup:** Before inserting a new unknown face, check if
embedding is close to existing unknown records (`DEDUP_SIMILARITY_THRESHOLD=0.50`).
If match found, update the existing record (add new image, bump `updated_at`)
instead of creating a duplicate.

**Indexes:**
- `faces`: index on `latest_embedding` for vector search, index on `role` and `tags`
- `events`: TTL index on `timestamp` with `expireAfterSeconds=7776000` (90 days),
  index on `camera_id`, `status`, `alert_level` for dashboard queries

> **Atlas Vector Search index** (create manually in Atlas UI, name
> `vector_index`): `path=latest_embedding`, `numDimensions=512`,
> `similarity=cosine`. The code must fall back to a Python cosine scan if the
> index is missing (so the system runs before the index is set up).

---

## 5. Phase plan

> Build phase by phase in order. Each phase builds on the previous one.

### Phase 1 — Camera capture & quality selection

**Goal:** Continuous camera loop that detects people, tracks them, and selects
the best face image per tracked person.

- `pipeline/detector.py`: load `yolov8n.pt` once; detect `person` class only;
  return list of person boxes + confidences.
- `pipeline/tracker.py`: run `model.track(frame, persist=True,
  tracker="bytetrack.yaml", classes=[0])`; return `[(track_id, box, conf)]`.
- `pipeline/track_state.py`: maintain a dict `{track_id: Track}`; for each
  tracked person, crop the person region, run quality scoring on the face within
  it, and keep the highest-scoring face crop as `best_face_crop`. Use a
  `threading.Lock` to protect the dict from concurrent access.
- Drop a track when its `track_id` hasn't been seen for `TRACK_TIMEOUT_SECS`
  OR when `track.first_seen + MAX_TRACK_SECS < now` (force-finalize long tracks);
  on drop, hand the finalized `Track` to the Quality → Matching → Decision chain.

**Progressive recognition (real-time alerts during active tracks):**

Every `RECOGNITION_INTERVAL_FRAMES` frames, for each active track where
`frames_since_last_recognition >= RECOGNITION_INTERVAL_FRAMES`:

1. If `best_face_crop` quality ≥ threshold, run SCRFD + ArcFace embedding.
2. Run vector search → `match_result`.
3. Run `decide(track, match_result)` → `DecisionResult`.
4. If `should_alert` and not yet alerted → dispatch alert **immediately**.
5. Store partial result on Track (`pending_embedding`, `pending_match`).

This enables real-time alerts for masked unknowns while they are still in frame.
The track-end finalization still runs and writes the audit event.

**Quality scoring:**
- blur_raw = `cv2.Laplacian(gray, cv2.CV_64F).var()` (reject < 60)
- brightness_raw = mean HSV-V (reject < 50 or > 230)
- face_area_raw = bbox pixel area (reject if face_area < 2500)
- Normalize each to 0-1 range:
  - blur_norm = min(blur_raw / QUALITY_BLUR_MAX, 1.0)
  - bright_norm = brightness_raw / 255.0
  - area_norm = min(face_area_raw / QUALITY_AREA_MAX, 1.0)
- `overall_score = blur_norm*0.60 + bright_norm*0.25 + area_norm*0.15`
- Final gate: InsightFace must confirm a real face (`det_score ≥ 0.70`) — this
  rejects reflections / window frames / posters.

**Acceptance:** Two people in frame simultaneously each get a distinct
`track_id` and their own best face. No double-processing of one person.

### Phase 2 — Recognition, MongoDB, Cloudinary

**Goal:** Turn a best-face image into an identity decision backed by a database.

- `utils/embedding_utils.py`: InsightFace `buffalo_l` loaded as a **singleton**
  (load once, never per-frame). `generate_embedding()` → 512-dim normed vector;
  use `DET_SCORE_MIN=0.50` for face detection, `EMBEDDING_DET_SCORE_MIN=0.70`
  as quality gate before generating embeddings.
- `utils/db_utils.py`: `$vectorSearch` pipeline on `vector_index`, cosine,
  with a Python cosine-scan fallback. **Threshold conversion is critical:**
  `MATCH_THRESHOLD` is raw cosine (−1..1); Atlas `vectorSearchScore` is
  `(1 + cosine)/2`, so filter on `atlas_threshold = (1 + MATCH_THRESHOLD) / 2`.
  Recommended `MATCH_THRESHOLD = 0.35`; **never above 0.45** (ArcFace
  same-person indoor scores land 0.30–0.55).
- **Threshold comparison function** — single source of truth for both paths:
  ```python
  def compare_similarity(raw_cosine: float, threshold: float = MATCH_THRESHOLD) -> bool:
      """Compare raw cosine similarity against threshold. Used by both Atlas and Python fallback."""
      return raw_cosine >= threshold

  def atlas_score_to_cosine(atlas_score: float) -> float:
      """Convert Atlas vectorSearchScore back to raw cosine for comparison."""
      return (atlas_score * 2) - 1
  ```
  Both the Atlas path and Python fallback MUST use `compare_similarity()` after
  converting Atlas scores to raw cosine. Do not compare raw values directly.
- Cloudinary upload → `secure_url`; store full face record in `faces`.

**Acceptance:** A previously enrolled face matches with score ≥ threshold and
returns the stored identity; a new face returns no match — both with **zero**
operator interaction.

### Phase 3 — Decision Agent & Alert Agent

**Goal:** Autonomous policy + alerting. This is what makes it "agentic" rather
than a recognition demo.

**`agents/decision_agent.py` — `decide(track, match_result) → DecisionResult`:**

Policy (evaluate in order; first match wins):

| Condition | status | alert_level | action |
|-----------|--------|-------------|--------|
| Match found AND tag `blacklist` | `blacklist` | **critical** | alert + log |
| Match found AND tag `authorized` | `authorized` | none | log only |
| Match found (known, not authorized) | `known_visitor` | low | log |
| No match AND visibility = `hidden` | `intentionally_hidden` | **high** | alert + log + save best frame |
| No match AND visibility = `partial` | `masked_unknown` | medium | log + auto-register |
| No match AND visibility = `visible` | `unknown` | medium | log + auto-register |

**Visibility classification (computed at track end):**

```python
if max_face_ratio >= VISIBLE_FACE_RATIO:
    visibility = "visible"
elif max_face_ratio >= PARTIAL_FACE_RATIO:
    visibility = "partial"
elif is_masked or face_detected_once:
    visibility = "partial"           # mask or distant face → partial
elif not face_detected_once and total_frames_seen >= MIN_TRACK_FRAMES:
    visibility = "hidden"
else:
    visibility = "unknown"
```

- Decisions are made **once per track** when the track ends. Never from a single frame.
- If face becomes visible at any point (`max_face_ratio >= VISIBLE_FACE_RATIO`), process
  recognition normally using `best_face_crop` — even if initially hidden.
- Masked person (mask only, upper face visible) → `partial`, not `hidden`.
- `authorized` wins regardless of visibility.
- Only classify `hidden` when no face was detected AND track is long enough.
  Brief appearances do not trigger intentionally_hidden.
- Distant people with detectable faces (small ratio but `face_detected_once` is True)
  are classified as `partial`, not `hidden`.

**Tag priority (evaluate in this exact order):**

1. `blacklist` — always wins, even if person is also `authorized`
2. `authorized` — no alert regardless of visibility
3. `known_visitor` — low alert
4. No tags — apply visibility-based rules below

**`agents/alert_agent.py` — `dispatch(decision, track) → bool`:**

- Pluggable channels selected by config: **console** (always), **email**
  (smtplib), **SMS** (Twilio), **webhook** (POST JSON to `ALERT_WEBHOOK_URL` —
  works with Slack/Discord/n8n).
- Alert payload: status, alert_level, timestamp, camera_id, track_id, image_url,
  reason, snapshot thumbnail.
- **Debounce:** never send more than one alert per track_id; rate-limit repeated
  same-level alerts to `ALERT_COOLDOWN_SECS` to avoid alert storms.
- Every alert also writes an `events` document (audit trail).

**Acceptance:** With camera running, an enrolled authorized person produces no
alert; an unknown unmasked person logs a medium event; an unknown **masked**
person fires a **high** alert through every configured channel exactly once.

### Phase 4 — Review dashboard & audit log

**Goal:** A web UI for operators — no more terminal-only output.

**Backend (`dashboard/backend/`, FastAPI):**
- `GET /api/events` — paginated audit log (filter by status, alert_level, date).
- `GET /api/faces` — enrolled identities (CRUD: enroll, edit name/role/tags,
  delete, mark authorized/blacklist).
- `GET /api/alerts` — active/recent high+ alerts.
- `WS /ws/live` — WebSocket pushing annotated frames + current tracks for a
  live view.
- `POST /api/faces/{person_id}/review` — operator confirms / relabels an
  auto-registered unknown.

**Frontend (`dashboard/frontend/`, React+Vite or Streamlit):**
- **Live view:** annotated camera feed with track boxes, IDs, names, alert
  badges.
- **Visitor log:** chronological events with thumbnails, identity, decision,
  alert level.
- **Alerts panel:** high/critical events front and center, with acknowledge.
- **Enrollment / people manager:** review unknowns, assign names/roles/tags,
  toggle authorized/blacklist.
- **Audit trail:** immutable `events` history for compliance.

**Acceptance:** Operator can watch the live feed, see a masked-unknown high
alert appear in real time, click it, view the snapshot, and either enroll the
person or acknowledge the alert — all from the browser.

---

## 6. Orchestration (how the agents connect)

Two acceptable approaches — pick based on time budget:

**A. Plain Python callbacks with queue decoupling (simplest).**
```
main.py
 └─ run_camera_agent()
        │  (YOLO + ByteTrack loop, per-track accumulation)
        │  (progressive recognition every RECOGNITION_INTERVAL_FRAMES)
        │
        ├──► recognition_worker (thread) — runs during-track recognition
        │
        └──► finalized_track_queue (queue.Queue)
               │
               └── worker_pool (ThreadPoolExecutor, 2-4 workers)
                     ├─ run_visibility_analyzer(track) ────────► visibility
                     ├─ run_matching_agent(best_face) ──────────► match_result
                     ├─ decide(track, match_result) ────────────► DecisionResult
                     ├─ if should_alert: alert_agent.dispatch()
                     ├─ if should_register: db_utils.store_face() + cloudinary
                     └─ db_utils.log_event()                    # always
```

The camera loop NEVER blocks on I/O. All MongoDB, Cloudinary, and alert work
runs in the worker pool. Frame rate stays constant regardless of DB latency.

**B. LangGraph state machine (recommended for "agentic" framing).**
Nodes: `capture → quality → match → decide → (alert | register) → log`. State
object carries the `Track`. Gives you human-in-the-loop hooks, retries, and a
clean graph the dashboard can visualize. Heavier to set up.

> Start with A to get an end-to-end working system, then optionally refactor to
> B. Do not let orchestration framework choice block early progress.

**Thread safety requirements:**
- Use `threading.Lock` around the track dict in `track_state.py`
- Wrap InsightFace singleton access with `threading.Lock` or run inference on
  a dedicated thread with a work queue
- Copy frame data before passing to WebSocket (OpenCV reuses buffers)

---

## 7. Masked-person detection & high alert (new requirement)

This is a required feature. Two parts: **detect the mask**, then **apply the
policy** (handled in §5 Phase 3 — Decision Agent & Alert Agent).

### Detecting a mask

Pick one (in increasing order of robustness):

1. **Geometric landmark heuristic (no extra model — start here).** SCRFD returns
   5 facial landmarks (eyes, nose, mouth corners). A mask occludes nose + mouth.
   NOTE: SCRFD does NOT return per-landmark confidence scores — use geometric
   analysis instead:
   ```python
   # landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
   nose_tip = landmarks[2]
   mouth_center = (landmarks[3] + landmarks[4]) / 2
   upper_face = (landmarks[0] + landmarks[1]) / 2  # eye midpoint

   lower_face_height = abs(mouth_center[1] - nose_tip[1])
   upper_face_height = abs(upper_face[1] - nose_tip[1])

   # If lower face region is abnormally small → mask likely
   if lower_face_height / (upper_face_height + 1e-6) < 0.3:
       is_masked = True
   ```
   Cheap, no training, good enough for a first cut.
2. **Lightweight mask classifier (recommended).** A small CNN (e.g. MobileNetV2
   fine-tuned on a mask/no-mask dataset) run on the face crop → `is_masked`
   bool + confidence. Many pretrained mask classifiers exist; wrap one in
   `pipeline/face.py`.
3. **Detection-confidence signal.** Masks lower ArcFace/SCRFD detection
   confidence. A face that is clearly a person (YOLO is confident) but whose
   SCRFD `det_score` is borderline AND lower-face landmarks are weak is a strong
   mask candidate. Use `DET_SCORE_MIN=0.50` for detection,
   `EMBEDDING_DET_SCORE_MIN=0.70` for embedding quality gate.

Set `is_masked` on the embedding result and on the `Track`. Combine signals:
mask = (classifier says mask) OR (geometric heuristic says lower face occluded).

### The high-alert rule

In the Decision Engine: **unknown identity (no DB match above threshold) AND
`is_masked == True` → `status="masked_unknown"`, `alert_level="high"`,
`should_alert=True`.** Save the snapshot to `unknown_faces/` and Cloudinary,
write an `events` doc, and dispatch through the Alert Agent.

Important nuances:
- A **masked but recognized authorized** person → `authorized`, no alert. (Mask
  alone is not suspicious; mask + *unknown* is.)
- If recognition is unreliable because of the mask, treat as **unknown** (fail
  toward alerting, not toward silently authorizing).
- Escalate to **critical** if a masked unknown is detected repeatedly or
  loiters (track lifetime > `LOITER_SECS`).

---

## 8. Configuration (`.env` / `config/settings.py`)

`config/settings.py` loads `.env` with `python-dotenv` and exposes constants +
`validate_config()`. **Never hardcode secrets. Commit `.env.example`, never
`.env`.** Ensure `.gitignore` excludes `.env`, `captures/`, `models/`, `*.pt`.

```ini
# ── MongoDB ──────────────────────────────────────────
MONGODB_URI=mongodb+srv://<user>:<pass>@cluster.mongodb.net
MONGODB_DATABASE=surveillance
MONGODB_COLLECTION=faces
MONGODB_EVENTS_COLLECTION=events

# ── Cloudinary ───────────────────────────────────────
CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=

# ── Models ───────────────────────────────────────────
YOLO_MODEL=yolov8n.pt
YOLO_DEVICE=cpu                 # or "0" for CUDA GPU 0
INSIGHTFACE_MODEL=buffalo_l
INSIGHTFACE_DET_SIZE=640
INSIGHTFACE_PROVIDER=CPUExecutionProvider

# ── Detection / tracking ─────────────────────────────
PERSON_CONF_THRESHOLD=0.5
TRACK_TIMEOUT_SECS=8.0          # drop a track unseen this long
MAX_TRACK_SECS=300              # force-finalize track after this many seconds
DET_SCORE_MIN=0.50              # face detection threshold (lower for detection)
EMBEDDING_DET_SCORE_MIN=0.70    # quality gate before generating embeddings

# ── Progressive recognition ─────────────────────────
RECOGNITION_INTERVAL_FRAMES=20  # run face recog every N frames per track

# ── Quality scoring ───────────────────────────────────
QUALITY_BLUR_MAX=1000           # Laplacian variance cap for normalization
QUALITY_AREA_MAX=10000          # face area cap for normalization

# ── Matching ─────────────────────────────────────────
MATCH_THRESHOLD=0.35            # raw cosine; do NOT exceed 0.45
DEDUP_SIMILARITY_THRESHOLD=0.50 # merge unknowns above this similarity

# ── Mask / decision ──────────────────────────────────
MASK_DETECTION=heuristic        # heuristic | classifier
LOITER_SECS=30                  # masked unknown beyond this → critical

# ── Face visibility ──────────────────────────────────
VISIBLE_FACE_RATIO=0.025          # face_ratio >= this = visible
PARTIAL_FACE_RATIO=0.010          # face_ratio >= this = partial
MIN_TRACK_FRAMES=15               # min frames before hidden classification (was 30)

# ── Alerting ─────────────────────────────────────────
ALERT_CHANNELS=console,webhook  # console,email,sms,webhook
ALERT_WEBHOOK_URL=
ALERT_COOLDOWN_SECS=60
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=
ALERT_EMAIL_TO=
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM=
ALERT_SMS_TO=

# ── Camera ───────────────────────────────────────────
CAMERA_INDEX=0
FRAME_WIDTH=640
FRAME_HEIGHT=480
CAMERA_ID=cam_01

# ── Local LLM (Ollama) ──────────────────────────────
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b          # swap to qwen3.5:4b for better reasoning
OLLAMA_TIMEOUT=120              # seconds
```

---

## 9. Build order (do this, in this sequence)

1. **Scaffold + config.** Repo layout (§3), `requirements.txt` (§2),
   `.env.example` + `.gitignore`, `config/settings.py` with `validate_config()`.
2. **Data contracts (§4).** Define the dataclasses / dict shapes. Everything
   keys off these.
3. **Utils.** `image_utils.py` (blur/brightness/crop/save/draw),
   `embedding_utils.py` (InsightFace singleton + embed + compare),
   `db_utils.py` (Mongo CRUD + vector search + fallback + `log_event`).
4. **Camera & tracking pipeline.** `detector.py` (YOLO), `tracker.py` (ByteTrack),
   `track_state.py`, `camera_agent.py`, `quality_agent.py`. Verify two people
   get distinct stable IDs and best faces.
5. **Recognition pipeline.** `matching_agent.py` — embedding + DB search + Cloudinary.
   Verify enroll + re-match round-trips through Atlas.
6. **Decision & alerting.** `face.py` mask detection, `decision_agent.py` policy,
   `alert_agent.py` channels. Verify the masked-unknown high alert end to end.
7. **Dashboard.** FastAPI backend + dashboard frontend reading `events`/`faces`,
   live WebSocket feed, operator review/enroll.
8. **Hardening.** Single-load models (no per-frame reload), alert debounce,
   secret hygiene, graceful camera release on exit.
9. **LLM integration.** Ollama HTTP client (`utils/llm_client.py`), NL alert
   summaries in `alert_agent.py`, enhanced reports in `report.py`, conversational
   chat endpoint (`dashboard/backend/routes/chat.py`), `ChatPanel.jsx` frontend.
   All LLM calls have template fallback. Model swap via `OLLAMA_MODEL` env var.

---

## 10. Best practices

- **Load models once.** Load YOLO once at startup, InsightFace as a singleton.
  Never reload models inside a per-frame loop.
- **Thread safety.** Use `threading.Lock` around the track dict and InsightFace
  singleton access. Camera loop and worker pool run concurrently.
- **Decouple I/O from camera loop.** Use `queue.Queue` + `ThreadPoolExecutor`
  for MongoDB, Cloudinary, and alert work. Camera loop must never block.
- **Autonomous decisions.** Decisions must be autonomous without blocking on
  `input()`. Operators review events asynchronously on the dashboard.
- **Use composite track IDs.** Use `f"{camera_id}_{session_epoch}_{byte_track_id}"`
  to prevent collision after camera restarts.
- **Generous face crops.** InsightFace is more reliable on full-frame /
  loosely-cropped context. Feed the recognition stage a generous crop or the
  full frame, not a tight face box.
- **Threshold conversion.** Atlas `vectorSearchScore = (1+cosine)/2`. Convert
  back to raw cosine before comparing: `raw_cosine = (atlas_score * 2) - 1`.
  Always use `compare_similarity()` — never compare raw values directly.
- **Reasonable thresholds.** Keep `MATCH_THRESHOLD` ≤ 0.45 — above this
  rejects genuine same-person matches under indoor lighting.
- **Secret hygiene.** `.env` must be gitignored from commit #1; only
  `.env.example` is tracked.
- **Meaningful directories.** Either write to `known_faces/` / `unknown_faces/`
  meaningfully (save unknown/masked snapshots there for the dashboard) or rely
  on Cloudinary + Mongo as the durable store and document that.
- **Track-level visibility, not single-frame.** Use max_face_ratio across the
  entire track. Never classify intentional hiding from a single frame.
- **Hidden means no face detected in a long track.** Only classify `hidden` when
  face_detected_once is False AND total_frames_seen >= MIN_TRACK_FRAMES.
  Brief appearances do not trigger intentionally_hidden.
- **Auto-registration dedup.** Before inserting a new unknown face, check if
  embedding is close to existing unknown records. Merge instead of duplicating.

---

## 11. Acceptance criteria (definition of done, whole system)

1. Camera runs continuously; `Q` (or API stop) shuts down cleanly, releasing the
   device.
2. Two people in frame simultaneously each receive a distinct, stable
   `track_id` and are processed independently.
3. An enrolled authorized person is recognized and produces **no** alert.
4. An unknown, unmasked person is logged as `unknown` (medium) and
   auto-registered for later review — no operator interaction.
5. An unknown **masked** person fires a **high** alert through every configured
   channel **exactly once** per track, with a snapshot saved to Cloudinary and
   an `events` doc written.
6. A masked but *authorized* person does **not** alert.
7. The dashboard shows the live feed, the visitor/audit log, and lets an
   operator review/enroll an unknown — all without touching the terminal.
8. No model is loaded inside any per-frame loop; no secret is committed to git.
9. An unknown person whose face is NEVER detected during a sufficiently long
    track (>= 15 frames) fires a **high** `intentionally_hidden` alert.
10. A person who appears briefly (< 15 frames) with no face detected is NOT
    classified as `intentionally_hidden`.
11. A masked person (mask only) is classified as `partial`, not `hidden`.
12. A distant person with small face_ratio but detectable face is NOT
    classified as `intentionally_hidden`.
13. A masked unknown person triggers a **real-time alert** during the track
    (not just at track end) via progressive recognition.
14. Camera frame rate stays constant regardless of MongoDB/Cloudinary latency
    (I/O is decoupled via queue).
15. Track IDs are unique across camera restarts (composite key).
16. Auto-registered unknowns are deduplicated — same person appearing multiple
    times does not create duplicate face records.
17. LLM generates natural-language alert summaries visible in console, email, SMS,
    WebSocket broadcasts, and dashboard notifications — with template fallback if
    Ollama is unavailable.
18. Chat endpoint (`POST /api/chat`) answers natural-language questions about
    surveillance data (stats, recent events, unknown persons, visit history) with
    LLM-polished responses — with template fallback if Ollama is unavailable.
19. LLM model can be swapped via single env var change (`OLLAMA_MODEL`).