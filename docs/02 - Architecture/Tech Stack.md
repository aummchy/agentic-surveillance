# Tech Stack

> Every technology used in the system and how it connects.

## Core stack

Full detailed table: see [ARCHITECTURE.md](../../docs/ARCHITECTURE.md) (canonical reference).

| Component | Technology | File | How it's used |
|-----------|-----------|------|---------------|
| **Person detection** | YOLOv8 (`yolov8s`) | `pipeline/tracker.py` | Singleton model, detects person class only (class 0). OpenVINO IR export for GPU acceleration. |
| **Multi-object tracking** | ByteTrack | `config/bytetrack_surveillance.yaml` | Cross-frame person ID stability. Tuned for fixed-camera surveillance: `track_high_thresh=0.45`, `track_buffer=60`. |
| **Face detection** | InsightFace SCRFD (`buffalo_l`) | `utils/embedding_utils.py` | CLAHE → SCRFD detection → 512-dim ArcFace embedding. Singleton loaded once. |
| **Face embedding** | ArcFace (512-dim L2-normalized) | `utils/embedding_utils.py` | Quality-gated: only overwrites if det_score improves by ≥0.05. |
| **Mask detection** | Geometric heuristic | `utils/embedding_utils.py` | Landmark ratio: lower_face/upper_face < 0.3 → masked. No classifier. |
| **Vector search** | MongoDB Atlas `$vectorSearch` | `utils/db_utils.py` | HNSW index on `latest_embedding` (512d cosine). Fallback to Python numpy scan on failure. |
| **Database** | MongoDB Atlas | `utils/db_utils.py` | 3 collections: `faces`, `events`, `visit_memory`. Thread-safe lazy singleton client. |
| **Image storage** | Cloudinary + local fallback | `utils/image_utils.py` | Uploads face crops + full frames. Falls back to `captures/` directory. |
| **Local LLM** | Ollama (Gemma 3 4B / Qwen 3.5 4B) | `utils/llm_client.py` | NL alert summaries + dashboard chat. HTTP client with connection pooling. Optional — system works without it. |
| **Dashboard API** | FastAPI + Uvicorn | `dashboard/backend/main.py` | REST endpoints + WebSocket. Port 8000. CORS to localhost:5173. |
| **Dashboard frontend** | React + Vite | `dashboard/frontend/` | Live video feed via WebSocket, event history, face management. |
| **Configuration** | `.env` + `config.jsonc` | `config/settings.py` | Env vars for secrets, JSONC for tunables. Custom JSONC parser (strips comments). |
| **Logging** | structlog (3-tier) | `config/settings.py` | Terminal (compact ANSI), JSON file (machine-readable), debug file (verbose). |

## Python key libraries

| Library | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 inference + ByteTrack integration |
| `insightface` | SCRFD face detection + ArcFace embedding |
| `onnxruntime` | InsightFace inference backend |
| `opencv-python` | Video capture, image processing, CLAHE, JPEG encoding |
| `pymongo` | MongoDB driver (sync) |
| `fastapi` | Dashboard REST API |
| `uvicorn` | ASGI server for FastAPI |
| `structlog` | Structured logging with processors |
| `numpy` | Array operations, cosine similarity |
| `httpx` | HTTP client for Ollama (connection-pooled) |
| `requests` | Webhook alerts (simpler than httpx) |
| `python-dotenv` | Load `.env` file |
| `cloudinary` | Image upload (optional) |

## Hardware acceleration

| Device | Component | How |
|--------|-----------|-----|
| CPU | YOLO (default) | `YOLO_DEVICE=cpu` |
| Intel Arc iGPU | YOLO (OpenVINO) | Export to OpenVINO IR, `OPENVINO_DEVICE=GPU` |
| NVIDIA GPU | YOLO (CUDA) | `YOLO_DEVICE=0` |
| CPU | InsightFace | Always CPU (`CPUExecutionProvider`) |

## Data format: embeddings

| Stage | Format | Dimensions |
|-------|--------|-----------|
| Raw from ArcFace | float32 numpy array | 512 |
| L2-normalized | float32 numpy array, ‖v‖=1 | 512 |
| Stored in MongoDB | JSON array of floats | 512 |
| Query to Atlas | JSON array of floats | 512 |
| Atlas score | `(1 + raw_cosine) / 2` → [0,1] | scalar |
| Raw cosine | `(atlas_score × 2) - 1` → [-1,1] | scalar |

## See also
- [[System Overview]] — how it all fits together
- [[Data Flow]] — per-frame walkthrough
- [[All Config Settings]] — every configuration option
