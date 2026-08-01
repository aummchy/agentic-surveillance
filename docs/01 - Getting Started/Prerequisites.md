# Prerequisites

## Required

| Component | Version | Why |
|-----------|---------|-----|
| **Python** | 3.11 | InsightFace/onnxruntime wheels unreliable on other versions |
| **MongoDB Atlas** | Any | Vector search index for face matching |
| **Node.js** | 18+ | Dashboard frontend build |
| **Camera** | USB webcam or RTSP stream | Video input |

## Python packages

```bash
pip install -r requirements.txt
```

Key dependencies:
- `ultralytics` — YOLOv8 person detection
- `insightface` — SCRFD face detection + ArcFace embedding
- `onnxruntime` — InsightFace backend
- `pymongo` — MongoDB driver
- `fastapi` + `uvicorn` — Dashboard API
- `structlog` — Structured logging
- `opencv-python` — Video capture + image processing

## MongoDB Atlas setup

1. Create a cluster at https://cloud.mongodb.com
2. Create database `surveillance` with collections: `faces`, `events`, `visit_memory`
3. Create Vector Search index:
   - Name: `vector_index`
   - Collection: `faces`
   - Field: `latest_embedding`
   - Dimensions: 512
   - Similarity: cosine
4. Set `MONGODB_URI` in `.env`

## Camera

- **USB webcam**: `CAMERA_INDEX=0` (default)
- **IP camera / phone**: `CAMERA_SOURCE=http://<phone-ip>:8080/video` (via IP Webcam app)
- **RTSP stream**: `CAMERA_SOURCE=rtsp://<ip>:554/stream`

## Optional

| Component | Purpose |
|-----------|---------|
| **Ollama** | Local LLM for NL alert summaries + dashboard chat |
| **Intel Arc iGPU** | GPU acceleration for YOLO via OpenVINO |
| **Cloudinary** | Cloud image storage (falls back to local `captures/`) |

### Ollama setup
```bash
ollama pull gemma3:4b    # or qwen3.5:4b
```
Set `OLLAMA_MODEL` in `.env` to switch models.

### OpenVINO export
```bash
yolo export model=models/yolov8s.pt format=openvino half=True
```
Set `YOLO_MODEL=models/yolov8s_openvino_model/` and `OPENVINO_DEVICE=GPU`.
