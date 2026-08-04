# System Overview

> The system is an AI-powered single-camera surveillance pipeline that detects people, tracks them across frames, recognizes their faces against a known-person database, makes autonomous security decisions, and dispatches alerts — all in real-time.

## What it does, end to end

```
Camera frame
    │
    ▼
YOLOv8 person detection (finds people in frame)
    │
    ▼
ByteTrack tracking (assigns stable IDs across frames)
    │
    ▼
Progressive face recognition (every 20 frames per track)
    │   ├─ InsightFace SCRFD face detection
    │   ├─ Face quality scoring (blur, brightness, area)
    │   ├─ ArcFace 512-dim embedding generation
    │   ├─ MongoDB Atlas Vector Search (find similar known faces)
    │   ├─ Memory lookup (visit history, patterns)
    │   ├─ Confidence scoring (weighted formula, 1-100)
    │   └─ Policy decision (9-rule priority tree)
    │
    ▼
When person leaves frame (track expires):
    ├─ Final match + decision
    ├─ Store face in MongoDB (auto-register unknowns)
    ├─ Record visit in memory
    ├─ Dispatch alerts (console, email, SMS, webhook)
    └─ Broadcast to dashboard via WebSocket
```


SCRFD** stands for ==**Sample and Computation Redistribution for Efficient Face Detection**==
## The two halves

The system runs as **two concurrent halves** connected by a `queue.Queue`:

| Half | Thread | What it does | Blocking? |
|------|--------|-------------|-----------|
| **Camera** | Main thread | Capture frames, detect, track, run progressive recognition | Yes (this IS the main loop) |
| **Workers** | 2 background threads | Finalize tracks, match, decide, store, alert | No (queue + executor) |

This decoupling ensures the camera loop **never blocks** on slow I/O (MongoDB, Cloudinary, alerts).

## Key design principles

1. **Load models once** — YOLO in `tracker.py`, InsightFace as singleton in `embedding_utils.py`. Never reload in per-frame loops.

2. **Thread safety** — `TrackState` uses `threading.Lock`. Camera thread and worker pool run concurrently.

3. **I/O decoupled from camera** — MongoDB, Cloudinary, alerts run via `queue.Queue` + workers. Camera loop must never block.

4. **Quality gates before embedding storage** — Invalid faces (blur < 40, brightness outside 35-255, area < 1200px²) never get embeddings stored or searched.

5. **Confidence never downgrades** — Only upgrades across recognition passes. Critical alerts always update.

6. **One decision per track** — Recognition + decision runs once when track ends (or progressively every 20 frames). Not per-frame.

## Source files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, wires everything |
| `agents/camera_agent.py` | Camera loop + progressive recognition |
| `agents/track_processor.py` | Track finalization + dashboard broadcasting |
| `agents/finalizer.py` | Final embedding retry on track expiry |
| `agents/timing.py` | Thread-safe timing diagnostics |
| `pipeline/tracker.py` | YOLO + ByteTrack |
| `pipeline/face.py` | Face ratio calculation |
| `pipeline/recognition_pipeline.py` | Orchestrates face detect → quality → embed → match → decide |
| `pipeline/quality_agent.py` | Face quality scoring (validity gates + weighted composite) |
| `pipeline/track_state.py` | Thread-safe track dictionary |
| `pipeline/models.py` | Data classes (Track, MatchResult, DecisionResult, etc.) |
| `config/settings.py` | All configuration loading + validation |
| `dashboard/backend/main.py` | FastAPI REST + WebSocket server |

## See also
- [[Data Flow]] — per-frame step-by-step walkthrough
- [[Thread Architecture]] — every thread and its responsibilities
- [[MongoDB Schema]] — collections and indexes
- [[Tech Stack]] — all technologies used
