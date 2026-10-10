# Thread Architecture

> Every thread in the system, what it does, and how they coordinate.

## Thread map

```
Main thread
│   CameraAgent._loop()
│   ├── cap.read() — blocking
│   ├── track_persons() — YOLO + ByteTrack
│   ├── track_state.update() — thread-safe dict
│   ├── Progressive recognition scheduling (submit to executor)
│   ├── get_expired_tracks() → submit finalization to executor
│   ├── draw_annotations()
│   └── broadcast_frame() — JPEG encode + WebSocket
│
├── Recognition executor (4 threads)
│   ├── recognition_worker.progressive_recognition() — face detect → quality → embed → match → decide
│   └── TrackFinalizer.finalize_track() — final embedding retry
│
├── Track worker pool (2 threads — `runtime/track_workers.py` `NUM_WORKERS = 2`)
│   └── `TrackWorkers._worker_loop()` — queue consumer: final match → decide → store → alert → broadcast
│
├── JPEG encode executor (2 threads, inside TrackProcessor)
│   └── _encode_and_broadcast() — resize + JPEG encode + WebSocket send
│
├── Alert executor (2 threads)
│   └── _send_async() — LLM summary + email/SMS/webhook
│
├── Uvicorn server (1 thread)
│   └── FastAPI REST + WebSocket server (port 8000)
│
└── Startup background threads (daemon, runtime/background.py)
    ├── check_llm_background() — ping Ollama
    ├── run_startup_checks() — Atlas index check (embedding backfill exists
    │                           in utils/db_search.py but is not invoked —
    │                           disabled 2026-10-03)
    ├── prewarm_yolo() — load YOLO model
    └── prewarm_insightface() — load InsightFace model
```

> **Note:** the camera loop runs **on the main thread** — `camera.start()`
> blocks it for the whole session; there is no separate camera thread.
> Other documents' "camera thread" = the main thread while it is inside
> `CameraAgent._loop()`.

## Thread safety mechanisms

| Mechanism | Location | Protects |
|-----------|----------|----------|
| `TrackState._lock` | `pipeline/track_state.py:16` | `_tracks` dictionary (all track mutations) |
| `Track._lock` | `pipeline/models.py:48` | Per-track fields (embedding, decision, face crop) |
| `TrackState._in_flight` | `pipeline/track_state.py:18` | Tracks with active recognition (prevents premature removal) |
| `InsightFaceSingleton._lock` | `utils/embedding_utils.py:20` | One-time model initialization |
| `InsightFaceSingleton._inference_lock` | `utils/embedding_utils.py:21` | Serializes InsightFace inference (not thread-safe) |
| `_collection_locks` | `utils/db_utils.py:18-22` | Per-collection MongoDB connection creation |
| `_alert_lock` | `agents/alert_agent.py:19` | Alert dedup timestamp access |
| `_client_lock` | `utils/db_utils.py:17` | MongoDB client creation |
| `TrackWorkGate._recognizing` | `agents/track_work_gate.py:27` | Set of tracks currently being recognized (claim inside the worker) |
| `TrackWorkGate._finalized` | `agents/track_work_gate.py:28` | Set of tracks already submitted for finalization |

## Critical invariant: Camera loop never blocks

All slow operations (MongoDB queries, Cloudinary uploads, alert dispatch, LLM calls) run in background threads or executors. The camera loop only does:
1. Read frame (fast, ~16ms for 60fps)
2. YOLO detection + ByteTrack (~16-128ms depending on device)
3. Update track state (fast, lock-protected dict operation)
4. Submit recognition to executor (fast, just enqueuing)

The recognition executor handles face detection, embedding, matching, and decision — all off the main thread.

## Queue flow

```
Camera loop (main thread)   Worker threads (×2)
     │                               │
     │  track_queue.put(track)  ──►  │  track_queue.get()
     │                               │  process(track)
     │                               │    ├─ match → MongoDB
     │                               │    ├─ recognize → scoring
     │                               │    ├─ decide → policy
     │                               │    ├─ store_face → MongoDB
     │                               │    ├─ record_visit → MongoDB
     │                               │    ├─ dispatch → alerts
     │                               │    └─ broadcast_event → WebSocket
```

## Graceful shutdown sequence

```
1. camera.stop()              ← sets _running=False, releases capture (runs in start()'s finally)
2. workers.stop()             ← sets _shutdown_event → consumers exit (runtime/track_workers.py)
3. api.request_exit()         ← flips uvicorn should_exit → serving stops
4. workers.drain()            ← queue.join() — waits for all pending tracks
5. track_processor.shutdown() ← stops JPEG encode executor
6. alert_shutdown()           ← stops alert executor
7. llm_shutdown()             ← closes httpx client
8. close_client()             ← closes MongoDB client
9. api.finalize(timeout=5)    ← joins uvicorn thread + closes asyncio loop
```

## See also
- [[CURRENT_ARCHITECTURE]] — high-level architecture
- [[Data Flow]] — per-frame walkthrough
- `pipeline/track_state.py` — thread-safe track management
