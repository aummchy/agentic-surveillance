# Tracker (YOLO + ByteTrack)

> Person detection and multi-object tracking. YOLOv8 detects people in each frame, ByteTrack assigns stable IDs across frames.

**File**: `pipeline/tracker.py` (97 lines)

## Function: `track_persons(frame, persist=True)`

```python
def track_persons(frame: np.ndarray, persist: bool = True) -> list:
```

### Input
- `frame`: BGR numpy array (1280×720)
- `persist`: True = ByteTrack maintains IDs across frames

### Processing

```
1. model = get_model()  ← singleton YOLO model (loaded once)
2. Set OPENVINO_DEVICE env var (for OpenVINO IR models)
3. results = model.track(
    frame,
    persist=True,
    tracker="config/bytetrack_surveillance.yaml",
    classes=[0],                    ← person class only
    conf=PERSON_CONF_THRESHOLD,     ← default 0.40
    iou=0.45,
    device=YOLO_DEVICE,             ← "cpu" or "intel:GPU"
    verbose=False
)
4. Extract boxes, IDs, confidences from results
5. Return list of dicts
```

### Output
```python
[
    {"track_id": 1, "box": (x1, y1, x2, y2), "confidence": 0.82},
    {"track_id": 2, "box": (x1, y1, x2, y2), "confidence": 0.71},
]
```

## Model loading: `get_model()`

```python
def get_model() -> YOLO:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = YOLO(settings.YOLO_MODEL)
    return _model
```

**Thread-safe singleton** with double-checked locking. Model loaded once, never reloaded.

## ByteTrack configuration

**File**: `config/bytetrack_surveillance.yaml`

Tuned for indoor fixed-camera surveillance:
```yaml
track_high_thresh: 0.45      # High confidence threshold for matching
track_low_thresh: 0.10       # Low threshold for tentative tracks
new_track_thresh: 0.50       # Threshold for creating new tracks
track_buffer: 60             # Frames to keep lost tracks alive
match_thresh: 0.80           # Matching threshold for IoU
fuse_score: true             # Fuse detection score into tracking
```

Key tuning choices:
- `track_buffer=60`: Keeps lost tracks alive for ~2 seconds at 30fps, handles brief occlusions
- `new_track_thresh=0.50`: Prevents noise from creating spurious tracks
- `match_thresh=0.80`: Strict matching reduces ID switches

## Composite track IDs

**File**: `pipeline/track_state.py:34-35`

```python
def make_composite_id(camera_id, byte_track_id):
    return f"{camera_id}_{session_epoch}_{byte_track_id}"
```

Example: `cam_01_1782040060_3`

- `camera_id`: from settings (default `cam_01`)
- `session_epoch`: `int(time.time())` at startup (unique per session)
- `byte_track_id`: 0, 1, 2, ... from ByteTrack (resets on restart)

This ensures track IDs are unique across camera restarts.

## OpenVINO GPU acceleration

```bash
# Export PyTorch model to OpenVINO IR
yolo export model=models/yolov8s.pt format=openvino half=True

# Set in .env
YOLO_MODEL=models/yolov8s_openvino_model/
YOLO_DEVICE=cpu                # Ultralytics requirement for OpenVINO
OPENVINO_DEVICE=GPU            # Actual OpenVINO accelerator
```

Performance: ~128ms → ~16ms on Intel Arc iGPU (8× speedup).

## See also
- [[System Overview]] — where tracking fits
- [[Data Flow]] — Phase 1: frame capture
- [[Track State]] — thread-safe track dictionary
- [[All Config Settings]] — detection-related settings
