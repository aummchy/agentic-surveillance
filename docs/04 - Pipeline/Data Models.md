# Data Models

> Dataclasses used throughout the system. Every major component has a typed data structure for inputs and outputs.

**File**: `pipeline/models.py` (232 lines)

## Track

The central data object. One per detected person, alive for the duration of their appearance.

```python
@dataclass
class Track:
    track_id: str                    # composite: "{cam_id}_{epoch}_{bt_id}"
    first_seen: float                # time.time() when first detected
    last_seen: float                 # time.time() of last detection
    person_box: tuple                # (x1, y1, x2, y2) bounding box

    # Face data (accumulated over time)
    best_face_crop: np.ndarray       # best quality face crop
    best_face_score: float           # quality score of best face (0-1)
    best_full_frame: np.ndarray      # full frame when best face was captured
    best_frame_jpeg: bytes           # pre-encoded JPEG of best full frame
    fallback_frame_jpeg: bytes       # first-frame JPEG (always captured)

    # Recognition data
    is_masked: bool
    embedding: Optional[list]        # 512-dim ArcFace embedding
    embedding_det_score: float       # detection score for quality-gated updates
    decision: Optional[str]          # "known" / "unknown" / "verified" / etc.
    confidence: int                  # highest confidence seen (never downgrades)
    person_name: Optional[str]       # name from match result
    alerted: bool                    # True once alert has been sent

    # Progressive recognition cache
    pending_recognition: Optional[dict]
    pending_match_result: Optional[MatchResult]
    pending_memory_context: Optional[dict]

    # Track statistics
    total_frames_seen: int
    frames_with_detectable_face: int
    face_detected_once: bool
    max_face_ratio: float
    best_face_ratio: float
    best_face_crop_path: str
    visibility: str                  # "visible" / "partial" / "hidden" / "unknown"

    # Timing
    last_recognition_frame: int
    last_recognition_quality: float
    last_recognition_status: str
    last_recognition_time: float
    rescan_attempts: int
    expired_reported: bool
    max_track_secs: float            # default 300

    _lock: threading.Lock
```

### Key methods
- `is_expired(timeout_secs)` — True if unseen for >timeout_secs
- `is_max_lifetime_exceeded()` — True if alive >max_track_secs
- `update_confidence_if_higher(new)` — atomic, only upgrades
- `mark_alerted_once()` — atomic, returns True only on first call
- `snapshot()` → `TrackSnapshot` — thread-safe copy of all fields

## TrackSnapshot

Immutable copy of Track state. Created at finalization to avoid holding locks during slow I/O.

```python
@dataclass
class TrackSnapshot:
    # Same fields as Track (minus _lock)
    # Created by track.snapshot()
```

## QualityResult

Output of face quality assessment.

```python
@dataclass
class QualityResult:
    blur_score: float        # Laplacian variance
    brightness: float        # mean HSV V-channel
    face_area: int           # height × width
    is_valid: bool           # passes all validity gates
    overall_score: float     # weighted composite [0,1]

    @classmethod
    def invalid(cls):
        return cls(blur_score=0, brightness=0, face_area=0,
                   is_valid=False, overall_score=0)
```

## MatchResult

Output of vector search matching.

```python
@dataclass
class MatchResult:
    person_id: Optional[str]
    name: Optional[str]
    role: Optional[str]              # "visitor", "authorized", etc.
    tags: list                       # ["auto_registered", "verified", "blacklist"]
    similarity_score: float          # raw cosine similarity
    image_url: Optional[str]
    matched: bool                    # similarity >= MATCH_THRESHOLD
    verified: bool                   # "verified" in tags
    alert_level: str                 # "low", "medium", "high", "critical"
    second_best_similarity: Optional[float]
    margin: Optional[float]          # top1 - top2
    candidate_count: int
    all_candidates: list             # [{name, similarity}] top-N
```

## DecisionResult

Output of policy decision.

```python
@dataclass
class DecisionResult:
    status: str              # "verified" | "known_visitor" | "blacklist" | "unknown" | etc.
    alert_level: str         # "none" | "low" | "medium" | "high" | "critical"
    person_id: Optional[str]
    name: Optional[str]
    reason: str              # human-readable explanation
    should_alert: bool
    should_register: bool
    nl_summary: str          # filled by LLM later
```

## RecognitionResult

Output of recognition scoring.

```python
@dataclass
class RecognitionResult:
    status: str              # "known" | "unknown" | "uncertain"
    confidence: float        # 0-100
    similarity: float        # raw cosine
    face_quality: float      # 0-1
    is_masked: bool
    track_duration: float    # seconds
    reason: str              # human-readable explanation
```

## See also
- `pipeline/track_state.py` — manages Track objects
- `pipeline/recognition_pipeline.py` — produces PipelineResult (contains all these types)
- [[Data Flow]] — how data flows through these structures
