# Data Models

> Dataclasses used throughout the system, plus the state model: what information
> exists at each pipeline stage and who writes/reads it.
> Verified against `pipeline/models.py` on 2026-10-03 (Phase 2 audit).

**File**: `pipeline/models.py` (271 lines)

Status values are `Status` enum ints from `config/status.py` everywhere —
never raw strings. Visibility values are the `Visibility` enum.

## Track

The central data object. One per detected person, alive for the duration of their appearance.

```python
@dataclass
class Track:
    # Identity / lifetime
    track_id: str                    # composite: "{cam_id}_{epoch}_{bt_id}_{generation}"
    first_seen: float                # time.time() when first detected
    last_seen: float                 # time.time() of last detection
    person_box: tuple                # (x1, y1, x2, y2) bounding box
    byte_track_id: int               # raw ByteTrack ID (before composite wrapping)
    generation: int                  # reuse generation (0 = first with this bt_id)

    # Face evidence (accumulated over time)
    best_face_crop: Optional[np.ndarray]
    best_face_score: float           # quality score of best face (0-1)
    best_full_frame: Optional[np.ndarray]
    best_frame_jpeg: Optional[bytes] # pre-encoded JPEG of best full frame (~50KB)
    fallback_frame_jpeg: Optional[bytes]  # first-frame JPEG (always captured)
    best_person_crop_jpeg: Optional[bytes]  # person crop for display
    best_face_crop_path: str         # debug crop path ("" when disabled)
    max_face_ratio: float
    best_face_ratio: float
    face_detected_once: bool
    frames_with_detectable_face: int
    is_masked: bool

    # Recognition / identity
    embedding: Optional[list]        # 512-dim ArcFace embedding
    embedding_det_score: float       # detection score for quality-gated updates
    decision: Optional[int]          # Status enum value (policy's last decision)
    confidence: int                  # highest confidence seen (never downgrades)
    person_name: Optional[str]       # name from match result
    person_name_similarity: float    # sim of the name source (never downgrades)
    visibility: Visibility           # VISIBLE / PARTIAL / HIDDEN / UNKNOWN

    # Alerting
    alerted: bool                    # True once alert has been sent
    last_alert_time: float           # timestamp of last dispatch (cooldown)
    image_url: Optional[str]         # Cloudinary URL (set after upload)

    # Progressive recognition cache (upgrade-only, see state model below)
    pending_recognition: Optional[dict]
    pending_match_result: Optional[MatchResult]
    pending_memory_context: Optional[dict]

    # Recognition throttle state
    last_recognition_quality: float  # face quality at last pass
    last_recognition_status: int     # Status enum (actually the POLICY status)
    last_recognition_time: float
    rescan_attempts: int
    expired_reported: bool           # already reported by get_expired_tracks()

    # Stats / lifetime
    total_frames_seen: int
    max_track_secs: float            # default 300
    _finalized: bool                 # set once by mark_finalized_once()

    _lock: threading.Lock
```

### Key methods

- `is_expired(timeout_secs)` — True if unseen for >timeout_secs
- `is_max_lifetime_exceeded()` — True if alive >max_track_secs
- `update_confidence_if_higher(new)` — atomic, only upgrades
- `mark_alerted_once()` — atomic, returns True only on first call
- `mark_finalized_once()` — atomic, third duplicate-finalization guard;
  survives TrackState removal
- `snapshot()` → `TrackSnapshot` — thread-safe copy of all fields

## TrackSnapshot

Immutable copy used by finalization so slow I/O runs without holding locks.

It is **not** simply "Track minus `_lock`": it also omits
`person_name_similarity`, `last_alert_time`, and `_finalized` (those are only
needed on the live object), and it **copies** `best_face_crop` /
`best_full_frame` as new numpy arrays and `embedding` as a new list, so later
writes on another thread cannot change what finalization sees.

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
    def invalid(cls): ...     # zeros + is_valid=False (assessment impossible)
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
    status: int = Status.UNKNOWN     # Status enum value, NOT a string
    alert_level: str                 # "none" | "low" | "medium" | "high" | "critical"
    person_id: Optional[str]
    name: Optional[str]
    reason: str                      # human-readable explanation
    should_alert: bool
    should_register: bool
    nl_summary: str                  # filled by LLM later
```

## RecognitionResult

Output of the recognition agent.

```python
@dataclass
class RecognitionResult:
    status: int = Status.UNKNOWN     # Status enum value, NOT a string
    confidence: float                # 0-100
    similarity: float                # raw cosine
    face_quality: float              # 0-1
    is_masked: bool
    track_duration: float            # seconds
    reason: str

    def to_dict(self) -> dict: ...   # rounded for serialization;
                                     # this dict is what pending_recognition holds
```

## DedupStatus / DedupResult

Outcome of auto-registration deduplication (`deduplicate_identity`).

```python
class DedupStatus(Enum):
    MERGED = "merged"   # existing person covers this embedding — store nothing
    NEW = "new"         # genuinely new identity — safe to store
    FAILED = "failed"   # dedup unavailable — abort to avoid duplicates

@dataclass
class DedupResult:
    status: DedupStatus
    person_id: Optional[str] = None   # set when MERGED
    reason: str = ""
    similarity: float = 0.0
```

---

# State model

What information exists at each stage of the pipeline, where it lives, and
who writes and reads it. "Single write site" means exactly one place in the
code is allowed to mutate that state.

## Stage 1 — Frame

- **Shape:** raw BGR `np.ndarray`, resized to `FRAME_WIDTH × FRAME_HEIGHT`.
- **Lifetime:** one camera-loop iteration. The recognition pool receives a
  `frame.copy()`; nothing persists.
- **Thread:** camera thread (writes), recognition worker (read-only copy).

## Stage 2 — Detection

- **Shape:** `list[dict]` from `tracker.track_persons()` — raw ByteTrack id,
  box, confidence. Ephemeral; only boxes that survive IoU dedup move on.
- **Thread:** camera thread only.

## Stage 3 — Track

- **Shape:** `Track` inside `TrackState._tracks[composite_id]`.
- **Written by:** `TrackState.update()` (camera thread, every frame);
  setters for all accumulated fields below.
- **Read by:** scheduling checks (`_maybe_schedule_recognition`,
  `TrackFinalizer.finalize_expired`), `snapshot()` at finalization.
- **Rule:** every mutation goes through `TrackState` under its lock;
  two-lock ordering `TrackState._lock` → `track._lock`, never reversed.

## Stage 4 — Face evidence

- **Fields:** `best_face_crop`, `best_face_score`, `best_full_frame`,
  `best_frame_jpeg`, `fallback_frame_jpeg`, `best_person_crop_jpeg`,
  `best_face_crop_path`, `max_face_ratio`, `best_face_ratio`,
  `face_detected_once`, `frames_with_detectable_face`, `is_masked`,
  `visibility`.
- **Written by:** `_handle_pipeline_result` → `set_best_face` (hysteresis:
  needs +0.03 to replace) and `update_face_visibility`;
  `_classify_visibility_inplace` on expiry (reads under `TrackState._lock`).
- **Read by:** quality throttle (compare against `last_recognition_quality`),
  `retry_embedding` (crop + frame), debug saves, finalization snapshot,
  dashboard crops.

## Stage 5 — Embedding

- **Fields:** `embedding` (512 floats), `embedding_det_score`.
- **Written by:** `TrackState.set_embedding` — quality-gated, a new
  embedding only overwrites if its `det_score` beats the existing one by
  ≥ `EMBEDDING_DET_SCORE_IMPROVEMENT_MIN` (0.05). Callers:
  `_handle_pipeline_result` (progressive) and `retry_embedding` (final retry,
  via the injected callback).
- **Read by:** `run_matching_from_embedding` (both chains),
  `_handle_registration` (dedup + `store_face`).
- **Never written** for a face that failed the validity gates.

## Stage 6 — Match

- **Fields:** `MatchResult`, cached on the track as `pending_match_result`;
  plus `person_name` / `person_name_similarity`.
- **Written by:** single site `_handle_pipeline_result` →
  `set_pending_match_result` (upgrade rule: new sim ≥ existing, so a no-match
  with sim 0 can never clobber a real match) and `set_person_name`
  (upgrade-only on similarity).
- **Read by:**
  - `TrackProcessor._run_matching` — reuse instead of a second vector search
  - `TrackProcessor._run_recognition_and_memory` — `fresh_match` derives from
    `pending_match_result is None` and gates trust in the other two caches
  - `CameraAgent._should_skip_recognition` — high-confidence and rescan checks
  - `RecognitionPipeline.run` — its own high-confidence early return
    (read under `track._lock`)
  - `decide()` receives it as an argument

## Stage 7 — Recognition

- **Fields:** `pending_recognition` (a `RecognitionResult.to_dict()`),
  `confidence`, and the throttle trio `last_recognition_quality` /
  `last_recognition_status` / `last_recognition_time` (+ `rescan_attempts`).
- **Written by:** single site `_handle_pipeline_result` →
  `set_pending_recognition_data` (upgrade rule: strictly higher confidence;
  equal is dropped) and, from `_handle_decision_and_alert`,
  `set_recognition_snapshot` (no upgrade rule — always describes the most
  recent pass) + `update_confidence_if_higher`.
  **Caveat:** `last_recognition_status` receives the **policy** decision
  status, not the recognition agent's own classification — the two can differ.
- **Read by:** `TrackProcessor._run_recognition_and_memory` (reuse),
  `_should_skip_recognition` (rescan backoff + quality comparison),
  final `decide()` via its argument.

## Stage 8 — Policy (per pass)

- **Fields:** `decision` (`Status` int), `alerted`, `last_alert_time`.
- **Written by:** `_handle_decision_and_alert` → `set_decision` (not written
  when the track is already RESOLVED, so a resolved status never downgrades),
  `mark_alerted_once` / `last_alert_time` on actual dispatch.
- **Read by:** `_should_skip_recognition` (`decision in RESOLVED_STATUSES`),
  resolved checks before scheduling and inside the worker,
  `TrackProcessor.process` passes inputs to the final `decide()`.

## Stage 9 — Finalization

- **Shape:** `TrackSnapshot` → fresh or reused `MatchResult` +
  `recognition dict` + `memory context` → `DecisionResult` — all locals of
  `TrackProcessor.process()`.
- **Written by:** `snapshot()` (one consistent copy), `mark_finalized_once`,
  `set_person_name` if the final match is better, `last_alert_time` if the
  alert fires.
- **Read by:** registration, visit recording, alert dispatch, event log,
  WebSocket broadcasts — in that order (registration → visit → alert →
  `log_event` → `broadcast_event`).

## Stage 10 — Person / Event (MongoDB)

- **`faces`** — written by `store_face` (new identity) after
  `deduplicate_identity` approves (`NEW`); embedding history and quality-gated
  `latest_embedding` updates live here. Read by vector search.
- **`events`** — written by `log_event` (one row per finalization, plus the
  no-embedding early exit). Read by dashboard queries.
- **`visit_memory`** — written by `record_visit` / `MemoryAgent`; read by the
  memory lookup of later passes.

## pending_* summary table

| Field | Type | Single write site | Upgrade rule | Readers |
|---|---|---|---|---|
| `pending_match_result` | `MatchResult` | `_handle_pipeline_result` → `set_pending_match_result` | new sim ≥ existing (no-match can't clobber) | `_run_matching`, `_run_recognition_and_memory` (`fresh_match`), `_should_skip_recognition`, `RecognitionPipeline.run` (own high-conf return), `decide()` (by argument) |
| `pending_recognition` | `dict` (`RecognitionResult.to_dict()`) | `_handle_pipeline_result` → `set_pending_recognition_data` | strictly higher confidence | `_run_recognition_and_memory`, `decide()` (by argument) |
| `pending_memory_context` | `dict` | `_handle_pipeline_result` → `set_pending_memory_context` | `is_known` True beats False; both known → higher `visit_count` | `_run_recognition_and_memory`, `alert_agent` (alert payload) |

All three are set **only** at finalization-bound write-back on a recognition
worker, always under `TrackState._lock`, and all three are read through
`snapshot()` by the finalization thread — the caches exist so finalization
does not repeat MongoDB/LLM work the progressive pass already did.

## See also

- `pipeline/track_state.py` — manages Track objects and the upgrade rules
- `pipeline/recognition_pipeline.py` — produces `PipelineResult` (carries all
  these types for one progressive pass)
- [[Data Flow]] — how data moves through these structures step by step
