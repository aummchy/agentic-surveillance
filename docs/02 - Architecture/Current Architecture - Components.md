# Current Architecture — Components (Sections 4–7)

> Part of [`CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md). Section numbers follow the shared scheme: this file holds sections 4–7.
>
> **Source of truth:** current source code and tests. Facts that could not be proven from the source are marked `UNKNOWN — NEEDS VERIFICATION`.
>
> **Last verified:** 2026-10-03

**Navigation:** [Hub](../CURRENT_ARCHITECTURE.md) · [Components](Current%20Architecture%20-%20Components.md) · [Recognition](Current%20Architecture%20-%20Recognition.md) · [Decision Flow](Current%20Architecture%20-%20Decision%20Flow.md) · [Platform](Current%20Architecture%20-%20Platform.md) · [Code State](Current%20Architecture%20-%20Code%20State.md)

---

# 4. Major Components

## 4.1 Camera and Detection

### File

`agents/camera_agent.py` (669 lines)

### Current responsibilities

* Camera/video capture: `_open_capture()` (`:112`), frame property application (`:144`), unconditional resize to 1280×720 before inference.
* YOLO/ByteTrack execution through `pipeline/tracker.py`, once per frame.
* Track lifecycle: `TrackState.update()` results, IoU overlap dedup (`OVERLAP_IOU_THRESHOLD` 0.5), expired-track sweep (`_finalize_expired_tracks`, `:427`).
* Recognition scheduling: `_maybe_schedule_recognition` (`:316`) — every `RECOGNITION_INTERVAL_FRAMES` (20) frames (`:313`), gated so a pass only starts if face quality improved by `MIN_QUALITY_IMPROVEMENT` (0.10) over `last_recognition_quality`.
* Progressive recognition coordination: worker body `_progressive_recognition` (`:504`), result write-back `_handle_pipeline_result` (`:550`) — the **sole write-back site** for progressive results.
* Mid-track alerting: `_handle_decision_and_alert` (`:636`) → resolves the track image (`:678`) → `alert_dispatch(track, result.decision, image_url)` (`:679`).
* Frame broadcasting coordination via the `on_frame_annotated` callback.
* Track finalization coordination: `_finalize_expired_tracks` / `_finalize_track` (`:741`) — final embedding retry (`retry_embedding`, `:759`), then `on_track_finalized` → `main.handle_track_finalized` → `track_processor.enqueue`.
* Debug diagnostics (`_log_duplicate_diagnostics`, `:442`) and FPS stats (`_update_fps_stats`, `:472`).

### Important state

* `TrackState` (active tracks registry) — shared, lock-protected.
* `_recognizing_tracks` — set of track IDs currently being recognized; **added inside the worker** (`:538`), discarded at cleanup (`:733`).
* `_finalized_track_ids` — set of track IDs already finalized; add/discard protocol at `:438-439`, `:734-735`, `:768`; keeps ByteTrack ID reuse safe.
* `_frame_count` — drives the recognition cadence.
* Camera/session info: camera ID from settings, capture handle, FPS counters.

### Threads / concurrency

* Capture, detection, tracking, scheduling: camera thread.
* Recognition passes and finalization retry: `_recognition_executor` = `ThreadPoolExecutor(RECOGNITION_MAX_WORKERS=4)` (`:87-88`).
* Finalization does **not** run inline — it ends in an enqueue onto `track_queue` consumed by one of the two `TrackProcessor` threads.

### Known architectural concern

`camera_agent.py` currently contains camera/tracking responsibilities **and** recognition orchestration (scheduling, worker body, write-back, decision/alert handling, cleanup). This creates coupling between:

```text
Camera → Tracking → Recognition scheduling
```

This is documented as an architectural concern only. No change is implied by this document.

### Test coverage

**No test file imports `CameraAgent`.** Its behavior is only exercised by live runs.

---

# 5. Detection and Tracking

## `pipeline/tracker.py`

### Responsibility

YOLO-based person detection and ByteTrack tracking.

### Input

A video frame (already resized to 1280×720 by `camera_agent`).

### Output

Raw tracked detections (boxes, track IDs, scores) returned to `camera_agent._loop`.

### Implementation facts

* `get_model()` — YOLO singleton (PyTorch `.pt` or OpenVINO IR), created once on first use; prewarmed by `main._prewarm_yolo`.
* `track_persons()` — one detect + ByteTrack step per call.
* ByteTrack parameters: `config/bytetrack_surveillance.yaml` (`track_high_thresh=0.45`, `track_buffer=60`, `new_track_thresh=0.50`).

### Important distinction

Detection answers:

> Where is a person in this frame?

Tracking answers:

> Which detection corresponds to an existing tracked object?

Tracking does NOT determine the person's persistent identity. It only associates detections with transient track IDs.

---

# 6. Track Identity

## Track ID

The system uses a composite track identifier produced by `TrackState.make_composite_id` (`pipeline/track_state.py`, pure formatter):

```text
{camera_id}_{session_epoch}_{byte_track_id}_{generation}
```

* `session_epoch` — captured once when `TrackState` is constructed, so IDs stay unique across camera restarts **within a process**.
* `generation` — handles ByteTrack ID reuse within a session (incremented when a raw `bt_id` is reused by a departed-then-returned person).

Example from live logs: `cam_01_1791026990_3` style prefixes with a generation suffix (e.g. `..._3_0`).

### Purpose

The track ID identifies a tracked object during a surveillance session.

It is not a persistent person identity.

---

## Person ID

A `person_id` comes from the identity/recognition system and database (`faces` collection; auto-registration stores `person_id = snap.track_id` at `track_processor.py:402` for first-time identity creation).

Conceptually:

```text
Track ID  !=  Person ID
```

* A single track may produce evidence for a person identity.
* **Known hazard:** the broadcast alert payload puts `track.track_id` under the JSON key `"person_id"` (`track_processor.py:461-467` — the code itself documents this). The frontend depends on this key. Events store the real `person_id` separately (`track_processor.py:501`).

---

# 7. Track State

## `pipeline/track_state.py` (535 lines) and `pipeline/models.py` (`Track`)

### Responsibility

`TrackState` is the thread-safe registry of active tracks; `Track` (dataclass, `pipeline/models.py:27`, ~109 field lines) is the per-track record it owns.

### Current state includes information related to

* **Tracking** — `track_id`, `first_seen`, `last_seen`, `person_box`, `byte_track_id`, `generation`, `total_frames_seen`, `max_track_secs`, `visibility`.
* **Face observations** — `best_face_crop`, `best_face_score`, `best_face_ratio`, `max_face_ratio`, `face_detected_once`, `frames_with_detectable_face`, `best_person_crop_jpeg`, `is_masked`.
* **Frame/image** — `best_full_frame`, `best_frame_jpeg`, `fallback_frame_jpeg`, `best_face_crop_path`, `image_url`.
* **Embeddings** — `embedding`, `embedding_det_score`.
* **Recognition** — `last_recognition_quality/status/time`, `rescan_attempts`, `confidence`, `decision`, `person_name`, `person_name_similarity`.
* **Pending results** — `pending_recognition`, `pending_match_result`, `pending_memory_context` (populated by progressive passes, consumed at finalization).
* **Finalization** — `alerted`, `last_alert_time`, `_finalized`, `expired_reported`.
* **Persistence/dashboard** — `image_url`, release-on-finalize bookkeeping.

### Concurrency

`TrackState` uses locking because state is accessed by the camera thread, the 4 recognition workers, and the 2 queue-consumer threads.

* Two-lock ordering: `TrackState._lock` may be taken before `track._lock`, never the reverse.
* `pending_*` fields have lock-protected **upgrade-only** setters (a weaker result never replaces a stronger one; critical alerts exempt the confidence gate).
* JPEG encoding happens outside the locks.

### Important invariants (current implementation protects against)

* Updating pending results with weaker results — upgrade-only setters.
* Duplicate finalization — `_finalized_track_ids` protocol plus the `Track._finalized` flag.
* Invalid concurrent state mutation — `TrackState._lock` around dictionary and field writes.
* Confidence downgrade — max-upgrade gate at the setters/write-back sites.

### Architectural concern

The `Track`/track-state representation contains information belonging to multiple conceptual concerns:

```text
Tracking · Face observation · Recognition · Identity · Policy · Presentation · Persistence
```

This is currently implemented as **one connected dataclass** rather than several independent models. Splitting it is a parked, approval-gated item (Phase 4 #4 in `plan.md`). Documented as an observation only.
