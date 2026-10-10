# Current Architecture — Components (Sections 4–7)

> Part of [`CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md). Section numbers follow the shared scheme: this file holds sections 4–7.
>
> **Source of truth:** current source code and tests. Facts that could not be proven from the source are marked `UNKNOWN — NEEDS VERIFICATION`.
>
> **Last verified:** 2026-10-10

**Navigation:** [Hub](../CURRENT_ARCHITECTURE.md) · [Components](Current%20Architecture%20-%20Components.md) · [Recognition](Current%20Architecture%20-%20Recognition.md) · [Decision Flow](Current%20Architecture%20-%20Decision%20Flow.md) · [Platform](Current%20Architecture%20-%20Platform.md) · [Code State](Current%20Architecture%20-%20Code%20State.md)

---

# 4. Major Components

## 4.1 Camera and Detection

### File

`agents/camera_agent.py` (375 lines) — the frame loop and scheduling only. The progressive recognition path it schedules lives in `agents/recognition_worker.py` (306 lines), and the split also created `agents/capture.py`, `agents/recognition_throttle.py`, `agents/track_work_gate.py` and `agents/track_finalization.py` (Phase 4 refactor, completed 2026-10-10).

### Current responsibilities

* Camera/video capture via `agents/capture.py`: `camera_source()` (`:18`), `open_capture()` (`:31`), `apply_frame_props()` (`:64`), plus the unconditional resize to 1280×720 before inference.
* YOLO/ByteTrack execution through `pipeline/tracker.py`, once per frame.
* Track lifecycle: `TrackState.update()` results, IoU overlap dedup (`OVERLAP_IOU_THRESHOLD` 0.5), expired-track sweep (`TrackFinalizer.finalize_expired`, `agents/track_finalization.py:38`).
* Recognition scheduling: `CameraAgent._maybe_schedule_recognition` (`camera_agent.py:266`, body in `agents/recognition_worker.py:47`) — every `RECOGNITION_INTERVAL_FRAMES` (20) frames, gated so a pass only starts if face quality improved by `MIN_QUALITY_IMPROVEMENT` (0.10) over `last_recognition_quality`.
* Progressive recognition coordination (bodies in `agents/recognition_worker.py`): worker entry `progressive_recognition` (`:107`), result write-back `handle_pipeline_result` (`:153`) — the **sole write-back site** for progressive results.
* Mid-track alerting: `handle_decision_and_alert` (`agents/recognition_worker.py:240`) → resolves the track image (`:281`) → `alert_dispatch(track, result.decision, image_url)` (`:282`).
* Frame broadcasting coordination via the `on_frame_annotated` callback.
* Track finalization coordination (`agents/track_finalization.py`): `finalize_expired` / `cleanup_after_recognition`, both ending in `finalize_track` (`:85`) — final embedding retry (`retry_embedding`), then the `handle_track_finalized` closure in `main.py` → `track_processor.enqueue(workers.queue, track)`.
* Debug diagnostics (`_log_duplicate_diagnostics`, `camera_agent.py:286`) and FPS stats (`_update_fps_stats`, `:316`).

Each of the four worker-path methods above still exists on `CameraAgent` as a one-line delegate to `agents/recognition_worker.py`, so references to `CameraAgent.<method>` (including the tests) keep working.

### Important state

* `TrackState` (active tracks registry) — shared, lock-protected.
* `_recognition_executor` = `ThreadPoolExecutor(RECOGNITION_MAX_WORKERS=4)` (`camera_agent.py:90`).
* `TrackWorkGate` (`agents/track_work_gate.py`) — owns the two sets that used to live in this file: the recognizing set (**marked inside the worker**, not at submit time, released at cleanup) and the finalized set (add/discard protocol, keeps ByteTrack ID reuse safe).
* `_frame_count` — drives the recognition cadence.
* Camera/session info: camera ID from settings, capture handle, FPS counters.

### Threads / concurrency

* Capture, detection, tracking, scheduling: camera thread.
* Recognition passes and finalization retry: `_recognition_executor` (4 workers).
* Finalization does **not** run inline — it ends in an enqueue onto `track_queue` consumed by one of the two `TrackProcessor` threads.

### Known architectural concern

**Resolved 2026-10-10.** `camera_agent.py` previously mixed camera/tracking responsibilities with recognition orchestration (scheduling, worker body, write-back, decision/alert, cleanup). The worker path now lives in `agents/recognition_worker.py`; scheduling helpers, throttle checks, exactly-once claims and finalization moved to their own modules in steps 1–4. What remains in `camera_agent.py` is the frame loop plus one-line delegates.

### Test coverage

`tests/test_camera_agent.py` (38 characterization tests covering the throttle, scheduling, write-back order, decision/alert branches, finalization guards and worker entry) plus `tests/test_track_work_gate.py` (9 gate tests, including a 16-thread claim-contention test).

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
* Duplicate finalization - `TrackWorkGate._finalized` protocol plus the `Track._finalized` flag.
* Invalid concurrent state mutation — `TrackState._lock` around dictionary and field writes.
* Confidence downgrade — max-upgrade gate at the setters/write-back sites.

### Architectural concern

The `Track`/track-state representation contains information belonging to multiple conceptual concerns:

```text
Tracking · Face observation · Recognition · Identity · Policy · Presentation · Persistence
```

This is currently implemented as **one connected dataclass** rather than several independent models. Splitting it is a parked, approval-gated item (Phase 4 #4 in `plan.md`). Documented as an observation only.
