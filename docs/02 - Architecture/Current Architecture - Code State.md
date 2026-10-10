# Current Architecture — Code State (Sections 27–30)

> Part of [`CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md). Section numbers follow the shared scheme: this file holds sections 27–30.
>
> **Source of truth:** current source code and tests. Facts that could not be proven from the source are marked `UNKNOWN — NEEDS VERIFICATION`.
>
> **Last verified:** 2026-10-03
>
> These sections are **observations, not approved refactoring tasks.** What is approved to change is tracked in `plan.md` and `docs/REFACTOR_PLAN.md`.

**Navigation:** [Hub](../CURRENT_ARCHITECTURE.md) · [Components](Current%20Architecture%20-%20Components.md) · [Recognition](Current%20Architecture%20-%20Recognition.md) · [Decision Flow](Current%20Architecture%20-%20Decision%20Flow.md) · [Platform](Current%20Architecture%20-%20Platform.md) · [Code State](Current%20Architecture%20-%20Code%20State.md)

---

# 27. Current Architectural Duplication

The most important known duplication is the **split recognition implementation**:

```text
                 Recognition
                     |
          +----------+----------+
          |                     |
          v                     v
 RecognitionPipeline      TrackProcessor
 (mid-track, needs a      (finalization, snapshot
  live frame)              + pending_* caches)
          |                     |
       stages                stages
          |                     |
   face → quality → embed   match → recognize → memory
   → match → memory              → decide
   → recognize → decide          (+ register/visit/alert/log)
          |                     |
          +----------+----------+
                     |
          Similar outcome: a DecisionResult
```

Specifically, parts of:

```text
matching · recognition · memory · policy
```

are implemented through both the progressive recognition path ([Section 8.1](Current%20Architecture%20-%20Recognition.md#81-progressive-recognition-path)) and the finalization path ([Section 8.2](Current%20Architecture%20-%20Recognition.md#82-track-finalization-recognition-path)).

### Current status

**Verified architectural duplication.** Both paths exist, both call `matching_agent` / `RecognitionAgent` / `MemoryAgent` / `policy.decide`, and `track_processor` re-implements its own orchestration rather than calling `RecognitionPipeline`.

### Not yet decided

Whether the two paths should:

* share one implementation (`RecognitionPipeline.run_final(track, snapshot)` — the parked Phase 4 #1 idea),
* use separate entry points over shared stage functions, or
* retain some intentional differences (e.g. finalization's reliance on `pending_*` caches).

This must be determined from a line-by-line behavioral comparison and tests before code is moved. The comparison is the stated precondition in `plan.md`.

---

# 28. Current Architectural Concerns

These are observations, not approved refactoring tasks.

## High priority

### 1. Multiple recognition execution paths

* `camera_agent.py` → `recognition_worker.py` → `RecognitionPipeline.run(frame, track)` (4 recognition workers)
* `track_processor.py` → inline `_run_matching` + `_run_recognition_and_memory` (2 queue-consumer threads)

See Section 27.

### 2. Track state is overloaded

`Track` (≈109 field lines) and `TrackState` connect tracking, face evidence, frame/image blobs, embeddings, recognition, pending results, finalization flags, and presentation/persistence fields through one dataclass (Section 7). Splitting it is parked as Phase 4 #4.

## Medium priority

### 3. Configuration/default duplication

Duplicated defaults exist in both `settings.py` and `config.jsonc`; two verified pairs already diverge (`TRACK_TIMEOUT_SECS` 3.0 vs 15.0, `JPEG_QUALITY_BROADCAST` 65 vs 85 — Section 23). Whether other keys diverge: `UNKNOWN — NEEDS VERIFICATION` (full diff not done). Parked as Phase 4 #5.

### 4. Identity registration / deduplication

The auto-registration path runs `deduplicate_identity(embedding)` against `DEDUP_SIMILARITY_THRESHOLD` (0.45) before `store_face` (Section 16). **Whether the same real-world person can still be registered more than once under different identities is `UNKNOWN — NEEDS VERIFICATION`** — no behavioral test covers this question.

### 5. Recognition/policy threshold consistency

`CONFIDENCE_KNOWN_MIN` = 70 vs `KNOWN_VISITOR_CONFIDENCE` = 80 gate related concepts ("known" vs "known visitor") in different places. Reconciling them is parked (Phase 4 #2).

## Additional verified observations

### 6. Zero tests for the hot path's two biggest files

No test file imports `CameraAgent` or `TrackProcessor` (Section 30). Their behavior is currently evidenced only by live runs.

### 7. `LLM_ENABLED` gating is inconsistent

The flag is consulted only by `is_available()`, not by `generate()` (Section 22). Current value in `config.jsonc` is `false`. Intent needs confirming (M3) before code changes.

---

# 29. Known Fixed Issues

The following issues have previously been reported or fixed and should **NOT** automatically be treated as current bugs. Check the code before reopening them.

## Re-verified against the code on 2026-10-03

* Pending match / recognition / memory results are protected against weaker replacements (lock-protected upgrade-only setters in `TrackState`).
* Duplicate recognition scheduling has guards (`TrackWorkGate` recognizing claim + `pending_recognition` check; worker-start resolved-check is the real backstop).
* Track finalization has duplicate-finalization protection (three guards: `TrackWorkGate` finalize claim, `Track.mark_finalized_once()`, `TrackFinalizer.finalize_track` - `agents/track_work_gate.py`, `agents/track_finalization.py`) — the "Duplicate Finalization (Visit Inflation)" problem doc is stale; the issue is Resolved.
* Track ID and Person ID are distinct concepts (with the documented `"person_id"`-carries-`track_id` broadcast hazard).
* WebSocket origin validation on `/ws/live` (`live.py:82-85`, close code 4003).
* CORS limited to explicit methods/headers (`dashboard/backend/main.py:21-22`).
* Policy RULE 5 returns `unknown` — auto-registered unknowns are no longer promoted to `known_visitor`.
* Alert dispatch C1+H5 fixed: dead `track.alerted` guard removed, `AlertLevel` imported; two-layer dedup contract (caller marks first, cooldown stamped at allow-time) tested in `tests/test_alert_dispatch.py`.
* Incident reports H1+H2 fixed: naive/aware datetime crash and N+1 `visit_history` query; tested in `tests/test_report.py` (2026-10-03).
* Confidence never downgrades (max-upgrade gate at `TrackState` setters + camera write-back).
* `OFFICE_DAYS` env var casts to a list of ints; PyMongo uses `ReturnDocument.AFTER` in memory upserts.

## Documented as fixed in `AGENTS.md` — not re-verified in this pass

* LLM client shutdown uses `get_running_loop()` + `asyncio.run()` fallback.
* Lock-protected setters for `cached_embedding` / `pending_recognition`.
* Dead code removal pass (`get_embedding_for_track`, unused imports).
* Status string→int migration (2026-08-05); `scripts/migrate_status_ints.py`.
* Quality-gated recognition skip (`agents/recognition_throttle.py` early return).

### Known doc conflict

`AGENTS.md` "Recent fixes" says `JPEG_QUALITY_BROADCAST` was raised 50 → **90**; the effective value in code today is **85** (`config.jsonc:113`). The code wins (source-of-truth rule); the AGENTS note is stale on this number.

Historical details live separately in `docs/HISTORICAL_DEBUG_NOTES.md` — historical, not authoritative.

---

# 30. Current Tests

## Inventory (verified — `pytest tests/ -q`, **144 tests, 10 files**, 2026-10-10)

| File | Tests | What it evidences |
|------|------:|-------------------|
| `tests/test_recognition.py` | 27 | `matching_agent`, `policy.decide`, `RecognitionAgent`/`recognize`, `scoring.compute_confidence`, normalize/clip |
| `tests/test_recognition_pipeline.py` | 16 | end-to-end `RecognitionPipeline.run` with mocked DB/embedding |
| `tests/test_thread_safety.py` | 16 | `TrackState` lock behavior, concurrent field updates, `update_visit_memory` |
| `tests/test_recognition_pipeline_stages.py` | 9 | pipeline stage units (quality, match, decision → `PipelineResult`) |
| `tests/test_track_finalizer.py` | 9 | finalization decisions: dispatch/cooldown, policy, memory, DB helpers |
| `tests/test_report.py` | 7 | incident reports: H1 datetime regression, H2 batch consumption, endpoint 200 |
| `tests/test_embedding_history.py` | 7 | `db_faces.update_face` embedding-history behavior |
| `tests/test_alert_dispatch.py` | 6 | alert dispatch: mark-before-dispatch, cooldown, payload, one-shot marking |
| `tests/test_camera_agent.py` | 38 | `CameraAgent` characterization: throttle, scheduling, write-back order, decision/alert, finalization guards, worker entry |
| `tests/test_track_work_gate.py` | 9 | `TrackWorkGate` exactly-once claims (incl. 16-thread contention) |

Shared fixtures in `tests/conftest.py`: `sample_track`, `sample_embedding`, mocked vector search, mocked memory upserts, `make_match_result`.

## Coverage gaps (verified — no test file imports these)

```text
agents/track_processor.py     (0 tests — the entire finalization sequence)
pipeline/tracker.py           (0 tests — YOLO/ByteTrack step)
config/settings.py            (0 dedicated tests)
utils/llm_client.py           (0 tests — fallbacks only observed live)
dashboard/backend/routes/*    (only reports via the TestClient test; faces/events/chat untested)
utils/db_events, db_search    (only exercised indirectly)
```

## How to use this document

Tests are **behavioral evidence** when evaluating proposed architectural changes. Before structural refactoring of any component, identify the tests that cover it and run them:

```bash
python -m pytest tests/ -q      # gate: 97 passed
```

If a proposed change has **no covering tests**, write the tests first — that is the current process rule (`AGENTS.md` §3, `plan.md` Phase 4 gates).
