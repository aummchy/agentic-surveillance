# Current Architecture — Decision Flow (Sections 15–20)

> Part of [`CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md). Section numbers follow the shared scheme: this file holds sections 15–20.
>
> **Source of truth:** current source code and tests. Facts that could not be proven from the source are marked `UNKNOWN — NEEDS VERIFICATION`.
>
> **Last verified:** 2026-10-03

**Navigation:** [Hub](../CURRENT_ARCHITECTURE.md) · [Components](Current%20Architecture%20-%20Components.md) · [Recognition](Current%20Architecture%20-%20Recognition.md) · [Decision Flow](Current%20Architecture%20-%20Decision%20Flow.md) · [Platform](Current%20Architecture%20-%20Platform.md) · [Code State](Current%20Architecture%20-%20Code%20State.md)

---

# 15. Policy Engine

## `agents/policy.py` (358 lines)

### Responsibility

Converts recognition and contextual information into an operational `DecisionResult` (status, alert level, reason, `should_alert`, `should_register`, NL summary). Entry point: `decide()`; `PolicyAgent` is a thin wrapper.

### Current rule inventory (verified — rule docstrings at `policy.py:189–372`)

| Rule | Condition | Outcome (summary) |
|------|-----------|-------------------|
| RULE 1 | Blacklisted person | highest priority, **critical** alert |
| RULE 2 | Authorized person | no alert |
| RULE 3 | Verified visitor | no alert |
| RULE 4a | Auto-registered self-match (similarity > `AUTO_REGISTERED_SIMILARITY`) | known path |
| RULE 4b | Known visitor (matched + memory confirms; `:286`: similarity ≥ `KNOWN_VISITOR_SIMILARITY` 0.85 **or** confidence ≥ `KNOWN_VISITOR_CONFIDENCE` 80) | known_visitor |
| RULE 5 | Matched but not memory-confirmed | returns **`unknown`** (intentional since 2026-07-09 — auto-registered unknowns must not be promoted) |
| RULE 6 | Intentionally hidden | hidden |
| RULE 7 | Masked / partial visibility | masked_unknown |
| RULE 8 | After-hours unknown | higher alert |
| RULE 9 | Unknown during office hours | default low/no alert |

Priority order: blacklist > authorized > verified > auto-registered self-match > known visitor > matched-not-confirmed > hidden > masked/partial > after-hours unknown > office-hours unknown.

### Important distinction

Policy determines **what the system should do** with a recognition result. It should not independently become another face-recognition pipeline (see `docs/ARCHITECTURE_RULES.md`).

### Current concern

Recognition and policy thresholds are **not numerically aligned**: `CONFIDENCE_KNOWN_MIN` = 70 (policy "known" gate) vs `KNOWN_VISITOR_CONFIDENCE` = 80 (RULE 4b gate). Documented as concern §28.5; reconciling them is a parked Phase 4 item.

### Callers / threading

Called from **both** recognition chains (`recognition_pipeline._run_policy` and `track_processor.process`). Pure function of its inputs — safe on any thread; mutates nothing.

---

# 16. Track Finalization

## `agents/track_processor.py` (472 lines)

### Responsibility

Processes a track after its active lifecycle ends, and relays the preview frame.

Two entry points:

* `process(track)` (`:151`) — the closing sequence, runs on one of the **two** `track_queue` consumer threads.
* `handle_frame(frame)` (`:122`) — throttled JPEG preview relay (every `FRAME_SKIP` = 2 frames, resized to `BROADCAST_WIDTH×HEIGHT`, quality `JPEG_QUALITY_BROADCAST`), encoded on the 2-thread JPEG pool.

### Current `process()` sequence (verified call order)

```text
snapshot from TrackState
  → resolve image URLs (track_processor.py:182-183, Cloudinary → local capture fallback)
  → _run_matching                (:264)
  → _run_recognition_and_memory  (:291, fresh RecognitionAgent per track :319)
  → policy.decide()
  → _handle_registration         (:212 call; def :338)
        → deduplicate_identity(snap.embedding)   (:370, DEDUP_SIMILARITY 0.45)
        → _store_new_face → store_face (:401-402; person_id = snap.track_id for new IDs)
  → _record_visit                (visit memory, dedup via MIN_VISIT_GAP_SECS)
  → _dispatch_alert              (:441-451, alert_agent.dispatch)
  → _broadcast_alert             (:457, "person_id" key carries track_id)
  → _broadcast_event             (:482)
  → _log_event                   (:518, events collection)
```

### Threading

* Finalization runs on **two** consumer threads — two different tracks may be processed concurrently. The body is *not* fully serialized (an earlier version of this document's claim of "one consumer thread" was wrong; corrected 2026-10-03).
* Shared state touched across the two threads: MongoDB collections, the asyncio broadcast loop (posted only while alive), per-track `Track` objects (normally distinct per task).

### Architectural concern

This module combines **multiple responsibilities** (matching, recognition, memory, registration, visits, alerts, event logging, broadcasting) and overlaps with `RecognitionPipeline`. It has **zero direct test coverage** — no test file imports `TrackProcessor`. It is one of the main candidates for future refactoring (documented observation only).

### Known quirks (verified in code)

* `fresh_match = snap.pending_match_result is None` (`:306`) — one flag gates trust in all three `pending_*` fields.
* `_broadcast_alert` puts `track.track_id` under the payload key `"person_id"` (`:467`) — established frontend contract, documented as a hazard in the code itself.
* `shutdown()` sets `_shutdown_event` (`:110`) that **nothing in the class reads** (`:105` documents this).
* A fresh `RecognitionAgent()` is constructed per track (`:319`).

---

# 17. Final Embedding Retry

## `agents/finalizer.py`

### Responsibility

`retry_embedding()` — one last InsightFace embedding attempt for a track that never produced a usable embedding. Tries the best face crop first, then the full frame **only if** `ENABLE_FULL_FRAME_FALLBACK` is enabled (currently `false` in both `config.jsonc:55` and the `settings.py:309` default — the frame-fallback branch is dead in the current configuration).

### Ownership (verified — not open)

* **Called by:** `TrackFinalizer.finalize_track` (`agents/track_finalization.py:85`) only: `retry_embedding(track, set_embedding=self._track_state.set_embedding)` (`:103`).
* **Runs on:** a recognition-pool worker (finalization is submitted to the same 4-thread pool).
* The result is written back through the injected `TrackState.set_embedding` callback (lock-protected).

It is a step **inside** the camera-side finalization coordination (Section 4.1), executed *before* the track is handed to `track_processor` via the queue. Its relationship to `track_processor.py` is sequential (retry first, then enqueue), not overlapping.

---

# 18. Memory

## `agents/memory.py` (232 lines) + `utils/db_memory.py`

### Responsibility

`MemoryAgent` (defined at `memory.py:40`) maintains visit/history information for identities: visit lookup, visit recording, typical-hours patterns, and the confidence boost for returning visitors.

### Instances

Two independent instances, each held by its own component:

* `recognition_pipeline.py:131` (progressive path)
* `track_processor.py:94` (finalization path)

### Storage

MongoDB `visit_memory` collection (one document per `person_id`), via `utils/db_memory.py` (`get_visit_history`, `update_visit_memory`, `get_or_create_memory`).

### Purpose

Memory provides historical context to recognition/decision logic (confidence boost, RULE 4b confirmation, visit counting). It is distinct from **track state** (per-track, transient) and from **persistent person identity** (`faces`).

### Known issue (parked)

**H7 (Phase 3 review):** `update_visit_memory` never writes `avg_similarity` — it stays at its init value `0.0`, and all consumers gate on `> 0`, so similarity-drift detection (`MEMORY_CONSISTENCY_HIGH_SIM` / `LOW_SIM` / `SIMILARITY_DROP_RATIO`, `memory.py:177-189`) is currently **inert**. Compute-or-retire decision pending (Wave A, item 5). Not fixed by this document.

---

# 19. Database Layer

## MongoDB modules

| Module | Responsibility |
|--------|----------------|
| `utils/db_client.py` | lazy, thread-safe MongoDB connection singleton + collection getters (`MongoClient` created without `tz_aware` — stored datetimes are naive UTC) |
| `utils/db_faces.py` | face/person CRUD: `store_face`, `update_face` (quality-gated `latest_embedding` overwrite), `deduplicate_identity` (`:211`), embedding history |
| `utils/db_events.py` | event logging + dashboard stats (`get_events_with_faces`, `get_stats`) |
| `utils/db_memory.py` | visit-memory CRUD (see Section 18) |
| `utils/db_search.py` | Atlas `$vectorSearch` (index `vector_index`, 512d cosine) + Python cosine fallback + embedding backfill |
| `utils/db_utils.py` | **re-export facade** — consumers import from here even though the logic lives in the five domain modules above |

### Database collections

```text
faces          (vector-indexed: latest_embedding)
events
visit_memory   (person_id unique)
```

Full field semantics: Appendix C of the [Hub](../CURRENT_ARCHITECTURE.md#appendix-c--mongodb-collections) and `docs/02 - Architecture/MongoDB Schema.md`.

### Threading / concerns

* One shared connection, used concurrently by recognition workers, the 2 queue consumers, and API routes.
* The facade hides which domain module owns a function — read `db_utils.py` first when locating logic.
* **Invariants:** Atlas score conversion (Section 12); quality gates before embedding storage; PyMongo `ReturnDocument.AFTER` (not `return_document=True`) in memory upserts.

---

# 20. Alerts

## `agents/alert_agent.py` (199 lines)

### Responsibility

Dispatches alerts through the configured channels (`settings.ALERT_CHANNELS`, default `["console", "webhook"]`). Alerts are downstream effects of decisions — dispatch never determines identity.

### Current dispatch contract (rewritten 2026-10-03, tested)

```text
caller:  track.mark_alerted_once()   → True only the first time (one-shot, in-track lock)
caller:  dispatch(track, decision, image_url)
             │
             ├─ decision.should_alert == False        → return False (flag already burned)
             ├─ decision.alert_level not in AlertLevel → warn "dispatch_invalid_alert_level"
             ├─ should_send_alert(track_id, level, status)
             │     per-key cooldown stamped AT ALLOW-TIME
             │     keys: {track}:{level} or {track}:unverified:{level}
             │     window: ALERT_COOLDOWN_SECS (60), stale keys pruned (_prune_stale_alerts)
             ├─ build payload (track_id, status, alert_level, person_id, name, reason,
             │                 is_masked, timestamp, camera_id, image_url, memory_context)
             ├─ console channel: synchronous (_alert_console)
             ├─ LLM NL summary: generated synchronously BEFORE thread dispatch
             │     (template fallback if Ollama unavailable)
             └─ email / SMS / webhook: _alert_executor (2 threads)
```

### Call sites (verified)

* `agents/recognition_worker.py:282` — mid-track alert (e.g., blacklist while the track is live).
* `track_processor.py:451` — finalization alert.

### Known characteristics

* The cooldown cache is **module-global** (`_alert_timestamps`), pruned by age so it cannot grow unbounded.
* The NL summary generation is inline on the calling worker — when Ollama is down this adds roughly 4 seconds per dispatch attempt (observed 2026-10-03); the console line itself is not blocked by the network channels.

### Test coverage

`tests/test_alert_dispatch.py` (6 tests) — mark-before-dispatch, cooldown suppression, payload shape, `should_send_alert` allow-then-suppress, one-shot marking.
