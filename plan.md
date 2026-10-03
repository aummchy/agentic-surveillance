# plan.md — Phased roadmap

Status legend: ✅ done · 🔄 in progress · ⬜ not started

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Instruction cleanup + verification (docs only) | ✅ 2026-10-03 |
| 1 | Readability pass, zero behavior changes | ⬜ |
| 2 | Current-state documentation | ⬜ |
| 3 | Review-only, no code changes | ⬜ |
| 4 | Structural refactor (approval-gated) | ⬜ |

**Standing invariants for Phases 0–3:** 84 tests green (`python -m pytest tests/ -q`) ·
zero behavior changes · no commits unless explicitly requested · no structural code
change without a written proposal and explicit approval.

**Operating rules:** see `AGENTS.md` → "AI Development Workflow".
**Evidence for what is/ isn't actually broken:** `docs/11 - Issues/Review 2026-10-03 - External AI.md`.
**Historical (non-authoritative) debug notes:** `docs/HISTORICAL_DEBUG_NOTES.md`.

---

## Phase 0 — Instruction cleanup + verification ✅

Docs/instruction files only. No Python touched.

- [x] `AGENTS.md`: condensed **AI Development Workflow** section inserted near the top
      (source of truth, allowed/not-allowed, one-change-at-a-time, architecture approval,
      response format); stale "Current priorities: fix code issues" replaced with current phase;
      note that AGENTS.md is the only auto-loaded instruction file.
- [x] `senior.md` → `archive/senior.md` with an "ARCHIVED — NOT AN ACTIVE INSTRUCTION FILE" header.
- [x] `see.md` → `docs/HISTORICAL_DEBUG_NOTES.md` with a "NOT AUTHORITATIVE" header.
- [x] Deleted stale `repomix-output.json` / `repomix-output.xml` (kept `repomix.config.json`).
- [x] Verification memo: `docs/11 - Issues/Review 2026-10-03 - External AI.md`
      (every external claim verified → already-fixed / correct / residual gap).
- [x] This file (`plan.md`).

**Deliberately NOT changed in Phase 0:** any `.py` file, thresholds, config values,
`TODO.md` items, `docs/02 - Architecture/*` (deferred to Phase 2).

---

## Phase 1 — Readability pass (zero behavior changes)

Order (established pattern on the smallest, best-tested file first):

| # | File | Lines | Status |
|---|------|-------|--------|
| 1 | `pipeline/recognition_pipeline.py` | 392 | ✅ 2026-10-03 (commit `57b46df`) |
| 2 | `pipeline/track_state.py` | 454 | ✅ 2026-10-03 |
| 3 | `agents/camera_agent.py` | 545 | ✅ 2026-10-03 |
| 4 | `agents/track_processor.py` | 370 | ✅ 2026-10-03 |

**Phase 1 status: all 4 core files complete (readability pass done).**

Coverage note: file 1 is covered by 9 stage tests in
`tests/test_recognition_pipeline_stages.py`; full suite is 84.

Per-file loop, strictly sequential:

```
understanding report  →  your approval  →  readability edits  →  tests (84)  →  Changed/Verified/Not-Changed report  →  next file
```

Understanding report must cover: responsibility, importers, imports, owned/mutated state,
calling threads, inputs/outputs, side effects, covering tests.

**Allowed edits:** module/function/class docstrings · intent/invariant comments ·
type hints on public signatures (no runtime effect) · section headers ·
removal of clearly unused imports · formatting.

**Forbidden in Phase 1:** renames (even "provably safe") · line-number references in prose
(use function names — line numbers rot) · logic/import-behavior changes ·
docstring claims about bugs · any fix found along the way (write it in the report instead).

**Findings recorded during Phase 1 (documented, deliberately NOT fixed):**

From file 1 `pipeline/recognition_pipeline.py`:
- `skip_reason` is annotated/returned as plain `str` but semantically a
  `config/status.py:SkipReason` member; equality with the enum only works because
  `SkipReason` is a `StrEnum`. Phase 4 candidate.
- `_select_best_face` picks `crop_faces[0]` — verified *not* a bug:
  `embedding_utils.detect_faces_raw` sorts descending by `det_score`.
- `RecognitionMetrics` is constructed twice (once in `_detect_face`, a fresh
  embed/db-only one in `_build_embedding`); `run()` copies detection timings across
  by hand. Documented at the handoff site. Phase 4 candidate for a cleaner shape.

From file 2 `pipeline/track_state.py`:
- `get_active_track_ids_and_boxes` docstring/type hint were wrong (claimed
  `(byte_track_id, person_box)` / `Tuple[int, tuple]`; the value is the composite
  id, i.e. identical to the key). **Corrected in Phase 1** — annotation + doc only.
  The redundant value itself (value duplicates key) is a Phase 4 note.
- `remove()` has no callers in production or tests, and does not clear `_in_flight`.
  Documented as unused; NOT deleted.
- `set_embedding`'s `(bool, reason)` return is consumed by `finalizer.py` but ignored
  by `camera_agent`. Documented.
- `update()` matches tracks by `byte_track_id` alone (camera_id unused in the match —
  correct for single-camera) via a linear scan under the global lock. Documented.
- `set_person_name` cannot distinguish similarity 0.0 from "never set". Documented.
- Lock-order invariant (`self._lock` → `track._lock`, never reversed) now stated
  explicitly at module level, plus why `_classify_visibility_inplace` needs only
  `self._lock`.

From file 3 `agents/camera_agent.py`:
- **No test file imports `CameraAgent`** — zero direct coverage. `_loop`,
  `_process_tracks`, `_should_skip_recognition`, `_handle_decision_and_alert`,
  `_cleanup_recognition_track`, `_finalize_track` are untested. Biggest testing
  gap found so far → Phase 3.
- Stale comment at `_finalized_track_ids` ("Intentionally not pruned") directly
  contradicted by `_finalize_track`, which discards the id after finalization.
  **Comment corrected in Phase 1** (approved); behavior untouched.
- Unused import `Any` **removed**.
- `_should_skip_recognition` mutates `track.rescan_attempts` — a predicate with a
  side effect. Documented, not refactored.
- `set_recognition_snapshot` is fed the **policy** decision status while
  `_should_skip_recognition` reads it as if it were the recognition status —
  same RECOG/POLICY conflation as verification memo Claim 8. → Phase 3/4.
- `_recognizing_tracks` is set at worker start, not at submit time, so the
  queue-time check alone is not airtight; the worker-start resolved-check is the
  real backstop. Documented (this is the corrected form of `see.md` claim #1).
- `_process_tracks` rebuilds the active-track snapshot once per detection
  (quadratic per frame). Note only.

From file 4 `agents/track_processor.py` (Phase 1 complete):
- **No test file imports `TrackProcessor`** — zero direct coverage of
  `process()` (the whole finalization chain), `handle_frame`, `_handle_registration`,
  etc. Combined with CameraAgent this is the largest testing gap → Phase 3.
- This file is the **second implementation of the recognition chain** (verification
  memo Claim 3): `_run_matching` + `_run_recognition_and_memory` + `decide()`
  re-run what `RecognitionPipeline.run()` does, against a snapshot + `pending_*`
  results instead of a live frame. **Now documented in the module docstring**;
  unification remains Phase 4 (`run_final()`).
- `RecognitionAgent()` constructed fresh **per track** at `_run_recognition_and_memory`
  — the only non-reusing site in the system. Documented (already a Phase 4 item).
- `fresh_match` is derived solely from `pending_match_result is None` and gates
  trust in `pending_recognition` **and** `pending_memory_context` — three fields
  coupled to one flag. Documented.
- `_record_visit` mutates `memory_context["visit_count"]` **in place** after the
  memory agent returns, so the event log shows the post-increment value.
  Documented as deliberate.
- `_broadcast_alert` stores `track.track_id` under the payload key **`"person_id"`**
  — an established frontend contract (Phase 3/4 decision: rename + frontend together,
  or leave). Documented as a hazard, payload **not** changed.
- `_broadcast_event` uses `hasattr`/`getattr` for `alerted`/`person_name` but reads
  `best_face_crop_path` bare — inconsistent defensiveness. Documented.
- `_log_event` rebuilds `reason` from the status label instead of `decision.reason`,
  so rows read "Track finalized: <status>". Documented.
- `shutdown()` sets `_shutdown_event` but **nothing in this class reads it**
  (main.py owns the real event; `handle_frame` guards on loop-alive). Documented.
- `enqueue()` is a one-line `Queue.put` wrapper used once — documented as such.

Carried forward (pre-existing):
- `track_processor.py` `_run_recognition_and_memory` constructs a fresh
  `RecognitionAgent()` per finalization.
- Two recognition implementations exist (see verification memo, Claim 3).

---

## Phase 2 — Current-state documentation

1. **Create `docs/CURRENT_ARCHITECTURE.md`** — as-built inventory, NOT the target architecture.
   - Top: runtime flow diagram (main.py → CameraAgent → Tracker → TrackState →
     progressive RecognitionPipeline → TrackProcessor → Policy → DB/alerts/dashboard).
   - Then per component: `File` / `Responsibilities currently performed` / `Called by` /
     `Mutates` / `Threading` / `Known concerns`.
   - Components: main.py, camera_agent, tracker, track_state, recognition_pipeline,
     track_processor, policy, recognition/scoring, alert, finalizer, db layer, settings,
     logging, dashboard backend/frontend.
2. **Absorb-then-delete 3 overlapping docs** (content absorbed into #1 first):
   - `docs/ARCHITECTURE.md` (124-line tech-stack table)
   - `docs/02 - Architecture/System Overview.md` (83 lines)
   - `docs/02 - Architecture/Tech Stack.md` (66 lines, subset pointer)
   - Fix every inbound link: `README.md:92`, `SYSTEM_INDEX.md:3`, `docs/00 - Home.md:14,80`,
     wiki-links `[[System Overview]]` in Thread Architecture / Tracker (YOLO + ByteTrack) /
     3-Tier Logging / Base Agent / LLM Client, and `Tech Stack.md:7`.
   - Keep (audit only, unique content): `docs/02 - Architecture/Data Flow.md`,
     `Thread Architecture.md`, `MongoDB Schema.md`.
3. **Audit in place — no new filenames:**
   - `docs/02 - Architecture/Data Flow.md` verified line-by-line against code.
   - `docs/04 - Pipeline/Data Models.md` upgraded to carry the STATE_MODEL content
     (what information exists at each stage: Frame → Detection → Track → face evidence →
     Embedding → Match → Recognition → Policy → Finalization → Person/Event,
     including `pending_*` fields and who writes/reads them).
4. **Create `docs/ARCHITECTURE_RULES.md`** — semantic boundaries only:
   Identity (track_id ≠ person_id) · Detection · Tracking · Recognition ·
   Verification (weak result must not replace stronger accumulated evidence) ·
   Policy (must not implement a second recognition pipeline) · Storage.
5. **Create `docs/REFACTOR_PLAN.md`** — parked Phase 4 candidates, each marked
   "requires Phase 3 review + approval": `run_final()` consolidation · threshold pair
   70 vs 80 · `TODO.md` Items 4–5 · `Track` split · `settings.py` default de-duplication ·
   per-call `RecognitionAgent()`.

Gate: all links resolve, 84 tests still pass (no code touched in Phase 2).

---

## Phase 3 — Review-only (no code changes)

Ask for an architecture review that **must not modify code**. Output format:

```
CRITICAL | HIGH | MEDIUM | LOW
file · function · problem · evidence (file:line) · impact · recommendation
```

Must be triaged by me + you against the Phase 0 verification memo (so already-fixed
issues are not re-litigated) and the Phase 2 docs. Result: an approved, prioritized
list that becomes Phase 4 input. Anything not on the approved list stays parked.

---

## Phase 4 — Structural refactor (one approval-gated proposal at a time)

Each item starts with a proposal in the required format
(`CURRENT: file → responsibility → caller → state` / `TARGET: …`) and waits for
explicit approval before any edit. One item, then tests, then report, then the next.

1. **`RecognitionPipeline.run_final(track, snapshot)`** — make `track_processor.process()`
   use it; delete `_run_matching` / `_run_recognition_and_memory`.
   Precondition: line-by-line progressive-vs-finalization comparison written up first
   (Phase 1 reports supply this), so intentional differences are preserved.
2. **Threshold alignment** — reconcile `CONFIDENCE_KNOWN_MIN=70` with
   `KNOWN_VISITOR_CONFIDENCE=80` (policy.py:286 secondary gate).
3. **Identity embedding lifecycle** — `TODO.md` Items 4 & 5 (MERGED-path embedding update,
   matched-person embedding refresh).
4. **`Track` split** — separate tracking / face / pending-result / identity /
   decision / presentation concerns in `pipeline/models.py`.
5. **`settings.py` defaults de-duplication** — remove the duplicated-number reading hazard
   without changing any effective value.
6. Final cleanup: dead code, `TODO.md` reconciliation, docs re-verification.

---

*Created 2026-10-03 (Phase 0). Update the status table as phases complete.*
