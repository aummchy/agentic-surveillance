# TODO.md — Code Quality Backlog

Items extracted from `senior.md` rules. Each item is a small, low-risk change.
All items must preserve behavior (rule 1) and pass all tests.

---

## Item 1: ✅ Break up large functions

**Status:** Done — 6 functions decomposed in Phase 4.

- `_decide()` in `policy.py` → 10 rule methods + `DecisionContext` dict
- `update_visit_memory()` in `db_memory.py` → 3 helpers + `_STATUS_RANK`
- `_detect_face()` in `recognition_pipeline.py` → 4 helpers
- `_loop()` in `camera_agent.py` → 7 helpers
- `_progressive_recognition()` in `camera_agent.py` → 3 helpers
- `process()` in `track_processor.py` → already decomposed (9 helper methods)

---

## Item 2: ✅ Break up track_processor.py

**Status:** Done — `process()` is already decomposed into 9 helper methods
(`_run_matching`, `_update_person_name`, `_run_recognition_and_memory`,
`_handle_registration`, `_store_new_face`, `_record_visit`, `_dispatch_alert`,
`_broadcast_alert`, `_broadcast_event`, `_log_event`).

---

## Item 3: ✅ Deduplicate db_search.py result construction

**Status:** Done — extracted `_build_search_result(all_results)` helper. Both Atlas
and fallback paths now call the same function.

**File:** `utils/db_search.py:vector_search()`

**Problem:** The result dict construction (lines 70-78 and 91-98) is identical
between the Atlas path and the fallback path. Same `top2`/`margin` computation
and same `matches`/`all_candidates` list building.

**Proposed change:** Extract a `_build_search_result(all_results)` helper that
computes `top2`, `margin`, `matches`, and `all_candidates` from `all_results`.
Both paths call the same helper.

**Risk:** Very low — pure deduplication, no logic change.

**Verification:** `python -m pytest tests/ -v --tb=short`

---

*Created: 2026-08-06*

---

## Item 4: MERGED path should update existing record with better embedding

**Status:** TODO — deferred for now.

**File:** `agents/track_processor.py:_handle_registration()`

**Problem:** When `deduplicate_identity()` returns MERGED (existing person found),
the new (potentially better quality) embedding is completely discarded.
The person's record keeps the old, poor embedding forever.

**Impact:** Vicious cycle — poor initial embedding → low similarity on subsequent
encounters → shows as UNKNOWN → no embedding update → stuck forever.

**Proposed change:** In the MERGED path (line 240-244), call `update_face()` to
update the existing record's embedding if the new one has higher quality.

**Risk:** Medium — changes auto-registration behavior. Needs testing with real data.

**Verification:** `python -m pytest tests/ -v --tb=short`

---

## Item 5: Add embedding update for matched persons

**Status:** TODO — deferred for now.

**File:** `agents/track_processor.py:process()`

**Problem:** Once a person is registered, their `latest_embedding` in MongoDB is
never refreshed through normal operation. Embeddings become stale over time
(appearance changes, aging, lighting).

**Impact:** Progressive degradation of matching accuracy over weeks/months.

**Proposed change:** In `process()`, after `match_result.matched=True`, call
`update_face()` to update MongoDB with the new embedding (quality-gated).

**Risk:** Medium — changes normal pipeline flow. Needs testing to ensure no
regression in matching accuracy.

**Verification:** `python -m pytest tests/ -v --tb=short`
