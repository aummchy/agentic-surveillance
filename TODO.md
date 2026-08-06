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
