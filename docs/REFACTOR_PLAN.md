# Refactor Plan (Phase 4 candidates)

> **Nothing here is approved.** Every item is parked pending Phase 3 review and
> explicit approval per `plan.md` and `AGENTS.md` §4 (architecture changes
> require a written proposal: `CURRENT: file → responsibility → caller → state`
> / `TARGET: …`, then wait for approval). Phase 3 is review-only and must not
> modify code.
>
> This file absorbs the open items from the deleted `TODO.md` (Items 4–5) and
> the Phase 4 list from `plan.md`.

## Status legend

- 🅿️ parked — documented, not reviewed
- 📋 reviewed — triaged in Phase 3, awaiting approval
- ✅ approved — explicit go-ahead given, one at a time

---

## 1. `RecognitionPipeline.run_final(track, snapshot)` — consolidate the dual recognition chains 🅿️

**Origin:** verification memo Claim 3; Phase 1 finding (both `recognition_pipeline.py`
and `track_processor.py`).

Make `track_processor.process()` call a finalization entry point on
`RecognitionPipeline`, then delete `_run_matching` / `_run_recognition_and_memory`.
Precondition: a line-by-line progressive-vs-finalization comparison is written up
first (the Phase 1 reports supply the material) so intentional differences —
no live frame, `pending_*` reuse, snapshot inputs — are preserved exactly.

## 2. Threshold alignment: `CONFIDENCE_KNOWN_MIN=70` vs `KNOWN_VISITOR_CONFIDENCE=80` 🅿️

**Origin:** `plan.md` Phase 4; verification memo residual.

Policy accepts KNOWN at 70 confidence but its known-visitor secondary gate uses 80,
so the two gates disagree about what "known" means. Reconcile to one number without
changing which real classifications pass (behavior must be verified before/after).

## 3. Identity embedding lifecycle (former `TODO.md` Items 4 & 5) 🅿️

### Item 4 — MERGED path should update the existing record's embedding

**Where:** `agents/track_processor.py` → `_handle_registration()`.

**Problem:** when `deduplicate_identity()` returns MERGED, the new (potentially
better-quality) embedding is discarded entirely — the existing record keeps its
old, possibly poor embedding forever.

**Consequence:** vicious cycle — poor embedding → low similarity → person shows as
UNKNOWN → no embedding update → stuck.

**Proposed change:** in the MERGED branch, call `update_face()` to refresh the
existing record's embedding when the new one has higher quality.

**Risk:** medium — changes auto-registration behavior; needs real data testing.

### Item 5 — refresh embeddings for matched persons

**Where:** `agents/track_processor.py` → `process()`.

**Problem:** once a person is registered, `latest_embedding` is never refreshed
during normal operation, so embeddings go stale (appearance change, aging,
lighting) and matching accuracy degrades over weeks/months.

**Proposed change:** after `match_result.matched=True`, quality-gated
`update_face()` with the new embedding.

**Risk:** medium — touches normal pipeline flow; needs regression testing on
matching accuracy.

**Verification for both:** `python -m pytest tests/ -v --tb=short` plus real-data
matching comparison.

## 4. `Track` split 🅿️

`pipeline/models.py` `Track` is a god object (40+ fields). Separate tracking /
face / pending-result / identity / decision / presentation concerns. Structural —
requires its own proposal showing current vs target ownership per section.

## 5. `settings.py` defaults de-duplication 🅿️

Some defaults are duplicated between `config/settings.py` and
`config/config.jsonc`, creating a "which number wins" hazard. Remove the
duplication **without changing any effective value** (all effective values must
be identical before and after).

## 6. Final cleanup 🅿️

Dead code sweep, documentation re-verification against code, and closing out this
file's items after each lands.

---

## Related, deferred by decision

- **`"person_id"` payload key carries `track.track_id`** in
  `TrackProcessor._broadcast_alert` — established frontend contract; rename only
  together with the frontend (Phase 3 review decides).
- **Per-track `RecognitionAgent()` construction** in
  `TrackProcessor._run_recognition_and_memory` — fold into item 1.
- **O(n²) active-track snapshot rebuild** in `CameraAgent._process_tracks` —
  measure before touching.
