# Review 2026-10-03 — External AI architecture assessment (verified)

**Scope:** An external AI reviewed `agentic-surveillance-src.zip` (source-only snapshot,
81 files) and proposed a 4-stage plan (document → fix state bugs → consolidate pipeline →
folder refactor). This memo checks every claim against the current source code.

**Rule:** code is the source of truth. Findings below are evidence-based, not inherited.
Companion documents: `docs/HISTORICAL_DEBUG_NOTES.md` (old `see.md`, now marked
non-authoritative) and `plan.md` (the approved roadmap).

---

## Verdict summary

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 1 | 8,028 Python LOC (excl. venv) | ✅ Correct | measured 2026-10-03 |
| 2 | Seven largest files / line counts | ✅ Correct | camera_agent 545, track_state 454, policy 406, recognition_pipeline 392, track_processor 370, settings 393, models 271 |
| 3 | Two recognition implementations exist | ✅ **Correct — real duplication** | see "Duplication" below |
| 4 | `Track` model is overloaded | ✅ Correct | see "Track overload" below |
| 5 | Progressive recognition race / duplicate queueing | ⚠️ **Already fixed** | `agents/camera_agent.py:242-256`, `:357-363`, `:293-299` |
| 6 | Worse result overwrites better result | ⚠️ **Already fixed** | `pipeline/track_state.py:394-442`, `pipeline/models.py:43,87` |
| 7 | `track_id` vs `person_id` ownership confusion | ⚠️ **Already handled** | composite IDs + `MatchResult.person_id` |
| 8 | RECOG vs POLICY threshold divergence | ⚠️ **Largely fixed, one residual** | `agents/policy.py:274-283` |
| 9 | Identity double-registration | ⚠️ **Guarded, two residual gaps** | `utils/db_faces.py:211-261` |
| 10 | `settings.py` duplicate defaults | ✅ Real but deliberate | documented config chain |
| 11 | Immediately split `Track` | ❌ Rejected — Phase 4 only | see `plan.md` |
| 12 | Immediately consolidate pipeline (`run_final`) | ❌ Deferred — Phase 4 | see `plan.md` |

---

## Claim 3 — Duplication: TWO recognition implementations ✅ CONFIRMED

**Progressive path** (mid-track, every 20 frames):
`agents/camera_agent.py:379` → `pipeline/recognition_pipeline.py:83 run()` =
detect → quality → embed → match → memory → recognize → policy.

**Finalization path** (track ends): `agents/track_processor.py:105-112` re-runs the
same four stages inline:

- `_run_matching()` — `track_processor.py:167-172`
- `_run_recognition_and_memory()` — `track_processor.py:182-214`
  (constructs a **fresh `RecognitionAgent()` per call** at `:197`, unlike the
  pipeline which reuses one instance)
- `decide()` — `track_processor.py:112`

`RecognitionPipeline` is imported from exactly **one** place
(`agents/camera_agent.py:13,379`); `track_processor.py` never calls it.

**Why this is not simply "duplicate code":** the two paths take different inputs —
the pipeline needs a live `frame`; finalization usually has a snapshot + cached
`pending_*` results and no reliable frame. Any consolidation must preserve that
difference. → Recorded in `plan.md` Phase 4 as `RecognitionPipeline.run_final()`,
explicitly gated on a line-by-line progressive-vs-finalization comparison first.

**Duplicate-queueing note:** `track_processor.py:197` re-instantiating
`RecognitionAgent()` per track is a trivially safe cleanup candidate (the agent is
stateless) — but it is still a behavior-adjacent change, so Phase 1 only notes it,
Phase 4 fixes it.

## Claim 4 — `Track` overload ✅ CONFIRMED

`pipeline/models.py` (271 lines) `Track` carries at least six unrelated concerns:
tracking state (boxes, scores, frame counters), face state (best face crop/score,
visibility, rescan counters), embedding + `pending_match_result` /
`pending_recognition` / `pending_memory_context`, identity (`person_name`,
`person_name_similarity`), decision/alert state (`decision`, `confidence`,
`last_alert_time`), presentation (`image_url`, `person_crop_url`,
`best_full_frame`). → Phase 4 candidate, gated behind a proposal.

## Claim 5 — Progressive race ⚠️ ALREADY FIXED

`see.md` claimed: worker-start has no resolved check, `_recognizing_tracks` only
guards at queue time, finalization can run twice.

Current code:

- Schedule-time guard: `agents/camera_agent.py:242-256` (`already_recognizing` check
  before `submit`)
- **Worker-start resolved check: `agents/camera_agent.py:357-363`** —
  bails if `track.decision in RESOLVED_STATUSES` (this is `see.md` fix #2, implemented)
- Double-finalization guard: `agents/camera_agent.py:293-299` —
  `_finalized_track_ids` checked *and* added under one lock before submit

**Verdict:** `see.md` item #2 is historical. Do not reimplement.

## Claim 6 — Downgrade / no-match clobber ⚠️ ALREADY FIXED

`see.md` root cause: "set_pending_match_result / set_pending_recognition_data
unconditionally overwrite". Current code has only-upgrade guards:

- `pipeline/track_state.py:394-407` — `set_pending_match_result`: accepts only if
  `new.similarity >= existing.similarity`; comment: "Never let a no-match (sim=0)
  clobber a real match."
- `pipeline/track_state.py:409-427` — `set_pending_memory_context`: only-upgrade
  (`is_known`, `visit_count`)
- `pipeline/track_state.py:429-442` — `set_pending_recognition_data`: accepts only if
  `new_conf > existing_conf`
- `pipeline/models.py:43,87` — `Track.confidence` "Highest confidence seen — only
  upgrades, never downgrades"
- Early exit for already-great matches: `pipeline/recognition_pipeline.py:91-97`
  (`skip_reason="high_confidence"`) and `agents/camera_agent.py:264-270`

**Verdict:** `see.md` item #1 is historical. Do not reimplement.

## Claim 7 — track_id vs person_id ⚠️ HANDLED

Composite track IDs `{camera_id}_{session_epoch}_{byte_track_id}` (unique across
restarts) are tracking-scope; `MatchResult.person_id` is database-scope. Both are
carried explicitly through `snapshot()` and `_log_event()`
(`agents/track_processor.py:129-131`). No confusion found in current code.

## Claim 8 — RECOG vs POLICY divergence ⚠️ LARGELY FIXED, ONE RESIDUAL

`see.md` proposed: "make policy defer to the recognition agent's known status".
Implemented at `agents/policy.py:273-283`:

```python
# If recognition already classified as "known", respect that
if ctx["rec_status"] >= Status.KNOWN:
    return DecisionResult(status=Status.KNOWN_VISITOR, ...)
```

with `rec_status` taken from the recognition result at `agents/policy.py:163-168`.
So RECOG=KNOWN (confidence ≥ 70, `config/settings.py:294`) now yields
POLICY=KNOWN_VISITOR — the divergence seen in the old trk=2 log is closed.

**Residual (documented, not fixed):** `agents/policy.py:286` still has an independent
gate `similarity >= 0.85 or confidence >= 80`
(`config/config.jsonc:143-144`). It is now only a *secondary* path for
`rec_status < KNOWN`, so it no longer contradicts recognition — but the two thresholds
(`CONFIDENCE_KNOWN_MIN=70` vs `KNOWN_VISITOR_CONFIDENCE=80`) still encode different
beliefs about "known". → `plan.md` Phase 3 (review-only) item.

## Claim 9 — Identity double-registration ⚠️ GUARDED, TWO RESIDUAL GAPS

`see.md` #4 proposed dedup before registration. Implemented:
`agents/track_processor.py:235` calls `deduplicate_identity()`
(`utils/db_faces.py:211-261`) *before* `_store_new_face()`; threshold
`DEDUP_SIMILARITY_THRESHOLD = 0.45` (`config/config.jsonc:15`); verified persons are
never silently merged (`db_faces.py:240-244`); Atlas failure falls back to Python scan,
and if both fail registration is **aborted** (`track_processor.py:245-249`) rather than
creating a duplicate.

**Residual gaps (already tracked in `TODO.md`, not fixed here):**

1. `TODO.md` Item 4 — MERGED path discards the new, possibly better embedding.
2. `TODO.md` Item 5 — `latest_embedding` never refreshed for matched persons.

Both are behavior changes → Phase 3/4 territory.

## Claim 10 — `settings.py` duplicate defaults ✅ REAL, DELIBERATE

`config/settings.py` duplicates every default that `config/config.jsonc` also carries
(e.g. `DEDUP_SIMILARITY_THRESHOLD` 0.40 in settings vs 0.45 in jsonc,
`KNOWN_VISITOR_SIMILARITY` 0.85/0.85). This is the documented priority chain
(`.env` > `config.jsonc` > hardcoded default), not an accident — but the duplicated
numbers are a genuine reading hazard. → Phase 4 candidate (make settings.py defaults
provably authoritative or derive them), **not** a silent cleanup.

---

## Rejected / deferred recommendations

| External recommendation | Disposition | Reason |
|---|---|---|
| Split `Track` now | ❌ Phase 4 | Requires Phase 2 state documentation + written proposal first |
| `RecognitionPipeline.run_final()` consolidation now | ❌ Phase 4 | Two paths have different inputs (frame vs snapshot); merge before understanding = deleting intentional differences |
| Trust `see.md` bugs | ❌ Rejected | Items #1, #2 verified already fixed (above) |
| Stage 1 = write new architecture doc without auditing existing | ❌ Changed | Three overlapping architecture docs already exist; Phase 2 audits/absorbs instead of adding a fourth |

## Open items routed to `plan.md`

- Phase 1: readability pass, order `recognition_pipeline.py` → `track_state.py` → `camera_agent.py` → `track_processor.py`
- Phase 3 (review-only): threshold pair 70 vs 80; `TODO.md` Items 4–5; per-call `RecognitionAgent()` construction
- Phase 4 (approval-gated): `run_final()` consolidation, `Track` split, `settings.py` default de-duplication

---

*Compiled 2026-10-03. Evidence lines verified against working tree at that date.*
