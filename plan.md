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

1. `pipeline/recognition_pipeline.py` (392 lines, ~20 stage tests)
2. `pipeline/track_state.py` (454)
3. `agents/camera_agent.py` (545)
4. `agents/track_processor.py` (370)

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

**Known notes to record during Phase 1 (not fix):**
- `track_processor.py:197` constructs a fresh `RecognitionAgent()` per finalization.
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
