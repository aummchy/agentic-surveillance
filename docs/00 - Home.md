# Surveillance System - Map of Content

> AI-powered single-camera surveillance: YOLOv8 person detection → ByteTrack tracking → InsightFace face recognition → autonomous decision engine → alerts + local LLM for NL summaries.

---

## Canonical documents (read in this order)

| File | What it covers |
|------|---------------|
| [`../AGENTS.md`](../AGENTS.md) | Operating contract: workflow, invariants, do/don't |
| [[CURRENT_ARCHITECTURE]] | Current architecture snapshot: overview, runtime flow, concurrency — plus 5 linked detail files (components, recognition, decision, platform, code state) |
| [`../plan.md`](../plan.md) | Phased roadmap: what is done, what is next, gates |
| [[REFACTOR_PLAN]] | Phase 4 candidates — **nothing here is approved** |
| [[HISTORICAL_DEBUG_NOTES]] | Old debugging notes — **not authoritative, verify first** |
| [[Issues]] | Issue tracker (ISSUE-1..19 + verified-fixed history) |

> Rule of thumb when docs and code disagree: **code wins** (see `AGENTS.md` §0).

## Getting Started

- [[Quick Start]]
- [[Prerequisites]]
- [[Configuration System]]
- [[Environment Variables]]

## Architecture

- [[CURRENT_ARCHITECTURE]] — overview, runtime flow, concurrency, appendices (hub)
  - [[Current Architecture - Components]] — major components, tracking, track state (§4–7)
  - [[Current Architecture - Recognition]] — the two recognition paths, pipeline, quality, matching (§8–14)
  - [[Current Architecture - Decision Flow]] — policy, finalization, memory, DB, alerts (§15–20)
  - [[Current Architecture - Platform]] — dashboard, LLM, configuration, status (§21–24)
  - [[Current Architecture - Code State]] — duplication, concerns, fixed issues, tests (§27–30)
- [[Data Flow]] — per-frame step-by-step walkthrough
- [[Thread Architecture]] — every thread and its responsibilities
- [[MongoDB Schema]] — collections and indexes

## Pipeline

- [[Data Models]] — data/state at each pipeline stage (`pending_*` fields, who writes/reads what)

## Formulas

- [[Confidence Scoring]]
- [[Face Quality Scoring]]
- [[Vector Search & Matching]]
- [[Memory Boost]]
- [[Policy Rules]]
- [[Atlas Score Conversion]]

## Logging

- [[Terminal Output Reference]] — event formats seen in the console

## Problems (Phase 3 triage input — line refs dated, verify against code)

- [[00 - Problems Home]]
  - Pipeline — [[Recognition Bottleneck (26s Full-Frame Scan)]] · [[Identities Never Generated (No Embeddings)]] · [[Database Latency]]
  - Track — [[Duplicate Finalization (Visit Inflation)]] · [[Same Person Becomes Multiple Tracks]] · [[Fragmentation Detection is a No-op]]

## Source Code Layout

```
main.py                          Entry point, wires everything
agents/                          AI orchestration (camera, matching, recognition,
                                 scoring, policy, memory, alerts, report, timing,
                                 finalizer, track_processor)
pipeline/                        Computer vision (tracker, recognition_pipeline,
                                 track_state, quality_agent, models)
utils/                           Shared utilities (db_* modules, embedding,
                                 image, llm)
config/                          Settings loader + config.jsonc + ByteTrack YAML
dashboard/                       FastAPI backend + React frontend
tests/                           97 pytest tests (8 files)
```

## Quick Navigation by Concept

| Concept | Start here |
|---------|------------|
| How does detection work? | [[CURRENT_ARCHITECTURE]] → `pipeline/tracker.py` |
| How does face recognition work? | [[Data Flow]] → [[Vector Search & Matching]] |
| How is confidence calculated? | [[Confidence Scoring]] |
| What triggers an alert? | [[Policy Rules]] → `agents/policy.py` |
| How does the dashboard work? | `dashboard/backend/` → [[Data Flow]] |
| What are all the thresholds? | [[CURRENT_ARCHITECTURE]] Appendix B |
| How is the system configured? | [[Configuration System]] |
