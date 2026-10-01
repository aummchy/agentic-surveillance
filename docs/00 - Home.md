# Surveillance System — Map of Content

> AI-powered single-camera surveillance: YOLOv8 person detection → ByteTrack tracking → InsightFace face recognition → autonomous decision engine → alerts + local LLM for NL summaries.

---

## Getting Started
- [[Quick Start]]
- [[Prerequisites]]
- [[Configuration System]]
- [[Environment Variables]]

## Architecture
- [[System Overview]]
- [[Data Flow]]
- [[Thread Architecture]]
- [[MongoDB Schema]]
- [[Tech Stack]]

## Agents (AI Orchestration)
- [[Base Agent]]
- [[Camera Agent]]
- [[Track Processor]]
- [[Matching Agent]]
- [[Recognition Agent]]
- [[Scoring Module]]
- [[Policy Agent]]
- [[Memory Agent]]
- [[Alert Agent]]
- [[Finalizer]]

## Pipeline (Computer Vision)
- [[Tracker (YOLO + ByteTrack)]]
- [[Recognition Pipeline]]
- [[Face Detection & Embedding]]
- [[Quality Assessment]]
- [[Track State]]
- [[Data Models]]

## Utilities
- [[Database Utils]]
- [[Embedding Utils]]
- [[Image Utils]]
- [[LLM Client]]

## Formulas
- [[Confidence Scoring]]
- [[Face Quality Scoring]]
- [[Vector Search & Matching]]
- [[Memory Boost]]
- [[Policy Rules]]
- [[Atlas Score Conversion]]

## Dashboard
- [[Backend API]]
- [[Frontend]]
- [[WebSocket Live Feed]]

## Logging
- [[3-Tier Logging]]
- [[Terminal Output Reference]]
- [[Calculation Log]]

## Reference
- [[All Thresholds]]
- [[All Config Settings]]
- [[Common Gotchas]]

## Problems
- [[00 - Problems Home]]
  - Pipeline → [[Recognition Bottleneck (26s Full-Frame Scan)]] · [[Identities Never Generated (No Embeddings)]] · [[Database Latency]]
  - Track → [[Duplicate Finalization (Visit Inflation)]] · [[Same Person Becomes Multiple Tracks]] · [[Fragmentation Detection is a No-op]]

## Issues (open tracker)
- [[Issues]] — pipeline logic issues + verified-fixed history (single tracker, re-verified 2026-10-01)

## Canonical References (root)
| File | What it covers |
|------|---------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Tech stack table, connection patterns, MongoDB collections, key thresholds |
| [GOAL.md](GOAL.md) | Priority (pipeline first), use cases, commercial maturity, planned improvements |
| [FEATURES.md](FEATURES.md) | Complete inventory of implemented features |
| [REFERENCES.md](REFERENCES.md) | Open-source recognition pipelines to study (DeepFace, InsightFace, CompreFace, Frigate) |
| [CODEREFERENCE.md](CODEREFERENCE.md) | File-by-file code map + metrics — dated refactoring snapshot, verify before relying on line numbers |

---

## Source Code Layout

```
main.py                          ← Entry point, wires everything
agents/                          ← AI orchestration (7 agents + scoring + finalizer)
pipeline/                        ← Computer vision (tracker, face, quality, models)
utils/                           ← Shared utilities (DB, embedding, image, LLM)
config/                          ← Settings loader + config.jsonc + ByteTrack YAML
dashboard/                       ← FastAPI backend + React frontend
tests/                           ← 84 pytest tests
```

## Quick Navigation by Concept

| Concept | Start here |
|---------|-----------|
| How does detection work? | [[Tracker (YOLO + ByteTrack)]] |
| How does face recognition work? | [[Face Detection & Embedding]] → [[Vector Search & Matching]] |
| How is confidence calculated? | [[Confidence Scoring]] |
| What triggers an alert? | [[Policy Rules]] → [[Alert Agent]] |
| How does the dashboard work? | [[Backend API]] → [[WebSocket Live Feed]] |
| What are all the thresholds? | [[All Thresholds]] |
| How is the system configured? | [[Configuration System]] |
