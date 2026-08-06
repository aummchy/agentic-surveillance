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
- [[Decision Agent]]
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

## Canonical References (root)
| File | What it covers |
|------|---------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Tech stack table, connection patterns, MongoDB collections, key thresholds |

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
