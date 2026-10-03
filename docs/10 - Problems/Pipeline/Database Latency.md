# Database Latency

> **Stale refs:** file:line citations date from 2026-08 — verify against current code before acting (Phase 3 re-verification).

**Category:** Pipeline · **Severity:** Low · **Status:** Open

## What's wrong
`db_ms = 121–695ms` per matching call — MongoDB Atlas vector search round trips to the cloud. Not the main bottleneck, but on top of a 25s recognition it makes everything feel ~2× worse than it needs to be.

## Real log evidence (`recognition_timing`)

| track | db_ms |
|-------|-------|
| 8  | 591 |
| 2  | 695 |
| 13 | 121 |
| 15 | 272 |

## Fix (later)
- Cache recent Atlas match results (e.g. short TTL keyed by embedding hash) to avoid repeat round trips for the same person within a track.
- Consider a local index if the cloud round trip stays slow.

## Related
- `utils/db_*`
- [[Vector Search & Matching]]
