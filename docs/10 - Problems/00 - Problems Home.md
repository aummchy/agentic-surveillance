# Problems — Map of Content

> Every known problem in the surveillance system, organized by subsystem.
> Each note contains: what's wrong, the deep reason (with file:line), real log evidence, consequence, and fix.
>
> **Staleness warning:** file:line citations in these notes date from 2026-08 and
> may no longer match the code — verify before acting (see `AGENTS.md` §0).
> Already known stale: the Duplicate Finalization note cites a
> `_finalized_track_ids &= active_track_ids` pruning step that is no longer in
> the code. Statuses are unreviewed; re-verification happens in Phase 3 (`plan.md`).

---

## Pipeline
- [[Recognition Bottleneck (26s Full-Frame Scan)]]
- [[Identities Never Generated (No Embeddings)]]
- [[Database Latency]]

## Track
- [[Same Person Becomes Multiple Tracks]]
- [[Fragmentation Detection is a No-op]]
- [[Duplicate Finalization (Visit Inflation)]]

---

## The root cause in one paragraph

Recognition took **25–29 seconds** — now reduced to **6.4s mean** after crop expansion + fallback removal. The dominant remaining bottleneck is `INSIGHTFACE_DET_SIZE=1280` upscaling small crops before SCRFD detection. This slow detection causes a 63-second queue backlog, which directly causes **[[Duplicate Finalization (Visit Inflation)]]**: stale workers finish 30+ seconds after track removal, each triggering a duplicate finalize that inflates visit counts. Combined with a timeout mismatch (`TRACK_TIMEOUT_SECS=15.0` wall-clock vs `track_buffer=60` frames at variable FPS), one person can produce 4–5 separate Track objects in a single run.
