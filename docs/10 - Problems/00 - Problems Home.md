# Problems — Map of Content

> Every known problem in the surveillance system, organized by subsystem.
> Each note contains: what's wrong, the deep reason (with file:line), real log evidence, consequence, and fix.

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

Recognition took **25–29 seconds** — now reduced to **6.4s mean** after crop expansion + fallback removal. The dominant remaining bottleneck is `INSIGHTFACE_DET_SIZE=1280` upscaling small crops before SCRFD detection. This slow detection causes a 63-second queue backlog, which directly causes **[[Duplicate Finalization (Visit Inflation)]]**: stale workers finish 30+ seconds after track removal, each triggering a duplicate finalize that inflates visit counts. Combined with a timeout mismatch (`TRACK_TIMEOUT_SECS=3.0` wall-clock vs `track_buffer=60` frames at variable FPS), one person can produce 4–5 separate Track objects in a single run.
