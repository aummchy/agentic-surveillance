# Identities Never Generated (No Embeddings)

**Category:** Pipeline · **Severity:** High · **Status:** Resolved (2026-07-30)

## What was wrong

In earlier logs, `embed_ms = 0.0` on every track. The embedding was **never generated** because no face ever passed the threshold. The whole identity/memory system never got used — it just spammed "unknown" events.

## The deep reason (resolved)

- The face was only searched in the tiny person crop → failed because crops were too small (see [[Recognition Bottleneck (26s Full-Frame Scan)]]).
- `faces_on_crop = 0` in 100% of runs → full-frame fallback triggered → 26s per recognition.
- The full-frame fallback was so slow it consumed the entire recognition budget.

## Real log evidence (current run, 2026-08-01)

`embed_ms = 0.0` was a **red herring** — the embedding extraction (`tolist()`) is instant because InsightFace computes it during `app.get()`. The expensive part is the detection, not the embedding.

Current state (after crop expansion + fallback removal):
- `faces_on_crop=0`: **0%** (all crops find faces)
- `vector_search_started`: **41** (all 41 recognitions searched embeddings)
- `match_found`: **28** (matches for Alice, Solo, Bob, Unknown, Intruder)
- `track_embedding_present`: **9** (embeddings stored at finalization)
- `no_embedding_after_retries: 18` — legitimate (person's face never visible, `face_detected=False`)

## What fixed it

1. **Crop expansion 20%** (`recognition_pipeline.py:159-164`) — ensures face is fully within crop
2. **Full-frame fallback removed** (`ENABLE_FULL_FRAME_FALLBACK: false`) — stops wasting time on full-frame scan
3. **CLAHE skip for tiny crops** (<200px) — avoids destroying features

## Remaining note

`EMBEDDING_DET_SCORE_MIN = 0.40` still filters out weak detections (< 0.40 det_score). This is intentional — it prevents garbage embeddings from low-confidence faces. No change needed.

## Related
- [[Face Detection & Embedding]]
- [[Embedding Utils]]
- [[Vector Search & Matching]]
- [[Recognition Bottleneck (26s Full-Frame Scan)]]
