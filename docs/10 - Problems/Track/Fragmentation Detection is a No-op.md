# Fragmentation Detection is a No-op

> **Stale refs:** file:line citations date from 2026-08 — verify against current code before acting (Phase 3 re-verification).

**Category:** Track · **Severity:** Medium · **Status:** Open

## What's wrong
When a split is detected, nothing is done. `track_state.py:59-79` computes IoU against existing tracks when a new track is created and logs `nearest_track_id` / `nearest_iou` — then does nothing. It's **debug-only**. No merge, no cancellation of the stale track, no embedding comparison.

There is also no embedding-based re-identification at the track level: `TrackState` never asks "is this new ID the same face as a recently-dead track?" That check only happens at finalization, and only for unknown auto-registration. A known/verified person fragmented across tracks still produces 2+ separate Track objects (duplicate events/alerts per fragment).

## Real log evidence
```
track_created: new track 25, nearest_iou=0.305, time_gap_ms=0.0 vs existing track 24
```
A genuine ByteTrack spatial split (a person cut in two), detected and then ignored.

## Consequence
Duplicate fragments pile up, each re-running the 26s recognition and producing separate finalization/alert events.

## Fix options
- **Option B — real fragmentation detection in TrackState (medium):** replace the debug-only IoU block with active logic: when a new track overlaps an existing track by IoU > threshold AND the existing track is stale (>1s no update), merge — delete the old track's identity and hand its accumulated state (embedding, best face, person_name) to the new track. Needs a `merge_tracks()` method + careful lock handling.
- **Option C — embedding-based track re-ID (most correct, most work):** when a track expires, keep its embedding in a short-lived "recently departed" cache (~10s). When a new track spawns, compare its first good embedding against that cache before treating it as a new identity. Reuses `compare_similarity()`. Catches fragments even without spatial overlap.
- **Recommendation:** B if duplicates are still a problem after the timeout + recognition fixes. C is overkill for a single-cam system.

## Related
- `pipeline/track_state.py`
- [[Same Person Becomes Multiple Tracks]]
- `utils/embedding_utils.py`
