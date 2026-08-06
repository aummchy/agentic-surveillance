# Same Person Becomes Multiple Tracks

**Category:** Track · **Severity:** High · **Status:** Open

## What's wrong
One physical person produces 2+ `Track` objects, so one person = many fragments, each re-running the 25s recognition.

Normally one unique `byte_track_id` → exactly one `Track` (`track_state.py:34` builds `composite_id = f"{camera_id}_{session_epoch}_{byte_track_id}"`). Three scenarios break that:

1. **Occlusion / exit-frame → re-enter** (`TRACK_TIMEOUT_SECS = 15.0`): hidden >15s → track expires and finalizes. ByteTrack's `track_buffer` is 60 frames, so on reappearance ByteTrack hands out a **new ID** → new Track. Biggest source of duplicates.
2. **ID switch (fragmentation)**: ByteTrack swaps IDs between overlapping people, or splits one trajectory across two IDs. Each ID = new Track.
3. **Long presence > `MAX_TRACK_SECS` = 300 (5 min)**: `get_expired_tracks()` finalizes any track older than 5 minutes (`track_state.py:115` → `is_max_lifetime_exceeded()`). Person standing 15 min = 3 separate Track objects.

Also: app restart resets everything — `session_epoch = int(time.time())` makes even the same ByteTrack ID a brand-new composite ID.

## The deep reason — timeout mismatch
- `TRACK_TIMEOUT_SECS = 15.0` — **wall-clock** (app expires Track after 15s).
- `track_buffer = 60` — **frame-count** (ByteTrack keeps ID for 60 frames).

These only align at exactly 30 FPS. At the real (low) FPS, 60 frames ≈ far more than 3 seconds, so ByteTrack still *owns* the ID while the app already deleted the track → same ID "reincarnates" as a brand-new Track. Combined with the 26s recognition (person walks off before recognition finishes), this repeats constantly.

## Real log evidence (`surveillance.debug.log`)
Same ByteTrack ID created 3–4 times in ~90 seconds:

```
bt_track_id=22  created 13:31:45 → removed → created again 13:31:56 → created again 13:32:01
bt_track_id=14  created 13:31:41 → recreated 13:32:11 → recreated 13:32:26
bt_track_id=5   appears in 4 recognition cycles
```

## Consequence
Each finalized Track can auto-register an identity (`store_face`) or record a visit → duplicate identities and inflated visit counts. **See also [[Duplicate Finalization (Visit Inflation)]] for the concrete mechanism — a track finalized 4× inflates visit_count by 4 in one run.**

## Fix options
- **Option A — align the timeouts (smallest change):** set `TRACK_TIMEOUT_SECS` to match `track_buffer` at expected FPS, e.g. 2.0s at 30fps. Eliminates the coexist/reincarnation window. Doesn't solve fragmentation itself.
- **Option D — raise track_buffer to exceed app timeout:** `track_buffer: 90` (3s at 30fps) so ByteTrack never releases an ID before the app timeout. Eliminates reincarnation at the cost of slower recovery after true ID switches.
- **Recommended combo:** A + D (~2 lines total) to make both layers consistent, plus the recognition fix in [[Recognition Bottleneck (26s Full-Frame Scan)]] so finalization happens while the person is still visible.
- Proposed: `TRACK_TIMEOUT_SECS` 15.0 → 10.0.

## Related
- [[Track State]]
- [[Tracker (YOLO + ByteTrack)]]
- [[Fragmentation Detection is a No-op]]
