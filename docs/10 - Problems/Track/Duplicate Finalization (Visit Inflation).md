# Duplicate Finalization (Visit Inflation)

> **Stale refs:** file:line citations date from 2026-08 — verify against current
> code before acting. The cited pruning step `_finalized_track_ids &= active_track_ids`
> is no longer present; a three-guard scheme (`mark_finalized_once` +
> `_finalized_track_ids` + in-flight checks) now exists. Status needs Phase 3 re-verification.

**Category:** Track · **Severity:** Critical · **Status:** Open

## What's wrong

One physical person triggers **4–5 finalize calls** per track, each incrementing `visit_count` and logging `track_finalized` with identical data. Visit counts inflate by 3–5 per person per run.

**Real log evidence** (`surveillance.jsonl`, run 2026-08-01 09:11:32–09:15:19 UTC):

```
track_finalized ×4  for cam_01_1785575492_6  (identical data each time)
visit_recorded ×5   for cam_01_1785358203_6  visit_count: 33→34→35→36→37
```

**Consequence:**
- Visit history inflated → wrong confidence boosts, wrong memory profiles
- Duplicate `track_finalized` events / duplicate `face_crop_saved` writes
- Potential duplicate alerts (mitigated by `mark_alerted_once()` but not by finalization)

---

## Trace: Track 6 lifecycle through the code

| Time | Function | What happened |
|------|----------|---------------|
| 09:11:42 | `TrackState.update()` (track_state.py:50) | Created, added to `_tracks`. `active_count=6` |
| 09:11:42–54 | `_loop()` (camera_agent.py:118) | Frames refresh `last_seen` — track alive |
| ~09:11:54 | `_loop()` interval | `_progressive_recognition` submitted (worker 1) |
| 09:12:00 | worker 1 starts | `begin_recognition` → `_in_flight[6]=1` |
| 09:12:07 | worker 1 done | `decision=None` (skip). `end_recognition` → `_in_flight=0`. Track still active → **no removal**. Line 437 guard: `get(6) is not None` → **no finalize** ✓ |
| ~09:11:55 | `_loop()` interval | worker 2 submitted |
| 09:12:15 | worker 2 starts | `_in_flight[6]=1` |
| ~09:12:1x | `get_expired_tracks()` (track_state.py:111) | Track expired (person left frame). Sets `expired_reported=True` but **defers removal** (`_in_flight>0`) |
| 09:12:23 | worker 2 done | `decision=known_visitor`, conf=92. `end_recognition` → `_in_flight=0` → **removes track 6** ("expired_after_recognition"). Line 437: `not in _finalized_track_ids` ✓, `get(6) is None` ✓ → **submit `_finalize_track` #1** ✓ |
| 09:12:23+ | `_loop()` next frame | **line 245-248: `_finalized_track_ids &= active_track_ids` → erases track 6's ID** (not in active set) |
| 09:12:25 | worker 3 starts | `_in_flight[6]=1` (track object already gone from `_tracks`) |
| 09:12:32 | worker 3 done | line 437: `not in _finalized_track_ids` (**True — pruned**), `get(6) is None` (True) → **submit `_finalize_track` #2** ✗ |
| 09:12:41 / 09:12:49 | workers 4, 5 start | same pattern |
| 09:12:46 / 09:12:56 | workers 4, 5 done | **`_finalize_track` #3, #4** ✗✗ |
| 09:12:58–09:13:56 | `_finalize_track` ×4 | 4× `track_finalized` + `record_visit` → visits 33→37 |

---

## Root cause — the pruning erases the guard

The system has **two** "already finalized" guards. Both fail for the stale-worker path.

### Guard 1: `_finalized_track_ids` (CameraAgent set)

Set when `_finalize_track` is submitted (line 240 or 438). **But pruned every frame** at line 245-248:

```python
# camera_agent.py:245-248
active_track_ids = {t.track_id for t in all_tracks}
self._finalized_track_ids &= active_track_ids  # <-- removes removed tracks
```

Once a track is removed from `_tracks` (via `end_recognition` cleanup at track_state.py:195), it's no longer in `all_tracks` → pruned from `_finalized_track_ids` → every stale worker sees "not finalized" and finalizes again. **❌**

### Guard 2: `expired_reported` (Track field, models.py:33)

Set when `get_expired_tracks()` returns a track. Survives on the Track object. But the stale-worker path (camera_agent.py:437) **does not check it**:

```python
# camera_agent.py:437 (the broken guard)
if track.track_id not in self._finalized_track_ids and self.track_state.get(track.track_id) is None:
    # ^^ This checks the pruned set, not expired_reported
    self._finalized_track_ids.add(track.track_id)
    self._recognition_executor.submit(self._finalize_track, track)
```

Only the `get_expired_tracks()` path (line 117) uses `expired_reported`. The finally-block path ignores it. **❌**

### Why both fail

The stale-worker path (line 437) was designed for: "track expired while recognition was running, so `get_expired_tracks()` deferred removal; now that recognition finished, finalize it once." But it doesn't distinguish "the last recognition worker" from "a stale worker that finished after the track was already handled." Every worker finishing after removal triggers a new finalize.

### The queue makes it worse

`INSIGHTFACE_DET_SIZE=640` (changed from 1280) → crop detect takes ~2.2s/recognition. With 2 workers, tracks queue up ~20s deep. A track removed at 09:12:23 still has workers finishing at 09:12:25, 09:12:27 — 4 seconds of stale finalization.

---

## Fix: Track-level `_finalized` flag

The Track object outlives its removal from `_tracks` — stale workers hold references to it. A flag on it survives the `_finalized_track_ids` pruning. Mirrors the existing `mark_alerted_once()` pattern (`models.py:63`).

### Change 1 — `pipeline/models.py`

Add field and method to `Track`:

```python
_finalized: bool = False  # new field

def mark_finalized_once(self) -> bool:       # new method
    with self._lock:
        if not self._finalized:
            self._finalized = True
            return True
        return False
```

### Change 2 — `agents/camera_agent.py:444`

Guard `_finalize_track` at the top:

```python
def _finalize_track(self, track: Track):
    if not track.mark_finalized_once():
        logger.debug("finalize_skipped_duplicate", track_id=track.track_id)
        return
    # ... existing code unchanged
```

### Why this works

1. **Covers both submit paths** — expiry loop (line 241) and stale-worker finally (line 440) both go through `_finalize_track`. One guard, all paths.
2. **Survives removal** — Track object referenced by stale workers still has the flag, immune to `_finalized_track_ids` pruning.
3. **Thread-safe** — lock-protected, exactly one worker wins concurrent races.
4. **Handles reincarnation** — a new Track for the same byte_track_id gets `mark_finalized_once() = True` (fresh object, `False` by default). The old Track's flag only guards the old finalization.
5. **Minimal diff** — ~10 lines, 2 files. `_finalized_track_ids` and its pruning stay untouched (harmless, not source of truth anymore).

---

## Related

- [[Same Person Becomes Multiple Tracks]]
- `pipeline/track_state.py`
- [[Fragmentation Detection is a No-op]]
- `agents/camera_agent.py`

---

## Appendix: Answers to the five lifecycle questions

### 1. Who creates a Track?

`TrackState.update()` at `track_state.py:50`. The Track is a `@dataclass` with 50+ fields (identity, temporal, spatial, recognition state) plus `_lock`. The lock exists because 4 actors write concurrently: camera loop, 2 recognition workers, finalizer.

### 2. Who stores it?

`TrackState._tracks: Dict[str, Track]`, guarded by `TrackState._lock`. `CameraAgent` never touches `_tracks` directly — always through `self.track_state.*`.

### 3. Who modifies a Track?

| Field | Writer | When |
|-------|--------|------|
| `last_seen`, `person_box`, `total_frames_seen` | camera loop via `update()` | every frame |
| `expired_reported`, `visibility` | `get_expired_tracks()` | expiry sweep |
| `best_face_crop/score/frame` | recognition worker (`set_best_face`) | after quality pass |
| `embedding`, `is_masked` | recognition worker / finalizer | recognition / finalize |
| `decision`, `confidence`, `pending_*` | recognition worker | after pipeline run |
| `alerted` | `mark_alerted_once()` (lock-protected) | critical alert |

### 4. Follow one Track

See the trace table above — Track 6 from creation through 4 duplicate finalizations.

### 5. How does the code remember "already finalized"?

Two mechanisms, both failing for the stale-worker path:
- `_finalized_track_ids` — pruned every frame, forgets removed tracks
- `expired_reported` — survives on Track, but not checked by the finally-block path

The fix adds a third, authoritative mechanism: `Track._finalized` — lives on the object, checked by `_finalize_track`, immune to pruning.
