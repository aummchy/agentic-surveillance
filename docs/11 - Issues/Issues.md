# Issues

> Single tracker for pipeline logic issues. Originally from the 2026-08-04 codebase review; absorbs `BUG_REPORT.md` (2026-08-01), all items re-verified against code 2026-10-01. ISSUE-18/19 added 2026-10-03 from a `videos/low_4.mp4` run. Each has severity, reproduction conditions, and proposed fix.

---

## ISSUE-1 — Hidden/masked persons never trigger alerts

**Severity:** Critical · **Status:** Open · **File:** `agents/track_processor.py:72-78`

### What's wrong

When a person never produces a usable face (hidden, masked with no detectable face), `snap.embedding is None` at finalization. `process()` returns early without ever calling `decide()`, so **Policy Rule 6 (`intentionally_hidden` → high alert) and Rule 7 (masked loitering → medium/high alert) never fire**. The visibility classifier runs correctly in `track_state.py:171-182`, setting `visibility="hidden"`, but the result is never acted upon.

```python
# track_processor.py:72-78
if snap.embedding is None:
    self._log_event(track, "unknown", "none", False, 0.0, image_url)
    return   # ← never calls decide(), never dispatches
```

### Root cause

Two-stage gate:
1. `camera_agent.py:333-339` — on `skip_reason == "low_quality"`, `set_best_face` is only called if `quality.is_valid` (False for low-quality faces), so no crop is stored for the finalizer.
2. `finalizer.py:retry_embedding` — `track.best_face_crop is None` → no embedding found → `track.embedding` stays None.
3. `track_processor.process` — early return on `None` embedding → no policy decision.

### Consequence

An intruder who keeps their face hidden or covered triggers **no alert at all**. Only a plain "unknown"/"none" event is logged. The whole "intentionally_hidden" status (policy Rule 6) is dead code in practice.

### Proposed fix

Make `process()` run `decide()` even when embedding is None, using the track's visibility classification. For no-embedding tracks, pass visibility-derived signals to the policy agent. Registration/skip-registration remains gated on embedding presence.

```python
# In track_processor.process(), replace the early return:
if snap.embedding is None:
    # Still run policy for visibility-based alerts (hidden/masked)
    match_result = MatchResult(matched=False)
    recognition_result = {"status": "unknown", "confidence": 0, "similarity": 0, "is_masked": snap.is_masked}
    decision = decide(track, match_result, recognition_result, {})
    if decision.should_alert and track.mark_alerted_once():
        dispatch(track, decision, image_url)
    self._log_event(track, decision.status, decision.alert_level, track.alerted, 0.0, image_url)
    return
```

---

## ISSUE-2 — Full-frame fallback result discarded when crop has weak faces

**Severity:** Low · **Status:** Open · **File:** `pipeline/recognition_pipeline.py:189-197`

### What's wrong

When the crop contains a face below embedding quality (det_score < 0.40), `fallback_used=True` runs full-frame detection. But the selection logic picks `crop_faces[0]` because `if crop_faces:` comes before `elif frame_faces:`, so the full-frame result is discarded. The weak crop face is then rejected at line 216 (`det_score < EMBEDDING_DET_SCORE_MIN`), and recognition is skipped entirely.

```python
# recognition_pipeline.py:189-197
if crop_faces:
    best = crop_faces[0]          # ← always wins
    detected_in_person_crop = True
elif frame_faces:
    best = frame_faces[0]         # ← never reached if crop_faces non-empty
```

### Consequence

Latent today (`ENABLE_FULL_FRAME_FALLBACK=false`). If the fallback is ever re-enabled, the full-frame detection is wasted — the crop's weak face is always preferred.

### Proposed fix

```python
best = None
if crop_faces and any(f["det_score"] >= settings.EMBEDDING_DET_SCORE_MIN for f in crop_faces):
    best = crop_faces[0]
    detected_in_person_crop = True
elif frame_faces:
    best = frame_faces[0]
elif crop_faces:
    best = crop_faces[0]
    detected_in_person_crop = True
```

---

## ISSUE-3 — Quality score computed on a different image than the embedding

**Severity:** Low · **Status:** Open · **File:** `pipeline/recognition_pipeline.py:225` + `utils/embedding_utils.py:100-102`

### What's wrong

The embedding comes from `detect_faces_raw(person_crop)` where `person_crop` is **CLAHE-enhanced** inside `detect_faces_raw`. But `face_crop` (used for quality gate and storage) is cut from the **raw** frame at `frame[fy1:fy2, fx1:fx2]`. The gate's blur/brightness scores don't reflect the image actually embedded.

### Consequence

Faces that would fail quality on the raw crop are stored as embeddings (from the enhanced version). Minor inconsistency — the gate is slightly more conservative than necessary.

### Proposed fix

Use the same CLAHE-enhanced crop for both quality and embedding, or extract quality from the CLAHE output. Low priority — existing behavior is conservative (slightly over-rejects) which is acceptable.

---

## ISSUE-4 — `active_ids` dead code in camera loop

**Severity:** Cosmetic · **Status:** Resolved · **File:** `agents/camera_agent.py:165,169`

### What's wrong

`active_ids` is built by adding `track.track_id` for each active track but is never read. Left over from the pruning logic (`_finalized_track_ids &= active_track_ids`) that was intentionally removed per the documentation update.

### Proposed fix

Remove the `active_ids` variable entirely (lines 165 and 169).

---

## ISSUE-5 — `EMBEDDING_CACHE_COSINE_THRESHOLD` in docs but not in code

**Severity:** Cosmetic · **Status:** Resolved · **File:** `docs/09 - Reference/All Thresholds.md:24`

### What's wrong

`docs/09 - Reference/All Thresholds.md` line 24 lists `EMBEDDING_CACHE_COSINE_THRESHOLD` referencing `recognition_pipeline.py`. This embedding cache was removed from the codebase but the docs were not updated. The constant does not exist in the current `pipeline/recognition_pipeline.py`.

### Proposed fix

Remove the row from `docs/09 - Reference/All Thresholds.md` and any other stale references.

**Closing note (2026-10-03):** the referenced document no longer exists — the whole
`docs/09 - Reference/` section was deleted in the documentation pruning. The stale
row is gone with it, so this issue is resolved by deletion.

---

## ISSUE-6 — `set_best_face` never called on `"low_quality"` path

**Severity:** Low · **Status:** Open · **File:** `agents/camera_agent.py:333-339`

### What's wrong

When `skip_reason == "low_quality"`, `quality.is_valid` is always False (that's why it was low quality), so `set_best_face` is never called. The track reaches finalization with `best_face_crop = None` and `embedding = None` → no face stored, no retry possible.

### Consequence

If a face is slightly below quality threshold (e.g., blur_score=38, just under QUALITY_VALID_BLUR_MIN=40), no crop is stored at all, even though the face may be recoverable by the finalizer.

### Proposed fix

Store the face crop on low_quality path even when quality is invalid, but only if `det_score >= DET_SCORE_RELAXED` (detectable face exists). This gives the finalizer a chance to retry with relaxed thresholds.

---

## ISSUE-7 — `DEDUP_SIMILARITY_THRESHOLD` default mismatch

**Severity:** Cosmetic · **Status:** Resolved · **Files:** `config/settings.py:410,592` vs `config/config.jsonc:15`

### What's wrong

`config/config.jsonc` sets `DEDUP_SIMILARITY_THRESHOLD: 0.5` (effective at runtime). `config/settings.py` default is `0.40`. The docs say `0.5`. Not a runtime bug (config.jsonc wins), but the settings.py default is misleading and inconsistent.

### Proposed fix

Change settings.py default to `0.5` to match config.jsonc and docs.

---

## ISSUE-8 — `repomix-output.*` stale snapshot files

**Severity:** Low · **Status:** Resolved · **Files:** `repomix-output.xml`, `repomix-output.json`

### What's wrong

These files contain stale snapshots of the codebase (pre-refactor). They reference `EMBEDDING_CACHE_COSINE_THRESHOLD`, old architecture trees, and old default values that no longer match the current code. They can cause confusion when searching for current state.

### Proposed fix

Delete `repomix-output.xml` and `repomix-output.json`, or add them to `.gitignore`.

---

## ISSUE-9 — Policy diverges from Recognition on mid-range matches

**Severity:** Low · **Status:** Open · **File:** `agents/policy.py:245-255`

### What's wrong

When the Recognition Agent classifies a matched person as `"uncertain"` (confidence 55-69, matched=True), the Policy Agent's RULE 5 maps matched-but-below-known-visitor to `status="unknown"`, overriding the recognition status. The event then carries person_id and name but status says "unknown".

### Consequence

Dashboard shows "Unknown: John Smith" — contradictory. Not a security bug (no alert, no registration) but a UX inconsistency. The current behavior is intentional per AGENTS.md fix (prevent auto-registered unknowns promoted to known_visitor).

### Proposed fix

Change RULE 5 matched-but-unconfirmed return to `status="uncertain"` instead of `status="unknown"` to stay consistent with the recognition classification.

---

## Verified Fixed (absorbed from BUG_REPORT.md)

Re-verified against code 2026-10-01 — no action needed, history kept here so `BUG_REPORT.md` can be deleted:

| Old # | Problem | Verified at |
|-------|---------|-------------|
| 2 | `result.quality` unguarded on success path | pipeline guarantees `quality` on success; guarded branch `camera_agent.py:392` |
| 3 | `_policy_agent` singleton race | lock present, `policy.py:383-391` |
| 5 | `dispatch()` wrong return values | `alert_agent.py:57-131` returns False/True correctly |
| 7 | Stale `pmr` via embedding cache | embedding cache removed entirely |
| 8 | `set_best_face` TOCTOU | design confirmed correct (docs-only fix) |
| 9 | LLM async client leak | now sync `httpx.Client` + `client.close()`, `llm_client.py:269-280` |
| 10 | Misleading mean-embedding comment | comment removed, `db_faces.py:95-105` |
| 13 | `FRAME_SKIP = 2` hardcoded in `live.py` | constant gone; skip handled in `track_processor.py` |
| 14 | f-string loggers in `live.py` | structured kwargs everywhere |
| 17 | `_finalized_track_ids` per-frame prune | prune removed; `discard()` after finalization, `camera_agent.py:545` |
| 19 | Recognition pool too small | now `settings.RECOGNITION_MAX_WORKERS`, `camera_agent.py:41` |
| 20 | ByteTrack ID reuse collision | discard after finalization verified |
| 21 | No per-track HIGH alert cooldown | `last_alert_time` + `ALERT_COOLDOWN_SECS`, `camera_agent.py:472-482` |

---

## Open Issues (absorbed from BUG_REPORT.md, re-verified 2026-10-01)

---

### ISSUE-10 — `submit_time` dead write in camera loop

**Severity:** 🟢 Low · **Status:** Open · **File:** `agents/camera_agent.py:249`

In `_loop()`, `submit_time = self._timing.record_submit(...)` is assigned but never read (the real read is `pop_submit()` in the worker at line 366). Wastes a dict slot; return value unused.

**Fix:** drop the assignment: `self._timing.record_submit(track.track_id)`.

---

### ISSUE-11 — `datetime.utcnow()` deprecated (Python 3.12+)

**Severity:** 🟡 Medium · **Status:** Open · **Files:** `utils/db_faces.py` (~10 sites: 45, 47, 54, 55, 74, 83, 91, 154, 159), `utils/db_memory.py:29,77`, `tests/test_thread_safety.py:157`

Naïve datetime, `DeprecationWarning` now, `AttributeError` in a future Python.

**Fix:** replace with `datetime.now(tz=timezone.utc)`.

---

### ISSUE-12 — String `last_seen` silently skips memory boost

**Severity:** 🟡 Medium · **Status:** Open · **File:** `agents/memory.py:101-103`

Only `isinstance(last_seen, datetime)` is handled; if MongoDB returns an ISO string, `days_since_last` stays `None` and all time-based confidence boosts are skipped.

**Fix:** add a `str` branch parsing with `datetime.fromisoformat()`.

---

### ISSUE-13 — `resolve_track_image_url` reads/writes `track.image_url` without lock

**Severity:** 🟡 Medium · **Status:** Open · **File:** `utils/image_utils.py:122-148`

Worker threads write `track.image_url` while the camera loop reads it in `draw_annotations` — data race.

**Fix:** double-checked locking with `track._lock`: fast-path read under lock, upload outside lock, write under lock only if still unset.

---

### ISSUE-14 — `_get()` returns raw strings when `default=[]`

**Severity:** 🟡 Medium · **Status:** Open · **File:** `config/settings.py:71-76`

Element-type casting only happens when `default` is non-empty; any list setting with an empty default gets strings from `.env`.

**Fix:** accept an explicit `elem_type` param (or fall back to `str`), instead of gating on `default`.

---

### ISSUE-15 — Mixed datetime APIs in policy track lifetime

**Severity:** 🟢 Low · **Status:** Open · **File:** `agents/policy.py:180`

`datetime.now().timestamp() - track.first_seen` mixes two APIs that both yield UTC epoch today, but breaks silently if either side changes (e.g. `utcnow()`).

**Fix:** `time.time() - track.first_seen`.

---

### ISSUE-16 — `os.environ` written every frame in hot path

**Severity:** 🟢 Low · **Status:** Open · **File:** `pipeline/tracker.py:51`

`os.environ["OPENVINO_DEVICE"] = ...` runs ~30×/s inside `track_persons()`; global mutation from the camera thread.

**Fix:** move into `get_model()` (runs once).

---

### ISSUE-17 — Collection init uses unlocked outer check

**Severity:** 🟢 Low · **Status:** Open · **File:** `utils/db_client.py:48-51` (+ same pattern for events/memory)

Double-checked locking with an unguarded outer read — safe under the GIL, fragile on free-threaded Python 3.13+.

**Fix:** drop the outer check; always take the (uncontended after first call) lock.

---

## Issues from the 2026-10-03 video-file run

> Observed running `python main.py` with `CAMERA_SOURCE=videos/low_4.mp4` (1920×1080, 29.97 fps, 973 frames) and 4 dashboard tabs. Evidence from `logs/surveillance.jsonl` (2004 lines, run ends 07:49:28).

---

## ISSUE-18 — WebSocket broadcast flood: 1080p/q95 frames with no backpressure

**Severity:** High → Medium (Tier 1 applied 2026-10-03) · **Status:** Partially fixed — Tier 2 open · **Files:** `config/config.jsonc:113,204-205`, `dashboard/backend/routes/live.py:16-45`, `agents/track_processor.py:62-80`, `dashboard/frontend/src/App.jsx`

### Observed

```
07:49:12 / 07:49:22  frame_broadcast                      (frames flowing)
07:49:25             video_complete videos/low_4.mp4
07:49:25             camera_stopped
07:49:27             client_disconnected ×4 (total 3,2,1,0)
07:49:27-28          204 × client_send_timeout   (64 × TimeoutError,
                                                  140 × 'Cannot call "send"
                                                  once a close message has been sent.')
07:49:28             ~40 × clients_dropped_silent count=4 remaining=0
07:49:28             shutdown_complete
```

### What's wrong

`config/config.jsonc` broadcast at full preview resolution **(state at time of observation — since changed to 960×540 q85, see Proposed fix)**:

```jsonc
"JPEG_QUALITY_BROADCAST": 95,     // line 113
"BROADCAST_WIDTH": 1920,          // line 204
"BROADCAST_HEIGHT": 1080,         // line 205
```

Measured on `videos/low_4.mp4`: one frame encodes to **433 KB JPEG → 577 KB base64 per WebSocket message**. At ~10 broadcasts/s × 4 connected clients that is ~23 MB/s of JSON the browser must `JSON.parse`, wrap in a data URL, and decode as a 1920×1080 JPEG — on a single main thread, alongside EventLog re-renders and `fetchStats()`.

Intended values (`640×360` @ q80) produce **50 KB** — an 11.5× reduction. AGENTS.md documents the 640×360 preview as the design; config.jsonc overrides it.

### Root cause

Four defects compound:

1. **Unbounded producer queue** — `agents/track_processor.py:62-80` submits an encode+broadcast job for every `FRAME_SKIP`-th frame with no check on whether previous sends completed. No flow control between camera loop and consumers. ~50 broadcast coroutines were still pending at shutdown.
2. **Cancelled sends corrupt the connection** — `live.py:19-25` wraps `client.send_text()` in `asyncio.wait_for(..., 3.0)`. On timeout `wait_for` **cancels** the in-flight `send`, leaving the `websockets` connection state such that every subsequent send raises `RuntimeError: Cannot call "send" once a close message has been sent` — the 140-error wave.
3. **Dead clients never closed, log spam** — `_broadcast` (live.py:28-45) only does `connected_clients.difference_update(...)`; it never calls `websocket.close()`. Each backlogged coroutine recomputes `disconnected` from its **own** results snapshot taken before the first one cleared the set, so ~40 of them log the identical `count=4 remaining=0` line.
4. **Frontend reconnect storm** — `App.jsx:90-99` retries after a fixed 500 ms with no backoff and without closing the prior socket first.

Loopback bandwidth is not the constraint (it does GB/s). The failure is **TCP backpressure**: the browser stops draining → its receive window closes → `transport.write()` stops progressing → `await send_text()` blocks → `wait_for` timers stack up → all fire in one burst. Because the app layer queues instead of dropping, latency only grows.

### Consequence

Live feed freezes/goes stale, `logs/surveillance.jsonl` gets 200+ warnings in 2 seconds at shutdown, ~115 MB of queued base64 strings resident in the event loop, and dashboard tabs churn through connect/drop cycles.

### Proposed fix

**Tier 1 — applied 2026-10-03** (config + frontend; fixes the observed flood):

1. ✅ **Config** — `BROADCAST_WIDTH: 960`, `BROADCAST_HEIGHT: 540`, `JPEG_QUALITY_BROADCAST: 85` (implemented as 960×540 q85 rather than 640×360 q80 to keep the full-width panel sharp): 564 KB → ~101 KB per message, 22.0 → 4.0 MB/s at 4 clients.
4. ✅ **Frontend** — exponential reconnect backoff (500 ms → cap 5 s) with stale-socket and unmount guards, close the existing socket before opening a new one, `setLiveFrame` throttled to one update per animation frame (latest frame wins).

**Tier 2 — still open** (makes the failure structurally impossible; not yet implemented):

2. **Latest-wins backpressure** — in `track_processor.handle_frame`, skip submitting when a broadcast is already in flight (atomic flag set by the producer, cleared by the coroutine). Queue depth stays 0–1; stale frames are dropped, never queued.
3. **`live.py`** — never cancel an in-flight send; use a per-client in-flight slot so only one message is outstanding per client. On failure, `await client.close()` (guarded) and discard. Rate-limit `client_send_timeout` to one log per client per state change, and log `clients_dropped_silent` once per batch against the *current* set. Lower `_SEND_TIMEOUT` from 3.0 s to ~1.0 s.
5. **Optional** — send binary WebSocket frames instead of base64-in-JSON to remove the +33% byte tax.

Without Tier 2 the unbounded producer queue remains: raising the resolution back, or a large number of dashboard tabs, can reintroduce the same failure mode. See also ISSUE-19 for the shutdown-time traceback.

---

## ISSUE-19 — Proactor `Exception in callback _start_serving` at shutdown (Windows)

**Severity:** Low · **Status:** Open · **File:** `main.py:126-173`

### What's wrong

```
Exception in callback BaseProactorEventLoop._start_serving.<locals>.loop(
  <_Overlapped...0.1', 63624))>) at ...\asyncio\proactor_events.py:836
```

At shutdown `main.py:159` sets `server.should_exit = True`, uvicorn closes the port-8000 listening socket while a `_ProactorBaseServeSocketLoop` overlapped `accept()` is still pending. When the completion callback fires on the closed socket it raises, and asyncio prints the raw `Exception in callback` traceback to stderr. Windows/Proactor only — harmless in effect, alarming in appearance.

### Root cause

- No exception handler installed on the manually created loop, so callback errors go to stderr instead of structlog.
- `main.py:169-171` aggravates it: `server_thread.join(timeout=5)` can expire while uvicorn is still draining, because the `/ws/live` handlers sit in `websocket.receive_text()` forever and never finish — then `loop.close()` runs against a loop the server thread is still using (`RuntimeError: Event loop is closed` from stragglers).
- `run_server()` never calls `asyncio.set_event_loop(loop)` for its thread.

### Consequence

Scary traceback at the end of every video-file run; in the worst case lingering tasks fail against a closed loop and mask real errors.

### Proposed fix

```python
def run_server():
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())

# after camera.start() returns:
server.should_exit = True
server_thread.join(timeout=10)
if server_thread.is_alive():
    server.force_exit = True          # uvicorn: stop waiting for idle conns
    server_thread.join(timeout=5)
if server_thread.is_alive():
    logger.warning("server_thread_still_alive")   # do NOT close the loop
else:
    loop.close()

loop.set_exception_handler(lambda lp, ctx: logger.debug("loop_exception", **ctx))
```

Also close every client in `live.connected_clients` before setting `should_exit`, so uvicorn can drain immediately instead of waiting on WebSocket handlers that never return.

Related: ISSUE-18 — the broadcast flood is what leaves clients wedged in `receive_text()` during shutdown.

---

## See also
- `AGENTS.md` (Gotchas) — operational issues
- [[CURRENT_ARCHITECTURE]] section 6 — threshold references
- `config/config.jsonc` — config references
