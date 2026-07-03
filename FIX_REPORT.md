# FIX_REPORT.md — Actionable Remediation Plan

Source: `ANOMALY.md` root-cause analysis. This document converts that analysis into a task list an
engineering AI agent can execute directly, in order. Each task lists the file, the exact change,
and how to verify it worked. **Do not skip verification steps** — several of these issues interact
(e.g. fixing MATCH_THRESHOLD without fixing camera resolution will still produce bad matches).

Work through phases in order. Each phase is independently testable/deployable.

---

## PHASE 1 — Configuration Fixes (do first, unblocks everything else)

### 1.1 Fix camera resolution mismatch
**File:** `.env`
**Change:**
```
FRAME_WIDTH=1280
FRAME_HEIGHT=720
```
**Why:** InsightFace `det_size=(1280,1280)` (`settings.py:216`) expects ~1280px input. Capturing at
640x480 forces internal upscaling, which introduces interpolation artifacts and degrades small-face
detection.
**Verify:** Confirm camera hardware/driver supports 1280x720 at acceptable FPS before deploying. If
the camera can't sustain 1280x720, lower `INSIGHTFACE_DET_SIZE` in `settings.py` to match actual
capture resolution instead — the two values must match, not necessarily be 1280.

### 1.2 Raise MATCH_THRESHOLD to a safe value
**File:** `.env` line 31
**Change:** `MATCH_THRESHOLD=0.25` → `MATCH_THRESHOLD=0.45`
**Why:** 0.25 cosine similarity on 512-d ArcFace embeddings (~75° angle) is far too permissive.
Causes both false positives (strangers matched as known) and inconsistent false negatives.
**Verify:** After deploy, spot check the "known_visitor" event log for a day — false-positive rate
should visibly drop. If real false negatives increase, tune within `0.40–0.60` (see `config.jsonc:14`),
don't drop back below 0.40.

### 1.3 Fix EMBEDDING_DET_SCORE_MIN override
**File:** `.env` line 49
**Change:** `EMBEDDING_DET_SCORE_MIN=0.30` → `EMBEDDING_DET_SCORE_MIN=0.40`
**Why:** Aligns with `config.jsonc` and `settings.py` defaults; stops low-confidence detections from
producing embeddings that feed the vector search.

### 1.4 Resolve the three-way config conflict permanently
**Files:** `.env`, `config.jsonc`, `config/settings.py`
**Action:**
- Pick ONE source of truth. Recommended: `.env` for secrets/deployment-specific values only
  (ports, DB URLs, camera IDs). Everything else (thresholds, sizes) belongs in `config.jsonc`.
- Remove `FRAME_WIDTH`, `FRAME_HEIGHT`, `MATCH_THRESHOLD`, `PERSON_CONF_THRESHOLD`,
  `EMBEDDING_DET_SCORE_MIN` from `.env` entirely once corrected, so `config.jsonc` values are what
  actually run.
- Add a startup log line in `config/settings.py` that prints the **effective** value and its source
  for every tunable setting, so future mismatches are caught by reading the logs instead of
  cross-referencing three files.
**Verify:** Grep `.env` after the change — it should no longer contain any of the five settings
above unless intentionally kept as the sole source of truth.

### 1.5 Fix validation default drift
**File:** `config/settings.py` line 177
**Change:** Validation should compare against the *config.jsonc* value, not a separate hardcoded
`0.40` literal, so validation and runtime can never silently diverge again.

---

## PHASE 2 — Recognition Logic Bugs

### 2.1 Tighten recognition decision bands
**File:** `agents/recognition.py` lines 97-167
**Action:** After MATCH_THRESHOLD is corrected to 0.45 (Phase 1.2), re-derive the "uncertain" band
(`MATCH_THRESHOLD * 0.8`) — confirm it still makes sense at the new threshold (0.45 * 0.8 = 0.36).
Add a unit test asserting: similarity 0.30 → "unknown", 0.40 → "uncertain", 0.50 → "known" (adjust
exact boundaries to your policy, but lock them down with a test so config changes can't silently
shift classification behavior again).

### 2.2 Fix policy agent threshold reference
**File:** `agents/policy.py` line 211
**Action:** No code change needed once `settings.MATCH_THRESHOLD` is corrected in Phase 1 — but add
a regression test that asserts `policy.py` and `recognition.py` always read the *same* threshold
value, to prevent future drift between the two modules.

### 2.3 Remove/clarify dead fallback path
**File:** `agents/camera_agent.py` line 178
**Action:** Either:
- (a) Remove the dead `frame_faces = []` branch and the unreachable fallback check on line 195, and
  document why full-frame fallback isn't needed (crop-based detection with `DET_SCORE_RELAXED=0.20`
  always succeeds when crop_faces is non-empty), OR
- (b) If full-frame fallback is actually desired behavior for low-quality crops, fix the logic so
  `frame_faces` is populated when `best` is None, e.g.:
```python
frame_faces = app.detect_faces_raw(frame, ...) if not crop_faces else []
```
  Pick (a) unless there's a known scenario where the crop misses a face the full frame would catch.
**Also:** Rename `frame_faces` if kept, since the name implies it's always populated (see 10c).

### 2.4 Reduce CLAHE overhead
**File:** `utils/embedding_utils.py` lines 36-52
**Action:** Cache/skip CLAHE when the input face crop already meets a minimum contrast metric, or
move CLAHE to run conditionally only on low-quality crops rather than unconditionally on every call.
Profile before/after with 2 recognition threads under load to confirm throughput improves.

---

## PHASE 3 — Dashboard Backend & Data Integrity

### 3.1 Fix `fetch()` error handling
**File:** `App.jsx` lines 29-36
**Change:**
```javascript
const res = await fetch('/api/events/stats');
if (!res.ok) throw new Error(`Stats fetch failed: ${res.status}`);
const data = await res.json();
setStats(data);
```
Wrap in try/catch, and on failure keep the last-known-good `stats` state instead of overwriting it
with an error object.

### 3.2 Fix image URL resolution
**Files:** `App.jsx:21`, `EventLog.jsx:79`, `UnknownPersons.jsx:53`, `VerifiedPersons.jsx:65`,
`VerifyModal.jsx:36`
**Action:**
- Replace hardcoded `http://localhost:8000/` with a relative path or an env-configured base URL
  (e.g. `import.meta.env.VITE_API_BASE_URL`).
- Extend the Vite dev proxy config to also cover `/captures` (currently only `/api` and `/ws` are
  proxied).
- Add the missing `getImageUrl()` call in `VerifyModal.jsx:36`.
- Move `getImageUrl` into a single shared `utils/api.js` (see 3.4) so all five files use one
  implementation.

### 3.3 Stop silently swallowing WebSocket errors
**File:** `App.jsx:72`
**Change:**
```javascript
} catch (e) {
  console.error('WebSocket message parse error:', e);
  // surface to UI: setConnectionWarning(true) or similar
}
```

### 3.4 Add a shared API client
**Files:** all frontend components
**Action:** Create `src/utils/api.js` using `axios` with a base URL and a shared response
interceptor for error handling. Migrate `App.jsx`'s raw `fetch()` calls to use it. Remove the
per-component ad hoc HTTP logic.

### 3.5 Fix WebSocket ping timer leak
**File:** `App.jsx:53-56`
**Change:** Before creating a new WebSocket on reconnect, clear the previous `ws._pingInterval`
with `clearInterval()`.

### 3.6 Fix alert notification timer overlap
**File:** `App.jsx:70`
**Change:** Store the timeout ID for the 8-second auto-dismiss and call `clearTimeout()` on it
before setting a new one whenever a new alert replaces `activeAlert`.

### 3.7 Add React error boundaries
**File:** `main.jsx`
**Action:** Wrap the app root (and ideally each major panel: EventLog, UnknownPersons,
VerifiedPersons, ChatPanel) in an `ErrorBoundary` component so one bad render doesn't unmount the
whole dashboard.

### 3.8 Wire up real-time event push
**File:** `dashboard/backend/routes/live.py` lines 52-75
**Action:** Call `broadcast_event()` from wherever events are currently written to MongoDB (likely
in the event-log write path in `main.py` or `agents/memory.py`), so the dashboard gets push updates
instead of relying on 10-second polling.

### 3.9 Add missing event-status filters
**File:** `EventLog.jsx:105-124`
**Action:** Add tabs/filters for `masked_unknown`, `blacklist`, and `intentionally_hidden` instead
of lumping them all under "Unknown".

### 3.10 Give chat panel conversation memory
**Files:** `ChatPanel.jsx`, `dashboard/backend/routes/chat.py:108`
**Action:** Persist message history client-side and send it with each request; update the backend
route to accept and use a `history` field when constructing the model prompt. Cap history length
(see 3.12) to bound payload size.

### 3.11 Fix "new person" flash timing
**File:** `UnknownPersons.jsx:71-75`
**Action:** Either extend the window (e.g. 30s) or, better, use a persistent "unseen" flag stored in
component/local state that clears only when the user views the item, rather than a fixed wall-clock
window that can be missed entirely.

### 3.12 Cap chat message list growth
**File:** `ChatPanel.jsx:5-7`
**Action:** Cap stored messages (e.g. keep last 100) or virtualize the message list to avoid render
slowdown in long sessions.

### 3.13 Remove dead `unknowns` state
**File:** `App.jsx:12`
**Action:** Remove the unused `unknowns` state and the `onUnknownsLoaded` callback plumbing; the
real state already lives in `UnknownPersons.jsx`.

### 3.14 Remove unused dependency
**File:** `package.json:14`
**Action:** Remove `lucide-react` if truly unused, or actually adopt it for icons instead of raw
Unicode characters (pick one — don't leave dead weight either way).

---

## PHASE 4 — Data Flow & Storage

### 4.1 Stop auto-registration bloat
**File:** `main.py` lines 140-165
**Action:** Before `store_face()` for `unknown`/`masked_unknown`, run a similarity check against
recent unknown entries (reuse `DEDUP_SIMILARITY_THRESHOLD`) and update the existing record's visit
count/embedding history instead of always inserting a new document. Consider also: don't
auto-register masked faces at all, since a masked face's embedding is inherently unreliable for
future matching.
**Verify:** Track `faces` collection growth rate before/after over a test period.

### 4.2 Raise embedding history cap or make it adaptive
**File:** `utils/db_utils.py` line 392
**Action:** Increase `$slice: -10` to a higher number (e.g. -25) or switch to weighting by quality
score so high-quality older embeddings aren't discarded in favor of newer low-quality ones. Confirm
`mean_embedding` recomputation logic still performs acceptably at the new size.

### 4.3 Make visit memory update atomic
**File:** `utils/db_utils.py` lines 591-648
**Action:** Replace the read-modify-write pattern with a single atomic MongoDB update using `$inc`
for visit counts and `$push`/`$slice` for history, eliminating the TOCTOU window between
`get_or_create_memory` and `update_one`.
```python
collection.update_one(
    {"person_id": person_id},
    {
        "$inc": {"visit_count": 1},
        "$push": {"visit_history": {"$each": [new_visit], "$slice": -N}},
        "$set": {"last_seen": now}
    },
    upsert=True
)
```

### 4.4 Fix track expiration race with recognition threads
**Files:** `agents/camera_agent.py` lines 140-152, `pipeline/track_state.py` lines 89-107
**Action:** Before removing a track in `get_expired_tracks()`, check whether a recognition thread
currently holds a reference to it (e.g. via a per-track "in-flight" flag/counter set before
recognition starts and cleared after). Defer removal until in-flight recognition completes, or hand
off the best-face data to the finalization step before deleting the track.

---

## PHASE 5 — Alerting

### 5.1 Fix global alert dedup suppressing critical alerts
**File:** `agents/alert_agent.py` lines 41-54
**Action:** Replace the single global `"global:unverified"` dedup key with a tiered scheme: keep
per-severity dedup keys (e.g. `"unverified:routine"` vs `"unverified:critical"`), so a critical
alert (blacklist/after-hours) is never suppressed by a preceding routine unknown-person alert.

### 5.2 Route console alerts through the logger
**File:** `agents/alert_agent.py` line 132
**Change:** Replace `print(...)` with the structlog logger already used elsewhere, so console
alerts land in `logs/surveillance.log` and can be correlated with other log output.

### 5.3 Fix `None` name in alert payloads
**File:** `agents/policy.py` lines 236-285
**Action:** In rules 6-9 (hidden, masked, after-hours, default unknown), set `name` to a descriptive
fallback string (e.g. `"Unidentified Person"`, `"Masked Person"`, `"After-Hours Unknown"`) instead of
leaving it `None`.

---

## PHASE 6 — Cleanup

### 6.1 Remove dead functions
Delete or clearly mark `@deprecated` and schedule removal for:
- `pipeline/face.py`: `detect_and_embed()`, `embed_only()`
- `utils/embedding_utils.py`: `embed_only()`
- `utils/image_utils.py`: `decode_image()`
- `dashboard/backend/routes/live.py`: `broadcast_event()` — **do not delete this one, wire it up
  per Phase 3.8 instead.**

### 6.2 Remove dead variables
- `main.py:237` — `worker_pool` global never assigned; remove or implement.
- `main.py:34,45` — `_broadcast_frame_counter` incremented but never read; remove or use it.

### 6.3 Fix naming/typos
- `config/settings.py:264` — `MASK_CONFIDENCE_PENALITY` → `MASK_CONFIDENCE_PENALTY` (update all
  references).

### 6.4 Clarify misleading event-loop call
**File:** `main.py:274`
**Action:** Remove or comment `asyncio.set_event_loop()` on the main thread if uvicorn's daemon
thread actually owns the running loop — the call currently misleads anyone debugging event-loop
issues.

### 6.5 Restore or remove missing spec reference
**File:** `AGENTS.md`
**Action:** Either add the missing `agent.md` build specification file, or remove the reference to
it.

### 6.6 Remove stale model file
**Action:** Delete `yolov8n.pt` (6MB) from the repo root and from git history if desired, since
`.env` points to `models/yolov8s.pt`.

---

## SUGGESTED EXECUTION ORDER FOR THE AI AGENT

1. Phase 1 (config) — nothing else should be tuned until config is trustworthy.
2. Phase 2 (recognition logic) — depends on Phase 1 thresholds being correct.
3. Phase 4.3 and 4.4 (thread safety) — independent, can run in parallel with Phase 3.
4. Phase 3 (dashboard) — mostly independent frontend/backend work, safe to parallelize internally.
5. Phase 4.1 and 4.2 (data flow) — do after auto-registration behavior is understood/tested.
6. Phase 5 (alerting) — depends on Phase 1/2 producing correct classifications first.
7. Phase 6 (cleanup) — last, purely hygiene, no functional risk.

## DEFINITION OF DONE

- [ ] `.env`, `config.jsonc`, and `settings.py` agree on every shared setting, with a single
      documented source of truth.
- [ ] Face match false-positive rate measurably drops after threshold correction (spot-check a
      sample of "known_visitor" events).
- [ ] Dashboard renders correctly when the backend returns a 500 (no blank sections).
- [ ] WebSocket errors are visible in console/logs, not silently swallowed.
- [ ] `faces` collection growth rate for unknown persons drops after dedup fix.
- [ ] Visit counts survive concurrent visits from the same person (add a race-condition test).
- [ ] A blacklisted/after-hours alert is never suppressed by a preceding routine alert.
- [ ] No `print()` calls remain in the alert path.
- [ ] Dead code and dead dependencies listed in Phase 6 are gone from the repo.
