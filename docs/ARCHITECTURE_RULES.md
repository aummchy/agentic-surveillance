# Architecture Rules

> **Semantic boundaries only** — what each part of the system is allowed to
> mean and do. Violating these is a design error even when all tests pass.
> These are constraints on *responsibility*, not on file layout; file-layout
> changes additionally require a proposal per `AGENTS.md` §4.
>
> Companion documents: `docs/CURRENT_ARCHITECTURE.md` (as-built),
> `docs/REFACTOR_PLAN.md` (approved-to-change candidates, all currently
> parked), `plan.md` (phased roadmap).

---

## 1. Identity — `track_id` is never a `person_id`

- A **track** is one continuous sighting of one blob in one camera session.
  Its id is the composite `{camera_id}_{session_epoch}_{bt_id}_{generation}`
  and it is born and dies with the track.
- A **person** is a long-lived identity row in the `faces` collection with a
  `person_id` and one or more embeddings.
- One person may produce many tracks (re-entry, occlusion, ID switches); one
  track never represents many people.
- Code must not use one where the other belongs: don't store a track id in a
  person field, don't treat a person_id as unique per sighting, don't log one
  under the other's name.
  - *Known wart:* `TrackProcessor._broadcast_alert` sends `track.track_id`
    under the payload key `"person_id"` — an established frontend contract,
    documented in `CURRENT_ARCHITECTURE.md` and parked in `REFACTOR_PLAN.md`.
    Do not copy this pattern into new payloads.
- Auto-registration is the only place the two are deliberately equal: a brand
  new person is created with `person_id = track_id`.

## 2. Detection — detection answers "where", never "who"

- `pipeline/tracker.py` (YOLO + ByteTrack) produces boxes and per-frame
  track ids only. It must never consult embeddings, memory, status, or
  policy.
- Person confidence thresholds (`track_high_thresh`, `PERSON_CONF_THRESHOLD`)
  belong to detection and must not double as identity thresholds.

## 3. Tracking — `TrackState` owns track truth

- All mutation of track state goes through `pipeline/track_state.py` and its
  lock discipline (`TrackState._lock` may nest inside/around `track._lock`,
  never in the reverse order).
- Track lifetime rules (expiry, deferral while recognition is in flight,
  generation handling, composite-id formatting) are decided here and nowhere
  else.
- No other module may reach into `Track` fields without the lock while a
  worker may be writing.

## 4. Recognition — one chain, two moments

- Recognition = detect face → validity gates → quality score → embedding →
  vector search → memory → confidence. The implementation lives in
  `pipeline/recognition_pipeline.py` (mid-track moment) and, for historical
  reasons, a second copy in `agents/track_processor.py` (finalization moment).
- Two call sites exist because finalization has no live frame. That is the
  *only* sanctioned difference; any third implementation is a violation.
- Consolidating the two into one entry point is `REFACTOR_PLAN.md` item 1 —
  do not start it informally.
- Embeddings are only stored or searched after passing the validity gates
  (blur, brightness, face area). A face that fails them produces no
  identity evidence at all.

## 5. Verification — weak evidence never overwrites strong evidence

- Confidence only upgrades across passes; a later weaker result may not
  replace an earlier stronger one (`TrackState` setters enforce this).
- Best-face selection keeps a hysteresis margin — a barely-better crop does
  not evict the incumbent.
- DB-level embedding overwrite (`latest_embedding`) is quality-gated the
  same way.
- Critical (blacklist) alerts are the one deliberate exception: they always
  update.

## 6. Policy — decides, never re-recognizes

- `agents/policy.py::decide()` is a pure function of (track state, match
  result, recognition result, memory context). It may re-rank and re-threshold
  those inputs; it may not run face detection, embeddings, vector search, or
  DB lookups.
- If policy needs a new signal, the signal must be produced by recognition or
  memory and passed in — not computed inside policy.
- Status comparison uses `config/status.py` (`Status` enum, higher = more
  trusted, `is_known = status >= 3`), never raw strings or numbers.

## 7. Storage — the DB layer speaks for persistence

- All MongoDB access goes through `utils/db_*` (imported via the
  `utils/db_utils.py` facade). No agent or pipeline module issues raw queries.
- Atlas vector scores are converted (`raw_cosine = (atlas_score * 2) - 1`)
  at the boundary before any threshold comparison.
- Deduplication (`deduplicate_identity`) decides whether a new embedding may
  become a new person; registration logic must honor its verdict
  (MERGED / FAILED / NEW) rather than inserting around it.

## 8. Threading — the camera loop never blocks

- The camera thread does capture, detection, tracking, and bookkeeping only.
  All I/O (MongoDB, Cloudinary, alerts, LLM, WebSocket) happens on workers
  behind queues/executors or on the API thread.
- Anything crossing threads does so via `TrackState` locks, `queue.Queue`, or
  `asyncio.run_coroutine_threadsafe` onto the (still-running) loop.
- One decision per track: recognition/decision runs progressively and once at
  finalization — never per frame.

## 9. Configuration — one place per value

- Tunables live in `config/config.jsonc`; secrets live in `.env`;
  `config/settings.py` only reads them (env > jsonc > default).
- Do not add a second hardcoded copy of a number that settings already
  resolves (the de-duplication hazard parked as `REFACTOR_PLAN.md` item 5).
- `MATCH_THRESHOLD` stays ≤ 0.45.
