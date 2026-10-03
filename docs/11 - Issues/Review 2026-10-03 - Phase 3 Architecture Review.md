# Phase 3 — Architecture review (read-only), 2026-10-03

**Rule:** no code was modified to produce this document. Every finding was verified
line-by-line against the working tree (subagent-collected candidates were re-verified
personally before inclusion; unverifiable candidates were dropped — see §G).

**Triage baseline:** `docs/11 - Issues/Review 2026-10-03 - External AI.md` (Phase 0 memo)
and the Phase 2 docs. Already-fixed items are not re-litigated (§D). Everything here is
either new or a confirmed-still-open tracker item.

**Format:** `severity · file · function · problem · evidence (file:line) · impact · recommendation`

---

## A. Findings

### CRITICAL

**C1 · `agents/alert_agent.py` · `dispatch()` — the alert pipeline is silently dead**
- **Problem:** Both callers mark the track alerted *before* calling `dispatch()`, and
  `dispatch()` rejects any track with `alerted=True`. Every call returns False at the
  guard; nothing after it ever executes.
- **Evidence:** guard `alert_agent.py:71-72`; mark-then-call `track_processor.py:450-451`
  and `camera_agent.py:674-679`; `mark_alerted_once()` sets `alerted=True`
  (`pipeline/models.py:92-101`). No third call site exists.
- **Impact:** console/email/SMS/webhook never fire; `_broadcast_alert` never runs
  (`track_processor.py:221-222`); `last_alert_time` never stamped (`:452-453`) — while
  `_log_event` still records `alerted=True` (`:224-225`) and the event row stores it.
  Dashboards and logs show alerts that were never dispatched. All 84 tests stay green:
  nothing calls `dispatch()` (see H9).
- **Recommendation:** dispatch first, mark only on success (or remove the in-dispatch
  guard and keep the caller-side mark as the sole one-shot); settle cooldown semantics
  (M13) in the same change. Note: `docs/11 - Issues/Issues.md` ISSUE-1's proposed-fix
  snippet (lines 44-45) contains this same mark-before-dispatch bug — fix the snippet.

**C2 · `agents/track_processor.py` · `process()` — ISSUE-1 confirmed: hidden/masked tracks never alert**
- **Problem:** When `snap.embedding is None` (hidden person, masked, no usable face),
  `process()` logs an event and returns before `decide()` — Policy Rules 6/7 never run.
- **Evidence:** `track_processor.py:187-193`; `docs/11 - Issues/Issues.md` ISSUE-1
  (status Open still correct; cited line numbers have drifted from 72-78 to 187-193).
- **Impact:** an intruder who keeps their face hidden or covered triggers no alert at all
  — the entire `intentionally_hidden` status is dead in practice.
- **Recommendation:** run policy on visibility-derived signals when embedding is None
  (ISSUE-1 proposal — but repair its alert-guard bug per C1 before implementing).

### HIGH

**H1 · `agents/report.py` · `_incident_report` → `_build_summary` — naive/aware TypeError**
- **Problem:** `now_utc = datetime.now(timezone.utc)` (aware) minus `last_seen` (naive,
  from MongoDB) raises `TypeError: can't subtract offset-naive and offset-aware datetimes`.
  The adjacent comment even says "Use naive UTC" but the code doesn't.
- **Evidence:** `report.py:279-282`; `last_seen` written naive (`utils/db_memory.py:77`
  `datetime.utcnow()`; `MongoClient` created without `tz_aware`, `utils/db_client.py:33`);
  only call site of `_build_summary` is `report.py:100`.
- **Impact:** `GET /api/reports/incidents` (`dashboard/backend/routes/reports.py:46-84`)
  returns HTTP 500 as soon as any listed event has a person with a visit record. The
  route catches only `RuntimeError`/`HTTPException`, so the TypeError escapes raw.
- **Recommendation:** normalize both sides (parse `last_seen` to aware UTC, or use naive
  `utcnow()`); add a test that feeds a memory doc into `_build_summary`.

**H2 · `dashboard/backend/routes/reports.py` + `agents/report.py` · `_fetch` / `_incident_report` — batched `visit_history` ignored (N+1 preserved)**
- **Problem:** The route batch-fetches all visit histories in one query, passes them as
  `visit_history`; `_incident_report` never reads that key and re-queries
  `get_visit_history(person_id)` per event instead.
- **Evidence:** batch built `reports.py:55-61`, passed `:78`; ignored + re-queried
  `report.py:87-96`.
- **Impact:** dead batch work + one Mongo query per event on every incidents request;
  this re-query is also what feeds the naive datetime into H1.
- **Recommendation:** consume `data["visit_history"]`; delete the per-event query.
  Fix together with H1.

**H3 · `main.py` + `dashboard/backend/` — unauthenticated API bound to `0.0.0.0` with destructive routes and captured imagery**
- **Problem:** The API listens on all interfaces with no authentication anywhere, while
  exposing identity management (PUT/DELETE/verify), PII, and a static mount of face crops.
- **Evidence:** `main.py:128` `host="0.0.0.0"`; zero `Depends`/auth/API-key code in
  `dashboard/backend/` (only CORS, `dashboard/backend/main.py:17-23`);
  destructive routes `routes/faces.py:113-152`; `app.mount("/captures", …)`
  (`main.py:32`); faces/events/memory PII via REST. CORS origins are limited to
  localhost (`config/config.jsonc:207`) but CORS binds browsers only — any LAN client
  using curl/python is unrestricted.
- **Impact:** any host on the network can read biometric PII, delete or verify
  identities, and download captured face images.
- **Recommendation:** decision needed — default bind `127.0.0.1`, or add token
  middleware. CORS ≠ authentication.

**H4 · `dashboard/backend/routes/chat.py` + `utils/db_faces.py` — full embedding arrays disclosed to clients and to the LLM**
- **Problem:** Read paths project out only `latest_embedding`; the `embeddings` array
  (≤25×512 floats) and `mean_embedding` (512 floats) survive into API responses and,
  for chat, into the Ollama prompt.
- **Evidence:** projection `{"latest_embedding": 0}` in `db_faces.py:178` and
  `faces.py:51`; chat puts faces in `data` (`chat.py:82-83`), serializes the whole dict
  into the prompt (`chat.py:101`) and returns it to the caller (`chat.py:141`).
- **Impact:** biometric-template disclosure (compounds H3: unauthenticated), prompt/token
  bloat (~13k floats per face), LLM context overflow on `unknown` queries.
- **Recommendation:** project embeddings out of these read paths; strip numeric arrays
  before prompt construction.

**H5 · `agents/alert_agent.py` · `dispatch()` — latent `NameError: AlertLevel` (masked by C1)**
- **Problem:** Line 74 references `AlertLevel`, which is never imported or defined in the
  module (only occurrence in the file; imports at `:1-13` lack it).
- **Evidence:** `alert_agent.py:74`; compare `camera_agent.py:33` which correctly imports
  it `from config.status import … AlertLevel`.
- **Impact:** any naive fix of C1, or any call with `alerted=False`, crashes dispatch with
  NameError. C1's dead guard is currently the only thing keeping this latent.
- **Recommendation:** `from config.status import Status, UNVERIFIED_STATUSES, AlertLevel`.

**H6 · `tests/test_track_finalizer.py` · `TestAlertDispatchConditions` — the failing unit has zero test coverage**
- **Problem:** No test in the suite calls `dispatch()` or `should_send_alert()`. The class
  named "alert dispatch conditions" only constructs `DecisionResult` and asserts its
  fields. Phase 1 separately found `CameraAgent` and `TrackProcessor` have no importing
  tests at all.
- **Evidence:** `tests/test_track_finalizer.py:165-192`; repo-wide grep: zero
  `dispatch(`/`should_send_alert(` calls in `tests/`; `plan.md` Phase 1 findings.
- **Impact:** C1 shipped with 84 green tests; the whole finalization→alert chain is
  unverified.
- **Recommendation:** add unit tests that pin the intended ordering (dispatch-then-mark
  or mark-then-dispatch — whichever C1's fix adopts), plus a `should_send_alert`
  cooldown test.

**H7 · `utils/db_memory.py` · `update_visit_memory` — `avg_similarity` never written; memory-consistency features inert**
- **Problem:** The atomic update never sets `avg_similarity` (stays at its init `0.0`);
  all consumers gate on `> 0`, so they never fire.
- **Evidence:** `$set` block `db_memory.py:100-106` (no `avg_similarity`); init `:42`,
  `:63`; read-back `:123`; consumers `agents/memory.py:177-189`
  (`MEMORY_CONSISTENCY_HIGH_SIM`/`LOW_SIM`, `MEMORY_SIMILARITY_DROP_RATIO`),
  `agents/report.py:218`, `:238`.
- **Impact:** similarity-drift detection silently dead; person reports always show
  `avg_similarity: 0`. `similarity_history` *is* written (`:94-98`) but never aggregated.
- **Recommendation:** derive the average from `similarity_history` inside the update
  (pipeline `$set`), or retire the feature explicitly — behavior change → Phase 4.

### MEDIUM

**M1 · `main.py` · `main()` — two queue consumers; docs claim one**
- `main.py:117` `for i in range(2)` starts **2** `worker_process_tracks` threads that
  call `track_processor.process()` concurrently (`main.py:79-91`). Contradicts
  `track_processor.py:14-16` and `:154`, `docs/CURRENT_ARCHITECTURE.md:43,81,84,170-175`,
  and the `plan.md` Phase 2 note ("1 queue-consumer thread not 2/4").
- **Impact:** false threading invariant. No live race found (`process()` holds no
  per-instance mutable state; `handle_frame` runs on the camera thread, `main.py:74-76`),
  but future code may trust the comment.
- **Recommendation:** correct the two docstrings, `CURRENT_ARCHITECTURE.md`, and
  `plan.md` (docs batch).

**M2 · `dashboard/backend/routes/chat.py` · `chat`/`chat_health` — sync LLM check on the event loop**
- `llm_client.is_available()` is synchronous httpx with `timeout=3.0`
  (`llm_client.py:247-263`), called directly in async handlers `chat.py:132` and `:150`
  (cached 10 s). Impact: a slow check stalls every dashboard request.
- **Recommendation:** `await asyncio.to_thread(...)` — the pattern already used at
  `chat.py:140`.

**M3 · `utils/llm_client.py` + `agents/alert_agent.py`/`agents/report.py` — `LLM_ENABLED` only gates `is_available()`; generators are ungated sync calls**
- `_generate` has no `LLM_ENABLED`/availability gate (only `is_available():250` checks
  it). `generate_nl_summary` runs synchronously on a finalization worker per alert
  (`alert_agent.py:108-114`); `generate_incident_summary`/`generate_executive_summary`
  per report (`report.py:104`, `:184`). Note: `config/config.jsonc:228` currently sets
  `"LLM_ENABLED": false`, so chat is permanently "unavailable" while the generators
  would still attempt HTTP if reached — inconsistent gating, and it contradicts
  AGENTS.md's expectation that chat/summaries work.
- **Impact:** with Ollama down, each alert/report pays full timeout+retry on one of the
  2 queue workers → finalization stalls.
- **Recommendation:** gate `_generate` on `LLM_ENABLED`; move NL summary off the worker
  (executor already exists); confirm `config.jsonc:228` is intended.

**M4 · `agents/alert_agent.py` · `should_send_alert`/`dispatch` — cooldown consumed at allow-time; `dispatch` returns True on channel failure**
- `alert_agent.py:58` stamps the cooldown before any send; `:131` returns True
  unconditionally after that; contrast the invariant claimed at
  `track_processor.py:446-448`. **Latent** (C1 makes dispatch unreachable today).
- **Recommendation:** settle semantics as part of C1's fix.

**M5 · `agents/memory.py` · `run` — local clock vs UTC storage**
- `memory.py:90` `datetime.now()` (local) compared against UTC-written
  `last_seen`/`typical_hours` (`db_memory.py:29,77`): `days_since_last` (`memory.py:103`)
  and `now.hour` vs stored UTC hours (`:106`). Impact: recency/time-pattern features
  skewed by machine TZ (±8 h here) — wrong `is_typical_time`, off-by-one visit days.
- **Recommendation:** single UTC source; pair with ISSUE-11 migration.

**M6 · `utils/db_memory.py` · `_compute_best_status` — read-then-update race, docstring overstates atomicity**
- `db_memory.py:171-176` reads `best_status`, then `:90-110` updates separately, while
  `:75` claims "a single atomic MongoDB operation". Two queue workers recording visits
  concurrently can regress `best_status` (lower trust overwrites higher).
- **Recommendation:** fold the status decision into the atomic update (aggregation
  pipeline `$set`) — Phase 4.

**M7 · `utils/scoring.py` · confidence logging — `logger` undefined (NameError)**
- `scoring.py:207` calls `logger.warning(...)`; the module defines no logger (imports
  `:7-13`). Impact: rare-path crash when raw cosine is outside `[-1, 1]`.
- **Recommendation:** `logger = structlog.get_logger(__name__)`.

**M8 · `utils/image_utils.py` · `save_image` — returns True when `cv2.imwrite` fails**
- `image_utils.py:110-111`: `imwrite` returns False instead of raising; the return value
  is ignored → success reported regardless. Impact: silent loss of capture/debug files.
- **Recommendation:** `return bool(cv2.imwrite(path, image))`.

**M9 · `dashboard/backend/routes/faces.py` · `list` — 400 re-raised as 500**
- `faces.py:61` raises HTTPException(400) inside the `try`; `except Exception` (`:64`)
  converts it to 500. The `except HTTPException: raise` pattern exists at `:76` but not
  in this route.
- **Recommendation:** add `except HTTPException: raise`.

**M10 · `utils/db_faces.py` · `update_face` — embedding/mean/quality branch never runs in production**
- Production callers pass only name/tags/alert_level (`faces.py:120-126`); no pipeline
  caller exists, so `db_faces.py:92-131` (embedding history, `mean_embedding` refresh,
  quality-gated `latest_embedding` update) is dead outside tests.
- **Impact:** identity embeddings frozen at registration — this *is*
  `docs/REFACTOR_PLAN.md` item 3 (TODO 4/5); tests assert behavior no production path
  exercises. **Verdict: cross-listed under the already-approved Phase 4 item 3 — no new
  item created.**

**M11 · `docs/11 - Issues/Issues.md` — two status/fix errors found during re-verification**
- **ISSUE-4:** marked Resolved with fix "remove `active_ids`" — but `active_ids` *is*
  read now (`camera_agent.py:300`, IoU dedup). Applying the documented fix would break
  dedup. → rewrite entry.
- **ISSUE-7:** marked Resolved, but the mismatch remains: `settings.py:100,265` default
  `0.40` vs `config.jsonc:15` `0.45` (docs once said 0.5). → reopen as accepted hazard
  or fold into Phase 4 item 5.
- **ISSUE-1** proposed-fix snippet contains C1's bug (see C1).

**M12 · `dashboard/backend` — unbounded events + captures growth**
- No TTL on `events` (per-request `count_documents`: `db_events.py:77`,
  `:119-122` — four counts per stats poll); `captures/` grows unbounded behind the
  static mount (`main.py:8-9,32`). Impact: latency and disk growth over weeks.
- **Recommendation:** TTL index + capture rotation — Phase 4.

**M13 · `dashboard/backend/routes/chat.py` · `ChatRequest.history` — unbounded list**
- No max length on `history` (`chat.py:29`); only the last 20 are used (`:68`) but the
  whole body is accepted and held. Fold into H3's auth/request-size posture.
- **Recommendation:** `max_length` on the field.

### LOW

| # | file · function · problem · evidence · recommendation |
|---|---|
| L1 | `pipeline/recognition_pipeline.py` · `_select_best_face` — **ISSUE-2 confirmed Open**: crop always wins (`:338-345`), so a weak crop face discards a good full-frame hit; latent while `ENABLE_FULL_FRAME_FALLBACK=false`. Issues.md line refs (189-197) stale → update. |
| L2 | `utils/embedding_utils.py` + `pipeline` quality gate — **ISSUE-3 confirmed Open**: quality scored on raw crop, embedding computed on CLAHE-enhanced image (`embedding_utils.py:100-102`); conservative direction, low. |
| L3 | `agents/camera_agent.py` · `_handle_skip_result` — **ISSUE-6 confirmed Open**: on `LOW_QUALITY`, `set_best_face` still requires `quality.is_valid` (`:571-577`), false by definition → crop never stored. Partial improvement exists: `NO_FACE`/`EMBEDDING_FAILED` with valid quality now store (`:571-576`). |
| L4 | `agents/camera_agent.py` — **ISSUE-10 confirmed Open**: dead assignment `submit_time =` (`:367`; real read is `pop_submit` at `:530`). |
| L5 | `utils/db_faces.py`, `utils/db_memory.py` — **ISSUE-11 confirmed Open**: `datetime.utcnow()` deprecated sites (`db_memory.py:29,77` + ~10 in `db_faces.py`). Naturally paired with M5's clock unification. |
| L6 | `agents/memory.py` — **ISSUE-12 confirmed Open**: `last_seen` handles only `datetime` instances (`:101-103`); ISO strings silently skip the boost. |
| L7 | `utils/image_utils.py` · `resolve_track_image_url` — **ISSUE-13 confirmed Open**: unlocked read/write of `track.image_url` (`:123-147`) vs camera-thread reads. |
| L8 | `config/settings.py` · `_get` — **ISSUE-14 confirmed Open**: element-cast only when `default` non-empty (`:71-76`); empty-default list settings yield strings. |
| L9 | `agents/policy.py` — **ISSUE-15 confirmed Open**: `datetime.now().timestamp() - track.first_seen` (`:180`). |
| L10 | `pipeline/tracker.py` · `track_persons` — **ISSUE-16 confirmed Open**: `os.environ["OPENVINO_DEVICE"] = …` per frame (`:51`). |
| L11 | `utils/db_client.py` — **ISSUE-17 confirmed Open**: unguarded outer read in double-checked locking (`:49-51`); same pattern `image_utils.py:14-16`. |
| L12 | `main.py` — **ISSUE-19 confirmed Open**: only `server.should_exit = True` present (`:159`); proposed fix (`set_event_loop`, `force_exit`, exception handler, closing WS clients) not applied. |
| L13 | `agents/recognition.py` · `run` — `face_quality=None` (fed at `track_processor.py:325`) reaches `round(None, 3)` (`pipeline/models.py:267`). Unreachable today only because embedding-present ⇒ quality-gate-valid ⇒ score > 0 (coupled to L3) — fragile latent type hazard. |
| L14 | `agents/policy.py` · class docstring — rule list (`:34-41`) mismatches the actual chain (`:138-155`): omits auto-registered/matched/after-hours; misnumbers 5-7 (real: 5 = matched, 6 = hidden, 7 = masked). |
| L15 | `utils/db_events.py` · `log_event` — `get_events_collection()` at `:39` sits outside the try, so init failure crashes the caller despite the docstring's "must never crash". |
| L16 | `utils/db_client.py` · `close_client` — resets `_client` only (`:37-43`); `_faces/_events/_memory_collection` globals keep references to the closed client (stale after in-process restart). |
| L17 | `dashboard/backend/routes/live.py` · origin check — accepts connections **without** an Origin header (`:84` `if origin and …`); non-browser clients bypass the whitelist. Fold into H3. |
| L18 | `config/settings.py` reading hazard — `TRACK_TIMEOUT_SECS` default `3.0` (`:141,:224`) vs `config.jsonc:47` `15.0`; another instance of memo Claim 10 → Phase 4 item 5. |
| L19 | `config/settings.py` · `_get` — malformed `.env`/jsonc value raises bare `ValueError` at import (`:79-80`) instead of a friendly validation error. |
| L20 | Docs status corrections: `docs/10 - Problems/Track/Duplicate Finalization` → **Resolved** (three guards verified: `camera_agent.py:438-439`, `:734-756`, `:768`; the cited pruning root cause is absent); Problems Home "root cause" paragraph is stale (claims it open); `CURRENT_ARCHITECTURE.md` worker count (M1). |

---

## B. `Issues.md` re-verification (ISSUE-1 … 19)

| ISSUE | Tracker says | Verified 2026-10-03 | Verdict |
|---|---|---|---|
| 1 Hidden/masked never alert | Open (Critical) | `track_processor.py:187-193` still returns before `decide()` | **Open → C2** |
| 2 Full-frame fallback discarded | Open (Low) | crop-first selection at `:338-345`; latent (fallback off) | **Open → L1** (refs stale) |
| 3 Quality vs embedding image mismatch | Open (Low) | CLAHE inside detect path `embedding_utils.py:100-102` | **Open → L2** |
| 4 `active_ids` dead code | Resolved | `active_ids` **is read** at `camera_agent.py:300` (IoU dedup) | **Rewrite → M11** (fix text dangerous) |
| 5 `EMBEDDING_CACHE_COSINE_THRESHOLD` docs | Resolved | `docs/09` deleted in Phase 2 | **Closed (by deletion)** |
| 6 `set_best_face` low_quality path | Open (Low) | still gated on `is_valid` (`camera_agent.py:571-577`) | **Open → L3** |
| 7 `DEDUP` default mismatch | Resolved | `settings.py:100,265` 0.40 vs `config.jsonc:15` 0.45 | **Reopen → M11** |
| 8 repomix stale files | Resolved | deleted in Phase 0 | **Closed** |
| 9 Policy/Recognition mid-range divergence | Open (Low) | RULE 5 → "unknown" (`policy.py:263`) — intentional per AGENTS.md | **Accepted by design**; doc's "uncertain" proposal optional → Phase 4 |
| 10 `submit_time` dead write | Open | `camera_agent.py:367` | **Open → L4** |
| 11 `utcnow()` deprecated | Open | `db_memory.py:29,77` + db_faces sites | **Open → L5** |
| 12 String `last_seen` skips boost | Open | `memory.py:101-103` | **Open → L6** |
| 13 `image_url` unlocked race | Open | `image_utils.py:123-147` | **Open → L7** |
| 14 `_get` empty-default lists | Open | `settings.py:71-76` | **Open → L8** |
| 15 Mixed datetime APIs (policy) | Open | `policy.py:180` | **Open → L9** |
| 16 `os.environ` per frame | Open | `tracker.py:51` | **Open → L10** |
| 17 Collection init unlocked outer check | Open | `db_client.py:49-51` | **Open → L11** |
| 18 WS broadcast flood | Partial (Tier 2 open) | Tier 1 config verified (960×540 q85); `wait_for` still at `live.py:21` | **Open (Tier 2) → Phase 4** |
| 19 Proactor shutdown traceback | Open | only `should_exit` at `main.py:159` | **Open → L12** |

## C. Problems notes re-verification (7 files)

| Note | Tracker says | Verified | Verdict |
|---|---|---|---|
| Duplicate Finalization (Visit Inflation) | Open (Critical) | Proposed fix implemented: `mark_finalized_once` guard `camera_agent.py:755`; prune removed, `discard()` at `:768`; stale-worker path `:734-735` intact; no test/call path bypasses `_finalize_track` | **RESOLVED → L20 docs update.** (Cross-track inflation from reincarnation stays under "Same Person Multiple Tracks".) |
| Same Person Becomes Multiple Tracks | Open (High) | timeout mismatch live: `TRACK_TIMEOUT_SECS=15.0` wall-clock (`config.jsonc:47`) vs `track_buffer: 60` frames (`bytetrack_surveillance.yaml:14`); `MAX_TRACK_SECS=300` | **Open → Phase 4 config alignment** (A+D option, behavior change) |
| Fragmentation Detection is a No-op | Open (Medium) | IoU computed then only logged: `track_state.py:159-179` | **Open → Phase 4 optional (Option B)** |
| Database Latency | Open (Low) | inherent Atlas round trip; no cache added | **Open → park (re-measure first)** |
| Identities Never Generated | Resolved | fixes still in place: `INSIGHTFACE_DET_SIZE=640` (`config.jsonc:77`), fallback off, CLAHE tiny-crop skip (`embedding_utils.py:100`) | **Resolved confirmed** |
| Recognition Bottleneck (26s) | Resolved | same as above | **Resolved confirmed** |
| Problems Home root-cause paragraph | — | describes Duplicate Finalization as live | **Stale → L20** |

## D. Triage against the Phase 0 memo (not re-litigated)

| Memo item | Disposition |
|---|---|
| Claim 5 progressive race / Claim 6 downgrade clobber / Claim 7 track-vs-person id | Already fixed — excluded per memo |
| Claim 3 dual recognition implementations | Already routed → Phase 4 item 1 (`run_final`) |
| Claim 4 `Track` overload | Already routed → Phase 4 item 4 |
| Claim 8 threshold pair 70 vs 80 | **Phase 3 verdict: still present, still only a secondary gate — keep as Phase 4 item 2** |
| Claim 9 TODO Items 4–5 (MERGED/embedding refresh) | **Phase 3 verdict: still present — see M10; keep as Phase 4 item 3** |
| Claim 10 `settings.py` duplicate defaults | **Phase 3 verdict: real, deliberate — keep as Phase 4 item 5** (new evidence: L18) |
| Per-call `RecognitionAgent()` (memo open item) | Confirmed at `track_processor.py:319` → folds into Phase 4 item 1 |
| Phase 1 findings (RECOG/POLICY status conflation via `set_recognition_snapshot`, `skip_reason` enum typing, `RecognitionMetrics` double-construction, `"person_id"` payload key, testing gaps) | Carried forward: conflation → Phase 4 item 1 review; enum/metrics/payload-key → Phase 4 item 6 cleanup; testing gaps → H6 |

## E. Approved priority list (Phase 4 input — approved by user 2026-10-03)

**Wave A — correctness & safety (new findings):**
1. **C1 + H5 + M4 + H6** — repair the alert dispatch chain end-to-end, with tests.
2. **H1 + H2** — incident-report crash + batch consumption (one change).
3. **H3 + H4 + L17 + M13** — security posture: bind/auth decision + embedding projection.
4. **C2** — hidden/masked persons reach policy (ISSUE-1).
5. **H7** — `avg_similarity`: compute or retire.

**Wave B — hygiene:** M1 (worker-count docs), M2–M3 (LLM blocking/gating + confirm
`LLM_ENABLED=false` intent), M5 (clock unification, with ISSUE-11), M6–M9,
M11–M12, M7, M8, L20 (docs statuses).

**Already-approved Phase 4 items (unchanged):** `run_final` consolidation · threshold
pair 70/80 · embedding lifecycle (M10) · `Track` split · settings de-dup (L18) ·
cleanup pass (L1–L16, ISSUE-18 Tier 2, timeout alignment, fragmentation option B).

## F. Not changed by this review

No `.py` file, config value, threshold, or database schema was modified. 84 tests were
untouched. This document and `plan.md`'s status table are the only edits.

## G. Candidate claims dropped during verification

- "scoring.py has unused imports (`os`/`datetime`/`threading`)" — all three are used
  (`scoring.py:15-23,56,139-145`). Rejected.
- "`memory._empty_result` erases `person_id`" — true that it returns `person_id: None`
  (`memory.py:226`), but no consumer reads `memory_context["person_id"]`; no impact.
  Rejected as a finding (noted only).
- "`Status` IntEnum renders as raw int in payloads" — by design (AGENTS.md status
  system). Rejected.
- "db lazy-init can cache partial state" — `MongoClient()` construction is lazy and
  non-failing; no partial state found. Rejected.
- "Llama/LLM called per-incident in a loop" — one executive-summary call per report and
  one NL call per alert (M3), not per-event loops. Claim corrected, then kept as M3.

---

*Compiled 2026-10-03. All evidence verified against the working tree at that date.*
