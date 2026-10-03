# Historical Debug Notes (formerly `see.md` at repo root)

> **⚠️ NOT AUTHORITATIVE**
>
> This document contains observations from previous debugging sessions.
> It is **NOT** authoritative for current behavior.
>
> Every issue described here must be verified against the current source code
> before any implementation. When a documented bug turns out to be fixed, mark it
> as historical — do **not** reimplement the fix.
>
> Verification status as of 2026-10-03: see
> `docs/11 - Issues/Review 2026-10-03 - External AI.md`, which checks each claim
> below against current code.

---

Analysis of the trk=2 log
Why a single track gets recognized 4× (the reprocessing you're asking about)
Progressive recognition is by design (every RECOGNITION_INTERVAL_FRAMES=20 + a rescan every RESCAN_INTERVAL_SECS=3s, up to MAX_RESCAN_ATTEMPTS=3). But the log shows passes pile up faster than they can resolve, because of a scheduling/completion race:
1. Scheduling happens on the camera loop; resolution lands async. camera_agent.py:220 guards with _recognizing_tracks, but that flag is only set when a worker starts — not when it's queued. With a 2-worker pool and 3–9s queue delays, multiple passes for the same track get submitted before the first one finishes and sets track.decision.
2. already_resolved is checked at schedule time, not run time. After pass 2 set known_visitor (14:23:21), passes 3 & 4 were already in the queue — and _progressive_recognition has no resolved-check at worker start (camera_agent.py:302-320), so they run anyway.
3. Rescan logic re-fires as long as last_recognition_status in ("unknown","uncertain") and a rescan slot remains — pass 1 (UNCERTAIN) → rescan; pass 3 (UNCERTAIN) → rescan again.
Other issues from comparing the trk=2 sequence
A. The final decision can DOWNGRADE from the best progressive result — the serious one.
- trk=3: reached KNOWN conf=92 (14:23:26), but FINAL shows conf=52 UNKNOWN sim=0.474 (the first pass).
- trk=10: reached KNOWN_VISITOR conf=95 (14:24:17), then a no-match pass (14:24:29, sim=0.000) clobbered pending_match_result/pending_recognition, and the final = MASKED_UNKNOWN conf=23 → dispatched a HIGH alert for a known visitor (14:25:03).
Root cause: set_pending_match_result/set_pending_recognition_data (track_state.py:283-305) unconditionally overwrite, and track_processor.process (track_processor.py:86-119) recomputes the decision from the last pending state — not track.confidence/track.decision, which do hold the max (models.py:59-64). The "confidence never downgrades" invariant only protects the max fields, not the pending data used at finalization. A worse/no-match pass overwrites a better one.
B. RECOG status vs POLICY status diverge. trk=2 pass at 14:23:44: RECOG=KNOWN conf=72 but POLICY=UNKNOWN. Policy thresholds (KNOWN_VISITOR_SIMILARITY=0.85, KNOWN_VISITOR_CONFIDENCE=80) are far stricter than the recognition agent's status classifier, so a matched person labeled KNOWN shows up UNKNOWN in POLICY/FINAL.
C. False HIGH alerts during the no-match window. trk=9/10/11/5/15/14/17 all logged alert=high ALERT for sim=0.000 passes (conf 17–23) — same people were later matched KNOWN at conf 75–95. Mostly logged-not-dispatched (progressive only dispatches critical), but trk=10's final did dispatch.
D. Two identities for the same scene. trk=2/10 → cam_01_1785391472_2; trk=3/6/9/11 → cam_01_1785358203_6. trk=10's sim=0.993 is suspiciously high (self-match), suggesting the same person was auto-registered twice.
Proposed fixes (ranked)
#	Fix	File(s)
1	Best-result finalization: gate pending writes — only overwrite pending_match_result if new sim ≥ existing; only overwrite pending_recognition if new conf > existing; never let a no-match (sim=0) clobber a good match	track_state.py, camera_agent.py
2	Resolve check at worker start: bail out of _progressive_recognition if track.decision in RESOLVED_STATUSES (closes the scheduling race; stops wasted inference)	camera_agent.py:302
3	Align thresholds: lower KNOWN_VISITOR_SIMILARITY/CONFIDENCE or make policy defer to the recognition agent's known status so RECOG and POLICY agree	config/config.jsonc, policy.py
4	Optional: identity dedup — tighten find_similar_unknowns merge so one person isn't registered as two identities	track_processor.py, db_utils.py
My recommendation: #1 + #2 are the core fixes (they stop downgraded finals and over-processing), #3 fixes the confusing RECOG/POLICY split. Want me to implement all of #1–#3, or start with a subset?