# Issues

> Pipeline logic issues found during 2026-08-04 codebase review. Each has severity, reproduction conditions, and proposed fix.

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

## See also
- [[Common Gotchas]] — operational issues
- [[All Thresholds]] — threshold references
- [[All Config Settings]] — config references
