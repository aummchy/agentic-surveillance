# Issues & Bug Tracker

**10 issues remaining (2 CRITICAL, 3 HIGH, 3 MEDIUM, 2 LOW). 96 FIXED.**

---

## Active Issues

| ID | Severity | File | Problem | Fix |
|----|----------|------|---------|-----|
| 96 | CRITICAL | `main.py:266` | `best_crop_path` NameError — variable never defined, caught by outer try, causes silent finalization failure (no event/alert/memory) | Change to `track.best_face_crop_path` |
| 50 | CRITICAL | `.env:12,19,20` | Live MongoDB URI + Cloudinary keys in `.env` — leaked via zip/tarball | Rotate credentials. Keep `.env.example` with placeholders only |
| 93 | HIGH | `pipeline/track_state.py:167` | `det_score + 0.05` margin blocks clear face from overwriting blurred embedding — detection score correlates with size/lighting, not blur | Gate on `quality.overall_score` instead of `det_score + 0.05` |
| 94 | HIGH | `main.py:155-165` | Finalization recognition call missing `top2`, `margin`, `name`, `track_id` — produces different confidence than progressive pass | Pass all 4 params from `match_result` |
| 89 | HIGH | `agents/camera_agent.py:394-448` | `_finalize_track` race: `finally` block double-submits without checking `_finalized_track_ids` | Add dedup check in `finally` block + atomic flag per track |
| 90 | MEDIUM | `utils/db_utils.py:215-225` | `len()` on 0-d numpy scalar raises `TypeError` in `auto_register_dedup` | Guard with `hasattr(embedding, '__len__')` or validate type after `tolist()` |
| 92 | MEDIUM | `main.py:200-215` | Face crops moved to `captures/face_crops/Unknown/` for auto-registered persons | Skip move when `person_name` is `None`, empty, or `"Unknown"` |
| 95 | MEDIUM | `main.py:110-234` | No cross-track dedup for known persons — fragmented tracks cause duplicate visits/alerts (unknown dedup exists at line 176-194) | Check `person_id` finalization recency before `record_visit()` and `dispatch()` |
| 87 | LOW | `agents/camera_agent.py:249` | Fake quality object via `type('Q', (), {...})()` — fragile to attribute access | Use `QualityResult(is_valid=False, overall_score=0.0)` |
| 88 | LOW | `.env` | `CAMERA_SOURCE` not set — falls back to local webcam silently | Add `CAMERA_SOURCE=` to `.env` |

---

## Remediation Plan

### Phase 0 — Critical Bug (do first)
1. **0.1** Fix NameError: `best_crop_path` → `track.best_face_crop_path` in `main.py:266`

### Phase 1 — Config (do first)
1. **1.1** Set `FRAME_WIDTH=1280`, `FRAME_HEIGHT=720` in `config/config.jsonc` (matches InsightFace det_size)
2. **1.2** Set `MATCH_THRESHOLD=0.45` in `config/config.jsonc` (0.25 too permissive)
3. **1.3** Set `EMBEDDING_DET_SCORE_MIN=0.40` in `config/config.jsonc` (align with defaults)

### Phase 2 — Recognition Logic
1. **2.1** Re-derive "uncertain" band at new threshold (0.45 × 0.8 = 0.36) in `agents/recognition.py:97-167`
2. **2.2** Add regression test: `policy.py` and `recognition.py` read same threshold
3. **2.3** Remove dead `frame_faces = []` fallback in `agents/camera_agent.py:178`
4. **2.4** Skip CLAHE when input already meets contrast threshold in `utils/embedding_utils.py:36-52`
5. **2.5** Change embedding gate from `det_score + 0.05` to `quality.overall_score` in `pipeline/track_state.py:167`
6. **2.6** Pass `top2`, `margin`, `name`, `track_id` in finalization recognition call in `main.py:155-165`

### Phase 3 — Dashboard
1. **3.1** Wrap `fetch()` in try/catch in `App.jsx:29-36` — keep last-known-good stats on failure
2. **3.2** Replace hardcoded `http://localhost:8000/` with relative URLs; extend Vite proxy for `/captures`
3. **3.3** Log WebSocket errors instead of swallowing in `App.jsx:72`
4. **3.4** Add `src/utils/api.js` shared Axios client; migrate raw `fetch()` calls
5. **3.5-3.14** Fix timer leaks (WS ping, alert dismiss), add ErrorBoundary, wire `broadcast_event()`, add event-status filters, give chat memory, cap messages, remove dead `unknowns` state, remove unused `lucide-react`

### Phase 4 — Data Flow
1. **4.1** Dedup auto-registration: check similarity before `store_face()` for unknowns in `main.py:140-165`
2. **4.2** Raise embedding history cap from `$slice: -10` to `-25` in `utils/db_utils.py:392`
3. **4.3** Make visit memory atomic: single `$inc`/`$push`/`$slice` update in `utils/db_utils.py:591-648`
4. **4.4** Check `_finalized_track_ids` before removing tracks in `pipeline/track_state.py:89-107`
5. **4.5** Cross-track dedup: check `person_id` finalization recency before `record_visit()` and `dispatch()` in `main.py`

### Phase 5 — Alerting
1. **5.1** Tiered dedup keys (`unverified:routine` vs `unverified:critical`) in `agents/alert_agent.py:41-54`
2. **5.2** Replace `print()` with structlog in `agents/alert_agent.py:132`
3. **5.3** Set fallback `name` in rules 6-9 of `agents/policy.py:236-285` instead of `None`

### Phase 6 — Cleanup
1. **6.1** Delete dead functions: `pipeline/face.py:detect_and_embed/embed_only`, `utils/embedding_utils.py:embed_only`, `utils/image_utils.py:decode_image`
2. **6.2** Delete dead vars: `main.py:237` (`worker_pool`), `main.py:34,45` (`_broadcast_frame_counter`)
3. **6.3** Fix typo: `MASK_CONFIDENCE_PENALITY` → `MASK_CONFIDENCE_PENALTY` in `config/settings.py:264`
4. **6.4** Remove misleading `asyncio.set_event_loop()` in `main.py:274`
5. **6.5** Remove or fix missing `agent.md` reference in `AGENTS.md`

---

## Definition of Done

- [ ] NameError in event broadcast fixed
- [ ] Blurred faces no longer inherit identity (embedding gate uses quality score)
- [ ] Config sources agree — single documented source of truth
- [ ] False-positive rate drops after threshold correction
- [ ] Dashboard renders on backend 500 (no blank sections)
- [ ] WebSocket errors visible in logs
- [ ] Dead code from Phase 6 removed
