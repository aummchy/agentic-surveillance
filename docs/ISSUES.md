# Issues & Bug Tracker

**7 issues remaining (1 CRITICAL, 1 HIGH, 3 MEDIUM, 2 LOW). 96 FIXED.**

---

## Active Issues

| ID | Severity | File | Problem | Fix |
|----|----------|------|---------|-----|
| 50 | CRITICAL | `.env:12,19,20` | Live MongoDB URI + Cloudinary keys in `.env` — leaked via zip/tarball | Rotate credentials. Keep `.env.example` with placeholders only |
| 89 | HIGH | `agents/camera_agent.py:394-448` | `_finalize_track` race: `finally` block double-submits without checking `_finalized_track_ids` | Add dedup check in `finally` block + atomic flag per track |
| 90 | MEDIUM | `utils/db_utils.py:215-225` | `len()` on 0-d numpy scalar raises `TypeError` in `auto_register_dedup` | Guard with `hasattr(embedding, '__len__')` or validate type after `tolist()` |
| 91 | MEDIUM | `agents/camera_agent.py:20` | `EMBEDDING_CACHE_COSINE_THRESHOLD=0.005` too tight — cache never fires | Relax to `0.02` |
| 92 | MEDIUM | `main.py:200-215` | Face crops moved to `captures/face_crops/Unknown/` for auto-registered persons | Skip move when `person_name` is `None`, empty, or `"Unknown"` |
| 87 | LOW | `agents/camera_agent.py:249` | Fake quality object via `type('Q', (), {...})()` — fragile to attribute access | Use `QualityResult(is_valid=False, overall_score=0.0)` |
| 88 | LOW | `.env` | `CAMERA_SOURCE` not set — falls back to local webcam silently | Add `CAMERA_SOURCE=` to `.env` |

---

## Remediation Plan

### Phase 1 — Config (do first)
1. **1.1** Set `FRAME_WIDTH=1280`, `FRAME_HEIGHT=720` in `config/config.jsonc` (matches InsightFace det_size)
2. **1.2** Set `MATCH_THRESHOLD=0.45` in `config/config.jsonc` (0.25 too permissive)
3. **1.3** Set `EMBEDDING_DET_SCORE_MIN=0.40` in `config/config.jsonc` (align with defaults)
4. **1.4** Remove `FRAME_WIDTH/HEIGHT/MATCH_THRESHOLD/PERSON_CONF_THRESHOLD/EMBEDDING_DET_SCORE_MIN` from `.env` — let `config.jsonc` be single source of truth
5. **1.5** Fix `config/settings.py:177` — validate against config.jsonc value, not hardcoded `0.40`

### Phase 2 — Recognition Logic
1. **2.1** Re-derive "uncertain" band at new threshold (0.45 × 0.8 = 0.36) in `agents/recognition.py:97-167`
2. **2.2** Add regression test: `policy.py` and `recognition.py` read same threshold
3. **2.3** Remove dead `frame_faces = []` fallback in `agents/camera_agent.py:178`
4. **2.4** Skip CLAHE when input already meets contrast threshold in `utils/embedding_utils.py:36-52`

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
6. **6.6** Delete stale `yolov8n.pt` from repo

---

## Definition of Done

- [ ] Config sources agree — single documented source of truth
- [ ] False-positive rate drops after threshold correction
- [ ] Dashboard renders on backend 500 (no blank sections)
- [ ] WebSocket errors visible in logs
- [ ] Dead code from Phase 6 removed
