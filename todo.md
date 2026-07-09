# TODO — Visitor Surveillance System

## Status: 1 remaining (env credentials). All recognition/throttle/confidence/visibility fixes complete (2026-07-09).

---

## Remaining

- [ ] **#50: Rotate `.env` credentials** — MongoDB URI, Cloudinary key/secret exposed in working directory

---

## Hardening (spec §9 Phase 8)

- [x] Move `yolov8n.pt` to `models/` directory
- [x] Alert debounce per track_id (thread-safe with `_alert_lock`)
- [x] Graceful camera release on exit
- [x] Model loading audit — no per-frame reload
- [x] `.env.example` with placeholder values
- [x] MongoDB client closed on shutdown
- [x] Alert executor shut down on process exit
- [x] uvicorn server stopped gracefully (`server.should_exit = True`)

---

## Missing Features (Phase 4 — Dashboard)

### Backend (`dashboard/backend/`)

- [x] FastAPI app setup
- [x] `GET /api/events` — paginated audit log
- [x] `GET /api/faces` — enrolled identities CRUD (`?status=all|unknown|verified`)
- [x] `GET /api/alerts` — active/recent high+ alerts
- [x] `WS /ws/live` — WebSocket pushing annotated frames + tracks
- [x] `event._id` properly serialized to React frontend
- [ ] `POST /api/faces/{person_id}/review` — operator relabels unknown

### Frontend (`dashboard/frontend/`)

- [x] Live view — annotated camera feed with track boxes, IDs, names, badges
- [x] Visitor log — chronological events with thumbnails (React list keys fixed)
- [x] Alerts panel — high/critical events with acknowledge
- [x] Enrollment manager — review unknowns, assign names/roles/tags
- [ ] Audit trail — immutable events history

---

## Performance & Thread Safety

- [x] Progressive recognition offloaded to thread pool (no camera blocking)
- [x] Finalize track offloaded to thread pool (no camera blocking)
- [x] Reuse match result from progressive recognition (no redundant vector search)
- [x] Alert timestamp cache pruning (no unbounded growth)
- [x] Alert cooldown check thread-safe with `_alert_lock`
- [x] `_recognizing_tracks`/`_finalized_track_ids` protected with `_track_sets_lock`
- [x] WebSocket broadcast uses defensive `list()` copy before iteration
- [x] Exception handlers include `exc_info=True` for stack traces
- [x] `_finalize_track` always calls `on_track_finalized` (even on error)
- [x] `process_finalized_track` always logs event (even on error)
- [x] `classify_visibility` runs before track removal from dict
- [x] `broadcast_alert` covers all alert-worthy detections (not just unknown/masked_unknown)
- [x] MongoDB lookups batched in `get_recent_incidents` (no N+1)
- [x] Dead code `agents/alert.py` deleted
- [x] Duplicate `PolicyAgent` singleton removed from `policy.py`
