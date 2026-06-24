# TODO — Visitor Surveillance System

## Status: All 39 issues fixed (2026-06-24)

No remaining issues. See `problem.md` for full audit history.

---

## Hardening (spec §9 Phase 8)

- [x] Move `yolov8n.pt` to `models/` directory
- [x] Alert debounce per track_id
- [x] Graceful camera release on exit
- [x] Model loading audit — no per-frame reload
- [x] `.env.example` with placeholder values
- [ ] Rotate exposed credentials (if `.env` was ever committed)

---

## Missing Features (Phase 4 — Dashboard)

### Backend (`dashboard/backend/`)

- [x] FastAPI app setup
- [x] `GET /api/events` — paginated audit log (filter by status, alert_level, date)
- [x] `GET /api/faces` — enrolled identities CRUD
- [x] `GET /api/alerts` — active/recent high+ alerts
- [x] `WS /ws/live` — WebSocket pushing annotated frames + tracks
- [ ] `POST /api/faces/{person_id}/review` — operator relabels unknown

### Frontend (`dashboard/frontend/`)

- [x] Live view — annotated camera feed with track boxes, IDs, names, badges
- [x] Visitor log — chronological events with thumbnails
- [x] Alerts panel — high/critical events with acknowledge
- [x] Enrollment manager — review unknowns, assign names/roles/tags
- [ ] Audit trail — immutable events history

---

## Performance Optimizations

- [x] Removed dead `detector.py`, unified to single YOLO model in `tracker.py`
- [x] Optimized face detection cascade — `detect_faces_raw()` runs detector once per image
- [x] Fixed N+1 query pattern — batched with `$in` query
- [x] All dashboard routes wrapped in `asyncio.to_thread()` for async MongoDB
- [x] Concurrent WebSocket broadcasting with `asyncio.gather()`
- [x] Async alert dispatch via `ThreadPoolExecutor`
- [x] Offloaded JPEG encoding to thread pool
- [x] Compressed JPEG in Track objects (~50KB vs ~921KB raw)
- [x] Progressive recognition offloaded to thread pool (no camera blocking)
- [x] Finalize track offloaded to thread pool (no camera blocking)
- [x] Reuse match result from progressive recognition (no redundant vector search)
- [x] Alert timestamp cache pruning (no unbounded growth)
