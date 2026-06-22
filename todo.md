# TODO — Visitor Surveillance System

## Status: Last reviewed 2026-06-22

---

## Bugs (fix immediately)

### 1. Duplicate alerts — `main.py:87`
**Status: PARTIALLY FIXED**
- `alert_agent.py:30-31` checks `track.alerted` and returns early, preventing actual duplicate sends
- However, `main.py:87` still calls `dispatch()` without checking `track.alerted` first (wasteful but not broken)
```python
# Current (main.py:87-89):
if decision.should_alert:
    dispatch(track, decision, image_url)
    track.alerted = True

# Should be:
if decision.should_alert and not track.alerted:
    dispatch(track, decision, image_url)
    track.alerted = True
```

### 2. `$push` overwrite — `db_utils.py:188-193`
**Status: STILL EXISTS**
When both `image_url` and `embedding` are provided, the second `$push` overwrites the first.
Images are silently lost from the update.
```python
# Bug (lines 188-193):
if image_url:
    update_ops["$push"] = {"images": {"url": image_url, "captured_at": datetime.utcnow()}}
if embedding:
    update_ops["$push"] = {"embeddings": embedding}  # overwrites images!

# Fix: merge into single $push
push_ops = {}
if image_url:
    push_ops["images"] = {"url": image_url, "captured_at": datetime.utcnow()}
if embedding:
    push_ops["embeddings"] = embedding
if push_ops:
    update_ops["$push"] = push_ops
```

### 3. `image_url` always `None` — `main.py:61-66`
**Status: STILL EXISTS**
Image saved to disk but `image_url` never assigned. `store_face` and `dispatch` both receive `None`.
```python
# Bug (lines 61-66):
image_url = None  # never updated
if track.best_full_frame is not None:
    ...
    save_image(track.best_full_frame, image_path)

# Fix: assign image_path to image_url (or upload to Cloudinary and use URL)
image_url = image_path
```

### 4. `crop_face_region` coordinate bug — `image_utils.py:39-40`
**Status: STILL EXISTS**
Subtracts `px1`/`py1` from already-absolute coordinates when clipping.
```python
# Bug (lines 39-40):
abs_fx2 = min(w, abs_fx2 - px1)  # px1 already added on line 32
abs_fy2 = min(h, abs_fy2 - py1)  # py1 already added on line 33

# Fix:
abs_fx2 = min(w, abs_fx2)
abs_fy2 = min(h, abs_fy2)
```

### 5. InsightFace lock contention — `embedding_utils.py:33`
**Status: STILL EXISTS**
Singleton lock serializes **all** face processing across all threads.
Multiple concurrent tracks block each other during `detect_and_embed()` and `embed_only()`.
- Consider using a queue-based approach or per-thread model instances

---

## Config Mismatches (`.env` vs spec)

| Setting | Current Default | Spec Value | Status |
|---------|----------------|------------|--------|
| `DET_SCORE_MIN` | 0.50 | 0.70 | **MISMATCH** — default in settings.py:62 is 0.50 |
| `MIN_TRACK_FRAMES` | 15 | 30 | **MISMATCH** — default in settings.py:78 is 15 |

Note: The `.env` file overrides these defaults, but the code defaults should match the spec.

---

## Security Issues

### .env contains real credentials
**Status: CRITICAL**
The `.env` file contains actual production credentials:
- MongoDB URI with password
- Cloudinary API key and secret

**Action needed:**
1. Rotate all credentials immediately
2. Create `.env.example` with placeholder values
3. Ensure `.env` is in `.gitignore` (it is)
4. Check git history to ensure `.env` was never committed

---

## Missing Features (Phase 4 — Dashboard)

### Backend (`dashboard/backend/`)
- [ ] FastAPI app setup
- [ ] `GET /api/events` — paginated audit log (filter by status, alert_level, date)
- [ ] `GET /api/faces` — enrolled identities CRUD
- [ ] `GET /api/alerts` — active/recent high+ alerts
- [ ] `WS /ws/live` — WebSocket pushing annotated frames + tracks
- [ ] `POST /api/faces/{person_id}/review` — operator relabels unknown

### Frontend (`dashboard/frontend/`)
- [ ] Live view — annotated camera feed with track boxes, IDs, names, badges
- [ ] Visitor log — chronological events with thumbnails
- [ ] Alerts panel — high/critical events with acknowledge
- [ ] Enrollment manager — review unknowns, assign names/roles/tags
- [ ] Audit trail — immutable events history

---

## Missing Files (per spec §3)

| File | Status | Priority |
|------|--------|----------|
| `dashboard/backend/` | Missing | High — Phase 4 |
| `dashboard/frontend/` | Missing | High — Phase 4 |
| `README.md` | Missing | Medium — documentation |
| `.env.example` | Missing | High — security |
| `models/` directory | Missing | Medium — organize weights |
| `pipeline/visibility_analyzer.py` | Missing | Low — logic in track_state.py (acceptable) |
| `pipeline/decision_engine.py` | Missing | Low — logic in decision_agent.py (acceptable) |

---

## Cloudinary Integration (incomplete)

**Status: NOT IMPLEMENTED**
- `utils/image_utils.py` has no Cloudinary upload function
- Images saved locally only, no `secure_url` stored in MongoDB
- `store_face` receives `image_url=None` always
- Config variables (`CLOUDINARY_CLOUD_NAME`, etc.) exist but are unused

---

## Hardening (spec §9 Phase 8)

- [ ] Move `yolov8n.pt` to `models/` directory
- [ ] Add alert debounce per track_id (partially done in `alert_agent.py`)
- [ ] Graceful camera release on unexpected exit
- [ ] Model loading audit — ensure no per-frame reload
- [ ] Create `.env.example` with placeholder values
- [ ] Rotate exposed credentials

---

## Priority Order

1. **CRITICAL: Rotate exposed credentials** (MongoDB, Cloudinary)
2. **Fix bug #2** — `$push` overwrite (data loss)
3. **Fix bug #3** — `image_url` always None (broken archival)
4. **Fix bug #4** — `crop_face_region` coordinate bug (broken face crops)
5. **Fix bug #1** — Add `track.alerted` check in main.py (cleanup)
6. **Fix config defaults** — DET_SCORE_MIN=0.70, MIN_TRACK_FRAMES=30
7. **Add Cloudinary upload** (image archival)
8. **Create `.env.example`** (security)
9. **Build dashboard** (Phase 4 — biggest gap)
10. **Add README.md**
11. **Hardening** (move yolov8n.pt, lock contention, edge cases)
