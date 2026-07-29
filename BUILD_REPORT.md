# Surveillance System — Build Report

## What Was Built

A complete multi-agent video surveillance system with:
- YOLOv8 person detection + ByteTrack tracking
- SCRFD face detection + ArcFace embeddings (InsightFace)
- MongoDB Atlas vector search for identity matching
- Autonomous decision engine (authorize/alert)
- Multi-channel alerting (console, email, SMS, webhook)
- Progressive recognition for real-time alerts

---

## Changes Made to Original Spec (agent.md)

### 1. Track ID → Composite Key
**Before:** `track_id: int` (bare ByteTrack ID)
**After:** `track_id: str` = `f"{camera_id}_{session_epoch}_{byte_track_id}"`

**Why:** ByteTrack resets IDs to 0 on camera restart. Two different people could share `track_id: 42` across sessions, making the audit trail unreliable.

### 2. Thread Safety
**Before:** No locking on track dict or InsightFace model
**After:** `threading.Lock` on `TrackState._tracks` and `InsightFaceSingleton._lock`

**Why:** Camera loop and worker pool run concurrently. Without locks, dict modification during iteration causes `RuntimeError`, and concurrent InsightFace inference causes segfaults.

### 3. Progressive Recognition
**Before:** Recognition only runs when track ends (person leaves frame)
**After:** Recognition runs every `RECOGNITION_INTERVAL_FRAMES=30` frames during active tracks

**Why:** Original design meant a masked unknown only got a HIGH alert AFTER they left. Now alerts fire while the person is still in frame.

### 4. I/O Decoupling (Queue + Worker Pool)
**Before:** `handle_track()` runs synchronously in camera loop (blocks on MongoDB/Cloudinary)
**After:** Finalized tracks go to `queue.Queue`, processed by `ThreadPoolExecutor` workers

**Why:** MongoDB query + Cloudinary upload takes 500ms-2s. Without decoupling, camera loop stalls and drops frames. Now frame rate stays constant.

### 5. Split Detection Scores
**Before:** Single `DET_SCORE_MIN=0.70` for both detection and embedding
**After:** `DET_SCORE_MIN=0.50` (detection) + `EMBEDDING_DET_SCORE_MIN=0.70` (embedding quality gate)

**Why:** Side profiles and low-light real faces legitimately score 0.4-0.6. Hard cutoff at 0.70 rejected legitimate faces, causing false "hidden" classifications.

### 6. Visibility Classification Fix
**Before:** `is_masked` short-circuits to "partial" even if face is clearly visible
**After:** Check `max_face_ratio` FIRST, then mask flag

**Why:** A masked person whose face IS clearly visible should be "visible" with a mask flag, not forced to "partial". The original logic contradicted the rule "if face becomes visible, process recognition normally."

### 7. MIN_TRACK_FRAMES: 30 → 15
**Before:** 30 frames (~1 second) before "hidden" classification
**After:** 15 frames (~0.5 seconds)

**Why:** A deliberate camera-avoider appearing for 29 frames (just under the threshold) only got medium alert instead of high. This was a security gap.

### 8. Geometric Mask Detection
**Before:** "Lower-face landmark confidence is low" — but SCRFD doesn't return per-landmark confidence
**After:** Geometric analysis: `lower_face_height / upper_face_height < 0.3`

**Why:** The original heuristic was unimplementable. SCRFD returns landmark coordinates, not confidence scores.

### 9. Threshold Conversion Function
**Before:** Ad-hoc threshold comparison, Atlas and Python fallback could use different scales
**After:** `compare_similarity()` + `atlas_score_to_cosine()` — single source of truth

**Why:** Atlas `vectorSearchScore = (1+cosine)/2`. Without explicit conversion, Python fallback uses raw cosine while Atlas uses transformed score, producing inconsistent matching.

### 10. DB Schema Fixes
- Removed `event_id` field (use MongoDB `_id`)
- Changed `embedding` → `embeddings` array + `latest_embedding` (multiple over time)
- Added `images` array instead of single `image_url`
- `track_id` in events is now composite string
- `similarity_score` stores actual value (not always 0.0 for unknowns)
- Added TTL index on `timestamp` (90 days)
- Added auto-registration dedup via `find_similar_unknowns()`

### 11. Tag Priority Order
**Before:** "First match wins" but no defined order for tags
**After:** Explicit: `blacklist` > `authorized` > `known_visitor`

**Why:** If a person has both `authorized` and `blacklist` tags, blacklist must win. Without defined order, implementation-dependent.

### 12. Max Track Lifetime
**Before:** No maximum — person standing still forever never triggers decision
**After:** `MAX_TRACK_SECS=300` — force-finalize after 5 minutes

**Why:** A security guard at a desk would accumulate a track that never ends, and the recognition pipeline never runs.

---

## What Worked Well

### 1. InsightFace Singleton Pattern
The singleton with thread lock works correctly. Model loads once at startup, inference is serialized. No crashes from concurrent access.

### 2. Progressive Recognition
The every-30-frames recognition tick fires real-time alerts. A masked unknown triggers HIGH alert while still in frame, not after leaving.

### 3. Worker Pool Decoupling
Camera loop runs at full FPS regardless of MongoDB/Cloudinary latency. Frame drops eliminated.

### 4. Composite Track IDs
No ambiguity after camera restart. Events in MongoDB are uniquely attributable to specific sessions.

### 5. Geometric Mask Detection
Simple ratio check (`lower_face_height / upper_face_height < 0.3`) works reasonably well for surgical masks. No extra model needed.

### 6. Vector Search with Fallback
Atlas vector search is primary, Python cosine scan is fallback. System works before Atlas index is set up.

### 7. Auto-Registration Dedup
Same unknown appearing multiple times updates existing record instead of creating duplicates.

---

## What Needs Improvement

### 1. Mask Detection Accuracy
The geometric heuristic is basic. A proper MobileNetV2 mask classifier would be more robust. False positives from scarves, high collars, or beards.

### 2. Embedding Quality
Single embedding per appearance. Over months, appearance changes (aging, weight, hairstyle) will degrade recognition. Need embedding update strategy.

### 3. Camera Failure Handling
No graceful degradation if camera disconnects, MongoDB is unreachable, or Cloudinary fails. Needs retry logic and fallbacks.

### 4. Dashboard
Not built yet. FastAPI backend + React/Streamlit frontend for live view, visitor log, alerts panel, enrollment manager.

### 5. Frame Rate Adaptation
Fixed 640x480 resolution. No mechanism to drop frames or reduce resolution when CPU is overloaded.

### 6. LangGraph Integration
Currently plain Python callbacks. LangGraph state machine would give human-in-the-loop hooks and retries.

### 7. Idempotency for Alerts
Alert debounce works per track, but same person creating multiple tracks triggers multiple alerts. Need `(person_id, status)` keying.

### 8. Image Cleanup
No strategy for old captures eating disk space. Cloudinary is primary store but local files accumulate.

---

## File Inventory (21 files)

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 120 | Entry point, worker pool, track finalization |
| `config/settings.py` | 95 | Load .env, expose constants, validate_config() |
| `pipeline/models.py` | 85 | Track, QualityResult, EmbeddingResult, MatchResult, DecisionResult |
| `pipeline/detector.py` | 40 | YOLOv8 person detection (singleton) |
| `pipeline/tracker.py` | 45 | ByteTrack wrapper |
| `pipeline/track_state.py` | 150 | Thread-safe track dict, visibility classification |
| `pipeline/quality_agent.py` | 35 | Blur/brightness/area scoring |
| `pipeline/face.py` | 30 | SCRFD detect + ArcFace embed + face ratio |
| `agents/camera_agent.py` | 180 | Camera loop + progressive recognition |
| `agents/matching_agent.py` | 50 | Embedding → MongoDB vector search |
| `agents/decision_agent.py` | 70 | Policy engine |
| `agents/alert_agent.py` | 150 | Console/email/SMS/webhook alerts |
| `utils/image_utils.py` | 100 | Crop/resize/draw/save |
| `utils/embedding_utils.py` | 140 | InsightFace singleton + threshold utils |
| `utils/db_utils.py` | 180 | MongoDB CRUD + vector search + dedup |
| `requirements.txt` | 30 | Dependencies |
| `.env.example` | 60 | Config template |
| `.gitignore` | 25 | Git ignore rules |

**Total:** ~1,585 lines of Python + config
