one track per person

1. Measured from logs/surveillance.debug.log — one recognition run takes 25–29 seconds(reason fallback to full face scan) now only 5-6.
2. Sample and Computation Redistribution Face/Person Detector

YOLO detects person boxes → ByteTrack assigns stable IDs → face recognition runs on the person crop (not the full frame)
Here's the exact flow:

1. Camera reads frame
2. YOLO detects person bounding boxes (not faces)
3. ByteTrack assigns stable IDs across frames
4. Every 20 frames, for each active track:
   a. Crop person from frame using ByteTrack's box
   b. InsightFace detects faces WITHIN that crop
   c. If no face in crop → fallback to full frame detection
   d. Quality check (blur, brightness, area)
   e. Embedding + MongoDB vector search
   The key insight: YOLO finds people, InsightFace finds faces. They're separate models running on different regions. YOLO gives you "there's a person here at (x1,y1,x2,y2)", then you crop that person and ask InsightFace "is there a face, and whose is it?"

After crop, we run first on the crop, then fall back to full frame only if no face found:
Step 1: person_crop = frame[by1:by2, bx1:bx2] ← crop from ByteTrack box

Step 2: crop_faces = detect_faces(person_crop) ← try crop first

Step 3: IF no face in crop:
frame_faces = detect_faces(frame) ← fallback to full camera frame

### 2 person crop and face crop

Two-stage crop:

1. Person crop (from ByteTrack box) → InsightFace runs here to find face location
2. Face crop (from full frame using face bbox) → quality + embedding run here

# Stage 1: find face location

person_crop = frame[y1:y2, x1:x2] # ByteTrack box
crop_faces = detect_faces(person_crop) # InsightFace → face bbox within person crop

# Stage 2: crop just the face from full frame

face_crop = frame[fy1:fy2, fx1:fx2] # face bbox → numpy slice
quality = compute_quality(face_crop) # runs on face only
embedding = get_embedding(face_crop) # ArcFace runs on face only
So InsightFace detection runs on person crop (faster, avoids background noise), but quality and embedding run on the actual face crop (just the face region).

## 3. problems

Here are the non-obvious things in your code:
Race Conditions

1. track.best_full_frame stores a numpy reference, not a copy (track_state.py:223) — if the camera thread reuses the frame buffer, the stored frame gets corrupted silently
2. Camera loop reads track.decision, track.pending_match_result without track.\_lock (camera_agent.py:168) — recognition workers write these concurrently
3. decide(track, ...) passes the live track, not the snapshot (track_processor.py:118) — PolicyAgent reads track.visibility and track.first_seen from the live object, which other threads may mutate
   Silent Failures
4. close_client() never resets \_faces_collection/\_events_collection (db_utils.py:34-41) — after close, get_faces_collection() returns a stale collection bound to the closed client
5. handle_frame() silently drops frames on encode failure (track_processor.py:44-62) — no log, no counter
6. llm_client.shutdown() leaks the async client (llm_client.py:396) — when uvicorn's event loop is running, asyncio.run() fails silently, client never closes
   Duplicate Code
7. compute_iou is defined in both track_state.py:21 AND image_utils.py:186 — identical implementations, different imports
   Config Inconsistencies
8. JPEG_QUALITY_BROADCAST = 90 in config.jsonc but default 65 in settings.py — AGENTS.md says "Fixed 50→90" but config.jsonc overrides to 90
9. FRAME_WIDTH/HEIGHT = 1920×1080 in config.jsonc but 1280×720 in settings.py defaults — docs describe 1280×720
10. DEBUG_RECOGNITION and DEBUG_DUPLICATE_BOXES are true in the committed config.jsonc — causes verbose debug logging on every frame, hurts performance
    Performance
11. store_face() does a redundant vector search (db_utils.py:326) — track_processor.py already ran run_matching_from_embedding() which does the same Atlas query, but skip_search=False means it runs again
12. draw_annotations() copies the full frame (image_utils.py:203) — frame.copy() is 6MB per frame, unnecessary since the camera loop creates new frames each iteration
13. \_ensure_log_dir() called on every confidence calculation (scoring.py:82) — does os.path.isdir() every time, should be done once at startup
    Security
14. Zero authentication on all API endpoints — anyone on the network can verify persons, delete faces, access live feed
15. Prompt injection via /api/chat (chat.py:102) — user message interpolated directly into LLM prompt with no sanitization
    Logic Bug
16. Office hours is inclusive on both ends (policy.py:83) — <= means hour 17 (5 PM) counts as office hours. Should be < for end time
17. update_visit_memory returns stale avg_similarity (db_utils.py:733) — the $push appends the new value but avg_similarity is never $set, so the returned value is always the previous average

unknown" → 1
"uncertain" → 2
"known" → 3
"known_visitor" → 4
"verified" → 5
"authorized" → 6
"blacklist" → 7
"masked_unknown" → 8
"intentionally_hidden" → 9
