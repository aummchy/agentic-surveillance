# Formulas & Data Flow

Complete reference for the end-to-end data flow, face quality scoring, matching, memory, recognition, and policy decision pipeline.

---

# Part 1: End-to-End Data Flow

# working.md — End-to-End Data Flow with Exact Formulas

---

## 1. Startup sequence (`main.py:295-372`)

```
1. settings.validate_config()          — checks MONGODB_URI, thresholds, channels
2. llm_available()                     — pings Ollama GET /api/tags (cached 10s)
3. check_atlas_search_index()          — warns if "vector_index" missing
4. backfill_missing_embeddings()       — fixes faces with empty latest_embedding
5. Start 2 worker threads             — worker_process_tracks() via queue.Queue
6. Start uvicorn (port 8000)          — FastAPI dashboard + REST + WebSocket
7. CameraAgent(on_track_finalized, on_frame_annotated)
8. camera.start()                     — blocks until Q press or camera disconnect
9. On exit: queue.join(), shutdown executors, close MongoDB, close LLM clients
```

---

## 2. Camera loop (`camera_agent.py:111-183`)

Each iteration:

```
frame = cap.read()                     — OpenCV VideoCapture
frame_count += 1

tracks = track_persons(frame)          — YOLOv8 + ByteTrack

For each track t:
    composite_id = f"{camera_id}_{session_epoch}_{byte_track_id}"
    track = track_state.update(camera_id, t["track_id"], t["box"])

    Every RECOGNITION_INTERVAL_FRAMES frames:
        If not already_resolved AND not high_confidence:
            Submit _progressive_recognition(frame.copy(), track) to executor

expired = track_state.get_expired_tracks()
    — Returns tracks where:
        is_expired = (now - last_seen) > TRACK_TIMEOUT_SECS   [default 3.0s]
        OR is_max_lifetime = (now - first_seen) > MAX_TRACK_SECS [default 300s]
    — Classifies visibility BEFORE removal
    — Removes from dict only if no in-flight recognition

For each expired track not already finalizing:
    Submit _finalize_track(track) to executor

annotated = draw_annotations(frame, all_tracks)
on_frame_annotated(annotated)          — JPEG encode + WebSocket broadcast
```

---

## 3. Person detection + tracking (`pipeline/tracker.py`)

```python
model = YOLO(settings.YOLO_MODEL)     — loaded once, module-level singleton
results = model.track(
    frame,
    persist=True,                      — ByteTrack maintains IDs across frames
    tracker="bytetrack.yaml",
    classes=[0],                       — person class only
    conf=PERSON_CONF_THRESHOLD,        — default 0.40
    iou=0.5,
    device=YOLO_DEVICE,                — "cpu" or "intel:GPU"
)
# Returns: [{"track_id": int, "box": (x1,y1,x2,y2), "confidence": float}]
```

---

## 4. Face detection + embedding + mask detection (`utils/embedding_utils.py`)

### 4.1 CLAHE contrast enhancement

```
Applied to every image BEFORE face detection.

1. Convert BGR → LAB color space
2. Extract L channel
3. Compute l_std = stddev(L)
4. IF l_std >= 40.0: skip CLAHE (contrast already adequate)
5. ELSE: apply CLAHE with:
       clipLimit = CLAHE_CLIP_LIMIT (default 2.0)
       tileGridSize = CLAHE_TILE_SIZE × CLAHE_TILE_SIZE (default 8×8)
6. Convert LAB → BGR
```

### 4.2 Face detection (SCRFD via InsightFace)

```
InsightFace singleton loaded once:
    model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(1280, 1280))

detect_faces_raw(image, min_score):
    1. Apply CLAHE to image
    2. faces = app.get(image)          — SCRFD face detection
    3. Filter: keep faces where det_score >= min_score
    4. For each face, extract:
       - det_score: float (detection confidence)
       - embedding: 512-dim L2-normalized vector (ArcFace)
       - bbox: (x1, y1, x2, y2) in pixel coords
       - is_masked: bool (geometric mask heuristic)
    5. Sort by det_score descending
```

### 4.3 Mask detection heuristic (`embedding_utils.py:66-81`)

```python
# landmarks from SCRFD: [left_eye, right_eye, nose, left_mouth, right_mouth]
nose_tip = landmarks[2]
mouth_center = (landmarks[3] + landmarks[4]) / 2
upper_face = (landmarks[0] + landmarks[1]) / 2

lower_face_height = |mouth_center.y - nose_tip.y|
upper_face_height = |upper_face.y - nose_tip.y|

ratio = lower_face_height / upper_face_height
is_masked = (ratio < MASK_RATIO_THRESHOLD)   # default 0.3
```

### 4.4 Progressive recognition two-stage detection (`camera_agent.py:204-232`)

```
Stage 1: Detect in person_crop (tighter, more focused)
    crop_faces = detect_faces_raw(person_crop, min_score=DET_SCORE_RELAXED=0.20)

Stage 2: If no embedding-grade face in crop, detect in full frame
    crop_has_embedding_quality = any(f["det_score"] >= EMBEDDING_DET_SCORE_MIN for f in crop_faces)
    frame_faces = detect_faces_raw(frame, min_score=DET_SCORE_RELAXED=0.20)
        ONLY IF crop_has_embedding_quality is False

Pick best face:
    1. Try crop_faces[0] (highest det_score) — detected_in_person_crop = True
    2. Else try frame_faces[0]
    3. Convert bbox to frame coordinates if detected in crop:
       frame_bbox = (fx1 + px1, fy1 + py1, fx2 + px1, fy2 + py1)

Gate: best["det_score"] must be >= EMBEDDING_DET_SCORE_MIN (0.40)
    — If below threshold, skip embedding generation
```

### 4.5 Quality-gated recognition skip (`camera_agent.py:306-331`)

```
After face detection and quality scoring:

IF quality.is_valid:
    → set_best_face() — store best face crop
    → Continue to embedding generation and search
ELSE:
    → logger.debug("skip_recognition_low_quality", ...)
    → RETURN (exits _progressive_recognition early)
    → try/finally ensures end_recognition() still runs (track cleanup)

This prevents:
  - Storing embeddings from blurry/dark/small faces
  - Searching MongoDB with low-quality embeddings
  - Incorrectly matching blurry faces to known persons
  - Caching bad match results on the track
```

### 4.6 Finalization face detection fallback (`camera_agent.py:401-448`)

```
If track has no embedding at finalization:
    1. Try detect_faces_raw(best_face_crop, min_score=DET_SCORE_RELAXED=0.20)
    2. If no faces in crop, try detect_faces_raw(best_full_frame, min_score=DET_SCORE_RELAXED=0.20)
    3. Pick best face with two-tier priority:
       a. First pass: any face with det_score >= EMBEDDING_DET_SCORE_MIN (0.40)
       b. Second pass: any face with det_score >= DET_SCORE_RELAXED (0.20)
    4. If found: track.embedding = best["embedding"].tolist()
```

---

## 5. Face quality scoring (`pipeline/quality_agent.py`)

```
blur_raw = Laplacian(gray, CV_64F).var()
    — Gray conversion: cvtColor(face_crop, COLOR_BGR2GRAY) if 3-channel
    — Laplacian variance: higher = sharper

brightness_raw = mean(HSV_V channel)
    — HSV conversion: cvtColor(face_crop, COLOR_BGR2HSV)
    — Take channel V (value/brightness)
    — Mean of all pixels: range 0–255

face_area = height × width (in pixels)

VALIDITY GATES (all must pass):
    blur_valid  = blur_raw   >= QUALITY_VALID_BLUR_MIN      (default 40)
    bright_valid = QUALITY_VALID_BRIGHTNESS_MIN <= brightness_raw <= QUALITY_VALID_BRIGHTNESS_MAX  (default 35–255)
    area_valid  = face_area  >= QUALITY_VALID_FACE_AREA_MIN  (default 1200)

NORMALIZATION:
    blur_norm   = min(max(blur_raw - QUALITY_BLUR_MIN, 0) / (QUALITY_BLUR_MAX - QUALITY_BLUR_MIN), 1.0)
                  — shifted: 40=weak, 350=maxed
    bright_norm = 1 - min(|brightness_raw - QUALITY_BRIGHTNESS_CENTER| / QUALITY_BRIGHTNESS_RADIUS, 1.0)
                  — center-radius: peak at 145, radius 110
    area_norm   = min(max(face_area - QUALITY_FACE_AREA_MIN, 0) / (QUALITY_AREA_MAX - QUALITY_FACE_AREA_MIN), 1.0)
                  — shifted: 1500=weak, 10000=maxed

WEIGHTED SCORE:
    overall_score = blur_norm   × QUALITY_WEIGHT_BLUR      (default 0.50)
                  + bright_norm × QUALITY_WEIGHT_BRIGHT     (default 0.25)
                  + area_norm   × QUALITY_WEIGHT_AREA       (default 0.25)

is_valid = blur_valid AND bright_valid AND area_valid
```

### 5.1 Best face selection (`pipeline/track_state.py:120-138`)

```
set_best_face(composite_id, face_crop, face_score, full_frame, face_ratio):
    1. Quick exit: if face_score <= track.best_face_score + 0.03 → return
       (requires >3% improvement to replace)
    2. Encode full_frame to JPEG (QUALITY_STORE=85) — outside lock
    3. Under lock: if face_score > track.best_face_score:
       - Replace best_face_crop, best_face_score, best_full_frame,
         best_face_ratio, best_frame_jpeg
```

---

## 6. Face ratio calculation (`pipeline/face.py`)

```python
face_area   = (fx2 - fx1) × (fy2 - fy1)
person_area = (px2 - px1) × (py2 - py1)
face_ratio  = face_area / person_area    (if person_area > 0, else 0.0)
```

---

## 7. Embedding cache optimization (`camera_agent.py:288-303`)

```
IF track has cached_embedding AND pending_match_result:
    a = array(cached_embedding, float32)
    b = array(new_embedding, float32)
    cos_distance = 1.0 - dot(a, b) / (||a|| × ||b||)

    IF cos_distance < EMBEDDING_CACHE_COSINE_THRESHOLD (0.005):
        Reuse previous match_result (skip Atlas query)
    ELSE:
        Run vector_search(new_embedding) → fresh match_result
        Update track.cached_embedding = new_embedding
```

---

## 8. Visibility classification (`pipeline/track_state.py:70-81`)

Classified when track expires (before removal):

```
IF max_face_ratio >= VISIBLE_FACE_RATIO (0.025):
    visibility = "visible"

ELIF max_face_ratio >= PARTIAL_FACE_RATIO (0.010):
    visibility = "partial"

ELIF is_masked OR face_detected_once:
    visibility = "partial"           — mask or distant detectable face

ELIF NOT face_detected_once AND total_frames_seen >= MIN_TRACK_FRAMES (15):
    visibility = "hidden"            — long track, never saw face

ELSE:
    visibility = "unknown"           — short track, no face detected
```

---

## 9. Vector search — Atlas path (`utils/db_utils.py:79-132`)

```python
pipeline = [
    {"$vectorSearch": {
        "index": "vector_index",
        "path": "latest_embedding",
        "queryVector": embedding,         # 512-dim L2-normalized
        "numCandidates": 150,             # VECTOR_SEARCH_CANDIDATES
        "limit": 5                        # VECTOR_SEARCH_LIMIT
    }},
    {"$addFields": {"score": {"$meta": "vectorSearchScore"}}}
]

# Atlas returns vectorSearchScore = (1 + raw_cosine) / 2
# Must convert back:
raw_cosine = (atlas_score × 2) - 1

# Then compare against threshold:
match = compare_similarity(raw_cosine)   # raw_cosine >= MATCH_THRESHOLD (0.45)
```

### 9.1 Python cosine fallback (`utils/db_utils.py:135-195`)

```
Triggered when Atlas $vectorSearch fails (index missing, timeout, etc.)

1. Load up to SCAN_LIMIT (500) face documents from MongoDB
2. For each stored embedding:
   query_emb  = embedding / (||embedding|| + 1e-6)    — L2-normalize
   stored_emb = stored / (||stored|| + 1e-6)           — L2-normalize
   similarity = dot(query_emb, stored_emb)              — cosine similarity

3. Sort by similarity descending
4. Return top matches where similarity >= MATCH_THRESHOLD (0.45)
```

---

## 10. Match result structure (`agents/matching_agent.py`)

```python
run_matching_from_embedding(embedding):
    matches = vector_search(embedding)       # Atlas or Python fallback
    if not matches: return MatchResult(matched=False)

    best = matches[0]
    return MatchResult(
        person_id      = best["person_id"],
        name           = best["name"],
        role           = best["role"],
        tags           = best["tags"],          # ["authorized", "blacklist", etc.]
        similarity_score = best["similarity_score"],  # raw cosine
        image_url      = best["image_url"],
        matched        = True,
        verified       = best["verified"],
        alert_level    = best["alert_level"]
    )
```

---

## 11. Memory Agent — confidence boost (`agents/memory.py:149-184`)

```
boost = 0.0

# Returning visitor bonus
IF visit_count > 0:
    boost += min(10, visit_count × 2)       — +2 per visit, cap at +10

# Recency bonus
IF days_since_last <= 7:    boost += 5
ELIF days_since_last <= 30: boost += 2

# Consistency bonus
IF avg_similarity > 0.8:    boost += 3
ELIF avg_similarity > 0.6:  boost += 1

# Pattern bonus
IF is_typical_time:          boost += 2
IF is_typical_camera:        boost += 1

# Penalty: current similarity much lower than average
IF avg_similarity > 0 AND current_similarity < avg_similarity × 0.7:
    boost -= 5

RETURN clamp(boost, -10, +20)
```

### 11.1 is_typical_time check

```
hour_counts = count occurrences of each hour in typical_hours
common_hours = top 3 most frequent hours
is_typical_time = any(|current_hour - h| <= 2 for h in common_hours)
```

### 11.2 is_known check

```
is_known = (visit_count > 0) AND (last_status in ["known", "verified", "authorized", "known_visitor"])
```

---

## 12. Recognition Agent — confidence computation (`agents/recognition.py`, `agents/scoring.py`)

### 12.1 Confidence formula (weighted normalization)

```
base = 0.65×sim_norm + 0.15×quality_norm + 0.10×track_norm + 0.05×memory_norm + 0.05×margin_norm
adjusted = base × (1 - 0.15×mask_norm)
confidence = int(round(1 + 99 × clip(adjusted, 0, 1)))
```

### 12.2 Normalization functions (`agents/scoring.py`)

```
sim_norm     = clip((raw_cosine - 0.25) / (0.80 - 0.25), 0, 1)
quality_norm = clip(face_quality, 0, 1)   — None/0 → fallback 0.50
track_norm   = min(track_seconds / 1.5, 1.0)   — saturates at 1.5s
memory_norm  = clip(memory_boost, 0, 20) / 20
margin_norm  = clip(margin / 0.30, 0, 1)   — None → 0.50
mask_norm    = 1.0 if masked else 0.0
```

### 12.3 Status mapping (`confidence_status`)

```
matched=True  + confidence >= 70  → "known"       (CONFIDENCE_KNOWN_MIN)
matched=True  + confidence >= 55  → "uncertain"    (CONFIDENCE_UNCERTAIN_MIN)
matched=True  + confidence <  55  → "unknown"
matched=False + confidence >= 55  → "uncertain"
matched=False + confidence <  55  → "unknown"
```

### 12.4 Max-confidence gate

Recognition never downgrades. If the new confidence score is lower than the existing value on the Track, the update is skipped. Critical alerts always update regardless.

### 12.5 Quality-gated throttle

Progressive recognition skips if face quality didn't improve by ≥ `MIN_QUALITY_IMPROVEMENT` (0.10) since the last recognition. Prevents wasted CPU when lighting/pose haven't changed.

---

## 13. Policy Agent — decision rules (`agents/policy.py:110-289`)

Evaluated in priority order (first match wins):

```
RULE 1: "blacklist" in tags
    → status="blacklist", alert_level="critical", should_alert=True

RULE 2: "authorized" in tags
    → status="authorized", alert_level="none", should_alert=False

RULE 3: verified == True
    → status="verified", alert_level="none", should_alert=False

RULE 4a: "auto_registered" in tags AND similarity > 0.65
    → status="known_visitor", alert_level="low", should_alert=False

RULE 4b: matched AND is_known_from_memory
    → status="known_visitor", alert_level="low", should_alert=False

RULE 5: matched (not auto-registered, not memory-confirmed)
    5a: similarity >= KNOWN_VISITOR_SIMILARITY (0.85) OR confidence >= KNOWN_VISITOR_CONFIDENCE (80)
        → status="known_visitor", alert_level="low"
    5b: similarity >= MATCH_THRESHOLD (0.45)
        → status="unknown", alert_level="low", should_register=False
    5c: else (unreachable)
        → status="unknown", alert_level="low", should_register=False

RULE 6: visibility == "hidden"
    → status="intentionally_hidden", alert_level="high", should_alert=True

RULE 7: is_masked OR visibility == "partial"
    IF track_lifetime > LOITER_SECS (30):
        → alert_level="high", reason="Masked unknown person loitering"
    ELSE:
        → alert_level="medium", reason="Unknown person with partial visibility or mask"
    → status="masked_unknown", should_alert=True, should_register=True

RULE 8: NOT is_office_hours OR NOT is_weekday
    → status="unknown", alert_level="high", should_alert=True
    (office_hours: OFFICE_HOURS_START=9 to OFFICE_HOURS_END=17, weekday=Mon-Fri)

RULE 9: else (unknown during office hours)
    → status="unknown", alert_level="medium", should_alert=True, should_register=True
```

---

## 14. Alert deduplication (`agents/alert_agent.py:41-55`)

```
For unverified statuses (unknown, masked_unknown, uncertain, intentionally_hidden):
    key = "unverified:{alert_level}"
    — All unknowns of same level share one cooldown timer
    — A routine unknown does NOT suppress a critical (blacklist) alert

For all other statuses:
    key = "{track_id}:{alert_level}"
    — Per-track, per-level dedup

IF now - last_alert_time < ALERT_COOLDOWN_SECS (60):
    → suppress (return False)
ELSE:
    → allow, update timestamp
```

### 14.1 Stale pruning

```
Every ALERT_COOLDOWN_SECS × 2 seconds:
    Remove entries older than ALERT_COOLDOWN_SECS × 2 from _alert_timestamps
    — Prevents unbounded memory growth
```

---

## 15. Progressive critical alert dispatch (`camera_agent.py:345-362`)

```
During progressive recognition (NOT at finalization):

IF decision.should_alert
   AND decision.alert_level == "critical"    — ONLY critical fires early
   AND NOT track.alerted
   AND NOT already_finalized:
    → dispatch(track, decision, image_url)   — immediate alert
    → set_decision(track_id, status, alerted=True)

ALL OTHER alerts:
    → deferred to finalization (main.py process_finalized_track)
    — Avoids premature alerts for verified/known users when early
      recognition attempts produce low similarity
```

---

## 16. Track finalization (`main.py:68-271`)

```
1. Upload image to Cloudinary (or save locally as fallback)
   — Uses pre-encoded JPEG bytes if available (avoids re-encoding)

2. IF track.embedding is None:
   → log_event("unknown", "none", alerted=False) and return

3. Reuse match_result from progressive recognition if available
   — ELSE: run_matching_from_embedding(track.embedding)

4. Reuse recognition_result if match was also reused
   — ELSE: RecognitionAgent.run({...})

5. Reuse memory_context if match was reused
   — ELSE: MemoryAgent.run({...})

6. decision = decide(track, match_result, recognition_result, memory_context)

7. IF decision.should_register:
   7a. IF status is "unknown" or "masked_unknown":
       — Check find_similar_unknowns(embedding) for dedup
       — If similar found: update_face() (merge into existing)
       — ELSE: store_face() (create new record)
   7b. ELSE: store_face() with name/role from match_result

8. IF match_result.matched:
   — memory_agent.record_visit(person_id, camera_id, status, similarity, is_masked)

9. Move face_crop to captures/face_crops/{person_name}/ if matched

10. IF decision.should_alert AND NOT track.alerted:
    — dispatch(track, decision, image_url) → returns True if sent
    — Broadcast alert_payload to dashboard via WebSocket

11. log_event(track_id, status, alert_level, alerted, similarity_score, ...)
12. Broadcast event_payload to dashboard via WebSocket
```

---

## 17. Auto-registration dedup (`utils/db_utils.py:198-229`)

```
find_similar_unknowns(embedding):
    1. Load up to 500 unknowns (role="unknown") sorted by created_at desc
    2. For each:
       query_emb  = embedding / (||embedding|| + 1e-6)
       stored_emb = stored / (||stored|| + 1e-6)
       similarity = dot(query_emb, stored_emb)
    3. Return matches where similarity >= DEDUP_SIMILARITY_THRESHOLD (0.40)

IF similar found: merge into first match (update_face)
ELSE: create new face record (store_face)
```

---

## 18. Face record storage (`utils/db_utils.py:287-352`)

```
store_face(...) {
    1. IF NOT skip_search:
       — Run vector_search(embedding, limit=3) for dedup check
       — For each match with similarity >= DEDUP_SIMILARITY_THRESHOLD (0.40):
         a. IF existing face is verified: SKIP (never overwrite verified)
         b. ELSE: merge via update_face() and return existing person_id

    2. Create new document:
       {
           person_id: str,
           name: str,
           role: str,
           embeddings: [embedding],              # list of 512-dim vectors
           latest_embedding: embedding,           # fastest search field
           mean_embedding: embedding,             # recomputed on update
           latest_embedding_quality: quality_score,
           embedding_model: "arcface",
           images: [{id: uuid, url: image_url, captured_at: now}],
           source: {camera_id, captured_at},
           tags: [...],                           # ["authorized", "blacklist", etc.]
           verified: False,
           alert_level: "low",
           created_at: now, updated_at: now
       }
}
```

### 18.1 Quality-gated embedding update (`utils/db_utils.py:382-391`)

```
update_face(...) {
    IF new_embedding AND quality_score provided:
        current_quality = existing.latest_embedding_quality
        IF quality_score > current_quality:
            → overwrite latest_embedding with new embedding
            — Higher quality embedding wins for vector search
}
```

### 18.2 Embedding history management (`utils/db_utils.py:395-414`)

```
Push new embedding to embeddings array:
    $push: { embeddings: { $each: [embedding], $slice: -EMBEDDING_HISTORY_CAP } }
    — Keeps only last 25 embeddings (FIFO)

Recompute mean_embedding:
    all_embs = existing.embeddings + [new_embedding]
    all_embs = all_embs[-25:]                      — last 25
    mean = mean(all_embs, axis=0)
    mean_embedding = mean / (||mean|| + 1e-6)       — L2-normalize
```

---

## 19. Memory visit recording (`utils/db_utils.py:601-662`)

Atomic MongoDB update (single round-trip):

```
find_one_and_update({person_id}, {
    $inc:  { visit_count: 1 },
    $push: {
        similarity_history: { $each: [similarity], $slice: -10 },
        status_history:     { $each: [{status, timestamp}], $slice: -10 },
        typical_hours:      { $each: [current_hour], $slice: -20 },
    },
    $addToSet: { typical_cameras: camera_id },
    $set: {
        last_seen: now,
        last_camera: camera_id,
        last_status: status,
        updated_at: now,
    },
}, upsert=True)
```

---

## 20. LLM integration (`utils/llm_client.py`)

### 20.1 Alert NL summary generation

```
POST /api/generate (Ollama)
System: "You are a surveillance alert system..."
Prompt: status, alert_level, person, camera, masked, reason, visit_history
Temperature: 0.2, Max tokens: 150
Retry: 3 attempts, timeout from settings.OLLAMA_TIMEOUT (30s)
Fallback: None (caller uses template string)
```

### 20.2 Chat completion

```
POST /api/chat (Ollama)
Messages: [system_prompt, user_message]
Temperature: 0.3, Max tokens: 1024
Retry: 3 attempts
Fallback: None (returns None on failure)
```

### 20.3 Availability check

```
GET /api/tags
Cached for 10 seconds (avoids hammering Ollama)
Returns True if status_code == 200
```

---

## 21. Dashboard broadcast pipeline (`dashboard/backend/routes/live.py`, `agents/track_processor.py`)

```
Per-frame pipeline (track_processor.py):
    — frame_counter += 1
    — if counter % FRAME_SKIP (2) != 0: return       ← skip before encode
    — preview = cv2.resize(frame, (640, 360))         ← downscale for WebSocket
    — cv2.imencode(".jpg", preview, ...)              ← 640×360 JPEG
    — submit encode task to executor

WebSocket /ws/live:

broadcast_frame(jpeg_bytes):
    — FRAME_SKIP = 2 (broadcasts every 2nd frame)
    — MAX_FRAME_SIZE = 1MB
    — base64 encode → JSON {"type": "frame", "data": "..."}

broadcast_event(event_dict):
    — JSON {"type": "event", "data": {...}}

broadcast_alert(alert_dict):
    — JSON {"type": "alert", "data": {...}}

Client management:
    — connected_clients: Set[WebSocket]
    — Auto-disconnect on send failure (1s timeout per client)
    — Ping/pong: client sends "ping" → server responds {"type": "pong"}
```

---

## 22. Thread architecture

```
Main thread:
    CameraAgent._loop()                 — blocking cap.read() loop

Recognition executor (2 threads):
    _progressive_recognition()          — InsightFace + matching + decision
    _finalize_track()                   — final embedding + finalization

Track worker pool (2 threads):
    process_finalized_track()           — Cloudinary + DB + alerts + events

JPEG encode + broadcast (inline in camera loop):
    on_frame_annotated()                — JPEG encode + WebSocket broadcast (no separate executor)

Alert executor (2 threads):
    _send_async()                       — LLM summary + email/sms/webhook

Uvicorn server (1 thread):
    FastAPI REST + WebSocket server

Thread safety:
    — TrackState._lock: threading.Lock around _tracks dict
    — TrackState._in_flight: tracks in-progress recognition count
    — InsightFaceSingleton._lock: one-time model initialization
    — _collection_locks: per-collection MongoDB connection locks
    — _alert_lock: alert dedup timestamp access
    — _client_lock: MongoDB client creation
```

---

## 23. All thresholds summary

| Threshold | Default | Purpose |
|-----------|---------|---------|
| PERSON_CONF_THRESHOLD | 0.40 | YOLO person detection confidence |
| DET_SCORE_MIN | 0.40 | Standard face detection threshold |
| DET_SCORE_RELAXED | 0.20 | Relaxed face detection fallback |
| EMBEDDING_DET_SCORE_MIN | 0.40 | Minimum det_score to generate embedding |
| MATCH_THRESHOLD | 0.45 | Raw cosine similarity for face match (max 0.45) |
| DEDUP_SIMILARITY_THRESHOLD | 0.40 | Merge unknowns above this similarity |
| VERY_HIGH_SIMILARITY | 0.90 | Definite known match |
| HIGH_CONFIDENCE_SIMILARITY | 0.85 | Skip re-recognition |
| KNOWN_VISITOR_SIMILARITY | 0.85 | Auto-escalate to known_visitor |
| KNOWN_VISITOR_CONFIDENCE | 80 | Alternative confidence threshold |
| BORDERLINE_FACE_QUALITY | 0.80 | Face quality for borderline case |
| MASK_CONFIDENCE_PENALTY | 0.85 | Multiply confidence for masked faces |
| MASK_RATIO_THRESHOLD | 0.30 | Geometric mask detection ratio |
| VISIBLE_FACE_RATIO | 0.025 | face_area/person_area >= this = visible |
| PARTIAL_FACE_RATIO | 0.010 | face_area/person_area >= this = partial |
| MIN_TRACK_FRAMES | 15 | Min frames before hidden classification |
| TRACK_TIMEOUT_SECS | 3.0 | Drop track unseen this long |
| MAX_TRACK_SECS | 300 | Force-finalize after this many seconds |
| LOITER_SECS | 30 | Masked unknown beyond this = high alert |
| QUALITY_BLUR_MIN | 40 | Minimum Laplacian variance for scoring normalization |
| QUALITY_BRIGHTNESS_MIN | 35 | Min HSV-V brightness (validity gate) |
| QUALITY_BRIGHTNESS_MAX | 255 | Max HSV-V brightness (validity gate) |
| RECOGNITION_INTERVAL_FRAMES | 20 | Run progressive recognition every N frames |
| MIN_QUALITY_IMPROVEMENT | 0.10 | Min quality improvement to trigger re-recognition |
| QUALITY_VALID_BLUR_MIN | 40 | Validity gate: min Laplacian variance |
| QUALITY_VALID_BRIGHTNESS_MIN | 35 | Validity gate: min HSV-V brightness |
| QUALITY_VALID_BRIGHTNESS_MAX | 255 | Validity gate: max HSV-V brightness |
| QUALITY_VALID_FACE_AREA_MIN | 1200 | Validity gate: min face area px² |
| QUALITY_BLUR_MIN | 40 | Blur normalization floor (shifted from 40) |
| QUALITY_BLUR_MAX | 350 | Blur normalization cap |
| QUALITY_BRIGHTNESS_CENTER | 145 | Brightness model center (peak score) |
| QUALITY_BRIGHTNESS_RADIUS | 110 | Brightness model radius |
| QUALITY_FACE_AREA_MIN | 1500 | Area normalization floor (shifted from 1500) |
| QUALITY_BLUR_MAX | 350 | Blur normalization cap |
| QUALITY_AREA_MAX | 10000 | Area normalization cap |
| QUALITY_WEIGHT_BLUR | 0.50 | Blur weight in overall score |
| QUALITY_WEIGHT_BRIGHT | 0.25 | Brightness weight |
| QUALITY_WEIGHT_AREA | 0.25 | Area weight |
| CLAHE_CLIP_LIMIT | 2.0 | CLAHE contrast clip limit |
| CLAHE_TILE_SIZE | 8 | CLAHE tile grid size |
| EMBEDDING_CACHE_COSINE_THRESHOLD | 0.005 | Skip Atlas if embedding nearly identical |
| ALERT_COOLDOWN_SECS | 60 | Min seconds between same-type alerts |
| EMBEDDING_HISTORY_CAP | 25 | Max past embeddings per person |
| OFFICE_HOURS_START | 9 | Office hours start (hour) |
| OFFICE_HOURS_END | 17 | Office hours end (hour) |
| OFFICE_DAYS | [0,1,2,3,4] | Mon–Fri |
| SCAN_LIMIT | 500 | Max docs for Python cosine fallback |
| VECTOR_SEARCH_CANDIDATES | 150 | Atlas HNSW candidate pool |
| VECTOR_SEARCH_LIMIT | 5 | Atlas top results returned |


---

# Part 2: Embedding & Matching

# Embedding & Matching — How It Works

## 1. What buffalo_l returns

For each detected face, InsightFace returns a **512-d raw embedding vector**:

```
e = [e₁, e₂, ..., e₅₁₂]
```

This is just 512 floats — a numeric fingerprint of the face, not a name.

## 2. L2 normalization

Before storing or searching, the raw embedding is L2-normalized:

```
|e| = √(e₁² + e₂² + ... + e₅₁₂²)

ê = e / |e|      → ‖ê‖ = 1
```

**What Atlas stores:** the L2-normalized vector `ê` (512 floats).

```json
{
  "person_id": "cam_01_1782040060_1",
  "name": "aum",
  "latest_embedding": [0.013, -0.082, ..., 0.031]
}
```

## 3. Cosine similarity = dot product

Since all stored and query vectors have norm = 1:

```
cos(θ) = ê₁ · ê₂ = a₁b₁ + a₂b₂ + ... + a₅₁₂b₅₁₂
```

This single number (range -1 to 1) is the **raw cosine similarity**. Higher = more similar.

## 4. What Atlas returns

Atlas $vectorSearch returns a normalized `vectorSearchScore` (range 0-1). Convert back:

```
raw_cosine = 2 × vectorSearchScore − 1
```

Example: `Atlas score 0.8615 → raw_cosine = 2(0.8615)−1 = 0.723`

## 5. End-to-end flow

```
Camera frame
  → SCRFD detect face
  → CLAHE contrast enhancement
  → ArcFace (buffalo_l): raw 512-d embedding e
  → L2 normalize: ê = e/|e|
  → Atlas $vectorSearch(ê, index="vector_index", metric=cosine)
  → Atlas returns top match + vectorSearchScore
  → raw_cosine = 2 × score − 1
  → compare_similarity(raw_cosine ≥ 0.45) → MatchResult
```

## 6. 3-person example

Stored vectors: êₐ (Aum), êᵣ (Rahul), êₚ (Priya)

New query êq is compared:

| vs | Dot product | Atlas score | Raw cosine |
|----|:-----------:|:-----------:|:----------:|
| êₐ | 0.723 | (1+0.723)/2 = 0.8615 | 0.723 |
| êᵣ | 0.281 | (1+0.281)/2 = 0.6405 | 0.281 |
| êₚ | 0.119 | (1+0.119)/2 = 0.5595 | 0.119 |

Best match = Aum (0.723 ≥ 0.45 threshold) → `MatchResult(matched=True, similarity=0.723)`

## 7. Why same-person isn't 1.0

ArcFace embeddings shift between frames due to: head rotation (±0.05), lighting (±0.03–0.10), expression (±0.02–0.08), crop alignment (±0.02–0.05).

| Comparison | Typical cosine range |
|-----------|:--------------------:|
| Same person | 0.40 – 0.85 |
| Different person | −0.20 – 0.25 |

# 2nd 

| Value                 | Symbol                    |                                    Range | Meaning                               |
| --------------------- | ------------------------- | ---------------------------------------: | ------------------------------------- |
| Raw embedding         | (e)                       |                           no fixed range | 512-d face vector from `buffalo_l`    |
| Normalized embedding  | (\hat e)                  | components usually in `[-1,1]`, norm = 1 | stored/search vector                  |
| L2 norm               | (|e|)                     |                                     (>0) | magnitude of raw embedding            |
| Raw cosine similarity | (\hat e_1 \cdot \hat e_2) |                                 `[-1,1]` | actual face similarity                |
| Atlas score           | `vectorSearchScore`       |                                  `[0,1]` | Atlas normalized search score         |
| Detection score       | `det_score`               |                                  `[0,1]` | face detector confidence              |
| Final confidence      | `confidence`              |                                `[0,100]` | your app-level recognition confidence |

| Raw cosine      | Meaning                                  |
| --------------- | ---------------------------------------- |
| **0.80 – 0.95** | extremely strong same-person match       |
| **0.65 – 0.80** | strong same-person match                 |
| **0.45 – 0.65** | possible / moderate same-person match    |
| **0.25 – 0.45** | weak similarity / often different people |
| **< 0.25**      | usually different people                 |


# new in current system 
Implementation Note — Defensive Re-normalization in Python Fallback Paths

Although the canonical pipeline stores and queries L2-normalized face embeddings, the Python fallback comparison paths (_python_cosine_scan, find_similar_unknowns, find_similar_faces) defensively re-normalize embeddings with np.linalg.norm(...) before cosine computation.

// See Part 3 §4 below for the current weighted normalization formula (replaces old linear scaling).

---

# Part 3: Recognition Pipeline Formulas

# Recognition Pipeline Formulas

Complete reference for the face quality → matching → memory → recognition → policy decision pipeline.

---

## 1. Face Quality Score

**File:** `pipeline/quality_agent.py` · `pipeline/models.py:48-53`

### Raw measurements

| Metric | Function | Code |
|--------|----------|------|
| Blur | Laplacian variance of grayscale crop | `cv2.Laplacian(gray, cv2.CV_64F).var()` |
| Brightness | Mean HSV Value channel | `np.mean(hsv[:, :, 2])` |
| Area | Height × Width of face crop | `h * w` |

### Normalization

Shifted so a face at the minimum validity threshold scores near zero, and the full range is usable:

```
blur_norm    = min(max(blur_raw - QUALITY_BLUR_MIN, 0) / (QUALITY_BLUR_MAX - QUALITY_BLUR_MIN), 1.0)
bright_norm  = 1 - min(|brightness_raw - QUALITY_BRIGHTNESS_CENTER| / QUALITY_BRIGHTNESS_RADIUS, 1.0)
area_norm    = min(max(face_area - QUALITY_FACE_AREA_MIN, 0) / (QUALITY_AREA_MAX - QUALITY_FACE_AREA_MIN), 1.0)
```

| Metric | Norm at min threshold | Norm at max |
|--------|:--------------------:|:-----------:|
| Blur (40 → 350) | `(40-40)/310 = 0` | `1.0` |
| Brightness (center=145, radius=110) | `1-|V-145|/110` | `1.0` at V=145 |
| Area (1500 → 10000) | `(1500-1500)/8500 = 0` | `1.0` |

### Validity check vs. overall_score

- **`is_valid`** = whether the face is usable (passes minimum thresholds: blur ≥ 40, brightness 35–255, area ≥ 1200).
- **`overall_score`** = how good the face is relative to the full range. A barely-valid face scores ~0.03 ("low"). An excellent face scores 0.8+ ("high").

### Examples

| Condition | Blur | Bright | Area | `overall_score` | Label |
|-----------|:----:|:------:|:----:|:---------------:|:-----:|
| Barely valid | 40 | 35 | 1200 | `0×0.50 + 0.182×0.25 + 0×0.25 = 0.05` | low |
| Typical | 200 | 120 | 3000 | `0.516×0.50 + 0.227×0.25 + 0.176×0.25 = 0.37` | low |
| Good | 500 | 150 | 5400 | `1.0×0.50 + 0.055×0.25 + 0.459×0.25 = 0.63` | medium |
| Excellent | 1000 | 200 | 10000 | `1.0×0.50 + 0.591×0.25 + 1.0×0.25 = 0.90` | high |

### Validity gate

`is_valid = True` only when **all three** pass:

| Check | Threshold |
|-------|-----------|
| `blur_valid` | `blur_raw >= 40` |
| `bright_valid` | `35 <= brightness_raw <= 255` |
| `area_valid` | `face_area >= 1200` px² |

### Quality level labels (used in recognition reason string)

| `overall_score` | Label |
|:---------------:|-------|
| `>= 0.8` | "high face quality" |
| `>= 0.5` and `< 0.8` | "medium face quality" |
| `< 0.5` | **"low face quality"** |

### Live thresholds (from `config/config.jsonc`)

| Setting | Value | Purpose |
|---------|-------|---------|
| `QUALITY_BLUR_MIN` | 40 | Min Laplacian variance for scoring normalization |
| `QUALITY_BLUR_MAX` | 350 | Normalization cap for blur |
| `QUALITY_BRIGHTNESS_CENTER` | 145 | Center of brightness model (peak score) |
| `QUALITY_BRIGHTNESS_RADIUS` | 110 | Radius of brightness model |
| `QUALITY_FACE_AREA_MIN` | 1500 | Min face area for scoring normalization |
| `QUALITY_AREA_MAX` | 10000 | Normalization cap for area |
| `QUALITY_WEIGHT_BLUR` | 0.50 | Weight of blur in score |
| `QUALITY_WEIGHT_BRIGHT` | 0.25 | Weight of brightness |
| `QUALITY_WEIGHT_AREA` | 0.25 | Weight of area |
| `QUALITY_VALID_BLUR_MIN` | 40 | Validity gate: min blur |
| `QUALITY_VALID_BRIGHTNESS_MIN` | 35 | Validity gate: min brightness |
| `QUALITY_VALID_BRIGHTNESS_MAX` | 255 | Validity gate: max brightness |
| `QUALITY_VALID_FACE_AREA_MIN` | 1200 | Validity gate: min face area |

---

## 2. Vector Search (Matching)

**Files:** `utils/db_utils.py` · `agents/matching_agent.py`

### Atlas $vectorSearch

```
collection     = faces
index          = vector_index
path           = latest_embedding
dimensions     = 512
similarity     = cosine
numCandidates  = 150
limit          = 5
```

### Score conversion

Atlas returns `vectorSearchScore` in range `[0, 1]` where:
```
vectorSearchScore = (1 + raw_cosine) / 2
```

To recover raw cosine similarity:
```
raw_cosine = (atlas_score × 2) - 1
```

The `similarity_score` in `MatchResult` is the **raw cosine** (range `[-1, 1]`, but 0 to 1 for non-opposite embeddings). Only the best match (highest score) is used.

### Match threshold

```
MATCH_THRESHOLD = 0.45
```

If `similarity_score >= 0.45`, the match is considered a hit. Otherwise it's a miss.

### MatchResult fields

```python
MatchResult(
    person_id,          # MongoDB _id of matched face
    name,               # Person's display name
    role,               # "visitor", "authorized", etc.
    tags,               # ["auto_registered", "verified", "blacklist", "authorized"]
    similarity_score,   # Raw cosine similarity (0.0 to 1.0)
    matched,            # True if similarity >= MATCH_THRESHOLD
    verified,           # True if "verified" in tags
    alert_level,        # "low", "medium", "high", "critical"
)
```

---

## 3. Memory Agent — Confidence Boost

**File:** `agents/memory.py:149-184`

### Boost formula

```
boost = 0

# Returning visitor bonus
boost += min(10, visit_count × 2)          # +2 per visit, max +10

# Recency bonus
if days_since_last <= 7:   boost += 5
elif days_since_last <= 30: boost += 2

# Consistency bonus
if avg_similarity > 0.8:   boost += 3
elif avg_similarity > 0.6: boost += 1

# Pattern bonuses
if is_typical_time:  boost += 2
if is_typical_camera: boost += 1

# Penalty: current match much worse than usual
if current_similarity < avg_similarity × 0.7:  boost -= 5

return clamp(boost, -10, +20)
```

### Range

| Component | Min | Max |
|-----------|-----|-----|
| Visit count | 0 | +10 |
| Recency | 0 | +5 |
| Consistency | 0 | +3 |
| Typical time | 0 | +2 |
| Typical camera | 0 | +1 |
| Penalty | -5 | 0 |
| **Total (clamped)** | **-10** | **+20** |

### Typical time check

```python
current_hour = now.hour
common_hours = top 3 most frequent visit hours
is_typical_time = any(abs(current_hour - h) <= 2 for h in common_hours)
```

---

## 4. Recognition Agent — Confidence & Status

**File:** `agents/recognition.py` · `agents/scoring.py`

### Decision flow

```
Face detected → vector search → match_result (similarity, top2, margin)
                                       │
                                       ▼
                              Memory Agent → memory_boost (-10 to +20)
                                       │
                                       ▼
                          compute_confidence(raw_cosine, quality, ...)
                                       │
                                       ▼
                          confidence_status(confidence, matched)
                                       │
                                       ▼
                              status = "known" | "uncertain" | "unknown"
```

### Confidence formula (weighted normalization)

```
base = 0.65×sim_norm + 0.15×quality_norm + 0.10×track_norm + 0.05×memory_norm + 0.05×margin_norm
adjusted = base × (1 - 0.15×mask_norm)
confidence = int(round(1 + 99 × clip(adjusted, 0, 1)))
```

**File:** `agents/scoring.py:compute_confidence()`

### Normalization functions

| Component | Function | Input | Output | Notes |
|-----------|----------|-------|--------|-------|
| `sim_norm` | `normalize_cosine(raw)` | raw cosine `[-1, 1]` | `[0, 1]` | `(raw - 0.25) / (0.80 - 0.25)`, clipped |
| `quality_norm` | `normalize_quality(q)` | `float [0, 1]` or `None` | `[0, 1]` | `None`/`0` → fallback `0.50` |
| `track_norm` | `normalize_track_duration(secs)` | seconds | `[0, 1]` | `min(secs / 1.5, 1.0)` — saturates at 1.5s |
| `memory_norm` | `normalize_memory(boost)` | boost `[-10, +20]` | `[0, 1]` | `clip(boost, 0, 20) / 20` |
| `margin_norm` | `normalize_margin(margin)` | margin `[0, 1]` | `[0, 1]` | `clip(margin / 0.30, 1.0)`, `None` → `0.50` |
| `mask_norm` | `normalize_mask(masked)` | bool | `0.0` or `1.0` | `1.0` if masked |

### Weight breakdown

| Component | Weight | Normalization | Effective range |
|-----------|:------:|:-------------:|:---------------:|
| Similarity | 0.65 | `[0.25..0.80]` → `[0..1]` | 0–0.65 |
| Face Quality | 0.15 | `[0..1]` direct | 0–0.15 |
| Track Duration | 0.10 | `secs / 1.5` saturated | 0–0.10 |
| Memory Boost | 0.05 | `[0..20]` → `[0..1]` | 0–0.05 |
| Margin | 0.05 | `[0..0.30]` → `[0..1]` | 0–0.05 |
| **Base total** | **1.00** | | **0–1.00** |
| Mask penalty | ×(1 - 0.15×mask) | | ×1.0 or ×0.85 |

### Thresholds

| Threshold | Value | Effect |
|-----------|-------|--------|
| `MATCH_THRESHOLD` | 0.45 | `raw_cosine >= 0.45` → `matched=True` |
| `CONFIDENCE_KNOWN_MIN` | 70 | `matched + conf >= 70` → `"known"` |
| `CONFIDENCE_UNCERTAIN_MIN` | 55 | `conf >= 55` → `"uncertain"` else `"unknown"` |

### Status determination

```python
def confidence_status(confidence, matched):
    if matched:
        if confidence >= 70:  return "known"
        if confidence >= 55:  return "uncertain"
        return "unknown"
    if confidence >= 55:  return "uncertain"
    return "unknown"
```

### Worked example

```
Input:
  raw_cosine    = 0.704
  face_quality  = 0.512
  track_seconds = 10.5s
  memory_boost  = 18.0
  margin        = 0.202
  is_masked     = False

Normalization:
  sim_norm     = (0.704 - 0.25) / 0.55 = 0.825
  quality_norm = 0.512
  track_norm   = min(10.5 / 1.5, 1) = 1.000
  memory_norm  = 18.0 / 20 = 0.900
  margin_norm  = 0.202 / 0.30 = 0.675
  mask_norm    = 0.0

Weighted sum:
  base = 0.65×0.825 + 0.15×0.512 + 0.10×1.000 + 0.05×0.900 + 0.05×0.675
       = 0.536 + 0.077 + 0.100 + 0.045 + 0.034
       = 0.792

  adjusted = 0.792 × (1 - 0.15×0) = 0.792
  confidence = 1 + 99 × 0.792 = 79
  status = "known"  (matched=True, 0.704 >= 0.45, confidence=79 >= 70)
```

### Match threshold gate

```
raw_cosine >= 0.45  →  matched = True
raw_cosine <  0.45  →  matched = False
```

Even if `matched=False`, confidence can still reach `"uncertain"` (>= 55) from quality + track + memory signals.

---

## 5. Policy Agent — Final Decision

**File:** `agents/policy.py`

### Rule priority (highest wins)

```
1. BLACKLIST   → status="blacklist",           alert="critical", should_alert=True
2. AUTHORIZED  → status="authorized",          alert="none",     should_alert=False
3. VERIFIED    → status="verified",            alert="none",     should_alert=False
4a. AUTO-REG  → "auto_registered" in tags AND sim > 0.65  → status="known_visitor", alert="low"
4b. KNOWN (memory) → matched AND is_known_from_memory     → status="known_visitor", alert="low"
5a. MATCHED (high)  → sim >= 0.85 OR conf >= 80           → status="known_visitor", alert="low"
5b. MATCHED (mid)   → sim >= 0.45 (not 5a)                → status="unknown",       alert="low"
5c. MATCHED (else)  → unreachable                          → status="unknown",       alert="low"
6. HIDDEN      → status="intentionally_hidden", alert="high"
7. MASKED      → status="masked_unknown",       alert medium/high (loitering >30s)
8. AFTER-HOURS → status="unknown",              alert="high"
9. OFFICE-HOURS→ status="unknown",              alert="medium"
```

### Policy decision fields

```python
DecisionResult(
    status,          # "verified" | "known_visitor" | "blacklist" | "unknown" | etc.
    alert_level,     # "none" | "low" | "medium" | "high" | "critical"
    should_alert,    # True = dispatch notification
    should_register, # True = store face in DB as new person
    reason,          # Human-readable explanation
)
```

---

## 6. Threshold Reference (all from `config/config.jsonc`)

### Matching

| Setting | Value | Effect |
|---------|-------|--------|
| `MATCH_THRESHOLD` | 0.45 | Min cosine similarity for a match |
| `DEDUP_SIMILARITY_THRESHOLD` | 0.40 | Min similarity to merge auto-registrations |

### Recognition

| Setting | Value | Effect |
|---------|-------|--------|
| `VERY_HIGH_SIMILARITY` | 0.90 | Definite known (Case 1) |
| `HIGH_CONFIDENCE_SIMILARITY` | 0.85 | Skip re-recognition in camera loop |
| `KNOWN_VISITOR_SIMILARITY` | 0.85 | Policy auto-escalates to known_visitor |
| `KNOWN_VISITOR_CONFIDENCE` | 80 | Policy auto-escalates if confidence >= 80 |
| `BORDERLINE_FACE_QUALITY` | 0.8 | Threshold for "high face quality" label |
| `MASK_CONFIDENCE_PENALTY` | 0.85 | Confidence multiplier for masked faces |

### Policy

| Setting | Value | Effect |
|---------|-------|--------|
| `OFFICE_HOURS_START` | 9 | After-hours alert if before this hour |
| `OFFICE_HOURS_END` | 17 | After-hours alert if after this hour |
| `OFFICE_DAYS` | Mon–Fri | Weekday check for after-hours |
| `LOITER_SECS` | 30 | Masked person loitering threshold |

### Face Detection

| Setting | Value | Effect |
|---------|-------|--------|
| `DET_SCORE_MIN` | 0.40 | Standard face detection threshold |
| `DET_SCORE_RELAXED` | 0.20 | Permissive fallback threshold |
| `EMBEDDING_DET_SCORE_MIN` | 0.40 | Min detection score to generate embedding |
| `RECOGNITION_INTERVAL_FRAMES` | 20 | Run recognition every N frames |

---

## 7. Pipeline Flow Diagram

```
Camera frame
    │
    ▼
YOLO person detection (track_persons)
    │
    ▼
ByteTrack tracking (track_state)
    │
    ▼
Progressive recognition (every 20 frames)
    │
    ├── Face detection (SCRFD via InsightFace)
    │       │
    │       ▼
    ├── Quality gate (compute_quality)
    │       │
    │       ▼
    ├── ArcFace embedding (512-dim)
    │       │
    │       ▼
    ├── Embedding cache check (cosine distance < 0.005? → reuse last match)
    │       │
    │       ▼
    ├── Atlas $vectorSearch → MatchResult
    │       │
    │       ▼
    ├── Memory Agent → confidence_boost (-10 to +20)
    │       │
    │       ▼
    ├── Recognition Agent → status + confidence (0-100)
    │       │
    │       ▼
    └── Policy Agent → DecisionResult
            │
            ▼
    ● CRITICAL alert? → dispatch immediately
    ● Otherwise → deferred to finalization
```


          Camera frame
               │
               ▼
     CLAHE contrast enhancement
               │
               ▼
     SCRFD face detection → det_score=0.87, bbox, landmarks
               │
               ▼
     Face alignment (affine transform via landmarks)
               │
               ▼
     ArcFace ResNet-100 → 512-dim embedding
               │
               ▼
     L2-normalize → ‖embedding‖ = 1
               │
               ▼
     Atlas $vectorSearch (HNSW index, cosine)
               │
               ▼
     (1 + dot_product) / 2  → Atlas score = 0.8615
               │
               ▼
     atlas_score_to_cosine → raw_cosine = (0.8615×2)-1 = 0.723
               │
               ▼
     compare_similarity(0.723 >= 0.45) → MatchResult(matched=True, similarity=0.723)
Want me to document this full flow in FORMULAS.md as a new section?
