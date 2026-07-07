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

### 4.5 Finalization face detection fallback (`camera_agent.py:401-448`)

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
    blur_valid  = blur_raw   >= QUALITY_BLUR_MIN          (default 30)
    bright_valid = QUALITY_BRIGHTNESS_MIN <= brightness_raw <= QUALITY_BRIGHTNESS_MAX  (default 30–240)
    area_valid  = face_area  >= QUALITY_FACE_AREA_MIN      (default 1600 = 40×40)

NORMALIZATION:
    blur_norm   = min(blur_raw / QUALITY_BLUR_MAX, 1.0)   — default max 1000
    bright_norm = brightness_raw / 255.0
    area_norm   = min(face_area / QUALITY_AREA_MAX, 1.0)  — default max 10000

WEIGHTED SCORE:
    overall_score = blur_norm   × QUALITY_WEIGHT_BLUR      (default 0.60)
                  + bright_norm × QUALITY_WEIGHT_BRIGHT     (default 0.25)
                  + area_norm   × QUALITY_WEIGHT_AREA       (default 0.15)

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

## 12. Recognition Agent — confidence computation (`agents/recognition.py`)

### 12.1 Case routing

```
CASE 1: similarity >= VERY_HIGH_SIMILARITY (0.90)
    confidence = min(95, 70 + (similarity - 0.90) × 250 + memory_boost)
    status = "known"

CASE 2: similarity >= MATCH_THRESHOLD (0.45)
    confidence = compute_confidence(...)
    IF confidence >= 70: status = "known"
    ELSE:                status = "uncertain"

CASE 3: face_quality >= BORDERLINE_FACE_QUALITY (0.8)
        AND similarity >= MATCH_THRESHOLD × 0.8 (= 0.36)
    confidence = 40 + similarity × 30 + memory_boost
    status = "uncertain"

CASE 4: else (low similarity)
    confidence = max(60, 100 - similarity × 100)
    status = "unknown"
```

### 12.2 Confidence formula (`_compute_confidence`)

```
sim_score = min(60, (similarity - MATCH_THRESHOLD) / (1.0 - MATCH_THRESHOLD) × 60)
    — Maps similarity from [MATCH_THRESHOLD, 1.0] to [0, 60]

quality_score = face_quality × 25
    — Maps face_quality [0, 1] to [0, 25]

duration_score = min(15, track_duration / 10)
    — Maps seconds [0, 150] to [0, 15]

confidence = sim_score + quality_score + duration_score + memory_boost

IF is_masked:
    confidence ×= MASK_CONFIDENCE_PENALTY (0.85)

confidence = clamp(confidence, 0, 100)
```

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

RULE 4: matched AND is_known_from_memory
    → status="known_visitor", alert_level="low", should_alert=False

RULE 5: matched (not memory-confirmed)
    5a: similarity >= KNOWN_VISITOR_SIMILARITY (0.85) OR confidence >= KNOWN_VISITOR_CONFIDENCE (80)
        → status="known_visitor", alert_level="low"
    5b: similarity >= MATCH_THRESHOLD (0.45)
        → status="known_visitor", alert_level="low"
    5c: else
        → status="uncertain", alert_level="low", should_register=True

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

## 21. Dashboard broadcast pipeline (`dashboard/backend/routes/live.py`)

```
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

JPEG encode executor (2 threads):
    _encode_and_broadcast()             — JPEG encode + WebSocket broadcast

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
| QUALITY_BLUR_MIN | 30 | Minimum Laplacian variance |
| QUALITY_BRIGHTNESS_MIN | 30 | Min HSV-V brightness |
| QUALITY_BRIGHTNESS_MAX | 240 | Max HSV-V brightness |
| QUALITY_FACE_AREA_MIN | 1600 | Min face area (40×40 px) |
| QUALITY_BLUR_MAX | 1000 | Blur normalization cap |
| QUALITY_AREA_MAX | 10000 | Area normalization cap |
| QUALITY_WEIGHT_BLUR | 0.60 | Blur weight in overall score |
| QUALITY_WEIGHT_BRIGHT | 0.25 | Brightness weight |
| QUALITY_WEIGHT_AREA | 0.15 | Area weight |
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
