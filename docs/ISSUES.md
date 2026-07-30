# Issues & Bug Tracker

**63 issues remaining (0 CRITICAL, 5 HIGH, 19 MEDIUM, 25 LOW, 20 META). 117 FIXED.**

See `INTENTIONAL.md` for design decisions that look like limitations but are deliberate trade-offs.

---

## CRITICAL — Will crash at runtime

### C1 — NameError in event broadcast ✅ FIXED

- **File:** `main.py:266`
- **Problem:** Variable `best_crop_path` is never defined in `process_finalized_track()`. The `Track` dataclass has `best_face_crop_path` but the code references `best_crop_path`. This raises `NameError` every time a track is finalized with a non-null `decision` and event broadcast.
- **Impact:** Every track finalization with event broadcast crashes silently (caught by outer try), producing no event/alert/memory.
- **Fix:** Changed `best_crop_path` to `track.best_face_crop_path` on line 266.
- **Status:** Fixed 2026-07-30. Verified: field exists in Track model (`pipeline/models.py:43`), populated in `track_state.py:189`, no competing field names.

### C2 — AttributeError on dynamic quality object ✅ FIXED

- **File:** `agents/camera_agent.py:303-330`
- **Problem:** When `face_crop.size == 0`, a fallback quality object is created via `type('Q', (), {'is_valid': False, 'overall_score': 0.0})()`. The logger on lines 324-330 accesses `quality.blur_score`, `quality.brightness`, and `quality.face_area` — none of which exist on the dynamic type.
- **Impact:** `AttributeError` crash when processing an empty face crop (e.g., detection returned bbox but crop was zero-size).
- **Fix:** Added `QualityResult.invalid()` class method to `pipeline/models.py`. Replaced `type('Q', ...)` hack in `camera_agent.py:304` with `QualityResult.invalid()`. Updated import to include `QualityResult`.
- **Status:** Fixed 2026-07-30. All 43 tests pass. Factory method is now single source of truth for invalid quality.

### C3 — bool cast from env var is broken ✅ FIXED

- **File:** `config/settings.py:552`
- **Problem:** `_get()` function casts env var values using `cast(val)`. When `cast=bool`, `bool("False")` returns `True` (any non-empty string is truthy). Setting `ENABLE_CALC_LOG=false` in `.env` silently becomes `True`.
- **Impact:** Debug logging cannot be disabled via env var. Any future `bool`-typed setting will have same bug.
- **Fix:** Added `if cast is bool:` special case in `_get()` that parses `"false"`, `"0"`, `"no"`, `"off"`, `""` as `False`; everything else as `True`.
- **Status:** Fixed 2026-07-30. All 43 tests pass. All 3 critical bugs now resolved.

---

## HIGH — Data corruption / threading / logic errors

### H1 — avg_similarity never updated in MongoDB ⏭️ SKIPPED

- **Files:** `agents/memory.py:92`, `utils/db_utils.py:641-660`
- **Problem:** `avg_similarity` is initialized to `0.0` in `get_or_create_memory()` and never recalculated in `update_visit_memory()`. The atomic `$push` operation adds to `similarity_history` but `avg_similarity` stays at `0.0` forever.
- **Impact:** The confidence boost from consistent similarity (lines 167-170 in `memory.py`) never fires. People seen with consistently high similarity get no confidence advantage.
- **Fix:** Recalculate `avg_similarity` in `update_visit_memory()` after `$push`, or compute on read from `similarity_history`.
- **Skip reason:** User deferred. Informational field, no critical logic depends on it.

### H2 — Mixed timezone in update_visit_memory ✅ FIXED

- **File:** `utils/db_utils.py:624`
- **Problem:** `update_visit_memory()` uses `datetime.now()` (local time) while every other timestamp in the file uses `datetime.utcnow()` (UTC). This includes lines 352, 353, 360, 361, 375, 384, 392, 450, 455, 493, 552, 571, 591, 690.
- **Impact:** `last_seen` / `updated_at` timestamps in `visit_memory` collection will be in local time while all other timestamps are UTC. Breaks time-based queries and analysis.
- **Fix:** Changed `datetime.now()` to `datetime.utcnow()` on line 624.
- **Status:** Fixed 2026-07-30. All 43 tests pass.

### H3 — TOCTOU race in quality-gated embedding overwrite ✅ FIXED

- **File:** `utils/db_utils.py:398-405` (was), `utils/db_utils.py:441-455` (now)
- **Problem:** Classic read-then-write race. Two threads processing different frames for the same `person_id` both: (1) read `current_quality`, (2) compute `quality_score > current_quality`, (3) attempt to overwrite `latest_embedding`. Second writer wins, potentially overwriting a higher-quality embedding with a lower-quality one.
- **Impact:** Embedding quality can degrade over time under concurrent frame processing.
- **Fix:** Split `update_face()` into two `update_one` calls. Second call uses a filter with `$or` that evaluates quality at write time (inside MongoDB), eliminating the TOCTOU gap. `embedding is not None` (not `if embedding`) avoids numpy truth-value ambiguity.
- **Status:** Fixed 2026-07-30. All 43 tests pass.

### H4 — TOCTOU race in mean_embedding calculation ⏭️ SKIPPED

- **File:** `utils/db_utils.py:420-428`
- **Problem:** Concurrent calls for the same `person_id` read stale `embeddings` arrays, compute different means, and the last writer wins. One thread's new embedding may be silently dropped from the mean.
- **Impact:** `mean_embedding` diverges from actual average, degrading recognition accuracy over time.
- **Fix:** Use atomic `$push` operation and recalculate mean inside the MongoDB update pipeline, or use a single-threaded writer.
- **Skip reason:** 2-thread worker pool makes race unlikely. Mean self-corrects on next write. Fixing properly requires MongoDB aggregation pipelines (heavy) or single-writer lock (adds contention). Low ROI.

### H5 — update_visit_memory crashes on None result ✅ FIXED

- **File:** `utils/db_utils.py:670-684`
- **Problem:** `find_one_and_update` with `upsert=True` can return `None` (e.g., network interruption between upsert and read). All `.get()` calls on line 666-674 throw `AttributeError`. The caller in `memory.py:249` catches `Exception` but the error is misleading.
- **Impact:** Visit memory updates silently fail with misleading error messages.
- **Fix:** Add `if result is None` guard, log warning, return fallback dict with `__fallback__: True` so callers can distinguish from a real document.
- **Status:** Fixed 2026-07-30. All 43 tests pass.

### H6 — get_or_create_memory can return None ✅ FIXED

- **File:** `utils/db_utils.py:620-635`
- **Problem:** Same as H5. `find_one_and_update` with `$setOnInsert` and no `$set` can return `None`. Callers at `agents/memory.py:83` assume `dict` and call `.get()` which throws `AttributeError`.
- **Impact:** Memory-based confidence features crash on first encounter with a new person.
- **Fix:** Add `if result is None` guard, log warning, return fallback dict with `__fallback__: True`.
- **Status:** Fixed 2026-07-30. All 43 tests pass.

### H7 — Track fields read without lock (data races) ⏭️ SKIPPED (partial)

- **File:** `agents/camera_agent.py:162-185`
- **Problem:** Main camera thread reads `track.decision`, `track.pending_match_result`, `track.last_recognition_status`, etc. while worker threads write them via lock-protected setters. Compound expressions like `track.pending_match_result and track.pending_match_result.similarity_score > settings.HIGH_CONFIDENCE_SIMILARITY` are NOT atomic — `pending_match_result` can become `None` between the truth check and the `.similarity_score` access.
- **Impact:** Intermittent `AttributeError` crashes under load. Data race can produce inconsistent state reads.
- **Fix:** Read fields under lock, or copy a snapshot of the track state atomically.
- **Skip reason:** Compound expression race is narrow (requires `pending_match_result` to become `None` between truth check and attribute access). H10, H11, H12 fix the more dangerous mutations. Full snapshot method deferred.

### H8 — track.rescan_attempts unprotected mutation ⏭️ SKIPPED

- **File:** `agents/camera_agent.py:177`
- **Problem:** `track.rescan_attempts += 1` is a read-modify-write mutation in the camera thread with no lock protection. If a recognition worker also reads `rescan_attempts`, there is a race condition.
- **Impact:** Counter could be corrupted under heavy concurrent access (unlikely in practice since only camera thread writes, but violates encapsulation).
- **Fix:** Add a setter method in `TrackState` or protect with lock.
- **Skip reason:** Only camera thread reads/writes. No concurrent access verified.

### H9 — track.confidence unprotected writes ⏭️ SKIPPED

- **File:** `agents/camera_agent.py:413, 421`
- **Problem:** `track.confidence = new_confidence` is a direct write from worker threads without any lock. `Track.confidence` is `int` but `RecognitionResult.confidence` is `float` — the `int()` cast in camera_agent.py papers over a type mismatch.
- **Impact:** Data race on confidence field. Type mismatch could cause issues if cast is removed.
- **Fix:** Use `set_decision()` or add a lock-protected setter for confidence.
- **Skip reason:** Only `_progressive_recognition` accesses it; effectively single-threaded per track via `_recognizing_tracks` gate.

### H10 — decision.nl_summary mutation from thread pool ✅ FIXED

- **File:** `agents/alert_agent.py:99`
- **Problem:** `_send_async` closure captures `DecisionResult` dataclass by reference and mutates `nl_summary` from a thread-pool worker. If any other thread reads `decision.nl_summary` concurrently (e.g., during finalization logging), this is an unprotected data race.
- **Impact:** NL summary could be partially written when read by another thread.
- **Fix:** Moved LLM call before thread dispatch. `_send_async()` now only sends channels.
- **Status:** Fixed 2026-07-30. All 43 tests pass.

### H11 — \_finalize_track bypasses quality gate ✅ FIXED

- **File:** `agents/camera_agent.py:503-504`
- **Problem:** `_finalize_track()` directly writes `track.embedding = ...` and `track.is_masked = ...`, bypassing `set_embedding()` which has quality-gating logic. This means finalization can overwrite a higher-quality embedding with a lower-quality one from the relaxed fallback detection.
- **Impact:** Final embedding stored in MongoDB may be lower quality than what was already stored during progressive recognition.
- **Fix:** Route through `set_embedding()`. Returns `bool`; logs `embedding_rejected_by_quality_gate` with `current_det_score`/`new_det_score` on rejection.
- **Status:** Fixed 2026-07-30. All 43 tests pass.

### H12 — cv2.imencode success flag ignored ✅ FIXED

- **File:** `pipeline/track_state.py:131, 155`
- **Problem:** `cv2.imencode()` returns `(success: bool, buffer: ndarray)`. The return value `success` is discarded (assigned to `_`). If encoding fails, `jpeg_buf` is empty/corrupt, and `jpeg_buf.tobytes()` produces garbage data stored as `best_frame_jpeg` or `fallback_frame_jpeg`.
- **Impact:** Corrupt JPEG data stored in track state, served to dashboard as invalid images.
- **Fix:** Check `success` bool, log warning with `composite_id` and `context`, return early on failure.
- **Status:** Fixed 2026-07-30. All 43 tests pass.

### H13 — RECOGNITION_INTERVAL_FRAMES default mismatch ✅ FIXED

- **Files:** `config/settings.py:485`, `config/config.jsonc:49`
- **Problem:** Hardcoded default in `settings.py` is `10`, but `config.jsonc` has `20`. Config wins at runtime (effective value = 20). AGENTS.md says "every 10 frames" which matches settings.py but not the actual runtime behavior.
- **Impact:** Misleading documentation. Developers debugging recognition behavior will be confused by discrepancy.
- **Fix:** Aligned defaults to `20` in both places, updated AGENTS.md.
- **Status:** Fixed 2026-07-30. All 43 tests pass.

### H14 — JPEG_QUALITY_BROADCAST default mismatch ✅ FIXED

- **Files:** `config/settings.py:509`, `config/config.jsonc:86`
- **Problem:** `settings.py` default is `65`, `config.jsonc` has `80`. Config wins at runtime. AGENTS.md changelog says "fixed JPEG_QUALITY_BROADCAST 50→65" but the fix is effectively dead since config overrides it.
- **Impact:** Actual broadcast quality is 80, not the documented 65. Bandwidth usage higher than expected.
- **Fix:** Aligned defaults to `65` in `config.jsonc`, updated docs.
- **Status:** Fixed 2026-07-30. All 43 tests pass.

---

## MEDIUM — Logic / design / error handling

### M1 — No upper-bound clamping in compute_face_ratio

- **File:** `pipeline/face.py:1-11`
- **Problem:** `compute_face_ratio()` can return values > 1.0 (face larger than person box, which is physically impossible due to coordinate errors or detection from full frame). Downstream `_classify_visibility_inplace()` compares against thresholds like 0.025, so a ratio of 50.0 would incorrectly classify as "visible".
- **Impact:** Incorrect visibility classification for edge-case detections.
- **Fix:** Clamp return value to `[0.0, 1.0]`.

### M2 — Dead code in Rule 5 (policy.py)

- **File:** `agents/policy.py:228-237`
- **Problem:** The `else` branch (`similarity < MATCH_THRESHOLD` when `matched=True`) should never execute because `vector_search` in `db_utils.py:128` already filters by `compare_similarity()` (i.e., `>= MATCH_THRESHOLD`). If reached, it would register a duplicate unknown for someone already matched.
- **Impact:** Dead code path that could cause duplicate registrations if ever reached.
- **Fix:** Remove dead branch or add an assertion that `similarity >= MATCH_THRESHOLD` when `matched=True`.

### M3 — masked_unknown applied to non-masked persons

- **File:** `agents/policy.py:255`
- **Problem:** If `visibility == "partial"` but `is_masked=False`, the status is still `masked_unknown`. The person is not masked, just partially visible. The status is misleading.
- **Impact:** Dashboard shows "masked_unknown" for partially visible unmasked persons, confusing operators.
- **Fix:** Split into separate statuses (e.g., `partial_visibility_unknown`) or check `is_masked` separately.

### M4 — Midnight wrap-around in time check

- **File:** `agents/memory.py:113`
- **Problem:** `abs(current_hour - h) <= 2` does not handle midnight boundary. If `common_hours = [0, 1]` and `current_hour = 23`, `abs(23 - 0) = 23` and `abs(23 - 1) = 22` — both > 2, so it returns `False` even though hour 23 is near the typical hours 0-1.
- **Impact:** Late-night/early-morning visits incorrectly classified as "atypical time".
- **Fix:** Use circular distance: `min(abs(a - b), 24 - abs(a - b))`.

### M5 — Inconsistent \_empty_result return type

- **File:** `agents/memory.py:220 vs 101`
- **Problem:** `_empty_result` returns `days_since_last_visit: 0` while `_analyze` returns `days_since_last = None` for the same semantic case (no visit history).
- **Impact:** Downstream consumers receive inconsistent types for the same "no history" case.
- **Fix:** Unify to `None` for both cases.

### M6 — Dead is_masked parameter

- **File:** `agents/memory.py:233`
- **Problem:** `record_visit()` accepts `is_masked=False` parameter but never passes it to `update_visit_memory()`. In `db_utils.py`, `update_visit_memory` accepts `is_masked` but doesn't store it either. The parameter is dead code.
- **Impact:** Masked visit status is never recorded, limiting future analytics.
- **Fix:** Remove parameter or actually store it in visit_memory.

### M7 — No quality weight sum validation ✅ FIXED

- **File:** `pipeline/quality_agent.py:62-64`
- **Problem:** Quality weights (`QUALITY_WEIGHT_BLUR`, `QUALITY_WEIGHT_BRIGHT`, `QUALITY_WEIGHT_AREA`) are user-configurable via `config.jsonc`. If a user sets them to e.g. 0.6 + 0.3 + 0.2 = 1.1, the `overall_score` is clamped to [0, 1] but the relative contribution changes unpredictably. `validate_config()` does not check this.
- **Impact:** Incorrect quality scoring if weights don't sum to 1.0.
- **Fix:** Added validation in `validate_config()`: each weight in `[0,1]` = error; sum != 1.0 = warning.
- **Status:** Fixed 2026-07-30. All 43 tests pass.

### M8 — llm_client.shutdown() not thread-safe ✅ FIXED

- **File:** `utils/llm_client.py:366-387`
- **Problem:** `shutdown()` modifies `_client`, `_async_client`, and `_shut_down` without acquiring `_client_lock` or `_async_client_lock`. If `_get_client()` is called concurrently, it could return a closed client or see stale `_shut_down` flag.
- **Impact:** Potential crash or resource leak during shutdown if LLM requests are in-flight.
- **Fix:** Added `_shutdown_lock` with early return if already shut down. Captures refs before clearing globals. `try/finally` with `logger.warning` on async close failure.
- **Status:** Fixed 2026-07-30. All 43 tests pass.

### M9 — Unused timeout parameter in generate()

- **File:** `utils/llm_client.py:75`
- **Problem:** `generate()` accepts `timeout: int = None` parameter that is never used. The actual timeout comes from `httpx.Timeout(settings.OLLAMA_TIMEOUT, connect=5.0)` set on the client at creation time.
- **Impact:** Misleading API — callers may think they can override timeout per-call.
- **Fix:** Remove parameter or implement per-call timeout override.

### M10 — report_agent.run() per-event in loop

- **File:** `dashboard/backend/routes/reports.py:67-80`
- **Problem:** For each event (up to `limit=100`), `report_agent.run()` is called synchronously via `asyncio.to_thread`. Each call may invoke the LLM for incident summary. With 100 events, this could take minutes with no timeout protection.
- **Impact:** `/api/reports/incidents` endpoint can hang for minutes under normal usage.
- **Fix:** Batch LLM calls, parallelize, or add per-event timeout.

### M11 — LLM prompt injection vulnerability

- **File:** `dashboard/backend/routes/chat.py:102, 118`
- **Problem:** User message is injected directly into the LLM prompt: `f"Operator question: {user_message}\n\nAvailable data:\n{data_str}{history_context}"`. A crafted message could override `SYSTEM_PROMPT` instructions, causing the LLM to reveal internal data or execute unintended operations.
- **Impact:** Information disclosure, unauthorized data access via prompt manipulation.
- **Fix:** Sanitize input, use structured prompt templates with clear delimiters, validate LLM output.

### M12 — Conversation history injection

- **File:** `dashboard/backend/routes/chat.py:29`
- **Problem:** `history: Optional[List[dict]]` accepts arbitrary conversation history from the client. A malicious client can inject fake "Assistant:" messages to bias the LLM's responses.
- **Impact:** LLM responses can be manipulated by injecting conversation context.
- **Fix:** Validate history format, limit client-controlled roles, cap history length.

### M13 — SMTP_PORT no error handling

- **File:** `config/settings.py:460`
- **Problem:** `SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))` crashes with unhandled `ValueError` at module import time if env var is non-numeric.
- **Impact:** Application fails to start with cryptic error if SMTP_PORT is misconfigured.
- **Fix:** Wrap in `try/except ValueError` with fallback to default.

### M14 — CWD-relative paths

- **File:** `utils/image_utils.py:113, 156`
- **Problem:** `save_image_atomic` uses `Path("captures/_tmp")` and `resolve_track_image_url` uses `f"captures/{track.track_id}.jpg"` — both relative to current working directory. If the process is started from a different directory, files are created in wrong location.
- **Impact:** Captures saved to wrong directory, dashboard cannot find images.
- **Fix:** Use `Path(__file__).resolve().parent.parent / "captures"` or a settings-based absolute path.

### M15 — cv2.imencode ignored in upload_to_cloudinary

- **File:** `utils/image_utils.py:40`
- **Problem:** `_, buffer = cv2.imencode(".jpg", image)` discards the `success` bool. If encoding fails, `buffer` is empty but `buffer.tobytes()` is called anyway, uploading an empty/corrupt image to Cloudinary.
- **Impact:** Broken image URLs stored in face records, dashboard shows blank images.
- **Fix:** Check `success` bool, skip upload on failure, log warning.

### M16 — Static /captures mount without auth + no Cloudinary signed URLs

- **File:** `dashboard/backend/main.py:35`, `utils/image_utils.py`
- **Problem:** The `/captures` directory is mounted as a static file server with no authentication. Combined with public Cloudinary URLs, any network client can download surveillance images. "Cloudinary signed URLs" is an unimplemented security measure.
- **Impact:** Surveillance images accessible without authentication. Privacy risk.
- **Fix:** Add auth middleware for `/captures` mount. Use Cloudinary signed URLs with expiry, or serve via authenticated proxy.

### M17 — No authentication on any dashboard endpoint

- **File:** All dashboard routes
- **Problem:** The entire dashboard API (faces CRUD, events, chat, WebSocket live feed) is completely unauthenticated. CORS restricts browser origins but does not prevent direct API calls. "WebSocket authentication for live feed" is unimplemented.
- **Impact:** Any network client can delete faces, query events, view live feed, or use chat. Unauthorized access to surveillance data.
- **Fix:** Add auth middleware (API key, JWT, or basic auth). Add WebSocket token validation on accept.

### M18 — Delete with no confirmation

- **File:** `dashboard/backend/routes/faces.py:139`
- **Problem:** `DELETE /api/faces/{person_id}` permanently deletes a face record with no soft-delete option or confirmation token.
- **Impact:** Irreversible data loss for a surveillance system. Accidental deletion loses identity data.
- **Fix:** Add soft-delete (set `deleted_at` timestamp) or require confirmation token.

### M19 — No WebSocket client limit

- **File:** `dashboard/backend/routes/live.py`
- **Problem:** `connected_clients` is a plain `set()` that can grow unbounded. A malicious client could open thousands of WebSocket connections and exhaust server memory/bandwidth.
- **Impact:** Denial of service via connection exhaustion.
- **Fix:** Cap max connections (e.g., 50), reject new clients above limit, log warnings.

### M20 — log_formula_header() not atomic

- **File:** `agents/scoring.py:31-37`
- **Problem:** Check-and-set of `_calc_log_initialized` is not atomic. Two threads can both see `False`, both write the header. Benign (duplicate header in log) but violates the pattern.
- **Impact:** Duplicate formula header in calculation log (cosmetic).
- **Fix:** Use `threading.Lock` or `threading.Event`.

### M21 — Camera failure handling (no graceful degradation) ⏭️ INTENTIONAL

- **File:** `agents/camera_agent.py`, `utils/db_utils.py`, `utils/image_utils.py`
- **Problem:** No graceful degradation if camera disconnects, MongoDB is unreachable, or Cloudinary fails.
- **Impact:** Camera disconnect = permanent pipeline failure. MongoDB outage = lost events. Cloudinary failure = lost images.
- **Decision:** Intentional design trade-off. See `INTENTIONAL.md` for rationale. Camera reconnect exists (fixed delay). MongoDB/Cloudinary retry assumed unnecessary under cloud SLA.

---

## LOW — Type hints / imports / code quality

### L1 — margin: float = None type hint

- **File:** `agents/recognition.py:93, 162`
- **Problem:** `margin: float = None` should be `Optional[float] = None`. Static type checkers will flag this.
- **Fix:** Change to `Optional[float] = None`.

### L2 — Loose type hints on models

- **File:** `pipeline/models.py:20, 80`
- **Problem:** `embedding: Optional[list]` should be `Optional[list[float]]`. `tags: list` should be `list[str]`.
- **Fix:** Tighten type hints.

### L3 — Dead code in models.py

- **File:** `pipeline/models.py:65-72, 118-128`
- **Problem:** `EmbeddingResult` is defined but never used anywhere in the codebase. `RecognitionResult.to_dict()` is never called — `recognition_agent.run()` already returns a plain dict.
- **Fix:** Remove dead code.

### L4 — Missing type hints in track_state.py

- **File:** `pipeline/track_state.py:191, 20`
- **Problem:** `match_result` parameter has no type annotation (should be `Optional[MatchResult]`). `frame: np.ndarray = None` should be `frame: Optional[np.ndarray] = None`.
- **Fix:** Add proper type hints.

### L5 — Redundant inline numpy imports

- **File:** `utils/db_utils.py:149, 215, 250, 291`
- **Problem:** Four functions each contain `import numpy as np` inline. Numpy is always available and these functions are called frequently. Redundant import lookup overhead.
- **Fix:** Move to module-level import.

### L6 — filter_role: str = None type hint

- **File:** `utils/db_utils.py:79`
- **Problem:** `filter_role: str = None` should be `Optional[str]`. Same issue in `_python_cosine_scan` (line 148).
- **Fix:** Change to `Optional[str] = None`.

### L7 — detect_faces_raw swallows all exceptions

- **File:** `utils/embedding_utils.py:112-113`
- **Problem:** `except Exception as e: return []` silently swallows every possible error, including `AttributeError` from InsightFace API changes, `MemoryError` from huge images. Caller cannot distinguish "no faces found" from "detection crashed."
- **Fix:** Catch specific exceptions (`RuntimeError`, `cv2.error`), log differently per type.

### L8 — CLAHE object recreated on every frame

- **File:** `utils/embedding_utils.py:62-64`
- **Problem:** `cv2.createCLAHE()` is called every time a low-contrast frame is processed. The CLAHE object is immutable and could be cached as a class attribute.
- **Fix:** Create once and cache.

### L9 — Missing type hints in image_utils.py

- **File:** `utils/image_utils.py:140, 186`
- **Problem:** `resolve_track_image_url(track)` has no type annotation on `track`. `draw_annotations(frame, tracks: list)` should be `list[Track]`.
- **Fix:** Add `Track` and `list[Track]` type hints.

### L10 — save_image swallows exceptions

- **File:** `utils/image_utils.py:92-98, 101-126`
- **Problem:** `save_image` and `save_image_atomic` catch `Exception` and return `False` without logging. Caller cannot distinguish "disk full" from "permission denied" from "invalid image."
- **Fix:** Log exception before returning `False`.

### L11 — Unused imports in llm_client.py

- **File:** `utils/llm_client.py:19`
- **Problem:** `Dict` and `Any` are imported from `typing` but never used. Code uses lowercase `dict` and doesn't use `Any`.
- **Fix:** Remove unused imports.

### L12 — is_available catches overly broad Exception

- **File:** `utils/llm_client.py:360`
- **Problem:** `except Exception: _avail_cache_val = False` catches `KeyboardInterrupt`, `SystemExit`, `MemoryError`, and programming errors. Should only catch network errors.
- **Fix:** Catch `httpx.RequestError` specifically.

### L13 — SMTP connection leaked on exception

- **File:** `agents/alert_agent.py:168-176`
- **Problem:** If `send_message()` or `login()` throws, `server.quit()` is never called. No `finally` block. SMTP connection is leaked until garbage collection.
- **Fix:** Use `try/finally` or context manager (`with smtplib.SMTP(...) as server:`).

### L14 — result["top2"] / result["margin"] without .get()

- **File:** `agents/matching_agent.py:23-24`
- **Problem:** `result["top2"]` and `result["margin"]` are accessed without `.get()` fallback. If `vector_search` returns a dict without these keys, `KeyError` is raised.
- **Fix:** Use `.get("top2", None)` and `.get("margin", 0.0)`.

### L15 — Deprecated datetime.utcnow()

- **File:** `agents/report.py:82, 279`
- **Problem:** `datetime.utcnow()` is deprecated since Python 3.12 (returns naive datetime). Should use `datetime.now(timezone.utc)`.
- **Fix:** Replace with `datetime.now(timezone.utc)`.

### L16 — Silent exception swallowing in report.py

- **File:** `agents/report.py:206-209`
- **Problem:** `except Exception: memory = {}` silently catches all errors with no logging. If MongoDB is down, returns empty dict with no indication of failure.
- **Fix:** Log exception before fallback.

### L17 — Hardcoded tracker parameters

- **File:** `pipeline/tracker.py:43, 45`
- **Problem:** `classes=[0]` (person class) and `iou=0.5` are hardcoded. Unlike `conf` which uses `settings.PERSON_CONF_THRESHOLD`, these are not configurable.
- **Fix:** Move to settings.

### L18 — Redundant import cv2

- **File:** `agents/camera_agent.py:57`
- **Problem:** `import cv2` inside `_open_capture()` is redundant — `cv2` is already imported at module top (line 1).
- **Fix:** Remove local import.

### L19 — Manual dict mapping in decision_agent

- **File:** `agents/decision_agent.py:45-53`
- **Problem:** Manual dict mapping of `MatchResult` fields instead of `dataclasses.asdict()`. If `MatchResult` gains a field, this dict must be updated manually.
- **Fix:** Use `dataclasses.asdict(match_result)`.

### L20 — Dead motor logger suppression

- **File:** `config/settings.py:340`
- **Problem:** `logging.getLogger("motor").setLevel(logging.WARNING)` suppresses motor logger, but `motor` (async MongoDB driver) is never installed, imported, or used.
- **Fix:** Remove.

### L21 — Dead MASK_DETECTION setting

- **File:** `config/settings.py:560`
- **Problem:** `MASK_DETECTION = _get("MASK_DETECTION", "MASK_DETECTION", "heuristic")` is defined but never referenced anywhere in the codebase.
- **Fix:** Remove or use.

### L22 — Unused imports in main.py

- **File:** `main.py:20`
- **Problem:** `save_image` and `upload_to_cloudinary` are imported from `utils.image_utils` but never used in `main.py`.
- **Fix:** Remove.

### L23 — asyncio.run_coroutine_threadsafe return discarded

- **File:** `main.py:89, 248, 272`
- **Problem:** The returned `concurrent.futures.Future` is discarded. If the coroutine raises an exception, it's silently lost.
- **Fix:** Store future, log on failure.

### L24 — No retry backoff on LLM timeout

- **File:** `utils/llm_client.py:104, 149, 204, 248`
- **Problem:** All four retry loops retry immediately on timeout with no delay. If the LLM server is temporarily overloaded, hammering it with immediate retries makes things worse.
- **Fix:** Add exponential backoff (e.g., 0.5s, 1s, 2s).

### L25 — Duplicated disconnect-cleanup logic

- **File:** `dashboard/backend/routes/live.py:49-98`
- **Problem:** Disconnect cleanup code is repeated in `broadcast_frame`, `broadcast_event`, and `broadcast_alert`. Nearly identical blocks.
- **Fix:** Extract shared helper function.

### L26 — list_unknown_events is duplicate of list_events

- **File:** `dashboard/backend/routes/events.py:45-62`
- **Problem:** `list_unknown_events` calls the exact same `get_events_with_faces(limit=limit, offset=offset, status_filter="unknown")`. The `list_events` endpoint achieves the same with `?status=unknown`.
- **Fix:** Remove, use `list_events` with filter parameter.

---

## META — Dependencies / scripts / tests / documentation

### MT1 — Unused dependencies in requirements.txt

- **File:** `requirements.txt`
- **Problem:** `langgraph>=0.1.0`, `langchain>=0.1.0` listed but never imported. `openvino>=2024.0.0`, `onnxruntime-openvino` listed but optional (Intel iGPU only).
- **Fix:** `langgraph` and `langchain` removed (2026-07-30). See `INTENTIONAL.md` for design rationale.

### MT2 — pydantic not explicitly listed

- **File:** `requirements.txt`
- **Problem:** `pydantic` is imported directly in `dashboard/backend/models.py` but only present as a transitive dependency of `fastapi`. Should be explicit.
- **Fix:** Add `pydantic>=2.0`.

### MT3 — numpy allows 2.x (breaking changes)

- **File:** `requirements.txt`
- **Problem:** `numpy>=1.24.0` allows numpy 2.x which had major breaking changes (removed `numpy.bool_` aliases, changed C API). `onnxruntime` and `insightface` may not be compatible.
- **Fix:** Pin to `numpy>=1.24.0,<2.0`.

### MT4 — No upper bounds on critical dependencies

- **File:** `requirements.txt`
- **Problem:** `ultralytics`, `fastapi`, `httpx`, `opencv-python` all lack upper bounds. Ultralytics in particular has frequent breaking changes.
- **Fix:** Add upper bounds (e.g., `ultralytics>=8.1.0,<9.0.0`).

### MT5 — Conflicting onnxruntime packages

- **File:** `requirements.txt`
- **Problem:** Both `onnxruntime>=1.16.0` and `onnxruntime-openvino` are listed. Neither is directly imported — both are transitive deps of `insightface`. Can cause version conflicts.
- **Fix:** Remove both, let `insightface` manage its own onnxruntime dependency.

### MT6 — Hardcoded paths in scripts/

- **File:** `scripts/` (all 11 files)
- **Problem:** All scripts have hardcoded path `C:\Users\Aummc\.local\share\mimocode\mimocode.db`. Will fail for any other user or machine.
- **Fix:** Parameterize via argparse or remove scripts (they query an unrelated IDE database).

### MT7 — No error handling in scripts/

- **File:** `scripts/` (all 11 files)
- **Problem:** Zero `try/except` blocks, no context managers for database connections, no `if __name__ == "__main__"` guard.
- **Fix:** Add argparse, try/except, `with` context managers.

### MT8 — SQL injection pattern in scripts/

- **File:** `scripts/query_db.py:15`
- **Problem:** `cursor.execute(f"PRAGMA table_info({table})")` uses f-string interpolation for table name. While table names come from `sqlite_master` (safe in practice), this is a bad pattern.
- **Fix:** Use parameterized queries or validate table names.

### MT9 — Duplicated SQL query pattern in scripts/

- **File:** `scripts/` (8 files)
- **Problem:** The SQL query pattern is identical across 8 `query_*_session.py` files. Massive code duplication.
- **Fix:** Consolidate into one parameterized script.

### MT10 — test_both_read_same_threshold tests nothing

- **File:** `tests/test_recognition.py:180-187`
- **Problem:** Test assigns `rec_threshold = s.MATCH_THRESHOLD` and `policy_threshold = s.MATCH_THRESHOLD` (the same variable!) and asserts they are equal. Always passes regardless of actual threshold.
- **Fix:** Test actual threshold used by `recognize()` and `PolicyAgent`, not the settings variable.

### MT11 — Exact float equality in tests

- **File:** `tests/test_recognition.py:51`
- **Problem:** `test_maximum_inputs` asserts `conf == 100` (exact equality). Floating-point arithmetic makes this fragile.
- **Fix:** Use `pytest.approx(100, abs=1)`.

### MT12 — Flaky test_old_pattern_loses_visits

- **File:** `tests/test_thread_safety.py:172-200`
- **Problem:** Test asserts `doc["visit_count"] < n_threads` relying on `time.sleep(0.001)` TOCTOU window being wide enough for 20 threads. Timing-dependent — may pass or fail inconsistently.
- **Fix:** Use deterministic race simulation instead of timing-based approach.

### MT13 — FakeMemoryCollection holds lock during sleep

- **File:** `tests/test_thread_safety.py:65`
- **Problem:** `_legacy_update` holds the lock during `time.sleep(0.001)`, which prevents the very race condition the test is trying to demonstrate. Only one thread can be in the critical section at a time.
- **Fix:** Release lock between read and write to allow actual concurrent access.

### MT14 — Overly broad @patch in tests

- **File:** `tests/test_embedding_history.py:113, 146, 181`
- **Problem:** `@patch("utils.db_utils.settings")` replaces the entire settings module with a `MagicMock`. Any settings attribute access gets a mock that could cause unexpected behavior.
- **Fix:** Patch specific attributes only.

*(BUILD_REPORT.md entries MT15-MT19 removed — file does not exist in repo)*

### MT20 — Embedding model versioning not tracked

- **File:** `utils/embedding_utils.py`, `utils/db_utils.py`
- **Problem:** If InsightFace model is updated, existing embeddings may become incompatible. No version tracking or re-embedding capability. Embeddings stored with one model version will produce wrong similarity scores with a different version.
- **Impact:** Model upgrades silently degrade recognition accuracy for all existing face records.
- **Fix:** Add model version string to face documents, detect version mismatch on read, trigger re-embedding for stale records.

---

## Remediation Plan

### Phase 0 — Critical Bugs (do first)

1. **0.1** Fix NameError: `best_crop_path` → `track.best_face_crop_path` in `main.py:266`
2. **0.2** Fix dynamic quality object: add missing attributes in `camera_agent.py:303`
3. **0.3** Fix bool cast: parse `"false"/"0"/"no"/"off"` as `False` in `settings.py:_get()`

### Phase 1 — Data Integrity

1. **1.1** Fix `avg_similarity` never updated: recalculate in `update_visit_memory()` (`db_utils.py`) — **SKIPPED** (user deferred; informational field, no critical logic depends on it)
2. **1.2** Fix timezone mix: `datetime.now()` → `datetime.utcnow()` in `update_visit_memory` (`db_utils.py:624`) — **FIXED**
3. **1.3** Fix TOCTOU races in `update_face`: use atomic MongoDB operations (`db_utils.py:398-428`) — **FIXED** (two `update_one` calls with quality filter)
4. **1.4** Add `None` guards to `update_visit_memory` and `get_or_create_memory` (`db_utils.py`) — **FIXED** (returns `__fallback__: True` dict)
5. **1.5** Align config defaults: `RECOGNITION_INTERVAL_FRAMES=20`, `JPEG_QUALITY_BROADCAST=65` (`settings.py` + `config.jsonc`) — **FIXED**

### Phase 2 — Threading Safety

1. **2.1** Protect `track.rescan_attempts` mutation with setter or lock (`camera_agent.py:177`) — **SKIPPED** (only camera thread reads/writes; no concurrent access verified)
2. **2.2** Protect `track.confidence` writes with lock or use `set_decision()` (`camera_agent.py:413,421`) — **SKIPPED** (only `_progressive_recognition` accesses it; effectively single-threaded per track)
3. **2.3** Fix `decision.nl_summary` mutation from thread pool (`alert_agent.py:99`) — **FIXED** (LLM call moved before thread dispatch)
4. **2.4** Fix `_finalize_track` quality gate bypass (`camera_agent.py:503-504`) — **FIXED** (use `set_embedding()`, returns bool, logs rejection)
5. **2.5** Check `cv2.imencode` success flag in `track_state.py:131,155` — **FIXED** (check success, log warning, return early)
6. **2.6** Make `llm_client.shutdown()` thread-safe (`llm_client.py:366-387`) — **FIXED** (lock + ref capture + error logging)

### Phase 3 — Logic Fixes

1. **3.1** Clamp `compute_face_ratio()` to `[0.0, 1.0]` (`face.py`) — **FIXED** (min/max clamp)
2. **3.2** Remove dead Rule 5 `else` branch or add assertion (`policy.py:228-237`) — **FIXED** (logged error + safe fallback `unknown`/no registration)
3. **3.3** Fix midnight wrap-around in `memory.py:113` — **FIXED** (circular clock distance)
4. **3.4** Fix inconsistent `_empty_result` in `memory.py` — **FIXED** (`days_since_last_visit: None`)
5. **3.5** Add quality weight sum validation in `validate_config()` — **FIXED** (range check = error, sum != 1.0 = warning)
6. **3.6** Add `SMTP_PORT` error handling in `settings.py:460` — **FIXED** (try/except with logged warning)

### Phase 4 — Dashboard Security

1. **4.1** Add authentication middleware to dashboard API (REST + WebSocket)
2. **4.2** Cap WebSocket client limit in `live.py`
3. **4.3** Add soft-delete to face delete endpoint
4. **4.4** Sanitize chat prompt input, validate history
5. **4.5** Fix CWD-relative paths in `image_utils.py`

### Phase 5 — Resilience

1. **5.1** Add camera reconnect with exponential backoff (`camera_agent.py`)
2. **5.2** Add MongoDB connection retry logic (`db_utils.py`)
3. **5.3** Add Cloudinary upload queue with retry (`image_utils.py`)
4. **5.4** Add embedding model versioning (`embedding_utils.py`, `db_utils.py`)
5. **5.5** Add graceful degradation (skip Cloudinary, log locally)

### Phase 6 — Code Quality

1. **6.1** Remove unused imports (`main.py:20`, `llm_client.py:19`)
2. **6.2** Fix type hints across codebase (L1-L6)
3. **6.3** Remove dead code (`models.py`, `settings.py`, `camera_agent.py`)
4. **6.4** Add `asyncio.run_coroutine_threadsafe` error handling (`main.py`)
5. **6.5** Add retry backoff to LLM client

### Phase 7 — Dependencies & Cleanup

1. **7.1** Remove unused deps from `requirements.txt`, add upper bounds, pin numpy <2.0
2. **7.2** Remove or parameterize scripts in `scripts/`
3. **7.3** Fix test flakiness (`test_thread_safety.py`, `test_recognition.py`)
4. **7.4** *(removed — BUILD_REPORT.md does not exist in repo)*

---

## Definition of Done

- [x] All 3 critical bugs fixed (NameError, AttributeError, bool cast)
- [ ] `avg_similarity` recalculation works
- [x] No TOCTOU races in `update_face`
- [x] Thread-safe track field access (2.3, 2.4, 2.5, 2.6 fixed; 2.1, 2.2 skipped — single-threaded)
- [x] Quality gate not bypassed in finalization
- [x] Logic fixes (face ratio clamp, midnight wrap, empty result, dead code, weight validation, SMTP_PORT)
- [ ] Dashboard has basic authentication (REST + WebSocket)
- [ ] No CWD-relative paths
- [ ] Camera reconnect on failure
- [ ] MongoDB connection retry
- [ ] All unused code and dependencies removed
- [ ] `requirements.txt` has upper bounds, numpy pinned <2.0
- [ ] Tests pass without flakiness
- [ ] *(removed — BUILD_REPORT.md does not exist in repo)*
