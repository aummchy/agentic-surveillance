# INTENTIONAL.md — Design Decisions (Not Bugs)

Documented trade-offs that look like limitations but are deliberate choices.

---

## Mask Detection

**Decision:** Geometric heuristic (lower_face/upper_face ratio < 0.3), not ML classifier.

**Why:** MobileNetV2 mask classifier adds model loading, inference overhead, and training data requirements. The geometric heuristic catches surgical masks reliably. False positives from beards/scarves/lighting are acceptable for an alert system — false positives are less dangerous than false negatives.

**File:** `utils/embedding_utils.py:75-90`

---

## Alert Behavior

**Decision 1:** Masked unknown starts at "medium", escalates to "high" after 30s loitering.

**Why:** A masked person walking through briefly (delivery, visitor) is not a threat. A masked person lingering (loitering) is. The escalation timer distinguishes transient vs deliberate presence.

**File:** `agents/policy.py:261-277`

**Decision 2:** Only CRITICAL (blacklist) alerts fire during active tracks. All others defer to finalization.

**Why:** Early recognition passes may produce low similarity scores (partial face, angle, blur). Dispatching HIGH/MEDIUM alerts during the track risks false positives for verified users when the first recognition pass is uncertain. Blacklist alerts are high-confidence and always dispatched immediately.

**File:** `agents/recognition_worker.py:240` (`handle_decision_and_alert`)

---

## Camera Handling

**Decision:** Fixed-delay reconnect (1s release + 2s retry), not exponential backoff.

**Why:** Indoor WiFi cameras (phone IP, RTSP stream) either work or are offline. Exponential backoff is for transient network failures — camera disconnection is typically a hard failure (app killed, network lost) requiring manual intervention. Fixed delay is sufficient.

**File:** `agents/camera_agent.py:155` (`_loop` reconnect block)

---

## External Services

**Decision 1:** No MongoDB connection retry.

**Why:** MongoDB Atlas cloud SLA (99.95%+). Connection failures are typically config errors (wrong URI, network down), not transient. A retry loop on a bad URI wastes resources. The existing `MongoClient` handles transient errors via PyMongo's built-in retryable reads/writes.

**File:** `utils/db_utils.py:30`

**Decision 2:** No Cloudinary upload retry.

**Why:** Upload failure is non-critical. The face crop is always saved locally to `captures/`. Cloudinary is archival, not operational. If Cloudinary is down, local files accumulate until it recovers.

**File:** `utils/image_utils.py`

---

## Frame Processing

**Decision:** Static 1280×720 resolution, no dynamic frame dropping.

**Why:** Indoor surveillance at fixed camera distance. 1280×720 is sufficient for face recognition at 3-5 meters. Dynamic resolution adds complexity for a scenario (CPU overload) that doesn't occur in a dedicated surveillance machine. If load becomes an issue, lower resolution in config rather than dynamically.

**File:** `config/config.jsonc:20-21`

---

## Storage

**Decision 1:** No image cleanup.

**Why:** Face crops are small (~50KB each) and used for operator review. Automatic cleanup risks deleting images before operators verify them. Disk management is an operator responsibility. If needed, add a cron job or TTL index in MongoDB.

**Decision 2:** Single-camera only.

**Why:** Project scope is single-camera indoor surveillance. Multi-camera requires cross-camera track correlation, centralized state, and unified event stream — significant architecture changes for a future requirement.

**File:** `config/config.jsonc:166`

---

## Match Threshold

**Decision:** `MATCH_THRESHOLD = 0.45` (cosine similarity).

**Why:** 0.45 is the standard ArcFace threshold used in production systems (InsightFace reference, MegaFace benchmarks, academic literature). ArcFace was designed and evaluated on controlled-capture datasets (LFW, MegaFace, IJB-C) where face quality is high — frontal, well-lit, aligned. In those conditions, genuine pairs score >0.70 and impostor pairs score <0.20, leaving a wide margin.

Surveillance footage is fundamentally different: unconstrained pose, variable lighting, motion blur, lower resolution. Cross-person cosine similarities of 0.50–0.65 are common under indoor lighting. A threshold of 0.45 deliberately prioritizes recall (catching genuine matches) over precision (rejecting impostors). This is acceptable because:
- The confidence formula (weighted normalization) independently down-ranks low-similarity matches via the `rec_status` system
- The policy layer gates actions (alerts, registration) on `rec_status`, not on the raw match
- False-positive name assignments are cosmetic — the status badge (KNOWN / UNCERTAIN / UNKNOWN) reflects actual confidence

Raising the threshold would reduce false-positive name assignments but would also increase false negatives for genuine matches under poor lighting — a worse outcome for a security system.

**File:** `config/config.jsonc:14`, `config/settings.py:428-430`

---

## Embeddings

**Decision 1:** Recency-only embedding history (no quality weighting).

**Why:** Quality-weighted trimming requires storing quality scores per embedding and sorting on every update. The current approach (keep last 25, compute mean) is simple and sufficient for short-term appearance consistency. Long-term appearance changes (aging, glasses) are not handled by either approach without re-embedding.

**File:** `utils/db_utils.py:402`

---

## Face Detection Size

**Decision:** `INSIGHTFACE_DET_SIZE=640`. Changed from 1280.

**Why:** SCRFD with det_size=1280 upscales small person crops to 1280×1280 buffer before inference — the compute cost is always ~1280² regardless of crop size. Measured mean: 3.2s per detection on CPU, 6.4s under queue contention (2 workers). Changing to 640 gives 30% faster inference (2.2s mean) and higher detection scores (0.780→0.859 on real crops). Full-frame fallback is disabled, so det_size only affects person-crop detection where faces are relatively large. OpenVINO EP confirmed broken (`openvino.dll` missing from onnxruntime plugin).

**File:** `config/config.jsonc:77`

**Decision 2:** No embedding model versioning.

**Why:** InsightFace model updates are rare and breaking changes are announced in release notes. The `embedding_model: "arcface"` field exists as a placeholder. Version-specific re-embedding is future work.

**File:** `utils/db_utils.py:351`
