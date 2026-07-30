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

**File:** `agents/camera_agent.py:438-443`

---

## Camera Handling

**Decision:** Fixed-delay reconnect (1s release + 2s retry), not exponential backoff.

**Why:** Indoor WiFi cameras (phone IP, RTSP stream) either work or are offline. Exponential backoff is for transient network failures — camera disconnection is typically a hard failure (app killed, network lost) requiring manual intervention. Fixed delay is sufficient.

**File:** `agents/camera_agent.py:133-148`

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

## Embeddings

**Decision 1:** Recency-only embedding history (no quality weighting).

**Why:** Quality-weighted trimming requires storing quality scores per embedding and sorting on every update. The current approach (keep last 25, compute mean) is simple and sufficient for short-term appearance consistency. Long-term appearance changes (aging, glasses) are not handled by either approach without re-embedding.

**File:** `utils/db_utils.py:402`

**Decision 2:** No embedding model versioning.

**Why:** InsightFace model updates are rare and breaking changes are announced in release notes. The `embedding_model: "arcface"` field exists as a placeholder. Version-specific re-embedding is future work.

**File:** `utils/db_utils.py:351`
