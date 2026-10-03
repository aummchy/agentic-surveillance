# Current Architecture — Recognition (Sections 8–14)

> Part of [`CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md). Section numbers follow the shared scheme: this file holds sections 8–14.
>
> **Source of truth:** current source code and tests. Facts that could not be proven from the source are marked `UNKNOWN — NEEDS VERIFICATION`.
>
> **Last verified:** 2026-10-03

**Navigation:** [Hub](../CURRENT_ARCHITECTURE.md) · [Components](Current%20Architecture%20-%20Components.md) · [Recognition](Current%20Architecture%20-%20Recognition.md) · [Decision Flow](Current%20Architecture%20-%20Decision%20Flow.md) · [Platform](Current%20Architecture%20-%20Platform.md) · [Code State](Current%20Architecture%20-%20Code%20State.md)

---

# 8. Current Recognition Architecture

The current system has **two recognition-related execution paths**.

This section is intentionally explicit because this is one of the main architectural issues in the codebase (see Section 27).

---

## 8.1 Progressive Recognition Path

### Entry

`agents/camera_agent.py:504` `_progressive_recognition` (runs on a recognition-pool worker).

### Flow

```text
Active Track + live frame
  |
  v
_scheduled by _maybe_schedule_recognition (camera_agent.py:316)
  |    every RECOGNITION_INTERVAL_FRAMES (20) frames,
  |    and only if quality improved by MIN_QUALITY_IMPROVEMENT (0.10)
  v
pipeline/recognition_pipeline.py :: RecognitionPipeline.run(frame, track)
  |
  +--> Face detection        (_detect_face)
  +--> Quality evaluation    (validity gates + score)
  +--> Embedding             (quality/det-score gated)
  +--> Vector matching       (matching_agent, injectable)
  +--> Memory                (MemoryAgent)
  +--> Recognition           (RecognitionAgent)
  +--> Policy decision       (policy.decide)
  |
  v
PipelineResult → camera_agent._handle_pipeline_result (:550)
  |    the SOLE write-back site for progressive results
  v
Track fields / pending_* caches (upgrade-only setters)
```

### Purpose

Allows recognition to occur **while a track is still active**, using the live camera frame.

### Important behavior

* Two gates before any work starts: frame cadence (every 20 frames) and quality improvement (≥ 0.10 over `last_recognition_quality`).
* `_should_skip_recognition` (`camera_agent.py:376`) also enforces `MAX_RESCAN_ATTEMPTS` — and, as its own docstring documents, it **mutates `track.rescan_attempts` as a side effect** (`:414`), i.e. a predicate with a side effect.
* Duplicate-run guards: `_recognizing_tracks` membership + `pending_recognition is not None` check.
* Upgrade-only rules prevent lower-quality/low-confidence results from replacing stronger stored results.

---

## 8.2 Track Finalization Recognition Path

### Entry

`agents/track_processor.py:151` `process()` (runs on one of the **two** `track_queue` consumer threads).

### Current flow

```text
Expired/ended Track (snapshot taken from TrackState)
    |
    v
image URLs resolved          (resolve_track_image_url / person crop)
    |
    v
_run_matching                (:264)  → matching_agent vector search
    |
    v
_run_recognition_and_memory  (:291)
    |    fresh_match = (snap.pending_match_result is None)   (:306)
    |    uses pending_recognition / pending_memory_context
    |    when the progressive pass already ran
    |    fresh RecognitionAgent() constructed per track (:319)
    v
policy.decide()              → DecisionResult
    |
    v
auto-register → record visit → dispatch alert → log event → broadcast
```

### Why it exists separately

A finalizing track usually has **no live frame left** — `RecognitionPipeline.run(frame, track)` requires a frame, which the expired track no longer provides. The finalization path therefore re-runs matching/recognition/memory/policy from the track **snapshot** plus any `pending_*` results the progressive pass cached, avoiding duplicate DB calls where possible.

### Architectural concern

This path **reimplements** matching → recognition → memory → policy — functionality that also lives inside `pipeline/recognition_pipeline.py`. The system currently has two implementations of parts of the recognition flow.

The exact behavioral differences between the two paths must be documented line-by-line before consolidation (that comparison is the stated precondition for Phase 4 item 1, `run_final()`, in `plan.md`).

---

# 9. Recognition Pipeline

## `pipeline/recognition_pipeline.py` (417 lines)

### Responsibility

Orchestrates one complete recognition pass: face detect → validity gates → quality scoring → embedding (gated) → vector search → memory → confidence scoring → policy decision. Returns `PipelineResult` (with `RecognitionMetrics` timing/decision data).

### Current conceptual flow

```text
Input track + frame
      |
      v
Face Detection  (_detect_face, _select_best_face, _extract_face_crop)
      |
      v
Face Quality    (_assess_quality → validity gates + score)
      |
      v
Embedding       (_build_embedding, quality/det-score gated)
      |
      v
Vector Match    (matching_fn, defaults to matching_agent)
      |
      v
Memory          (_lookup_memory)
      |
      v
Recognition     (_run_recognition → RecognitionAgent)
      |
      v
Policy Decision (_run_policy → policy.decide)
```

### Important facts

* **Called by:** `camera_agent._progressive_recognition` only. It mutates nothing itself — the caller writes results back to the `Track`.
* **Threading:** runs entirely on one recognition worker; reads track state under lock, writes nothing.
* **Quality selection:** `_select_best_face` picks between faces found in the person crop vs. the full frame; `_compute_frame_bbox` / `_expand_person_box` map crops back to frame coordinates.

### Known concern

`SkipReason` (a `StrEnum`) and plain `str` are both returned for skip reasons — the skip-reason types are inconsistent. Documented only; no change implied.

### Important rule

This is the clearest single implementation of the complete recognition sequence — but `track_processor.py` performs the same sequence independently (Section 8.2).

### Architectural concern

There is currently **no single exclusive owner** for all recognition execution paths.

---

# 10. Face Quality

## `pipeline/quality_agent.py`

### Responsibility

Determines whether a detected face is suitable for recognition, and scores usable faces.

### Current validity gates (reject before any embedding work)

* Blur: Laplacian variance ≥ `QUALITY_VALID_BLUR_MIN` (40).
* Brightness: V-channel mean in [35, 255].
* Face area: ≥ `QUALITY_VALID_FACE_AREA_MIN` (1200 px²).

### Quality score (weighted composite, 0–1)

```text
blur       50%  →  min(Laplacian_var / 350, 1.0)
brightness 25%  →  center-radius model (peak 145, radius 110)
area       25%  →  min(area / 10000, 1.0)
```

### Purpose

Quality gating prevents poor observations from entering recognition at all, and prevents poor-quality embeddings from replacing better stored evidence (embedding overwrite requires a det-score improvement ≥ `EMBEDDING_DET_SCORE_IMPROVEMENT_MIN`, 0.05).

### Also here

`compute_face_ratio()` — the geometric face-size helper — is defined in this module (it was inlined from the deleted `pipeline/face.py`).

---

# 11. Face Embedding

## `utils/embedding_utils.py`

### Responsibility

Provides the InsightFace face-detection + embedding functionality.

### Important implementation detail

InsightFace is loaded as a **singleton** (`SCRFD` detection + `ArcFace` embedding, `buffalo_l` model) rather than being repeatedly initialized per frame. CPU execution provider. CLAHE preprocessing runs before detection; mask detection uses the geometric landmark heuristic (lower/upper face ratio < `MASK_RATIO_THRESHOLD` 0.3), not a classifier.

### Output

A 512-dimension L2-normalized embedding vector used for identity matching.

### Invariant

Never reload this model inside a per-frame loop — the singleton is load-once.

---

# 12. Identity Matching

## `agents/matching_agent.py`

### Responsibility

Performs identity matching from a face embedding: `run_matching_from_embedding()`.

### Storage/search implementation

* MongoDB Atlas `$vectorSearch` against the `faces` collection, index `vector_index` (512-d, cosine) — `utils/db_search.py`.
* Python-side cosine fallback exists when Atlas search is unavailable.
* **Score conversion invariant:** Atlas returns `vectorSearchScore = (1 + cosine) / 2`; always convert back `raw_cosine = (atlas_score * 2) - 1` before comparing to `MATCH_THRESHOLD` (0.45).

### Callers

* `pipeline/recognition_pipeline.py` (injectable `matching_fn`, defaults to this module).
* `agents/track_processor.py:264` `_run_matching`.

### Important distinction

Matching produces an **identity candidate** with a similarity and a margin over the second-best candidate. A match alone does not mean the person is verified — recognition (Section 13) and policy (Section 15) consume it downstream.

### Thread/side effects

Read-only DB query; no mutation; runs synchronously on whichever worker calls it.

---

# 13. Recognition / Identity Classification

## `agents/recognition.py`

### Responsibility

Combines available identity evidence and classifies the recognition result: `RecognitionAgent` / `recognize()`.

### Relevant signals

* Similarity to the best match, match margin over the second best.
* Face quality.
* Track information (duration/observations).
* Memory (visit history, typical hours).
* Mask/partial visibility.

### Output

A recognition status (from `config/status.py`) plus a confidence (Section 14), consumed by `policy.decide()`.

### Instances

* Held by `RecognitionPipeline` (one per pipeline object).
* `track_processor.py:319` constructs a **fresh `RecognitionAgent()` per track** — the only per-track construction site in the system.

---

# 14. Confidence Scoring

## `agents/scoring.py`

### Current scoring model

Weighted normalization over five components (weights are `settings.WEIGHT_*`, effective values verified in code and in the formula header logged at startup):

```text
base     = 0.65×similarity + 0.15×quality + 0.10×track + 0.05×memory + 0.05×margin
adjusted = base × (1 − 0.15×mask)          # mask penalty
```

Each component is normalized to [0, 1] before weighting (`normalize_cosine`, etc.).

### Important notes

* The exact weights above are **verified** from `agents/scoring.py:68-77, 99-101` (they come from config/`settings.py` defaults: 0.65 / 0.15 / 0.10 / 0.05 / 0.05).
* A full per-computation tabular breakdown is written to `logs/calculation.log`.
* **The confidence "never downgrade" gate does NOT live here** — it lives in the callers (`TrackState` setters + `camera_agent` write-back). Remember this when reading `scoring.py` in isolation.
* Out-of-range raw cosine values are logged (`confidence_raw_cosine_out_of_range`) and clamped.
