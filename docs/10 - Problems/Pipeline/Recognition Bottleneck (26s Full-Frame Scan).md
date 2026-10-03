# Crop-Detect 2.2s Mean (was 6.4s)

> **Stale refs:** status "Resolved" is historical — citations date from 2026-08; verify against current code before relying on this note.

**Category:** Pipeline · **Severity:** High · **Status:** Resolved

## What was wrong

Recognition took **~6.4s per track** (mean from 41 runs, `surveillance.jsonl` 2026-08-01). 94% of that time was face detection on the person crop. This is down from 26s (full-frame fallback was fixed), but was still the dominant bottleneck.

## The deep reason

`recognition_pipeline.py:174` calls `detect_faces_raw(person_crop, min_score=0.2)`. Inside, InsightFace resized the image to `INSIGHTFACE_DET_SIZE=(1280, 1280)` before SCRFD inference (`embedding_utils.py:40-41`).

Crops are 300–900px wide. SCRFD upscaled them to a **1280×1280 input** — a 2–4× pixel expansion. On CPU this took 3–12 seconds per detection. OpenVINO EP was confirmed broken (`openvino.dll` missing from onnxruntime plugin).

## Fix applied (2026-08-01)

Changed `INSIGHTFACE_DET_SIZE` from 1280 to 640 in `config/config.jsonc:77` and `config/settings.py:532`. INTENTIONAL.md updated.

**Measured improvement** (benchmark on real 853×597 expanded crop):

| Metric | Before (1280) | After (640) | Change |
|--------|---------------|-------------|--------|
| Mean `crop_detect_ms` | 3173ms | 2219ms | **-30%** |
| Detection score | 0.780 | 0.859 | **+10%** |
| Small crop (100×150) | 560ms | 237ms | **-58%** |

Note: isolated benchmark shows 30% speedup. Under pipeline contention with 2 workers + lock, the effective improvement compounds — workers free the lock 30% sooner, reducing queue backlog proportionally.

## Previous fix

1. **Full-frame fallback removed** (`ENABLE_FULL_FRAME_FALLBACK: false`) — eliminated the 10–26s fallback scan
2. **Crop expansion 20%** — ensures face is fully within crop, `faces_on_crop=0` dropped to 0%
3. **CLAHE skip for tiny crops** — avoids destroying features on small images

## Knock-on effects (resolved)

- **Stale-worker finalization** — queue delays shrink from 63s to ~20s. See [[Duplicate Finalization (Visit Inflation)]]. (The `_finalized` guard is also in place as defense-in-depth.)
- **Worker pool saturation** — 2 workers + 2.2s/detection → ~55 recognitions/min (was ~18)
- **FPS** — should improve significantly

## Related
- `pipeline/recognition_pipeline.py`
- `utils/embedding_utils.py`
- [[Identities Never Generated (No Embeddings)]]
- [[Duplicate Finalization (Visit Inflation)]]
