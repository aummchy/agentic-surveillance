# References — Open-Source Recognition Pipelines

> Study these before changing our pipeline. Companion to [[GOAL]] §0 (pipeline first).

Grouped by how close they are to our stack.

## 1. Closest to our stack (CV pipeline end-to-end)

| Project | Pipeline | Why read it |
|---------|----------|-------------|
| [serengil/deepface](https://github.com/serengil/deepface) (23.5k★) | detect → align → normalize → represent → verify | The cleanest explanation of the 5 canonical face-recognition stages; 11 wrapped models incl. **Buffalo_L** (our InsightFace pack); `register/search` on Mongo/pgvector ≈ our Atlas vector search; `stream()` = real-time video with 5-frame stability rule; **`anti_spoofing=True` → `is_real`** (see §5) |
| [deepinsight/insightface](https://github.com/deepinsight/insightface) | SCRFD detect → landmark align → ArcFace embed → cosine match | The library we actually use — read `Face` analyze path + sample code to see det/rec exactly |
| [AdityaThakur-DA25M004/face-recognition-pipeline](https://github.com/AdityaThakur-DA25M004/face-recognition-pipeline-) | RetinaFace → 5-point align → AdaFace embed → FAISS search → duplicate/new decision | Same shape as ours incl. vector index + open-set decision |
| [taxaceaee/Face-Reconization](https://github.com/taxaceaee/Face-Reconization) | SCRFD → align → ArcFace embed → gallery cosine search → threshold calibration → unknown rejection | Good reference for open-set thresholds + unknown rejection (our UNCERTAIN/UNKNOWN logic) |
| [FuaadBashi/cctv-person-reidentification](https://github.com/FuaadBashi/cctv-person-reidentification) | YOLO → DeepSORT-style track → appearance embedding → cosine ReID | YOLO + tracking + identity telemetry (`tracks.csv`, `events.jsonl`) — mirrors our track/event logging |

## 2. Production-grade services (architecture patterns)

| Project | What it shows |
|---------|---------------|
| [exadel-inc/CompreFace](https://github.com/exadel-inc/CompreFace) (~8.3k★) | Dockerized face-recognition **server**: detection / recognition / verification REST APIs, embedding storage, plugins (mask, age, gender) — how to package recognition as a clean service |
| [blakeblackshear/frigate](https://github.com/blakeblackshear/frigate) | Full open-source NVR: motion gate → object detection → tracking → **zones** → face recognition (FaceNet/ArcFace) → events → recordings → alerts. Best reference for our planned **zone tracking** + operational event flow |

## 3. Multi-camera / re-ID research-style

| Project | What it shows |
|---------|---------------|
| [meenakshi4567/Person-Tracking-and-Identification-across-multiple-camera-field-of-views](https://github.com/meenakshi4567/Person-Tracking-and-Identification-across-multiple-camera-field-of-views) | YOLO → ByteTrack → crop → TorchReID → cross-camera global IDs (step-by-step scripts) |
| [sbreuers/detta](https://github.com/sbreuers/detta) | Detection + tracking feeding analysis modules with per-track temporal smoothing — how to attach analyzers cleanly onto track IDs |

## 4. Video-first (high-star) — building blocks for cross-frame verification

No high-star repo ships "video verification" as a single feature — it's always glue:
**frame loop → per-frame embedding → aggregate (mean / N-of-M vote) → threshold.** That is exactly what we do (one decision per track, progressive passes every 20 frames). Reference implementations of the pieces:

| Project | Stars | Piece it gives us |
|---------|-------|-------------------|
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | ~61k★ | YOLO detect + built-in `model.track()` (ByteTrack) — our `pipeline/tracker.py` wraps this |
| [roboflow/supervision](https://github.com/roboflow/supervision) | ~50k★ | frame iteration utilities (`sv.process_video`), track visualization, dataset tools — clean pattern for a video test harness |
| [blakeblackshear/frigate](https://github.com/blakeblackshear/frigate) | ~36k★ | NVR: per-track identity verified across frames (FaceNet) before emitting a "recognized person" event |
| [serengil/deepface](https://github.com/serengil/deepface) | 23.5k★ | `stream()` — only trusts an identity after the face is stable for 5 consecutive frames; simplest published cross-frame rule |
| [FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack) | ~6.6k★ | the tracker we already use (via ultralytics) — read the associator for how IDs survive occlusion |

Small dedicated cross-frame repos (0–few ★, code patterns only): `murukeshp/Immich-Face-Recognition-From-Video`, `ishaans04/Face-Recognition`, `DhavalMCA/Face-Recognition-System` (FrameVoter: N-of-M frame votes before accepting an ID).

**Cross-frame aggregation pattern** (generic, from these repos):

```text
per frame:  detect → quality gate → embed
per track:  collect embeddings over time
aggregate:  mean embedding  OR  N-of-M frames over threshold
decide:     once per track (verify against gallery), not per frame
```

## 5. Anti-spoofing / deepfake (threat + defenses)

Deepfakes/photos/screens can fool face verification: print attack, phone-screen attack, or a fake video injected into the stream. Our quality gates (blur/brightness/area) measure *image quality*, not *realness* — a sharp screen photo passes.

| Defense | Reference |
|---------|-----------|
| Real/fake classifier, one flag | [serengil/deepface](https://github.com/serengil/deepface) `anti_spoofing=True` → `is_real` on any task |
| Underlying model | [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) |
| Motion check (free, uses what we have) | require the face to change across ≥ N frames before verifying — blocks static photo/screen; we already collect frame history per track |
| Stream integrity | only trust the configured `CAMERA_SOURCE`; treat external/injected streams as untrusted |
| Liveness challenge (interactive auth only) | blink/head-turn — needed for dashboard login, not passive CCTV |

No single detector is reliable against the latest diffusion fakes — defense in depth (motion + classifier + stream integrity) beats any one check. Not implemented yet; candidate hardening after pipeline stability (GOAL §0).

## 6. How to use them

1. Read **DeepFace** first (conceptual stages) → then our `pipeline/recognition_pipeline.py` side by side.
2. Read **InsightFace** sample code to confirm our det → align → embed assumptions.
3. Skim **Frigate** docs/config for zone + event design before implementing zones.
4. Use **CompreFace** REST API locally (`docker-compose up`) as a behavior benchmark for gallery/unknown decisions.
5. Compare our cross-frame logic with **DeepFace `stream()`** (5-frame rule) and the §4 glue pattern — validate the 20-frame progressive interval.
6. Before any authentication feature: wire a §5 anti-spoofing gate (motion check first, `deepface anti_spoofing` second).
