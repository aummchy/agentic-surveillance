"""Single-track recognition pass: detect -> quality -> embed -> match -> memory -> recognize -> decide.

This module owns no state. It reads a Track's accumulated state under the
track's lock, does the heavy lifting (InsightFace inference, one MongoDB
vector search, memory read, recognition classification, policy decision),
and returns a fresh PipelineResult. Writing results back onto the Track is
the caller's job (see CameraAgent._handle_pipeline_result).

Threading: instances are created once and shared. CameraAgent submits run()
to its recognition ThreadPoolExecutor, so several different tracks may be
inside run() concurrently on different threads. Nothing here mutates shared
state; the only shared inputs are config settings (read-only) and the
InsightFace singleton.

Skip contract: run() may bail out early and return a PipelineResult with a
non-"success" skip_reason instead of a full result. Callers must check
skip_reason before consuming the other fields.
"""

from __future__ import annotations

import time
import numpy as np
import structlog
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import settings
from config.status import Status
from pipeline.models import Track, QualityResult, MatchResult, DecisionResult
from pipeline.quality_agent import compute_quality
from pipeline.quality_agent import compute_face_ratio
from utils.image_utils import crop_person
from utils.embedding_utils import get_insightface

logger = structlog.get_logger(__name__)


@dataclass
class RecognitionMetrics:
    """Wall-clock timings (milliseconds) for one recognition pass.

    Populated piecewise: _detect_face fills the detection/quality fields,
    _build_embedding replaces the object with an embed/db-only one, and run()
    copies the detection values across (see the metrics handoff noted there).
    Total is measured in run() itself.
    """

    # Face detection stage
    crop_detect_ms: float = 0.0
    fallback_detect_ms: float = 0.0
    detect_ms: float = 0.0
    # Face quality stage
    quality_ms: float = 0.0
    # Embedding + vector search stage
    embed_ms: float = 0.0
    db_ms: float = 0.0
    # Memory + recognition/policy stages
    memory_ms: float = 0.0
    recog_policy_ms: float = 0.0
    total_ms: float = 0.0
    # Detection diagnostics, not timings
    fallback_used: bool = False
    faces_on_crop: int = 0
    person_crop_shape: str = ""


@dataclass
class PipelineResult:
    """Everything one recognition pass produced, plus why it may be incomplete.

    skip_reason is the contract with the caller: "success" means every field
    below was populated; any other value means the pass bailed out early and
    only the fields mentioned in that branch are meaningful. Values are the
    SkipReason members from config/status.py (they are strings at runtime).
    """

    decision: Optional[DecisionResult] = None
    match: Optional[MatchResult] = None
    recognition: Optional[Dict[str, Any]] = None
    memory: Optional[Dict[str, Any]] = None
    embedding: Optional[list] = None
    quality: Optional[QualityResult] = None
    face_crop: Optional[np.ndarray] = None
    person_crop: Optional[np.ndarray] = None
    face_bbox: Optional[tuple] = None
    face_ratio: float = 0.0
    det_score: float = 0.0
    detected_in_person_crop: bool = False
    is_masked: bool = False
    metrics: RecognitionMetrics = field(default_factory=RecognitionMetrics)
    skip_reason: str = ""


def _embedding_to_list(embedding: Any) -> list:
    """Convert a numpy embedding (or anything iterable) into a plain list.

    MongoDB/JSON layers take lists; InsightFace hands back ndarrays.
    """
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return list(embedding)


class RecognitionPipeline:
    """Orchestrates the full recognition pipeline: detect → quality → embed → match → decide.

    Coordinates face detection, quality assessment, embedding generation,
    vector search matching, memory lookup, and policy decision into a
    single PipelineResult.

    Lifecycle: one instance is built by CameraAgent and reused for the whole
    session; the agent functions below are themselves stateless.

    Dependencies are injectable for tests (matching_fn, recognition_agent,
    memory_agent, decide_fn); the production defaults are imported inside
    __init__ rather than at module level to avoid import cycles.
    """

    def __init__(self, matching_fn: Optional[Callable] = None,
                 recognition_agent: Optional[Any] = None,
                 memory_agent: Optional[Any] = None,
                 decide_fn: Optional[Callable] = None) -> None:
        from agents.matching_agent import run_matching_from_embedding
        from agents.recognition import RecognitionAgent
        from agents.memory import MemoryAgent
        from agents.policy import decide

        self._matching_fn = matching_fn or run_matching_from_embedding
        self._recognition_agent = recognition_agent or RecognitionAgent()
        self._memory_agent = memory_agent or MemoryAgent()
        self._decide_fn = decide_fn or decide

    def run(self, frame: np.ndarray, track: Track) -> PipelineResult:
        """Execute the full recognition pipeline for a single track.

        Stage order: detect face -> assess quality -> embed + vector search
        -> memory lookup -> recognition -> policy decision.

        Returns a PipelineResult containing decision, match, recognition,
        memory, embedding, quality, and timing metrics.

        Four early exits, each with its own skip_reason; on those paths only
        the fields mentioned are populated:
          - "high_confidence"  a prior pass already matched strongly, so this
                               pass is redundant (nothing populated)
          - "no_face"          no usable face in the person box or frame
          - "low_quality"      face failed the validity gates (quality set)
          - "embedding_failed" no embedding could be produced (quality set)

        On "success", every field is populated. This method mutates nothing:
        the caller decides which results to write back onto the Track.
        """
        t_total = time.perf_counter()

        # Read under the track lock, then work with a local copy — another
        # worker for a different purpose may rewrite this field mid-pass.
        with track._lock:
            pmr = track.pending_match_result
        if pmr and pmr.similarity_score > settings.HIGH_CONFIDENCE_SIMILARITY:
            logger.debug("skip_recognition_high_confidence",
                         track_id=track.track_id,
                         similarity=pmr.similarity_score)
            return PipelineResult(skip_reason="high_confidence")

        face_det = self._detect_face(frame, track)
        if face_det is None:
            return PipelineResult(skip_reason="no_face")

        t_quality = time.perf_counter()
        quality = self._assess_quality(face_det.face_crop)
        quality_ms = round((time.perf_counter() - t_quality) * 1000, 1)
        face_det.metrics.quality_ms = quality_ms

        if not quality.is_valid:
            logger.debug("skip_recognition_low_quality",
                         track_id=track.track_id,
                         blur=quality.blur_score,
                         brightness=round(quality.brightness, 1),
                         area=quality.face_area)
            return PipelineResult(skip_reason="low_quality", quality=quality,
                                  metrics=face_det.metrics)

        embed_result = self._build_embedding(face_det, track)
        if embed_result is None:
            return PipelineResult(skip_reason="embedding_failed",
                                  quality=quality, metrics=face_det.metrics)

        match = embed_result.match_or_none
        metrics = embed_result.metrics
        # Metrics handoff: _build_embedding built a fresh, embed/db-only
        # RecognitionMetrics, so the detection timings recorded on
        # face_det.metrics have to be carried across by hand here.
        metrics.crop_detect_ms = face_det.metrics.crop_detect_ms
        metrics.fallback_detect_ms = face_det.metrics.fallback_detect_ms
        metrics.fallback_used = face_det.metrics.fallback_used
        metrics.faces_on_crop = face_det.metrics.faces_on_crop
        metrics.person_crop_shape = face_det.metrics.person_crop_shape
        metrics.quality_ms = face_det.metrics.quality_ms

        t_memory = time.perf_counter()
        memory = self._lookup_memory(track, match)
        metrics.memory_ms = round((time.perf_counter() - t_memory) * 1000, 1)

        t_recog = time.perf_counter()
        recognition = self._run_recognition(
            track, match, quality, memory, face_det
        )
        metrics.recog_policy_ms = round((time.perf_counter() - t_recog) * 1000, 1)

        decision = self._run_policy(track, match, recognition, memory)

        metrics.total_ms = round((time.perf_counter() - t_total) * 1000, 1)

        return PipelineResult(
            decision=decision,
            match=match,
            recognition=recognition,
            memory=memory,
            embedding=embed_result.embedding,
            quality=quality,
            face_crop=face_det.face_crop,
            person_crop=face_det.person_crop,
            face_bbox=face_det.frame_bbox,
            face_ratio=face_det.face_ratio,
            det_score=face_det.best.get("det_score", 0.0),
            detected_in_person_crop=face_det.detected_in_person_crop,
            is_masked=face_det.best.get("is_masked", False),
            metrics=metrics,
            skip_reason="success",
        )

    def _detect_face(self, frame: np.ndarray, track: Track) -> Optional["_FaceResult"]:
        """Detect the best face in the track's person box.

        Tries crop detection first, then optional full-frame fallback.
        Returns _FaceResult or None if no usable face found.

        "Best" means highest det_score: detect_faces_raw sorts its results
        descending by det_score, so index 0 is already the best. Returns None
        as well when the best detection's score is below
        EMBEDDING_DET_SCORE_MIN — good enough to log is not good enough to
        embed.
        """
        with track._lock:
            box = track.person_box
        if box is None or len(box) != 4:
            logger.debug("no_person_box", track_id=track.track_id)
            return None

        crop_box = self._expand_person_box(box, frame.shape)
        person_crop = crop_person(frame, crop_box)
        if person_crop.size == 0:
            logger.debug("empty_person_crop", track_id=track.track_id)
            return None

        app = get_insightface()

        t_before = time.perf_counter()
        crop_faces = app.detect_faces_raw(person_crop, min_score=settings.DET_SCORE_RELAXED)
        t_after_crop = time.perf_counter()

        crop_has_embedding_quality = any(f["det_score"] >= settings.EMBEDDING_DET_SCORE_MIN for f in crop_faces)
        fallback_used = not crop_has_embedding_quality
        if fallback_used and settings.ENABLE_FULL_FRAME_FALLBACK:
            frame_faces = app.detect_faces_raw(frame, min_score=settings.DET_SCORE_RELAXED)
        else:
            frame_faces = []
        t_after_fallback = time.perf_counter()

        crop_detect_ms = round((t_after_crop - t_before) * 1000, 1)
        fallback_detect_ms = round((t_after_fallback - t_after_crop) * 1000, 1) if fallback_used else 0.0
        person_crop_shape = f"{person_crop.shape[1]}x{person_crop.shape[0]}" if person_crop.size > 0 else "empty"
        faces_on_crop = len(crop_faces)

        best, detected_in_person_crop = self._select_best_face(crop_faces, frame_faces, track)

        if best is None:
            return None

        frame_bbox = self._compute_frame_bbox(best, crop_box, detected_in_person_crop)

        face_ratio = 0.0
        if frame_bbox:
            face_ratio = compute_face_ratio(frame_bbox, box)

        if best["det_score"] < settings.EMBEDDING_DET_SCORE_MIN:
            logger.debug("embedding_score_below_threshold",
                         track_id=track.track_id,
                         score=best["det_score"],
                         threshold=settings.EMBEDDING_DET_SCORE_MIN)
            return None

        face_crop = self._extract_face_crop(frame, person_crop, frame_bbox)

        metrics = RecognitionMetrics(
            crop_detect_ms=crop_detect_ms,
            fallback_detect_ms=fallback_detect_ms,
            detect_ms=crop_detect_ms + fallback_detect_ms,
            fallback_used=fallback_used,
            faces_on_crop=faces_on_crop,
            person_crop_shape=person_crop_shape,
        )

        return _FaceResult(
            best=best,
            face_crop=face_crop,
            person_crop=person_crop,
            frame_bbox=frame_bbox,
            face_ratio=face_ratio,
            detected_in_person_crop=detected_in_person_crop,
            metrics=metrics,
        )

    def _expand_person_box(self, box: tuple, frame_shape: tuple) -> tuple:
        """Expand person box by configured ratio to ensure face is within crop.

        Clips the expanded box to the frame boundary, so the result is always
        a valid (x1, y1, x2, y2) inside the image.
        """
        bx1, by1, bx2, by2 = map(int, box)
        w, h = bx2 - bx1, by2 - by1
        ex, ey = int(w * settings.PERSON_BOX_EXPANSION_RATIO), int(h * settings.PERSON_BOX_EXPANSION_RATIO)
        return (max(0, bx1 - ex), max(0, by1 - ey),
                min(frame_shape[1], bx2 + ex), min(frame_shape[0], by2 + ey))

    def _select_best_face(self, crop_faces: List[dict], frame_faces: List[dict], track: Track) -> Tuple[Optional[dict], bool]:
        """Select best face detection, preferring crop over full-frame.

        Returns (best_detection_dict, detected_in_person_crop) or (None, False).

        Preferring the crop is deliberate: a face found inside the person box
        belongs to the tracked person with certainty, whereas a full-frame hit
        may belong to someone standing behind them. Within either list the
        first entry is already the highest-scoring one (detect_faces_raw
        sorts by det_score descending), so no max() is needed here.
        """
        if crop_faces:
            best = crop_faces[0]
            logger.debug("face_found_crop", track_id=track.track_id, score=best["det_score"])
            return best, True
        if frame_faces:
            best = frame_faces[0]
            logger.debug("face_found_in_full_frame", track_id=track.track_id, score=best["det_score"])
            return best, False
        logger.debug("no_face_anywhere", track_id=track.track_id)
        return None, False

    def _compute_frame_bbox(self, best: dict, crop_box: tuple, detected_in_person_crop: bool) -> Optional[tuple]:
        """Convert face bbox from detection-local to full-frame coordinates.

        Detections on the person crop are relative to that crop's origin and
        must be offset; detections from the full-frame pass are already in
        frame coordinates. Returns None when the detection carries no bbox.
        """
        if not best.get("bbox"):
            return None
        fx1, fy1, fx2, fy2 = best["bbox"]
        if detected_in_person_crop:
            return (fx1 + crop_box[0], fy1 + crop_box[1],
                    fx2 + crop_box[0], fy2 + crop_box[1])
        return (fx1, fy1, fx2, fy2)

    def _extract_face_crop(self, frame: np.ndarray, person_crop: np.ndarray, frame_bbox: Optional[tuple]) -> np.ndarray:
        """Extract face crop from frame using computed bounding box.

        Falls back to the whole person crop when no frame-space bbox exists,
        so callers always get a non-empty array to run quality scoring on.
        """
        if frame_bbox:
            fx1, fy1, fx2, fy2 = frame_bbox
            return frame[fy1:fy2, fx1:fx2]
        return person_crop

    def _assess_quality(self, face_crop: np.ndarray) -> QualityResult:
        """Assess face quality (blur, brightness, area) for a face crop.

        Returns a QualityResult with is_valid flag and quality scores.
        """
        if face_crop.size > 0:
            quality = compute_quality(face_crop)
            logger.debug("face_quality",
                         score=quality.overall_score,
                         valid=quality.is_valid,
                         blur=quality.blur_score,
                         brightness=quality.brightness,
                         area=quality.face_area)
        else:
            quality = QualityResult.invalid()
        return quality

    def _build_embedding(self, face_det: "_FaceResult", track: Track) -> Optional["_EmbedResult"]:
        """Generate embedding from face crop and run vector search matching.

        Returns an _EmbedResult with embedding, match, and timing metrics.
        Note that the RecognitionMetrics returned here is a fresh, embed/db
        only object — run() copies the detection timings onto it afterwards.
        """
        t_embed = time.perf_counter()
        embedding_list = _embedding_to_list(face_det.best["embedding"])
        embed_ms = round((time.perf_counter() - t_embed) * 1000, 1)

        t_db = time.perf_counter()
        match = self._matching_fn(embedding_list, track_id=track.track_id)
        db_ms = round((time.perf_counter() - t_db) * 1000, 1)

        metrics = RecognitionMetrics(
            embed_ms=embed_ms,
            db_ms=db_ms,
        )

        return _EmbedResult(embedding=embedding_list, match=match, metrics=metrics)

    def _lookup_memory(self, track: Track, match: Optional[MatchResult]) -> Dict[str, Any]:
        """Look up visit history and memory context for a matched person.

        Returns empty dict if no match or person_id is available, and a
        skip marker instead of a lookup when the match is already
        high-confidence (re-reading visit history adds nothing there).
        """
        if not match or not match.matched or not match.person_id:
            return {}

        if match.similarity_score > settings.HIGH_CONFIDENCE_SIMILARITY:
            logger.debug("skip_memory_high_confidence",
                         track_id=track.track_id,
                         similarity=match.similarity_score)
            return {"skip_reason": "high_confidence", "confidence_boost": 0}

        return self._memory_agent.run({
            "person_id": match.person_id,
            "camera_id": settings.CAMERA_ID,
            "similarity": match.similarity_score,
            "status": Status.KNOWN if match.matched else Status.UNKNOWN,
        })

    def _run_recognition(self, track: Track, match: Optional[MatchResult],
                         quality: QualityResult,
                         memory: Dict[str, Any],
                         face_det: "_FaceResult") -> Dict[str, Any]:
        """Run the recognition agent to classify identity and confidence.

        Returns a dict with status, confidence, is_masked, and other fields.

        track_duration and face_quality feed the confidence formula; top2 and
        margin come from the match so the agent can judge separation between
        the best and second-best candidate.
        """
        track_duration = time.time() - track.first_seen
        face_quality = quality.overall_score if quality and quality.overall_score > 0 else None
        return self._recognition_agent.run({
            "similarity": match.similarity_score if match else 0.0,
            "is_masked": face_det.best.get("is_masked", False),
            "face_quality": face_quality,
            "track_duration": track_duration,
            "memory_context": memory,
            "top2": match.second_best_similarity if match and match.matched else None,
            "margin": match.margin if match and match.margin is not None else None,
            "name": match.name if match and match.matched else None,
            "track_id": track.track_id,
        })

    def _run_policy(self, track: Track, match: Optional[MatchResult],
                    recognition: Dict[str, Any],
                    memory: Dict[str, Any]) -> DecisionResult:
        """Run the policy agent to make a final decision.

        Returns a DecisionResult with status, alert_level, and action flags.

        The policy owns alerting and registration, not the recognition agent;
        it receives the recognition output as an input (see agents/policy.py).
        """
        return self._decide_fn(track, match, recognition, memory)


class _FaceResult:
    """Per-pass face detection outcome: what was found and where.

    Private transport type between _detect_face and _build_embedding. Uses
    __slots__ because it is allocated once per recognition pass.
    """
    __slots__ = ("best", "face_crop", "person_crop", "frame_bbox", "face_ratio",
                 "detected_in_person_crop", "metrics")

    def __init__(self, best: dict, face_crop: np.ndarray, person_crop: np.ndarray,
                 frame_bbox: Optional[tuple], face_ratio: float,
                 detected_in_person_crop: bool, metrics: RecognitionMetrics) -> None:
        self.best = best
        self.face_crop = face_crop
        self.person_crop = person_crop
        self.frame_bbox = frame_bbox
        self.face_ratio = face_ratio
        self.detected_in_person_crop = detected_in_person_crop
        self.metrics = metrics


class _EmbedResult:
    """Per-pass embedding outcome: the vector, its match, and the timings.

    Private transport type for _build_embedding's return; __slots__ for the
    same allocation reason as _FaceResult.
    """
    __slots__ = ("embedding", "match_or_none", "metrics")

    def __init__(self, embedding: list, match: Optional[MatchResult], metrics: RecognitionMetrics) -> None:
        self.embedding = embedding
        self.match_or_none = match
        self.metrics = metrics