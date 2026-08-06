from __future__ import annotations

import time
import numpy as np
import structlog
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

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
    crop_detect_ms: float = 0.0
    fallback_detect_ms: float = 0.0
    detect_ms: float = 0.0
    quality_ms: float = 0.0
    embed_ms: float = 0.0
    db_ms: float = 0.0
    memory_ms: float = 0.0
    recog_policy_ms: float = 0.0
    total_ms: float = 0.0
    fallback_used: bool = False
    faces_on_crop: int = 0
    person_crop_shape: str = ""


@dataclass
class PipelineResult:
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


def _embedding_to_list(embedding):
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return list(embedding)


class RecognitionPipeline:
    def __init__(self, matching_fn: Callable = None,
                 recognition_agent=None,
                 memory_agent=None,
                 decide_fn: Callable = None):
        from agents.matching_agent import run_matching_from_embedding
        from agents.recognition import RecognitionAgent
        from agents.memory import MemoryAgent
        from agents.policy import decide

        self._matching_fn = matching_fn or run_matching_from_embedding
        self._recognition_agent = recognition_agent or RecognitionAgent()
        self._memory_agent = memory_agent or MemoryAgent()
        self._decide_fn = decide_fn or decide

    def run(self, frame: np.ndarray, track: Track) -> PipelineResult:
        t_total = time.perf_counter()

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

    def _detect_face(self, frame: np.ndarray, track: Track):
        """Detect the best face in the track's person box.

        Tries crop detection first, then optional full-frame fallback.
        Returns _FaceResult or None if no usable face found.
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

    def _expand_person_box(self, box, frame_shape):
        """Expand person box by configured ratio to ensure face is within crop."""
        bx1, by1, bx2, by2 = map(int, box)
        w, h = bx2 - bx1, by2 - by1
        ex, ey = int(w * settings.PERSON_BOX_EXPANSION_RATIO), int(h * settings.PERSON_BOX_EXPANSION_RATIO)
        return (max(0, bx1 - ex), max(0, by1 - ey),
                min(frame_shape[1], bx2 + ex), min(frame_shape[0], by2 + ey))

    def _select_best_face(self, crop_faces, frame_faces, track):
        """Select best face detection, preferring crop over full-frame.

        Returns (best_detection_dict, detected_in_person_crop) or (None, False).
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

    def _compute_frame_bbox(self, best, crop_box, detected_in_person_crop):
        """Convert face bbox from detection-local to full-frame coordinates."""
        if not best.get("bbox"):
            return None
        fx1, fy1, fx2, fy2 = best["bbox"]
        if detected_in_person_crop:
            return (fx1 + crop_box[0], fy1 + crop_box[1],
                    fx2 + crop_box[0], fy2 + crop_box[1])
        return (fx1, fy1, fx2, fy2)

    def _extract_face_crop(self, frame, person_crop, frame_bbox):
        """Extract face crop from frame using computed bounding box."""
        if frame_bbox:
            fx1, fy1, fx2, fy2 = frame_bbox
            return frame[fy1:fy2, fx1:fx2]
        return person_crop

    def _assess_quality(self, face_crop: np.ndarray) -> QualityResult:
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

    def _build_embedding(self, face_det: "_FaceResult", track: Track):
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
        return self._decide_fn(track, match, recognition, memory)


class _FaceResult:
    __slots__ = ("best", "face_crop", "person_crop", "frame_bbox", "face_ratio",
                 "detected_in_person_crop", "metrics")

    def __init__(self, best, face_crop, person_crop, frame_bbox, face_ratio,
                 detected_in_person_crop, metrics):
        self.best = best
        self.face_crop = face_crop
        self.person_crop = person_crop
        self.frame_bbox = frame_bbox
        self.face_ratio = face_ratio
        self.detected_in_person_crop = detected_in_person_crop
        self.metrics = metrics


class _EmbedResult:
    __slots__ = ("embedding", "match_or_none", "metrics")

    def __init__(self, embedding, match, metrics):
        self.embedding = embedding
        self.match_or_none = match
        self.metrics = metrics