from __future__ import annotations
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config.status import Status


class DedupStatus(Enum):
    MERGED = "merged"
    NEW = "new"
    FAILED = "failed"


@dataclass
class DedupResult:
    status: DedupStatus
    person_id: Optional[str] = None
    reason: str = ""
    similarity: float = 0.0


@dataclass
class Track:
    track_id: str
    first_seen: float
    last_seen: float
    person_box: tuple
    byte_track_id: int = 0       # raw ByteTrack ID (before composite_id wrapping)
    generation: int = 0           # reuse generation (0 = first person with this bt_id)
    best_face_crop: Optional[np.ndarray] = None
    best_face_score: float = 0.0
    best_full_frame: Optional[np.ndarray] = None
    best_frame_jpeg: Optional[bytes] = None  # Compressed JPEG (~50KB vs ~921KB raw)
    fallback_frame_jpeg: Optional[bytes] = None  # First-frame fallback (always captured, quality-independent)
    is_masked: bool = False
    embedding: Optional[list] = None
    embedding_det_score: float = 0.0  # Detection score for quality-gated updates
    decision: Optional[int] = None
    confidence: int = 0  # Highest confidence seen — only upgrades, never downgrades
    person_name: Optional[str] = None  # Name from match result
    person_name_similarity: float = 0.0  # Highest sim for name — only upgrades, never downgrades
    alerted: bool = False
    last_alert_time: float = 0.0  # Timestamp of last alert dispatch (for per-track cooldown)
    image_url: Optional[str] = None  # Cloudinary URL (set after upload)
    last_recognition_quality: float = 0.0     # face quality at last recognition
    last_recognition_status: int = Status.UNKNOWN  # Status enum value
    last_recognition_time: float = 0.0        # timestamp of last recognition (for time-based rescan)
    rescan_attempts: int = 0                  # how many re-scan attempts used
    expired_reported: bool = False
    # True once this expired track has been reported by get_expired_tracks().
    # Prevents duplicate finalization while recognition is still in flight.
    _finalized: bool = False
    # Set once by mark_finalized_once(). Survives TrackState removal, so stale
    # recognition workers holding a reference cannot re-finalize.
    pending_recognition: Optional[dict] = None  # Phase 2.1: Recognition Agent output
    pending_match_result: Optional[MatchResult] = None  # Full match result from progressive recognition
    pending_memory_context: Optional[dict] = None  # Cached memory context from progressive recognition
    total_frames_seen: int = 0
    frames_with_detectable_face: int = 0
    face_detected_once: bool = False
    max_face_ratio: float = 0.0
    best_face_ratio: float = 0.0
    best_face_crop_path: str = ""
    best_person_crop_jpeg: Optional[bytes] = None  # JPEG bytes of person crop (for display)
    visibility: str = "unknown"
    max_track_secs: float = 300.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def is_expired(self, timeout_secs: float) -> bool:
        return (time.time() - self.last_seen) > timeout_secs

    def is_max_lifetime_exceeded(self) -> bool:
        return (time.time() - self.first_seen) > self.max_track_secs

    def update_confidence_if_higher(self, new_confidence: int) -> bool:
        with self._lock:
            if new_confidence > self.confidence:
                self.confidence = new_confidence
                return True
            return False

    def mark_alerted_once(self) -> bool:
        with self._lock:
            if not self.alerted:
                self.alerted = True
                return True
            return False

    def mark_finalized_once(self) -> bool:
        with self._lock:
            if not self._finalized:
                self._finalized = True
                return True
            return False

    def snapshot(self) -> TrackSnapshot:
        with self._lock:
            return TrackSnapshot(
                track_id=self.track_id,
                first_seen=self.first_seen,
                last_seen=self.last_seen,
                person_box=self.person_box,
                byte_track_id=self.byte_track_id,
                generation=self.generation,
                best_face_crop=np.array(self.best_face_crop, copy=True) if self.best_face_crop is not None else None,
                best_face_score=self.best_face_score,
                best_full_frame=np.array(self.best_full_frame, copy=True) if self.best_full_frame is not None else None,
                best_frame_jpeg=self.best_frame_jpeg,
                fallback_frame_jpeg=self.fallback_frame_jpeg,
                is_masked=self.is_masked,
                embedding=list(self.embedding) if self.embedding is not None else None,
                embedding_det_score=self.embedding_det_score,
                decision=self.decision,
                confidence=self.confidence,
                person_name=self.person_name,
                alerted=self.alerted,
                image_url=self.image_url,
                last_recognition_quality=self.last_recognition_quality,
                last_recognition_status=self.last_recognition_status,
                last_recognition_time=self.last_recognition_time,
                rescan_attempts=self.rescan_attempts,
                expired_reported=self.expired_reported,
                pending_recognition=self.pending_recognition,
                pending_match_result=self.pending_match_result,
                pending_memory_context=self.pending_memory_context,
                total_frames_seen=self.total_frames_seen,
                frames_with_detectable_face=self.frames_with_detectable_face,
                face_detected_once=self.face_detected_once,
                max_face_ratio=self.max_face_ratio,
                best_face_ratio=self.best_face_ratio,
                best_face_crop_path=self.best_face_crop_path,
                best_person_crop_jpeg=self.best_person_crop_jpeg,
                visibility=self.visibility,
                max_track_secs=self.max_track_secs,
            )


@dataclass
class TrackSnapshot:
    track_id: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    person_box: tuple = ()
    byte_track_id: int = 0
    generation: int = 0
    best_face_crop: Optional[np.ndarray] = None
    best_face_score: float = 0.0
    best_full_frame: Optional[np.ndarray] = None
    best_frame_jpeg: Optional[bytes] = None
    fallback_frame_jpeg: Optional[bytes] = None
    is_masked: bool = False
    embedding: Optional[list] = None
    embedding_det_score: float = 0.0
    decision: Optional[int] = None
    confidence: int = 0
    person_name: Optional[str] = None
    alerted: bool = False
    image_url: Optional[str] = None
    last_recognition_quality: float = 0.0
    last_recognition_status: int = Status.UNKNOWN
    last_recognition_time: float = 0.0
    rescan_attempts: int = 0
    expired_reported: bool = False
    pending_recognition: Optional[dict] = None
    pending_match_result: Optional[MatchResult] = None
    pending_memory_context: Optional[dict] = None
    total_frames_seen: int = 0
    frames_with_detectable_face: int = 0
    face_detected_once: bool = False
    max_face_ratio: float = 0.0
    best_face_ratio: float = 0.0
    best_face_crop_path: str = ""
    best_person_crop_jpeg: Optional[bytes] = None  # JPEG bytes of person crop (for display)
    visibility: str = "unknown"
    max_track_secs: float = 300.0


@dataclass
class QualityResult:
    blur_score: float
    brightness: float
    face_area: int
    is_valid: bool
    overall_score: float

    @classmethod
    def invalid(cls) -> "QualityResult":
        """Return a default invalid quality result.

        Use when quality assessment cannot be performed (e.g., empty crop or invalid ROI).
        """
        return cls(blur_score=0.0, brightness=0.0, face_area=0, is_valid=False, overall_score=0.0)


@dataclass
class MatchResult:
    person_id: Optional[str] = None
    name: Optional[str] = None
    role: Optional[str] = None
    tags: list = field(default_factory=list)
    similarity_score: float = 0.0
    image_url: Optional[str] = None
    matched: bool = False
    verified: bool = False
    alert_level: str = "low"
    second_best_similarity: Optional[float] = None
    margin: Optional[float] = None
    candidate_count: int = 0
    all_candidates: list = field(default_factory=list)  # [{name, similarity}] top-N from vector search


@dataclass
class DecisionResult:
    status: int = Status.UNKNOWN
    alert_level: str = "none"
    person_id: Optional[str] = None
    name: Optional[str] = None
    reason: str = ""
    should_alert: bool = False
    should_register: bool = False
    nl_summary: str = ""


@dataclass
class RecognitionResult:
    """Output from the Recognition Agent.

    Instead of a simple if/else on similarity, the agent considers
    multiple factors to make a structured decision.
    """
    status: int = Status.UNKNOWN        # Status enum value
    confidence: float = 0.0           # 0-100, how confident in the decision
    similarity: float = 0.0           # raw cosine similarity from ArcFace
    face_quality: float = 0.0         # quality score (0-1)
    is_masked: bool = False           # mask detected
    track_duration: float = 0.0       # seconds since first seen
    reason: str = ""                  # human-readable explanation

    def to_dict(self) -> dict:
        """Convert to dictionary for easy serialization."""
        return {
            "status": self.status,
            "confidence": round(self.confidence, 1),
            "similarity": round(self.similarity, 4),
            "face_quality": round(self.face_quality, 3),
            "is_masked": self.is_masked,
            "track_duration": round(self.track_duration, 1),
            "reason": self.reason,
        }
