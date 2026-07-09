from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Track:
    track_id: str
    first_seen: float
    last_seen: float
    person_box: tuple
    best_face_crop: Optional[np.ndarray] = None
    best_face_score: float = 0.0
    best_full_frame: Optional[np.ndarray] = None
    best_frame_jpeg: Optional[bytes] = None  # Compressed JPEG (~50KB vs ~921KB raw)
    is_masked: bool = False
    embedding: Optional[list] = None
    decision: Optional[str] = None
    person_name: Optional[str] = None  # Name from match result
    alerted: bool = False
    image_url: Optional[str] = None  # Cloudinary URL (set after upload)
    last_recognition_frame: int = 0
    pending_recognition: Optional[dict] = None  # Phase 2.1: Recognition Agent output
    pending_match_result: Optional[MatchResult] = None  # Full match result from progressive recognition
    pending_memory_context: Optional[dict] = None  # Cached memory context from progressive recognition
    total_frames_seen: int = 0
    frames_with_detectable_face: int = 0
    face_detected_once: bool = False
    max_face_ratio: float = 0.0
    best_face_ratio: float = 0.0
    best_face_crop_path: str = ""
    cached_embedding: Optional[list] = None  # Last embedding searched against Atlas
    visibility: str = "unknown"
    max_track_secs: float = 300.0

    def is_expired(self, timeout_secs: float) -> bool:
        return (time.time() - self.last_seen) > timeout_secs

    def is_max_lifetime_exceeded(self) -> bool:
        return (time.time() - self.first_seen) > self.max_track_secs


@dataclass
class QualityResult:
    blur_score: float
    brightness: float
    face_area: int
    is_valid: bool
    overall_score: float


@dataclass
class EmbeddingResult:
    embedding: Optional[np.ndarray] = None
    face_detected: bool = False
    detection_score: float = 0.0
    embedding_score: float = 0.0
    bbox: Optional[tuple] = None
    is_masked: bool = False
    error: Optional[str] = None


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


@dataclass
class DecisionResult:
    status: str = "unknown"
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
    status: str = "unknown"           # "known" | "unknown" | "uncertain"
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
