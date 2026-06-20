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
    is_masked: bool = False
    embedding: Optional[list] = None
    decision: Optional[str] = None
    alerted: bool = False
    last_recognition_frame: int = 0
    pending_embedding: Optional[list] = None
    pending_match: Optional[dict] = None
    total_frames_seen: int = 0
    frames_with_detectable_face: int = 0
    face_detected_once: bool = False
    max_face_ratio: float = 0.0
    best_face_ratio: float = 0.0
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


@dataclass
class DecisionResult:
    status: str = "unknown"
    alert_level: str = "none"
    person_id: Optional[str] = None
    name: Optional[str] = None
    reason: str = ""
    should_alert: bool = False
    should_register: bool = False
