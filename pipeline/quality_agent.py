import numpy as np
from pipeline.models import QualityResult
from utils.image_utils import compute_blur_score, compute_brightness
from config import settings


def _safe_norm(value: float, lo: float, hi: float) -> float:
    """Normalize value to [0, 1] with safety against bad config ranges."""
    if hi <= lo:
        return 0.0
    return min(max((value - lo) / (hi - lo), 0.0), 1.0)


def compute_quality(face_crop: np.ndarray) -> QualityResult:
    """Compute face quality score from a face crop.

    Returns QualityResult with:
      - is_valid: whether the face passes minimum validity gates
      - overall_score: weighted quality score 0.0-1.0
    """
    if face_crop is None or face_crop.size == 0:
        return QualityResult(
            blur_score=0.0,
            brightness=0.0,
            face_area=0,
            is_valid=False,
            overall_score=0.0
        )

    blur_raw = compute_blur_score(face_crop)
    brightness_raw = compute_brightness(face_crop)
    h, w = face_crop.shape[:2]
    face_area = h * w

    # ── Validity gates (separate from scoring) ────────────────
    blur_valid = blur_raw >= settings.QUALITY_VALID_BLUR_MIN
    bright_valid = (settings.QUALITY_VALID_BRIGHTNESS_MIN
                    <= brightness_raw
                    <= settings.QUALITY_VALID_BRIGHTNESS_MAX)
    area_valid = face_area >= settings.QUALITY_VALID_FACE_AREA_MIN
    is_valid = blur_valid and bright_valid and area_valid

    # ── Scoring normalization ─────────────────────────────────
    # Blur: 40=weak, 350=maxed
    blur_norm = _safe_norm(blur_raw, settings.QUALITY_BLUR_MIN, settings.QUALITY_BLUR_MAX)

    # Brightness: center-radius model (peak at 145, radius 110)
    if settings.QUALITY_BRIGHTNESS_RADIUS <= 0:
        bright_norm = 0.0
    else:
        bright_norm = 1.0 - min(
            abs(brightness_raw - settings.QUALITY_BRIGHTNESS_CENTER)
            / settings.QUALITY_BRIGHTNESS_RADIUS,
            1.0
        )

    # Area: 1500=weak, 10000=maxed
    area_norm = _safe_norm(face_area, settings.QUALITY_FACE_AREA_MIN, settings.QUALITY_AREA_MAX)

    # ── Weighted composite ────────────────────────────────────
    overall_score = min(max(
        settings.QUALITY_WEIGHT_BLUR * blur_norm
        + settings.QUALITY_WEIGHT_BRIGHT * bright_norm
        + settings.QUALITY_WEIGHT_AREA * area_norm,
        0.0
    ), 1.0)

    return QualityResult(
        blur_score=blur_raw,
        brightness=brightness_raw,
        face_area=face_area,
        is_valid=is_valid,
        overall_score=overall_score
    )
