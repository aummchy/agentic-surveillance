import numpy as np
from pipeline.models import QualityResult
from utils.image_utils import compute_blur_score, compute_brightness
from config import settings


def compute_quality(face_crop: np.ndarray) -> QualityResult:
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

    blur_valid = blur_raw >= settings.QUALITY_BLUR_MIN
    bright_valid = settings.QUALITY_BRIGHTNESS_MIN <= brightness_raw <= settings.QUALITY_BRIGHTNESS_MAX
    area_valid = face_area >= settings.QUALITY_FACE_AREA_MIN

    blur_norm = min(max(blur_raw - settings.QUALITY_BLUR_MIN, 0) / (settings.QUALITY_BLUR_MAX - settings.QUALITY_BLUR_MIN), 1.0)
    bright_norm = brightness_raw / 255.0
    area_norm = min(max(face_area - settings.QUALITY_FACE_AREA_MIN, 0) / (settings.QUALITY_AREA_MAX - settings.QUALITY_FACE_AREA_MIN), 1.0)

    overall_score = (blur_norm * settings.QUALITY_WEIGHT_BLUR
                     + bright_norm * settings.QUALITY_WEIGHT_BRIGHT
                     + area_norm * settings.QUALITY_WEIGHT_AREA)

    is_valid = blur_valid and bright_valid and area_valid

    return QualityResult(
        blur_score=blur_raw,
        brightness=brightness_raw,
        face_area=face_area,
        is_valid=is_valid,
        overall_score=overall_score
    )
