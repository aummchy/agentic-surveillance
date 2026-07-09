"""
Pure scoring functions for recognition confidence.

All tunables come from config.settings — no hardcoded values here.
"""

from config import settings


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_cosine(raw_cosine: float) -> float:
    """Map raw cosine [SIM_NORM_MIN..SIM_NORM_MAX] -> [0..1]."""
    return clip(
        (raw_cosine - settings.SIM_NORM_MIN) / (settings.SIM_NORM_MAX - settings.SIM_NORM_MIN),
        0.0, 1.0
    )


def normalize_quality(face_quality: float | None) -> float:
    """Map quality to [0..1]. None or 0.0 -> default fallback."""
    if face_quality is None or face_quality <= 0.0:
        return settings.DEFAULT_FACE_QUALITY
    return clip(face_quality, 0.0, 1.0)


def normalize_track_duration(track_seconds: float) -> float:
    """Map duration to [0..1]. Saturation at TRACK_SATURATION_SECS."""
    return clip(track_seconds / settings.TRACK_SATURATION_SECS, 0.0, 1.0)


def normalize_memory(memory_boost: float) -> float:
    """Map memory boost [0..MEMORY_NORM_MAX] -> [0..1]."""
    return clip(memory_boost, 0.0, settings.MEMORY_NORM_MAX) / settings.MEMORY_NORM_MAX


def normalize_mask(is_masked: bool) -> float:
    """Boolean mask -> penalty coefficient in [0..1]."""
    return 1.0 if is_masked else 0.0


def normalize_margin(margin: float | None) -> float:
    """Map margin [0..MARGIN_NORM_MAX] -> [0..1]. None -> 0.5 (neutral)."""
    if margin is None:
        return 0.5
    return clip(margin / settings.MARGIN_NORM_MAX, 0.0, 1.0)


def compute_confidence(
    raw_cosine: float,
    face_quality: float | None,
    track_seconds: float,
    memory_boost: float,
    is_masked: bool,
    margin: float | None = None
) -> int:
    """Compute confidence score 1-100 from normalized components."""
    sim_norm = normalize_cosine(raw_cosine)
    quality_norm = normalize_quality(face_quality)
    track_norm = normalize_track_duration(track_seconds)
    memory_norm = normalize_memory(memory_boost)
    margin_norm = normalize_margin(margin)
    mask_norm = normalize_mask(is_masked)

    base = (
        settings.WEIGHT_SIMILARITY * sim_norm +
        settings.WEIGHT_QUALITY * quality_norm +
        settings.WEIGHT_TRACK * track_norm +
        settings.WEIGHT_MEMORY * memory_norm +
        settings.WEIGHT_MARGIN * margin_norm
    )

    adjusted = base * (1.0 - settings.MASK_PENALTY_MAX * mask_norm)
    return int(round(1 + 99 * clip(adjusted, 0.0, 1.0)))


def is_match(raw_cosine: float) -> bool:
    """Binary identity gate: raw_cosine >= MATCH_THRESHOLD."""
    return raw_cosine >= settings.MATCH_THRESHOLD


def confidence_status(confidence: int, matched: bool) -> str:
    """Map confidence + match gate -> status string.

    matched=True  + confidence >= KNOWN_MIN   -> "known"
    matched=True  + confidence >= UNCERTAIN_MIN -> "uncertain"
    matched=True  + confidence <  UNCERTAIN_MIN -> "unknown"
    matched=False + confidence >= UNCERTAIN_MIN -> "uncertain"
    matched=False + confidence <  UNCERTAIN_MIN -> "unknown"
    """
    if matched:
        if confidence >= settings.CONFIDENCE_KNOWN_MIN:
            return "known"
        elif confidence >= settings.CONFIDENCE_UNCERTAIN_MIN:
            return "uncertain"
        return "unknown"
    if confidence >= settings.CONFIDENCE_UNCERTAIN_MIN:
        return "uncertain"
    return "unknown"
