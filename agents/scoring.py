"""
Pure scoring functions for recognition confidence.

All tunables come from config.settings — no hardcoded values here.
"""

import os
import datetime
import threading
from dataclasses import dataclass
from typing import Optional
from config import settings
from config.status import Status, STATUS_LABELS

_CALC_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "logs", "calculation.log"
)
_calc_log_initialized = False
_calc_log_lock = threading.Lock()


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class ScoreBreakdown:
    """All values needed for a calculation.log entry."""
    track_id: str
    raw_cosine: float
    face_quality: Optional[float]
    track_seconds: float
    memory_boost: float
    is_masked: bool
    margin: Optional[float]
    sim_norm: float
    quality_norm: float
    track_norm: float
    memory_norm: float
    margin_norm: float
    mask_norm: float
    base: float
    adjusted: float
    confidence: int
    matched: bool
    status: int


def log_formula_header():
    """Write the formula header to calculation.log once at startup."""
    global _calc_log_initialized
    if _calc_log_initialized:
        return
    if not getattr(settings, "ENABLE_CALC_LOG", False):
        return
    os.makedirs(os.path.dirname(_CALC_LOG_PATH), exist_ok=True)
    _calc_log_initialized = True

    lines = [
        "",
        "═" * 72,
        "                      CONFIDENCE SCORING FORMULA",
        "═" * 72,
        "",
        "  base = 0.65×sim + 0.15×quality + 0.10×track + 0.05×memory + 0.05×margin",
        "  adjusted = base × (1 - 0.15×mask)",
        "  confidence = int(round(1 + 99 × clip(adjusted, 0, 1)))",
        "",
        "─" * 72,
        f"  Weights:     sim={settings.WEIGHT_SIMILARITY}  "
        f"quality={settings.WEIGHT_QUALITY}  "
        f"track={settings.WEIGHT_TRACK}  "
        f"memory={settings.WEIGHT_MEMORY}  "
        f"margin={settings.WEIGHT_MARGIN}",
        f"  Normalize:   SIM=[{settings.SIM_NORM_MIN}..{settings.SIM_NORM_MAX}]  "
        f"TRACK_SAT={settings.TRACK_SATURATION_SECS}s  "
        f"MEM_MAX={settings.MEMORY_NORM_MAX}  "
        f"MAR_MAX={settings.MARGIN_NORM_MAX}",
        f"  Thresholds:  match>={settings.MATCH_THRESHOLD}  "
        f"known>={settings.CONFIDENCE_KNOWN_MIN}  "
        f"uncertain>={settings.CONFIDENCE_UNCERTAIN_MIN}  "
        f"unknown<{settings.CONFIDENCE_UNCERTAIN_MIN}",
        f"  Mask penalty: max={settings.MASK_PENALTY_MAX}",
        "═" * 72,
        "",
    ]

    with open(_CALC_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _format_log_lines(b: ScoreBreakdown) -> list:
    """Format a ScoreBreakdown into fixed-width log lines."""
    ts = _ts()
    status_label = STATUS_LABELS.get(b.status, str(b.status))
    w_sim = settings.WEIGHT_SIMILARITY * b.sim_norm
    w_qual = settings.WEIGHT_QUALITY * b.quality_norm
    w_track = settings.WEIGHT_TRACK * b.track_norm
    w_mem = settings.WEIGHT_MEMORY * b.memory_norm
    w_mar = settings.WEIGHT_MARGIN * b.margin_norm

    mask_label = f"{b.mask_norm:.1f} (masked)" if b.is_masked else f"{b.mask_norm:.1f} (unmasked)"
    mask_effect = f"× {1.0 - settings.MASK_PENALTY_MAX * b.mask_norm:.4f}" if b.is_masked else "→ no penalty"

    quality_raw = f"{b.face_quality:.3f}" if b.face_quality is not None and b.face_quality > 0 else "N/A"
    margin_raw = f"{b.margin:.3f}" if b.margin is not None else "N/A"
    margin_norm_str = f"{b.margin_norm:.3f}" if b.margin is not None else "0.500"

    C1, C2, C3, C4 = 20, 10, 12, 28
    sep = "─" * 72

    return [
        sep,
        f"  track={b.track_id}  │  {ts}",
        sep,
        f"  {'Component':<{C1}} {'Raw':>{C2}} {'Normalized':>{C3}} {'Weighted':>{C4}}",
        f"  {'─'*C1} {'─'*C2} {'─'*C3} {'─'*C4}",
        f"  {'Similarity':<{C1}} {b.raw_cosine:>{C2}.3f} {b.sim_norm:>{C3}.3f} {settings.WEIGHT_SIMILARITY}×{b.sim_norm:.3f} = {w_sim:.3f}",
        f"  {'Face Quality':<{C1}} {quality_raw:>{C2}} {b.quality_norm:>{C3}.3f} {settings.WEIGHT_QUALITY}×{b.quality_norm:.3f} = {w_qual:.3f}",
        f"  {'Track Duration':<{C1}} {b.track_seconds:>{C2-1}.1f}s {b.track_norm:>{C3}.3f} {settings.WEIGHT_TRACK}×{b.track_norm:.3f} = {w_track:.3f}",
        f"  {'Memory Boost':<{C1}} {b.memory_boost:>{C2}.1f} {b.memory_norm:>{C3}.3f} {settings.WEIGHT_MEMORY}×{b.memory_norm:.3f} = {w_mem:.3f}",
        f"  {'Margin':<{C1}} {margin_raw:>{C2}} {margin_norm_str:>{C3}} {settings.WEIGHT_MARGIN}×{b.margin_norm:.3f} = {w_mar:.3f}",
        f"  {'─'*C1} {'─'*C2} {'─'*C3} {'─'*C4}",
        f"  BASE = {w_sim:.3f} + {w_qual:.3f} + {w_track:.3f} + {w_mem:.3f} + {w_mar:.3f} = {b.base:.3f}",
        f"  MASK = {mask_label} {mask_effect}",
        f"  ADJUSTED = {b.base:.3f} × {1.0 - settings.MASK_PENALTY_MAX * b.mask_norm:.4f} = {b.adjusted:.3f}",
        f"  {'─'*70}",
        f"  CONFIDENCE = 1 + 99 × {b.adjusted:.3f} = {b.confidence}",
        f"  STATUS = {status_label}  (matched={str(b.matched).lower()}, threshold={settings.MATCH_THRESHOLD})",
        sep,
        "",
    ]


def _log_calculation(b: ScoreBreakdown):
    """Write a full calculation breakdown to calculation.log."""
    if not getattr(settings, "ENABLE_CALC_LOG", False):
        return
    os.makedirs(os.path.dirname(_CALC_LOG_PATH), exist_ok=True)
    lines = _format_log_lines(b)

    with _calc_log_lock:
        max_bytes = getattr(settings, "CALC_LOG_MAX_SIZE_MB", 10) * 1024 * 1024
        try:
            if os.path.exists(_CALC_LOG_PATH) and os.path.getsize(_CALC_LOG_PATH) > max_bytes:
                with open(_CALC_LOG_PATH, "w", encoding="utf-8") as f:
                    f.write("")
        except OSError:
            pass
        with open(_CALC_LOG_PATH, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_cosine(raw_cosine: float) -> float:
    """Map raw cosine [SIM_NORM_MIN..SIM_NORM_MAX] -> [0..1]."""
    return clip(
        (raw_cosine - settings.SIM_NORM_MIN) / (settings.SIM_NORM_MAX - settings.SIM_NORM_MIN),
        0.0, 1.0
    )


def normalize_quality(face_quality: Optional[float]) -> float:
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


def normalize_margin(margin: Optional[float]) -> float:
    """Map margin [0..MARGIN_NORM_MAX] -> [0..1]. None -> 0.5 (neutral)."""
    if margin is None:
        return 0.5
    return clip(margin / settings.MARGIN_NORM_MAX, 0.0, 1.0)


def compute_confidence(
    raw_cosine: float,
    face_quality: Optional[float],
    track_seconds: float,
    memory_boost: float,
    is_masked: bool,
    margin: Optional[float] = None,
    track_id: str = "unknown"
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
    confidence = round(1 + 99 * clip(adjusted, 0.0, 1.0))

    matched = is_match(raw_cosine)
    status = confidence_status(confidence, matched)

    _log_calculation(ScoreBreakdown(
        track_id=track_id,
        raw_cosine=raw_cosine,
        face_quality=face_quality,
        track_seconds=track_seconds,
        memory_boost=memory_boost,
        is_masked=is_masked,
        margin=margin,
        sim_norm=sim_norm,
        quality_norm=quality_norm,
        track_norm=track_norm,
        memory_norm=memory_norm,
        margin_norm=margin_norm,
        mask_norm=mask_norm,
        base=base,
        adjusted=adjusted,
        confidence=confidence,
        matched=matched,
        status=status,
    ))

    return confidence


def is_match(raw_cosine: float) -> bool:
    """Binary identity gate: raw_cosine >= MATCH_THRESHOLD."""
    return raw_cosine >= settings.MATCH_THRESHOLD


def confidence_status(confidence: int, matched: bool) -> int:
    """Map confidence + match gate -> Status enum value.

    matched=True  + confidence >= KNOWN_MIN   -> Status.KNOWN
    matched=True  + confidence >= UNCERTAIN_MIN -> Status.UNCERTAIN
    matched=True  + confidence <  UNCERTAIN_MIN -> Status.UNKNOWN
    matched=False + confidence >= UNCERTAIN_MIN -> Status.UNCERTAIN
    matched=False + confidence <  UNCERTAIN_MIN -> Status.UNKNOWN
    """
    if matched:
        if confidence >= settings.CONFIDENCE_KNOWN_MIN:
            return Status.KNOWN
        if confidence >= settings.CONFIDENCE_UNCERTAIN_MIN:
            return Status.UNCERTAIN
        return Status.UNKNOWN
    if confidence >= settings.CONFIDENCE_UNCERTAIN_MIN:
        return Status.UNCERTAIN
    return Status.UNKNOWN
