"""Video capture device I/O — open, configure, resolve the source.

Pure device helpers extracted from CameraAgent (Phase 4 split, step 1).
This module knows nothing about tracks, recognition, or finalization: it
answers only "which device" and "give me an open VideoCapture".

Consumed by agents/camera_agent.py (start + the _loop reconnect path).
"""

import cv2
import structlog

from config import settings

logger = structlog.get_logger(__name__)


def camera_source() -> str | int:
    """Return RTSP URL string if CAMERA_SOURCE is set, else CAMERA_INDEX int.

    The type of the return value decides downstream behavior: a str is a
    URL or file path (read timeouts applied, EOF ends the loop), an int
    is a device index (no timeouts, read failures trigger reconnect).
    """
    src = getattr(settings, "CAMERA_SOURCE", "")
    if src:
        return src
    return settings.CAMERA_INDEX


def open_capture() -> cv2.VideoCapture:
    """Open VideoCapture with configured backend (dshow/msmf/auto).

    Backend selection only applies to device indexes — network URLs and
    file paths always use OpenCV's default. On Windows, CAMERA_BACKEND
    picks the capture backend (dshow avoids the MSMF frame-drop error).

    For string sources, CAP_PROP_READ_TIMEOUT_MSEC is set so a stalled
    stream returns from read() instead of blocking the loop forever;
    a failed cap.set is non-fatal and only logged.
    """
    source = camera_source()
    backend = getattr(settings, "CAMERA_BACKEND", "")
    if backend and isinstance(source, int):
        be = getattr(cv2, f"CAP_{backend.upper()}", None)
        if be is not None:
            cap = cv2.VideoCapture(source, be)
        else:
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    # Set read timeout for network streams so read() doesn't block indefinitely
    if isinstance(source, str):
        for prop in ("CAP_PROP_READ_TIMEOUT_MSEC", "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            attr = getattr(cv2, prop, None)
            if attr is not None:
                try:
                    cap.set(attr, settings.CAMERA_READ_TIMEOUT_MS)
                except Exception as e:
                    logger.debug("camera_prop_set_failed", prop=prop, error=str(e))
    return cap


def apply_frame_props(cap: cv2.VideoCapture) -> str:
    """Set frame dimensions on the given capture and return actual resolution.

    The requested FRAME_WIDTH/FRAME_HEIGHT are a hint — the device may
    negotiate something else, so the value read back is what gets logged.
    Returns "unknown" if the capture is not open.
    """
    if cap and cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return f"{actual_w}x{actual_h}"
    return "unknown"
