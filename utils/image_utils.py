import cv2
import numpy as np
import structlog
from pathlib import Path
from config import settings

logger = structlog.get_logger(__name__)

_cloudinary_configured = False


def _init_cloudinary():
    global _cloudinary_configured
    if _cloudinary_configured:
        return True
    if not all([settings.CLOUDINARY_CLOUD_NAME, settings.CLOUDINARY_API_KEY, settings.CLOUDINARY_API_SECRET]):
        return False
    try:
        import cloudinary
        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET
        )
        _cloudinary_configured = True
        return True
    except Exception as e:
        logger.error("cloudinary_init_failed", error=str(e))
        return False


def upload_to_cloudinary(image: np.ndarray, folder: str = "surveillance") -> str | None:
    if not _init_cloudinary():
        return None
    try:
        import cloudinary.uploader
        _, buffer = cv2.imencode(".jpg", image)
        result = cloudinary.uploader.upload(
            buffer.tobytes(),
            folder=folder,
            resource_type="image"
        )
        return result.get("secure_url")
    except Exception as e:
        logger.error("cloudinary_upload_failed", error=str(e))
        return None


def upload_jpeg_to_cloudinary(jpeg_bytes: bytes, folder: str = "surveillance") -> str | None:
    """Upload pre-encoded JPEG bytes to Cloudinary (avoids re-encoding from numpy)."""
    if not _init_cloudinary():
        return None
    try:
        import cloudinary.uploader
        result = cloudinary.uploader.upload(
            jpeg_bytes,
            folder=folder,
            resource_type="image"
        )
        return result.get("secure_url")
    except Exception as e:
        logger.error("cloudinary_upload_failed", error=str(e))
        return None


def compute_blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if len(image.shape) == 3 else image
    return float(np.mean(hsv[:, :, 2] if len(hsv.shape) == 3 else hsv))


def crop_person(frame: np.ndarray, box: tuple) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    return frame[y1:y2, x1:x2].copy()


def resize_image(image: np.ndarray, target_size: tuple = (112, 112)) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def save_image(image: np.ndarray, path: str) -> bool:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, image)
        return True
    except Exception:
        return False


def _put_label_with_bg(img, text, pos, font_scale, color, thickness=1, bg_color=(0, 0, 0)):
    """Draw text with a filled background rectangle for readability."""
    (tx, ty) = pos
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    # Ensure label stays within frame bounds
    ty = max(th + 4, ty)
    cv2.rectangle(img, (tx, ty - th - 4), (tx + tw + 4, ty + 2), bg_color, -1)
    cv2.putText(img, text, (tx + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_annotations(frame: np.ndarray, tracks: list, decisions: dict = None) -> np.ndarray:
    if not tracks:
        return frame
    annotated = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.person_box)
        color = (0, 0, 255)
        border_thickness = 2
        is_verified = track.decision in ("authorized", "verified")

        if is_verified:
            color = (0, 255, 0)
            border_thickness = 3
        elif track.decision == "known_visitor":
            color = (0, 255, 255)
            border_thickness = 2
        if track.is_masked or track.decision in ("masked_unknown", "intentionally_hidden", "blacklist"):
            color = (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, border_thickness)

        name = getattr(track, 'person_name', None)
        short_id = track.track_id.rsplit("_", 1)[-1] if "_" in track.track_id else track.track_id
        label = f"ID:{short_id}"
        if name:
            label += f" {name}"

        if track.decision:
            if track.decision == "authorized":
                label += " [AUTHORIZED]"
            elif track.decision == "verified":
                label += " [VERIFIED]"
            elif track.decision == "known_visitor":
                label += " [KNOWN VISITOR]"
            elif track.decision == "blacklist":
                label += " [BLACKLIST]"
            elif track.decision == "intentionally_hidden":
                label += " [HIDDEN]"
            elif track.decision in ("unknown", "masked_unknown"):
                label += " [UNVERIFIED]"
            elif track.decision == "uncertain":
                label += " [UNCERTAIN]"
            else:
                label += f" [{track.decision}]"
        else:
            label += " [UNVERIFIED]"
        if track.is_masked:
            label += " MASK"

        if is_verified:
            _put_label_with_bg(annotated, label, (x1, y1 - 8), 0.55, (255, 255, 255), 2, (0, 140, 0))
        else:
            _put_label_with_bg(annotated, label, (x1, y1 - 8), 0.45, color, 1, (0, 0, 0))

    return annotated


def decode_image(image_bytes: bytes, max_size_mb: int = 10) -> np.ndarray:
    if len(image_bytes) > max_size_mb * 1024 * 1024:
        raise ValueError(f"Image size {len(image_bytes) / 1024 / 1024:.1f}MB exceeds limit of {max_size_mb}MB")
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
