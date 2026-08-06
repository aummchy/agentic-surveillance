import cv2
import numpy as np
import structlog
from pathlib import Path
from config import settings
from config.status import Status

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
            api_secret=settings.CLOUDINARY_API_SECRET,
            connection_pool_maxsize=10
        )
        _cloudinary_configured = True
        return True
    except Exception as e:
        logger.error("cloudinary_init_failed", error=str(e))
        return False


def upload_to_cloudinary(image: np.ndarray, folder: str = "surveillance") -> str | None:
    """Upload a numpy image to Cloudinary.

    Returns the secure URL, or None on failure.
    """
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
    """Compute blur score using Laplacian variance.

    Higher score = sharper image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(image: np.ndarray) -> float:
    """Compute brightness using HSV V-channel mean.

    Returns value in range [0, 255].
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if len(image.shape) == 3 else image
    return float(np.mean(hsv[:, :, 2] if len(hsv.shape) == 3 else hsv))


def crop_person(frame: np.ndarray, box: tuple) -> np.ndarray:
    """Crop a person region from the frame using the bounding box.

    Coordinates are clipped to frame boundaries.
    """
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    return frame[y1:y2, x1:x2].copy()


def save_image(image: np.ndarray, path: str) -> bool:
    """Save a numpy image to disk.

    Creates parent directories if needed. Returns True on success.
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, image)
        return True
    except Exception as e:
        logger.warning("save_image_failed", path=path, error=str(e))
        return False


def resolve_track_image_url(track) -> str | None:
    """Single source of truth for 'what photo represents this track'.

    Priority: existing image_url > best_frame_jpeg/fallback_frame_jpeg > best_full_frame > None.
    Cloudinary is source of truth; local file is fallback on upload failure.
    """
    if track.image_url:
        return track.image_url

    jpeg_data = track.best_frame_jpeg or track.fallback_frame_jpeg
    if jpeg_data:
        url = upload_jpeg_to_cloudinary(jpeg_data)
        if url:
            track.image_url = url
            return url
        # Cloudinary failed — persist locally as last resort
        path = f"captures/{track.track_id}.jpg"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(jpeg_data)
        track.image_url = path
        return path

    if track.best_full_frame is not None:
        url = upload_to_cloudinary(track.best_full_frame)
        if url:
            track.image_url = url
            return url
        # Cloudinary failed — save locally
        save_image(track.best_full_frame, f"captures/{track.track_id}.jpg")
        track.image_url = f"captures/{track.track_id}.jpg"
        return track.image_url

    return None


def resolve_track_person_crop_url(track) -> str | None:
    """Resolve URL for the person crop image (bounding-box area, not full frame).

    Priority: existing person_crop_url > best_person_crop_jpeg > fallback to resolve_track_image_url.
    """
    if getattr(track, "person_crop_url", None):
        return track.person_crop_url

    jpeg_data = getattr(track, "best_person_crop_jpeg", None)
    if jpeg_data:
        url = upload_jpeg_to_cloudinary(jpeg_data)
        if url:
            track.person_crop_url = url
            return url
        # Cloudinary failed — persist locally
        path = f"captures/crops/{track.track_id}_person.jpg"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(jpeg_data)
        track.person_crop_url = path
        return path

    # No person crop available — fall back to full frame
    return resolve_track_image_url(track)


def _put_label_with_bg(img, text, pos, font_scale, color, thickness=1, bg_color=(0, 0, 0)):
    """Draw text with a filled background rectangle for readability."""
    (tx, ty) = pos
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    # Ensure label stays within frame bounds
    ty = max(th + 4, ty)
    # Draw shadow outline for contrast on any background
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        cv2.putText(img, text, (tx + 2 + dx, ty - 2 + dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.rectangle(img, (tx, ty - th - 6), (tx + tw + 8, ty + 4), bg_color, -1)
    cv2.putText(img, text, (tx + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area2 = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def draw_annotations(frame: np.ndarray, tracks: list) -> np.ndarray:
    if not tracks:
        return frame
    annotated = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.person_box)
        color = (0, 0, 255)
        border_thickness = 3
        is_verified = track.decision in (Status.AUTHORIZED, Status.VERIFIED)

        if is_verified:
            color = (0, 255, 0)
            border_thickness = 4
        elif track.decision == Status.KNOWN_VISITOR:
            color = (0, 255, 255)
            border_thickness = 3
        elif track.decision is None:
            color = (255, 255, 0)
        if track.is_masked or track.decision in (Status.MASKED_UNKNOWN, Status.HIDDEN, Status.BLACKLIST):
            color = (0, 0, 255)

        # Black outline for contrast on any background
        cv2.rectangle(annotated, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), border_thickness + 2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, border_thickness)

        name = getattr(track, 'person_name', None)
        short_id = str(track.byte_track_id) if track.byte_track_id else (track.track_id.rsplit("_", 1)[-1] if "_" in track.track_id else track.track_id)
        label = f"ID:{short_id}"
        if name:
            label += f" {name}"

        if track.decision:
            if track.decision == Status.AUTHORIZED:
                label += " [AUTHORIZED]"
            elif track.decision == Status.VERIFIED:
                label += " [VERIFIED]"
            elif track.decision == Status.KNOWN_VISITOR:
                label += " [KNOWN VISITOR]"
            elif track.decision == Status.BLACKLIST:
                label += " [BLACKLIST]"
            elif track.decision == Status.HIDDEN:
                label += " [HIDDEN]"
            elif track.decision in (Status.UNKNOWN, Status.MASKED_UNKNOWN):
                label += " [UNVERIFIED]"
            elif track.decision == Status.UNCERTAIN:
                label += " [UNCERTAIN]"
            else:
                label += f" [{track.decision}]"
        else:
            label += " [SCANNING]"
        if track.is_masked:
            label += " MASK"

        if is_verified:
            _put_label_with_bg(annotated, label, (x1, y1 - 10), 0.65, (255, 255, 255), 2, (0, 140, 0))
        else:
            _put_label_with_bg(annotated, label, (x1, y1 - 10), 0.55, (255, 255, 255), 2, (0, 0, 0))

    return annotated
