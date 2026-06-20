import cv2
import numpy as np
from pathlib import Path
from config import settings


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


def crop_face_region(person_crop: np.ndarray, face_bbox: tuple, person_box: tuple) -> np.ndarray:
    fx1, fy1, fx2, fy2 = map(int, face_bbox)
    px1, py1, px2, py2 = map(int, person_box)

    abs_fx1 = px1 + fx1
    abs_fy1 = py1 + fy1
    abs_fx2 = px1 + fx2
    abs_fy2 = py1 + fy2

    abs_fx1 = max(0, abs_fx1)
    abs_fy1 = max(0, abs_fy1)

    h, w = person_crop.shape[:2]
    abs_fx2 = min(w, abs_fx2 - px1)
    abs_fy2 = min(h, abs_fy2 - py1)

    if abs_fx2 <= abs_fx1 or abs_fy2 <= abs_fy1:
        return person_crop

    return person_crop[abs_fy1:abs_fy2, abs_fx1:abs_fx2].copy()


def resize_image(image: np.ndarray, target_size: tuple = (112, 112)) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def save_image(image: np.ndarray, path: str) -> bool:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, image)
        return True
    except Exception:
        return False


def draw_annotations(frame: np.ndarray, tracks: list, decisions: dict = None) -> np.ndarray:
    annotated = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.person_box)
        color = (0, 255, 0)
        if track.is_masked:
            color = (0, 0, 255)
        if track.decision in ("masked_unknown", "intentionally_hidden", "blacklist"):
            color = (0, 0, 255)
        elif track.decision == "authorized":
            color = (0, 255, 0)
        elif track.decision == "known_visitor":
            color = (255, 165, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{track.track_id}"
        if track.decision:
            label += f" [{track.decision}]"
        if track.is_masked:
            label += " MASK"

        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated


def decode_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
