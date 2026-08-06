import io
import contextlib
import threading
import warnings
import structlog
import numpy as np
import cv2

# Suppress InsightFace FutureWarning about deprecated 'estimate' method
warnings.filterwarnings("ignore", message=".*estimate.*deprecated.*", category=FutureWarning)

from insightface.app import FaceAnalysis
from config import settings

logger = structlog.get_logger(__name__)


class InsightFaceSingleton:
    _instance = None
    _lock = threading.Lock()
    _inference_lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if not InsightFaceSingleton._initialized:
            with InsightFaceSingleton._lock:
                if not InsightFaceSingleton._initialized:
                    # Suppress InsightFace print noise during model loading
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.app = FaceAnalysis(
                            name=settings.INSIGHTFACE_MODEL,
                            providers=[settings.INSIGHTFACE_PROVIDER]
                        )
                        self.app.prepare(ctx_id=0, det_size=(settings.INSIGHTFACE_DET_SIZE,
                                                              settings.INSIGHTFACE_DET_SIZE))
                    InsightFaceSingleton._initialized = True
                    logger.info("insightface_loaded",
                                model=settings.INSIGHTFACE_MODEL,
                                provider=settings.INSIGHTFACE_PROVIDER,
                                det_size=settings.INSIGHTFACE_DET_SIZE)

    @staticmethod
    def _apply_clahe(image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve
        face detection and embedding quality in variable lighting.

        Skips CLAHE when the image already has good contrast (measured by the
        standard deviation of the L channel in LAB space). This avoids the
        overhead of color-space conversion + histogram equalization on frames
        that don't need it."""
        if image is None or len(image.shape) < 2:
            return image
        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            # Skip CLAHE if contrast is already adequate (std >= 40 is well-lit, high-contrast)
            l_std = float(l_channel.std())
            if l_std >= settings.CLAHE_SKIP_CONTRAST_THRESHOLD:
                return image
            clahe = cv2.createCLAHE(clipLimit=settings.CLAHE_CLIP_LIMIT,
                                     tileGridSize=(settings.CLAHE_TILE_SIZE, settings.CLAHE_TILE_SIZE))
            lab[:, :, 0] = clahe.apply(l_channel)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif len(image.shape) == 2:
            std = float(image.std())
            if std >= settings.CLAHE_SKIP_CONTRAST_THRESHOLD:
                return image
            clahe = cv2.createCLAHE(clipLimit=settings.CLAHE_CLIP_LIMIT,
                                     tileGridSize=(settings.CLAHE_TILE_SIZE, settings.CLAHE_TILE_SIZE))
            return clahe.apply(image)
        return image

    def _detect_mask_geometric(self, landmarks: np.ndarray) -> bool:
        if landmarks is None or len(landmarks) < 5:
            return False

        nose_tip = landmarks[2]
        mouth_center = (landmarks[3] + landmarks[4]) / 2
        upper_face = (landmarks[0] + landmarks[1]) / 2

        lower_face_height = abs(mouth_center[1] - nose_tip[1])
        upper_face_height = abs(upper_face[1] - nose_tip[1])

        if upper_face_height < 1e-6:
            return False

        ratio = lower_face_height / upper_face_height
        return ratio < settings.MASK_RATIO_THRESHOLD

    def detect_faces_raw(self, image: np.ndarray, min_score: float = 0.0) -> list:
        """Run face detection once and return all faces above min_score.
        Each result is a dict with keys: det_score, embedding, bbox, is_masked."""
        try:
            # Skip CLAHE on tiny crops (<200px) — 8x8 tile grid destroys features
            if min(image.shape[:2]) >= 200:
                image = self._apply_clahe(image)
            with self._inference_lock:
                faces = self.app.get(image)
            if not faces:
                return []
            results = []
            for f in faces:
                if f.det_score < min_score:
                    continue
                results.append({
                    "det_score": float(f.det_score),
                    "embedding": f.normed_embedding,
                    "bbox": tuple(map(int, f.bbox)),
                    "is_masked": self._detect_mask_geometric(f.landmark),
                })
            results.sort(key=lambda x: x["det_score"], reverse=True)
            return results
        except Exception as e:
            logger.error("detect_faces_raw_failed", error=str(e))
            return []


def get_insightface() -> InsightFaceSingleton:
    return InsightFaceSingleton()


def compare_similarity(raw_cosine: float, threshold: float = None) -> bool:
    if threshold is None:
        threshold = settings.MATCH_THRESHOLD
    return raw_cosine >= threshold


def atlas_score_to_cosine(atlas_score: float) -> float:
    return (atlas_score * 2) - 1
