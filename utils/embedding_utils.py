import threading
import structlog
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from config import settings
from pipeline.models import EmbeddingResult

logger = structlog.get_logger(__name__)


class InsightFaceSingleton:
    _instance = None
    _lock = threading.Lock()
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
                    self.app = FaceAnalysis(
                        name=settings.INSIGHTFACE_MODEL,
                        providers=[settings.INSIGHTFACE_PROVIDER]
                    )
                    self.app.prepare(ctx_id=0, det_size=(settings.INSIGHTFACE_DET_SIZE,
                                                          settings.INSIGHTFACE_DET_SIZE))
                    InsightFaceSingleton._initialized = True

    def detect_and_embed(self, image: np.ndarray) -> EmbeddingResult:
        try:
            h, w = image.shape[:2] if image is not None and len(image.shape) >= 2 else (0, 0)
            faces = self.app.get(image)

            if not faces:
                logger.debug("no_face_detected", width=w, height=h)
                return EmbeddingResult(
                    face_detected=False,
                    error="No face detected"
                )

            best_face = max(faces, key=lambda f: f.det_score)
            logger.debug("face_detected",
                        det_score=best_face.det_score,
                        threshold=settings.DET_SCORE_MIN,
                        faces_found=len(faces),
                        width=w,
                        height=h)

            if best_face.det_score < settings.DET_SCORE_MIN:
                logger.debug("detection_score_below_threshold",
                           score=best_face.det_score,
                           threshold=settings.DET_SCORE_MIN)
                return EmbeddingResult(
                    face_detected=True,
                    detection_score=float(best_face.det_score),
                    error=f"Detection score {best_face.det_score:.3f} below threshold"
                )

            embedding = best_face.normed_embedding
            bbox = tuple(map(int, best_face.bbox))

            is_masked = self._detect_mask_geometric(best_face.landmark)

            return EmbeddingResult(
                embedding=embedding,
                face_detected=True,
                detection_score=float(best_face.det_score),
                embedding_score=float(best_face.det_score),
                bbox=bbox,
                is_masked=is_masked
            )

        except Exception as e:
            logger.error("detect_and_embed_failed", error=str(e))
            return EmbeddingResult(
                face_detected=False,
                error=str(e)
            )

    def detect_and_embed_relaxed(self, image: np.ndarray, min_score: float = 0.20) -> EmbeddingResult:
        try:
            h, w = image.shape[:2] if image is not None and len(image.shape) >= 2 else (0, 0)
            faces = self.app.get(image)

            if not faces:
                return EmbeddingResult(
                    face_detected=False,
                    error="No face detected (relaxed)"
                )

            best_face = max(faces, key=lambda f: f.det_score)
            logger.debug("relaxed_face_detected",
                        det_score=best_face.det_score,
                        min_score=min_score,
                        faces_found=len(faces),
                        width=w,
                        height=h)

            if best_face.det_score < min_score:
                return EmbeddingResult(
                    face_detected=True,
                    detection_score=float(best_face.det_score),
                    error=f"Detection score {best_face.det_score:.3f} below relaxed threshold {min_score}"
                )

            embedding = best_face.normed_embedding
            bbox = tuple(map(int, best_face.bbox))
            is_masked = self._detect_mask_geometric(best_face.landmark)

            return EmbeddingResult(
                embedding=embedding,
                face_detected=True,
                detection_score=float(best_face.det_score),
                embedding_score=float(best_face.det_score),
                bbox=bbox,
                is_masked=is_masked
            )

        except Exception as e:
            logger.error("detect_and_embed_relaxed_failed", error=str(e))
            return EmbeddingResult(
                face_detected=False,
                error=str(e)
            )

    def embed_only(self, image: np.ndarray, det_score: float) -> EmbeddingResult:
        try:
            h, w = image.shape[:2] if image is not None and len(image.shape) >= 2 else (0, 0)
            faces = self.app.get(image)

            if not faces:
                logger.debug("embed_only_no_face", width=w, height=h)
                return EmbeddingResult(
                    face_detected=False,
                    error="No face detected"
                )

            best_face = max(faces, key=lambda f: f.det_score)
            logger.debug("embed_only_face_detected",
                        det_score=best_face.det_score,
                        threshold=settings.EMBEDDING_DET_SCORE_MIN)

            if best_face.det_score < settings.EMBEDDING_DET_SCORE_MIN:
                return EmbeddingResult(
                    face_detected=True,
                    detection_score=float(best_face.det_score),
                    embedding_score=float(best_face.det_score),
                    error=f"Embedding quality score {best_face.det_score:.3f} below threshold"
                )

            embedding = best_face.normed_embedding
            bbox = tuple(map(int, best_face.bbox))

            is_masked = self._detect_mask_geometric(best_face.landmark)

            return EmbeddingResult(
                embedding=embedding,
                face_detected=True,
                detection_score=det_score,
                embedding_score=float(best_face.det_score),
                bbox=bbox,
                is_masked=is_masked
            )

        except Exception as e:
            logger.error("embed_only_failed", error=str(e))
            return EmbeddingResult(
                face_detected=False,
                error=str(e)
            )

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
        return ratio < 0.3

    def detect_faces_raw(self, image: np.ndarray, min_score: float = 0.0) -> list:
        """Run face detection once and return all faces above min_score.
        Each result is a dict with keys: det_score, embedding, bbox, is_masked."""
        try:
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

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-6)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-6)
        return float(np.dot(emb1, emb2))


def get_insightface() -> InsightFaceSingleton:
    return InsightFaceSingleton()


def compare_similarity(raw_cosine: float, threshold: float = None) -> bool:
    if threshold is None:
        threshold = settings.MATCH_THRESHOLD
    return raw_cosine >= threshold


def atlas_score_to_cosine(atlas_score: float) -> float:
    return (atlas_score * 2) - 1
