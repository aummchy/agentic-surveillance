import numpy as np
from utils.embedding_utils import get_insightface
from pipeline.models import EmbeddingResult


def detect_and_embed(face_crop: np.ndarray) -> EmbeddingResult:
    app = get_insightface()
    return app.detect_and_embed(face_crop)


def embed_only(face_crop: np.ndarray, det_score: float) -> EmbeddingResult:
    app = get_insightface()
    return app.embed_only(face_crop, det_score)


def compute_face_ratio(face_bbox: tuple, person_box: tuple) -> float:
    fx1, fy1, fx2, fy2 = face_bbox
    face_area = (fx2 - fx1) * (fy2 - fy1)

    px1, py1, px2, py2 = person_box
    person_area = (px2 - px1) * (py2 - py1)

    if person_area <= 0:
        return 0.0

    return face_area / person_area
