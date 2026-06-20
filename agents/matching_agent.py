import numpy as np
from pipeline.models import MatchResult, EmbeddingResult
from utils.db_utils import vector_search
from utils.embedding_utils import get_insightface


def run_matching(embedding_result: EmbeddingResult) -> MatchResult:
    if not embedding_result.face_detected or embedding_result.embedding is None:
        return MatchResult(matched=False)

    embedding_list = embedding_result.embedding.tolist() if isinstance(embedding_result.embedding, np.ndarray) else embedding_result.embedding

    matches = vector_search(embedding_list)

    if not matches:
        return MatchResult(matched=False)

    best = matches[0]
    return MatchResult(
        person_id=best.get("person_id"),
        name=best.get("name"),
        role=best.get("role"),
        tags=best.get("tags", []),
        similarity_score=best.get("similarity_score", 0.0),
        image_url=best.get("image_url"),
        matched=True
    )


def run_matching_from_embedding(embedding: list) -> MatchResult:
    if not embedding:
        return MatchResult(matched=False)

    matches = vector_search(embedding)

    if not matches:
        return MatchResult(matched=False)

    best = matches[0]
    return MatchResult(
        person_id=best.get("person_id"),
        name=best.get("name"),
        role=best.get("role"),
        tags=best.get("tags", []),
        similarity_score=best.get("similarity_score", 0.0),
        image_url=best.get("image_url"),
        matched=True
    )
