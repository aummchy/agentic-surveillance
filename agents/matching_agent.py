import numpy as np
import structlog
from pipeline.models import MatchResult
from utils.db_utils import vector_search

logger = structlog.get_logger(__name__)


def run_matching_from_embedding(embedding: list, track_id: str = "unknown") -> MatchResult:
    if embedding is None:
        return MatchResult(matched=False)
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        return MatchResult(matched=False)

    result = vector_search(arr.tolist())
    matches = result["matches"]

    if not matches:
        return MatchResult(matched=False)

    best = matches[0]
    top2 = result["top2"]
    margin = result["margin"]

    logger.info("match_found",
                track_id=track_id,
                person_id=best.get("person_id"),
                name=best.get("name"),
                role=best.get("role"),
                tags=best.get("tags", []),
                similarity=round(best.get("similarity_score", 0.0), 4),
                top2=round(top2, 4) if top2 is not None else None,
                margin=round(margin, 4) if margin is not None else None,
                verified=best.get("verified", False),
                alert_level=best.get("alert_level", "low"))
    return MatchResult(
        person_id=best.get("person_id"),
        name=best.get("name"),
        role=best.get("role"),
        tags=best.get("tags", []),
        similarity_score=best.get("similarity_score", 0.0),
        image_url=best.get("image_url"),
        matched=True,
        verified=best.get("verified", False),
        alert_level=best.get("alert_level", "low"),
        second_best_similarity=top2,
        margin=margin,
        candidate_count=len(matches),
    )
