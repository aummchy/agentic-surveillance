import structlog
from pipeline.models import MatchResult
from utils.db_utils import vector_search

logger = structlog.get_logger(__name__)


def run_matching_from_embedding(embedding: list) -> MatchResult:
    if not embedding:
        return MatchResult(matched=False)

    matches = vector_search(embedding)

    if not matches:
        return MatchResult(matched=False)

    best = matches[0]
    logger.info("match_found",
                person_id=best.get("person_id"),
                name=best.get("name"),
                role=best.get("role"),
                tags=best.get("tags", []),
                similarity=round(best.get("similarity_score", 0.0), 4),
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
        alert_level=best.get("alert_level", "low")
    )
