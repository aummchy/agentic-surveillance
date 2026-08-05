"""Visit memory CRUD — visit history tracking per person."""

import structlog
from datetime import datetime, timedelta
from pymongo import ReturnDocument
from utils.db_client import get_memory_collection
from config import settings
from config.status import Status, LABEL_TO_STATUS

logger = structlog.get_logger(__name__)


def get_or_create_memory(person_id: str) -> dict:
    """Get existing memory or create a new one for a person (atomic upsert)."""
    collection = get_memory_collection()
    now = datetime.utcnow()

    result = collection.find_one_and_update(
        {"person_id": person_id},
        {"$setOnInsert": {
            "visit_count": 0,
            "first_seen": now,
            "last_seen": now,
            "last_camera": None,
            "last_status": None,
            "best_status": None,
            "typical_hours": [],
            "typical_cameras": [],
            "avg_similarity": 0.0,
            "similarity_history": [],
            "status_history": [],
            "created_at": now,
            "updated_at": now,
        }},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    if result is None:
        logger.warning("find_one_and_update returned None; returning fallback document",
                       person_id=person_id)
        return {
            "person_id": person_id,
            "visit_count": 0,
            "first_seen": now,
            "last_seen": now,
            "last_camera": None,
            "last_status": None,
            "typical_hours": [],
            "typical_cameras": [],
            "avg_similarity": 0.0,
            "similarity_history": [],
            "status_history": [],
            "created_at": now,
            "updated_at": now,
            "__fallback__": True,
        }
    return result


def update_visit_memory(person_id: str, camera_id: str, status: int,
                        similarity: float, is_masked: bool = False) -> dict:
    """Update memory after a visit using a single atomic MongoDB operation."""
    collection = get_memory_collection()
    now = datetime.utcnow()
    hour = now.hour
    status_entry = {"status": status, "timestamp": now}

    gap = getattr(settings, "MIN_VISIT_GAP_SECS", 0)
    if gap > 0:
        existing = collection.find_one(
            {"person_id": person_id},
            {"last_seen": 1, "visit_count": 1, "_id": 0},
        )
        if existing and existing.get("visit_count", 0) > 0 and existing.get("last_seen"):
            last_seen = existing["last_seen"]
            if isinstance(last_seen, datetime) and (now - last_seen) < timedelta(seconds=gap):
                logger.info("visit_suppressed_duplicate",
                            person_id=person_id,
                            last_seen=last_seen.isoformat(),
                            visit_count=existing.get("visit_count", 0),
                            gap_secs=gap)
                return {
                    "person_id": person_id,
                    "visit_count": existing.get("visit_count", 0),
                    "last_seen": last_seen,
                    "last_camera": camera_id,
                    "last_status": status,
                    "avg_similarity": existing.get("avg_similarity", 0.0),
                    "similarity_history": existing.get("similarity_history", []),
                    "status_history": existing.get("status_history", []),
                    "typical_hours": existing.get("typical_hours", []),
                    "typical_cameras": existing.get("typical_cameras", []),
                    "updated_at": now,
                    "suppressed": True,
                }

    _STATUS_RANK = {Status.AUTHORIZED: 6, Status.VERIFIED: 5, Status.KNOWN: 4,
                    Status.KNOWN_VISITOR: 3, Status.UNCERTAIN: 2, Status.UNKNOWN: 1}
    # Also map string labels for backward compatibility with old MongoDB data
    _STATUS_RANK.update({"authorized": 6, "verified": 5, "known": 4,
                         "known_visitor": 3, "uncertain": 2, "unknown": 0})
    current_best = collection.find_one(
        {"person_id": person_id},
        {"best_status": 1, "_id": 0},
    )
    prev_best = (current_best or {}).get("best_status")
    # Convert string statuses from old data to int
    if isinstance(prev_best, str):
        prev_best = LABEL_TO_STATUS.get(prev_best, Status.UNKNOWN)
    prev_rank = _STATUS_RANK.get(prev_best, -1) if prev_best else -1
    new_rank = _STATUS_RANK.get(status, 0)
    best_status = status if new_rank > prev_rank else (prev_best or status)

    result = collection.find_one_and_update(
        {"person_id": person_id},
        {
            "$inc": {"visit_count": 1},
            "$push": {
                "similarity_history": {"$each": [similarity], "$slice": -10},
                "status_history": {"$each": [status_entry], "$slice": -10},
                "typical_hours": {"$each": [hour], "$slice": -20},
            },
            "$addToSet": {"typical_cameras": camera_id},
            "$set": {
                "last_seen": now,
                "last_camera": camera_id,
                "last_status": status,
                "best_status": best_status,
                "updated_at": now,
            },
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )

    if result is None:
        logger.warning("find_one_and_update returned None; returning fallback document",
                       person_id=person_id, camera_id=camera_id)
        return {
            "person_id": person_id,
            "visit_count": 1,
            "last_seen": now,
            "last_camera": camera_id,
            "last_status": status,
            "best_status": status,
            "avg_similarity": 0.0,
            "similarity_history": [similarity],
            "status_history": [status_entry],
            "typical_hours": [hour],
            "typical_cameras": [camera_id],
            "updated_at": now,
            "__fallback__": True,
            "suppressed": False,
        }
    return {
        "person_id": person_id,
        "visit_count": result.get("visit_count", 1),
        "last_seen": now,
        "last_camera": camera_id,
        "last_status": status,
        "best_status": best_status,
        "avg_similarity": result.get("avg_similarity", 0.0),
        "similarity_history": result.get("similarity_history", []),
        "status_history": result.get("status_history", []),
        "typical_hours": result.get("typical_hours", []),
        "typical_cameras": result.get("typical_cameras", []),
        "updated_at": now,
        "suppressed": False,
    }


def get_visit_history(person_id: str) -> dict:
    """Get visit history for a person."""
    collection = get_memory_collection()
    return collection.find_one({"person_id": person_id}) or {}


def get_memory_stats() -> dict:
    """Get memory collection statistics."""
    collection = get_memory_collection()

    return {
        "total_persons": collection.count_documents({}),
        "known_persons": collection.count_documents({"visit_count": {"$gt": 1}}),
        "recent_unknowns": collection.count_documents({
            "last_status": {"$in": [Status.UNKNOWN, Status.MASKED_UNKNOWN]}
        }),
    }
