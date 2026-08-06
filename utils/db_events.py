"""Event logging and statistics queries."""

import structlog
from datetime import datetime, timezone
from typing import Optional
from utils.db_client import get_events_collection, get_faces_collection
from config.status import Status, LABEL_TO_STATUS

logger = structlog.get_logger(__name__)


def _extract_person_image(face: dict) -> Optional[str]:
    """Extract the first image URL from a face document's images array."""
    images = face.get("images")
    if not images:
        return None
    return images[0].get("url")


def _normalize_status(status_value) -> int:
    """Convert a MongoDB status value (old string or new int) to int."""
    if isinstance(status_value, int):
        return status_value
    if isinstance(status_value, str):
        converted = LABEL_TO_STATUS.get(status_value)
        if converted is not None:
            return int(converted)
        logger.warning("unknown_string_status", raw=status_value)
    return int(Status.UNKNOWN)


def log_event(track_id: str, camera_id: str, status: int, alert_level: str,
              person_id: str = None, name: str = None, is_masked: bool = False,
              similarity_score: float = 0.0, image_url: str = None,
              reason: str = "", alerted: bool = False,
              person_crop_url: str = None):
    """Insert an event document into MongoDB. Logs warning on failure — event
    logging is non-critical and must never crash the caller."""
    collection = get_events_collection()

    doc = {
        "track_id": track_id,
        "camera_id": camera_id,
        "timestamp": datetime.now(timezone.utc),
        "status": status,
        "alert_level": alert_level,
        "person_id": person_id,
        "name": name,
        "is_masked": is_masked,
        "similarity_score": similarity_score,
        "image_url": image_url,
        "person_crop_url": person_crop_url,
        "reason": reason,
        "alerted": alerted
    }

    try:
        collection.insert_one(doc)
    except Exception as e:
        logger.warning("event_insert_failed", track_id=track_id, camera_id=camera_id, error=str(e))


def get_events_with_faces(limit: int = 50, offset: int = 0,
                          status_filter: Optional[int] = None) -> dict:
    """Fetch events with enriched person data from the faces collection.

    Returns dict with keys: events, total, limit, offset.
    Old string statuses in MongoDB are normalized to int on read.
    """
    collection = get_events_collection()
    faces_collection = get_faces_collection()

    query = {}
    if status_filter is not None:
        query["status"] = status_filter

    total = collection.count_documents(query)
    events = list(collection.find(query)
                  .sort("timestamp", -1)
                  .skip(offset)
                  .limit(limit))

    person_ids = [e.get("person_id") for e in events if e.get("person_id")]
    face_map = {}
    if person_ids:
        faces = faces_collection.find(
            {"person_id": {"$in": person_ids}},
            {"person_id": 1, "name": 1, "verified": 1, "images": 1, "person_crop_url": 1}
        )
        for face in faces:
            face_map[face["person_id"]] = face

    for event in events:
        event["_id"] = str(event["_id"])
        event["status"] = _normalize_status(event.get("status"))
        pid = event.get("person_id")
        if pid and pid in face_map:
            face = face_map[pid]
            event["person_name"] = face.get("name")
            event["person_verified"] = face.get("verified", False)
            event["person_image"] = _extract_person_image(face)
            if face.get("person_crop_url"):
                event["person_crop_url"] = face["person_crop_url"]

    return {"events": events, "total": total, "limit": limit, "offset": offset}


def get_stats() -> dict:
    """Aggregate dashboard statistics.

    total_unknown: faces where verified is False or missing (not yet reviewed).
    """
    faces_col = get_faces_collection()
    events_col = get_events_collection()

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    return {
        "total_unknown": faces_col.count_documents({"verified": {"$ne": True}}),
        "total_verified": faces_col.count_documents({"verified": True}),
        "events_today": events_col.count_documents({"timestamp": {"$gte": today_start}}),
        "unknown_today": events_col.count_documents({
            "timestamp": {"$gte": today_start},
            "status": {"$in": [Status.UNKNOWN, Status.MASKED_UNKNOWN]}
        }),
    }
