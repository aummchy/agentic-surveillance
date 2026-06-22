from fastapi import APIRouter, Query
from typing import Optional
from dashboard.backend.models import EventsResponse
from utils.db_utils import get_events_with_faces

router = APIRouter()


@router.get("/stats")
async def get_stats():
    from utils.db_utils import get_stats as fetch_stats
    return fetch_stats()


@router.get("", response_model=EventsResponse)
async def list_events(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    result = get_events_with_faces(
        limit=limit,
        offset=offset,
        status_filter=status
    )
    return result


@router.get("/unknown")
async def list_unknown_events(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    result = get_events_with_faces(
        limit=limit,
        offset=offset,
        status_filter="unknown"
    )
    return result


@router.get("/alerts")
async def list_alerts(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    from utils.db_utils import get_events_collection
    collection = get_events_collection()

    query = {"alert_level": {"$in": ["high", "critical"]}}
    total = collection.count_documents(query)
    events = list(collection.find(query)
                 .sort("timestamp", -1)
                 .skip(offset)
                 .limit(limit))

    for event in events:
        event["_id"] = str(event["_id"])

    return {"events": events, "total": total, "limit": limit, "offset": offset}
