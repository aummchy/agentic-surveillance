from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from dashboard.backend.models import EventsResponse
from utils.db_utils import get_events_with_faces
import asyncio
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/stats")
async def get_stats():
    try:
        from utils.db_utils import get_stats as fetch_stats
        return await asyncio.to_thread(fetch_stats)
    except Exception as e:
        logger.error("stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch stats")


@router.get("", response_model=EventsResponse)
async def list_events(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    try:
        result = await asyncio.to_thread(
            get_events_with_faces,
            limit=limit,
            offset=offset,
            status_filter=status
        )
        return result
    except Exception as e:
        logger.error("list_events_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch events")


@router.get("/unknown")
async def list_unknown_events(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    try:
        result = await asyncio.to_thread(
            get_events_with_faces,
            limit=limit,
            offset=offset,
            status_filter="unknown"
        )
        return result
    except Exception as e:
        logger.error("list_unknown_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch unknown events")


@router.get("/alerts")
async def list_alerts(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    try:
        from utils.db_utils import get_events_collection

        def _fetch():
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

        return await asyncio.to_thread(_fetch)
    except Exception as e:
        logger.error("list_alerts_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch alerts")
