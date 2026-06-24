from fastapi import APIRouter, HTTPException, Query
from agents.report import ReportAgent
import asyncio
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter()
report_agent = ReportAgent()


@router.get("/reports/stats")
async def get_stats():
    """Get dashboard statistics."""
    try:
        result = await asyncio.to_thread(report_agent.run, {"report_type": "stats"})
        return result.get("stats", {})
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Server shutting down")


@router.get("/reports/summary")
async def get_summary(period: str = Query("daily", pattern="^(daily|weekly)$")):
    """Get daily or weekly summary."""
    try:
        result = await asyncio.to_thread(report_agent.run, {"report_type": "summary", "period": period})
        return result
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Server shutting down")


@router.get("/reports/person/{person_id}")
async def get_person_report(person_id: str):
    """Get report for a specific person."""
    try:
        result = await asyncio.to_thread(report_agent.run, {"report_type": "person", "person_id": person_id})
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Server shutting down")


@router.get("/reports/incidents")
async def get_recent_incidents(limit: int = Query(20, ge=1, le=100)):
    """Get recent incident reports."""
    from utils.db_utils import get_events_with_faces, get_memory_collection

    def _fetch():
        events_data = get_events_with_faces(limit=limit)
        events = events_data.get("events", [])

        # Batch-fetch all visit histories in one query (fix N+1)
        person_ids = list({e.get("person_id") for e in events if e.get("person_id")})
        memory_map = {}
        if person_ids:
            memory_col = get_memory_collection()
            for mem in memory_col.find({"person_id": {"$in": person_ids}}):
                memory_map[mem["person_id"]] = mem

        reports = []
        for event in events:
            person_id = event.get("person_id")
            visit_history = memory_map.get(person_id, {}) if person_id else {}
            result = report_agent.run({
                "report_type": "incident",
                "track_id": event.get("track_id"),
                "camera_id": event.get("camera_id"),
                "status": event.get("status"),
                "alert_level": event.get("alert_level"),
                "reason": event.get("reason"),
                "timestamp": event.get("timestamp"),
                "image_url": event.get("image_url"),
                "person_id": person_id,
                "name": event.get("person_name") or event.get("name"),
                "visit_history": visit_history,
            })
            reports.append(result)
        return {"reports": reports, "total": len(reports)}

    try:
        return await asyncio.to_thread(_fetch)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Server shutting down")
