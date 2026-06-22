from fastapi import APIRouter, HTTPException, Query
from agents.report import ReportAgent

router = APIRouter()
report_agent = ReportAgent()


@router.get("/reports/stats")
async def get_stats():
    """Get dashboard statistics."""
    result = report_agent.run({"report_type": "stats"})
    return result.get("stats", {})


@router.get("/reports/summary")
async def get_summary(period: str = Query("daily", pattern="^(daily|weekly)$")):
    """Get daily or weekly summary."""
    result = report_agent.run({"report_type": "summary", "period": period})
    return result


@router.get("/reports/person/{person_id}")
async def get_person_report(person_id: str):
    """Get report for a specific person."""
    result = report_agent.run({"report_type": "person", "person_id": person_id})
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/reports/incidents")
async def get_recent_incidents(limit: int = Query(20, ge=1, le=100)):
    """Get recent incident reports."""
    from utils.db_utils import get_events_with_faces
    events_data = get_events_with_faces(limit=limit)
    events = events_data.get("events", [])

    reports = []
    for event in events:
        result = report_agent.run({
            "report_type": "incident",
            "track_id": event.get("track_id"),
            "camera_id": event.get("camera_id"),
            "status": event.get("status"),
            "alert_level": event.get("alert_level"),
            "reason": event.get("reason"),
            "timestamp": event.get("timestamp"),
            "image_url": event.get("image_url"),
            "person_id": event.get("person_id"),
            "name": event.get("person_name") or event.get("name"),
        })
        reports.append(result)

    return {"reports": reports, "total": len(reports)}
