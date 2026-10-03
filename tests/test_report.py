"""Tests for the Reports component (agents/report.py + routes/reports.py).

Covers the Phase 3 review findings (H1, H2) and the report contract:
- H1: naive Mongo datetimes no longer crash _build_summary (aware-naive subtraction)
- H2: _incident_report consumes the batch-fetched visit_history, no N+1 query
- GET /api/reports/incidents returns 200 (was HTTP 500 with any visited person)
"""

from datetime import datetime, timedelta, timezone

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.report import ReportAgent
from config.status import Status


@pytest.fixture
def agent():
    return ReportAgent()


@pytest.fixture
def no_llm():
    with patch("agents.report.llm_client.generate_incident_summary", return_value=None):
        yield


NAIVE_LAST_SEEN = datetime.utcnow() - timedelta(days=5)

MEMORY_DOC = {
    "person_id": "p1",
    "visit_count": 12,
    "last_seen": NAIVE_LAST_SEEN,
    "avg_similarity": 0.61,
}


def build_summary(agent, last_seen, visit_count=3):
    return agent._build_summary(
        Status.BLACKLIST, "Alice", "cam_01",
        datetime.now(timezone.utc).isoformat(), visit_count, last_seen,
    )


def test_build_summary_naive_last_seen(agent, no_llm):
    """H1: naive datetime (Mongo style) must not raise TypeError."""
    summary = build_summary(agent, NAIVE_LAST_SEEN, visit_count=3)
    assert "last seen 5 days ago" in summary
    assert "visit #4" in summary


def test_build_summary_aware_last_seen(agent, no_llm):
    aware = datetime.now(timezone.utc) - timedelta(days=2)
    summary = build_summary(agent, aware, visit_count=1)
    assert "last seen 2 days ago" in summary


def test_build_summary_none_last_seen_first_time(agent, no_llm):
    summary = build_summary(agent, None, visit_count=0)
    assert "first-time visitor" in summary


def test_build_summary_seen_today(agent, no_llm):
    naive_now = datetime.utcnow() - timedelta(hours=1)
    summary = build_summary(agent, naive_now, visit_count=0)
    assert "previously seen today" in summary


def test_incident_consumes_batched_history_no_query(agent, no_llm):
    """H2: visit_history from the caller is used; get_visit_history never called."""
    with patch("agents.report.get_visit_history") as q:
        result = agent.run({
            "report_type": "incident",
            "track_id": "cam_01_1_7",
            "camera_id": "cam_01",
            "status": int(Status.BLACKLIST),
            "alert_level": "critical",
            "reason": "blacklist match",
            "timestamp": "2026-10-03T10:00:00+00:00",
            "image_url": None,
            "person_id": "p1",
            "name": "Bad Bob",
            "visit_history": MEMORY_DOC,
        })
        q.assert_not_called()

    assert result["report_type"] == "incident"
    assert result["details"]["visit_count"] == 12
    assert "visit #13" in result["summary"]


def test_incident_shape_keys(agent, no_llm):
    result = agent.run({
        "report_type": "incident",
        "track_id": "t1",
        "camera_id": "cam_01",
        "status": int(Status.UNKNOWN),
        "alert_level": "low",
        "reason": "unknown visitor",
        "timestamp": "2026-10-03T10:00:00+00:00",
        "person_id": None,
        "name": "Unknown",
        "visit_history": {},
    })
    for key in ("title", "summary", "recommendation", "severity", "details"):
        assert key in result
    assert result["details"]["visit_count"] == 0
    assert "first-time visitor" in result["summary"]


class FakeMemoryCollection:
    def __init__(self, docs):
        self._docs = docs

    def find(self, query):
        ids = query["person_id"]["$in"]
        return [d for d in self._docs if d["person_id"] in ids]


def test_incidents_endpoint_returns_200(no_llm):
    """H1 end-to-end: naive last_seen in Mongo previously yielded HTTP 500."""
    from dashboard.backend.routes import reports as reports_route

    app = FastAPI()
    app.include_router(reports_route.router, prefix="/api")

    events = [{
        "track_id": "cam_01_1_7",
        "camera_id": "cam_01",
        "status": int(Status.BLACKLIST),
        "alert_level": "critical",
        "reason": "blacklist match",
        "timestamp": "2026-10-03T10:00:00+00:00",
        "image_url": None,
        "person_id": "p1",
        "person_name": "Bad Bob",
    }]

    with patch("utils.db_utils.get_events_with_faces",
               return_value={"events": events, "total": 1, "limit": 20, "offset": 0}), \
         patch("utils.db_utils.get_memory_collection",
               return_value=FakeMemoryCollection([MEMORY_DOC])):
        client = TestClient(app)
        resp = client.get("/api/reports/incidents?limit=5")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    report = body["reports"][0]
    assert report["details"]["visit_count"] == 12
    assert "Bad Bob" in report["summary"]
