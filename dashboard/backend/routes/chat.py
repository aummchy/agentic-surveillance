"""
Dashboard Chat Endpoint — Conversational interface to the surveillance system.

Operators can ask natural language questions about events, persons, and system status.
The LLM processes the query and returns a structured response.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
import json
import structlog

from utils import llm_client
from config import settings
from utils.db_utils import (
    get_stats, get_events_with_faces, get_unknown_faces,
    get_visit_history, get_memory_stats, get_face_by_id,
)

logger = structlog.get_logger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message")


class ChatResponse(BaseModel):
    response: str
    data: Optional[dict] = None
    llm_available: bool = True


SYSTEM_PROMPT = """You are a surveillance system assistant. You answer operator questions about surveillance events, persons, and system status.

You have access to the following database query results. Use them to answer questions accurately.

Available data types:
- stats: {total_unknown, total_verified, events_today, unknown_today}
- events: list of recent events with status, camera_id, alert_level, person info
- unknown_faces: list of unverified persons
- person: visit history for a specific person (visit_count, typical_hours, typical_cameras, etc.)
- memory_stats: {total_persons, known_persons, recent_unknowns}

Rules:
1. Always answer based on the provided data. Do not make up information.
2. Keep responses concise (2-4 sentences max).
3. If the data is empty or unavailable, say so clearly.
4. Be professional and factual.
5. If you cannot determine the answer, say "I don't have enough information to answer that."
6. Do not use emojis or markdown formatting — plain text only.
"""


def _handle_query(user_message: str) -> tuple:
    """Process the user message, run any needed queries, return (response_text, data_dict).

    Returns (str, dict|None)
    """
    lower = user_message.lower().strip()

    # Route to appropriate data source based on query intent
    data = None

    if any(kw in lower for kw in ["stats", "status", "overview", "summary", "how many", "total"]):
        data = {"type": "stats", **get_stats()}

    elif any(kw in lower for kw in ["unknown", "unverified", "new"]):
        result = get_unknown_faces(limit=10)
        data = {"type": "unknown_faces", "faces": result.get("faces", []), "total": result.get("total", 0)}

    elif any(kw in lower for kw in ["recent", "events", "happened", "alerts"]):
        result = get_events_with_faces(limit=10)
        events = result.get("events", [])
        # Serialize datetime objects for JSON
        for e in events:
            if hasattr(e.get("timestamp"), "isoformat"):
                e["timestamp"] = e["timestamp"].isoformat()
            if hasattr(e.get("last_seen"), "isoformat"):
                e["last_seen"] = e["last_seen"].isoformat()
        data = {"type": "events", "events": events, "total": result.get("total", 0)}

    elif any(kw in lower for kw in ["memory", "known persons", "known people", "visitors"]):
        data = {"type": "memory_stats", **get_memory_stats()}

    # If we got data, format a prompt for the LLM with context
    if data:
        data_str = json.dumps(data, default=str, indent=2)
        prompt = f"Operator question: {user_message}\n\nAvailable data:\n{data_str}"
        reply = llm_client.chat_completion(prompt, system=SYSTEM_PROMPT, max_tokens=300)
        if reply:
            return reply, data
        # LLM failed — return raw data summary
        if data.get("type") == "stats":
            return (
                f"Unknown persons: {data.get('total_unknown', 0)}, "
                f"Verified: {data.get('total_verified', 0)}, "
                f"Events today: {data.get('events_today', 0)}, "
                f"Unknowns today: {data.get('unknown_today', 0)}",
                data,
            )
        return f"Here is the data I found: {json.dumps(data, default=str)[:500]}", data

    # No specific data query matched — general LLM response
    reply = llm_client.chat_completion(user_message, system=SYSTEM_PROMPT, max_tokens=300)
    if reply:
        return reply, None

    return "I'm unable to process your request right now. Please try again later.", None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Conversational endpoint for querying the surveillance system.

    Send a natural language question and get a response based on live data.
    """
    if not llm_client.is_available():
        return ChatResponse(
            response="LLM service (Ollama) is not available. Please ensure Ollama is running.",
            data=None,
            llm_available=False,
        )

    try:
        response_text, data = await asyncio.to_thread(_handle_query, request.message)
        return ChatResponse(response=response_text, data=data, llm_available=True)
    except Exception as e:
        logger.error("chat_endpoint_error", error=str(e))
        raise HTTPException(status_code=500, detail="Chat processing failed")


@router.get("/chat/health")
async def chat_health():
    """Check if the LLM backend is available."""
    available = llm_client.is_available()
    return {"available": available, "model": settings.OLLAMA_MODEL}
