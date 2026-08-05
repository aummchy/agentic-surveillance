import asyncio
import json
import base64
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Set
from config import settings

logger = structlog.get_logger(__name__)
router = APIRouter()

connected_clients: Set[WebSocket] = set()
MAX_FRAME_SIZE = 1024 * 1024  # 1MB max frame size
_frame_counter = 0

_SEND_TIMEOUT = 3.0


async def _send_or_disconnect(client, message, label=""):
    try:
        await asyncio.wait_for(client.send_text(message), timeout=_SEND_TIMEOUT)
        return client, True
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning("client_send_timeout", client=client, label=label, error=str(e))
        return client, False


async def _broadcast(message: str, label: str):
    """Send message to all connected clients, disconnecting silent ones."""
    if not connected_clients:
        return

    results = await asyncio.gather(
        *[_send_or_disconnect(c, message, label) for c in list(connected_clients)],
        return_exceptions=True
    )
    disconnected = set()
    for result in results:
        if isinstance(result, tuple):
            client, ok = result
            if not ok:
                disconnected.add(client)
    if disconnected:
        connected_clients.difference_update(disconnected)
        logger.warning("clients_dropped_silent", count=len(disconnected), remaining=len(connected_clients))


async def broadcast_frame(frame_data: bytes):
    global _frame_counter
    if not connected_clients:
        return

    _frame_counter += 1

    if len(frame_data) > MAX_FRAME_SIZE:
        logger.warning("frame_too_large", size=len(frame_data))
        return

    message = json.dumps({
        "type": "frame",
        "data": base64.b64encode(frame_data).decode("utf-8")
    })

    if _frame_counter % 100 == 1:
        logger.info("frame_broadcast", frame_num=_frame_counter, clients=len(connected_clients), size=len(frame_data))

    await _broadcast(message, "frame")


async def broadcast_event(event: dict):
    message = json.dumps({"type": "event", "data": event})
    await _broadcast(message, "event")


async def broadcast_alert(alert_data: dict):
    message = json.dumps({"type": "alert", "data": alert_data})
    await _broadcast(message, "alert")


@router.websocket("/live")
async def websocket_live(websocket: WebSocket):
    origin = websocket.headers.get("origin", "")
    allowed_origins = set(settings.ALLOWED_ORIGINS)
    if origin and origin not in allowed_origins:
        await websocket.close(code=4003, reason="origin not allowed")
        return

    await websocket.accept()
    connected_clients.add(websocket)
    logger.info("client_connected", total=len(connected_clients))

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("websocket_error", error=str(e))
    finally:
        connected_clients.discard(websocket)
        logger.info("client_disconnected", total=len(connected_clients))
