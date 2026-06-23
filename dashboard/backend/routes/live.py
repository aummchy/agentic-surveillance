import asyncio
import json
import base64
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Set

logger = logging.getLogger(__name__)
router = APIRouter()

connected_clients: Set[WebSocket] = set()
MAX_FRAME_SIZE = 1024 * 1024  # 1MB max frame size


async def broadcast_frame(frame_data: bytes):
    if not connected_clients:
        return

    if len(frame_data) > MAX_FRAME_SIZE:
        logger.warning("frame_too_large", size=len(frame_data))
        return

    message = json.dumps({
        "type": "frame",
        "data": base64.b64encode(frame_data).decode("utf-8")
    })

    async def _send(client):
        try:
            await asyncio.wait_for(client.send_text(message), timeout=0.5)
            return client, True
        except (asyncio.TimeoutError, Exception):
            return client, False

    results = await asyncio.gather(*[_send(c) for c in connected_clients], return_exceptions=True)
    disconnected = set()
    for result in results:
        if isinstance(result, tuple):
            client, ok = result
            if not ok:
                disconnected.add(client)
    connected_clients.difference_update(disconnected)


async def broadcast_event(event: dict):
    if not connected_clients:
        return

    message = json.dumps({
        "type": "event",
        "data": event
    })

    async def _send(client):
        try:
            await asyncio.wait_for(client.send_text(message), timeout=1.0)
            return client, True
        except (asyncio.TimeoutError, Exception):
            return client, False

    results = await asyncio.gather(*[_send(c) for c in connected_clients], return_exceptions=True)
    disconnected = set()
    for result in results:
        if isinstance(result, tuple):
            client, ok = result
            if not ok:
                disconnected.add(client)
    connected_clients.difference_update(disconnected)


async def broadcast_alert(alert_data: dict):
    if not connected_clients:
        return

    message = json.dumps({
        "type": "alert",
        "data": alert_data
    })

    async def _send(client):
        try:
            await asyncio.wait_for(client.send_text(message), timeout=1.0)
            return client, True
        except (asyncio.TimeoutError, Exception):
            return client, False

    results = await asyncio.gather(*[_send(c) for c in connected_clients], return_exceptions=True)
    disconnected = set()
    for result in results:
        if isinstance(result, tuple):
            client, ok = result
            if not ok:
                disconnected.add(client)
    connected_clients.difference_update(disconnected)


@router.websocket("/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"Client connected. Total clients: {len(connected_clients)}")

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connected_clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(connected_clients)}")
