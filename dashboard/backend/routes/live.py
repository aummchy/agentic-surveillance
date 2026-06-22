import asyncio
import json
import base64
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Set

logger = logging.getLogger(__name__)
router = APIRouter()

connected_clients: Set[WebSocket] = set()


async def broadcast_frame(frame_data: bytes):
    if not connected_clients:
        return

    message = json.dumps({
        "type": "frame",
        "data": base64.b64encode(frame_data).decode("utf-8")
    })

    disconnected = set()
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.add(client)

    connected_clients.difference_update(disconnected)


async def broadcast_event(event: dict):
    if not connected_clients:
        return

    message = json.dumps({
        "type": "event",
        "data": event
    })

    disconnected = set()
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.add(client)

    connected_clients.difference_update(disconnected)


async def broadcast_alert(alert_data: dict):
    if not connected_clients:
        return

    message = json.dumps({
        "type": "alert",
        "data": alert_data
    })

    disconnected = set()
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
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
