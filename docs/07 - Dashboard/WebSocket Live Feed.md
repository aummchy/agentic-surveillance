# WebSocket Live Feed

> Real-time video, events, and alerts broadcast to the dashboard via WebSocket.

## Frame broadcasting pipeline

**File**: `agents/track_processor.py:44-62`

```
1. Camera loop calls on_frame_annotated(annotated_frame)
2. frame_counter += 1
3. If counter % FRAME_SKIP (2) != 0 → skip (only every 2nd frame)
4. Submit to JPEG encode executor:
   a. preview = cv2.resize(frame, (640, 360))
   b. JPEG encode (quality=90)
   c. broadcast_frame(jpeg_bytes) via asyncio coroutine
```

## WebSocket protocol

### Client → Server
```
"ping"  →  Server responds {"type": "pong"}
```

### Server → Client

| Message type | Format | Content |
|-------------|--------|---------|
| `frame` | `{"type": "frame", "data": "base64..."}` | JPEG frame (640×360) |
| `event` | `{"type": "event", "data": {...}}` | Track event payload |
| `alert` | `{"type": "alert", "data": {...}}` | Alert payload |
| `pong` | `{"type": "pong"}` | Heartbeat response |

### Event payload
```json
{
    "track_id": "cam_01_1782040060_3",
    "camera_id": "cam_01",
    "status": "known",
    "alert_level": "low",
    "person_id": "...",
    "name": "John Doe",
    "similarity_score": 0.723,
    "image_url": "...",
    "timestamp": "2026-07-09T10:30:00Z"
}
```

### Alert payload
```json
{
    "person_id": "cam_01_1782040060_3",
    "status": "unknown",
    "name": "Unknown",
    "alert_level": "high",
    "image_url": "...",
    "timestamp": "2026-07-09T10:30:00Z",
    "camera_id": "cam_01",
    "reason": "Unknown person detected after hours",
    "nl_summary": "An unidentified individual was detected..."
}
```

## Client management

```python
connected_clients: Set[WebSocket] = set()
```

- Auto-disconnect on send failure (1s timeout per client)
- Ping/pong heartbeat
- CORS validation on connect

## Performance

- **Frame size**: ~50KB JPEG at quality 90
- **Max frame size**: 1MB cap
- **Broadcast rate**: Every 2nd frame (~15fps at 30fps camera)
- **Resolution**: 640×360 (downscaled from 1280×720)

## Origin validation

WebSocket `/ws/live` validates `Origin` header to prevent cross-origin hijacking.

## See also
- [[Backend API]] — WebSocket server setup
- [[Frontend]] — React client
- [[Track Processor]] — handle_frame() triggers broadcast
