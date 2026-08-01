# Backend API

> FastAPI server on port 8000. REST endpoints for faces, events, reports, chat. WebSocket for live video feed.

**File**: `dashboard/backend/main.py`

## Startup

Runs inside `main.py` as a daemon thread:
```python
config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
server = uvicorn.Server(config)
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
```

## REST endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/faces` | List known faces (with pagination) |
| GET | `/api/faces/{person_id}` | Get face by ID |
| POST | `/api/faces/{person_id}/verify` | Verify a face (set name, role, tags) |
| DELETE | `/api/faces/{person_id}` | Delete a face |
| GET | `/api/events` | List events (with pagination, status filter) |
| GET | `/api/stats` | Dashboard statistics (total unknown, verified, events today) |
| POST | `/api/chat` | Chat with LLM about surveillance data |
| GET | `/api/unknowns` | Get recent unknown faces |
| PUT | `/api/faces/{person_id}/alert-level` | Update alert level |

## WebSocket endpoints

| Path | Purpose |
|------|---------|
| `/ws/live` | Live video frame + events + alerts broadcast |

### Frame broadcast format
```json
{
    "type": "frame",
    "data": "base64-encoded-jpeg..."
}
```

### Event broadcast format
```json
{
    "type": "event",
    "data": {
        "track_id": "...",
        "status": "known",
        "name": "John",
        "alert_level": "low",
        ...
    }
}
```

### Alert broadcast format
```json
{
    "type": "alert",
    "data": {
        "person_id": "...",
        "status": "unknown",
        "alert_level": "high",
        "nl_summary": "Unknown person detected after hours",
        ...
    }
}
```

## CORS

Configured for localhost:5173 (Vite dev server).

## Chat flow

```
POST /api/chat
  → Parse intent from user message (keywords)
  → Fetch relevant data (stats, events, unknowns)
  → Build prompt with data context
  → LLM chat_completion()
  → Return response
```

## See also
- [[WebSocket Live Feed]] — live frame broadcasting
- [[Frontend]] — React dashboard
- [[LLM Client]] — chat completion backend
