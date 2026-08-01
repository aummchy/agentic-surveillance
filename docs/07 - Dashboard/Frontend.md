# Frontend

> React + Vite dashboard. Live video feed via WebSocket, event history, face management.

**Folder**: `dashboard/frontend/`

## Setup

```bash
cd dashboard/frontend
npm install
npm run dev
```

Opens at http://localhost:5173.

## Key features

- **Live video feed**: WebSocket connection to `/ws/live`, displays JPEG frames
- **Event stream**: Real-time events and alerts overlaid on video
- **Face management**: View unknown faces, verify/reject, set names
- **Event history**: Paginated list of all events with filters
- **Statistics**: Today's stats (unknowns, verified, events)
- **Chat**: Ask questions about surveillance data

## Technical notes

- `axios` pinned to `1.7.9` — versions ≥1.7.10 break Vite's esbuild
- WebSocket reconnection handled client-side
- Frames base64-decoded and displayed as `<img>` blobs

## See also
- [[Backend API]] — REST + WebSocket server
- [[WebSocket Live Feed]] — frame broadcasting details
