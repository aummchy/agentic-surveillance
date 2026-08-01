# MongoDB Schema

> Three collections in the `surveillance` database, each serving a distinct purpose.

## Collection: `faces`

The **person database**. Every known person has one document with their embeddings, metadata, and verification status.

```json
{
  "person_id": "cam_01_1782040060_3",
  "name": "John Doe",
  "role": "visitor",
  "tags": ["auto_registered", "verified"],
  "verified": true,
  "verified_at": "2026-07-09T10:30:00Z",
  "verified_by": "operator",
  "alert_level": "low",
  "embeddings": [[0.013, -0.082, ...]],     // Last 25 embeddings (FIFO cap)
  "latest_embedding": [0.013, -0.082, ...],  // Best quality embedding (for vector search)
  "mean_embedding": [0.012, -0.080, ...],    // L2-normalized mean of all embeddings
  "latest_embedding_quality": 0.72,           // Quality score of latest_embedding
  "embedding_model": "arcface",
  "images": [
    {
      "id": "uuid",
      "url": "https://res.cloudinary.com/...",
      "captured_at": "2026-07-09T10:30:00Z"
    }
  ],
  "source": {
    "camera_id": "cam_01",
    "captured_at": "2026-07-09T10:30:00Z"
  },
  "quality_scores": {},
  "created_at": "2026-07-09T10:30:00Z",
  "updated_at": "2026-07-09T10:30:00Z"
}
```

### Indexes
| Index | Type | Purpose |
|-------|------|---------|
| `vector_index` | Atlas Vector Search | 512-dim cosine search on `latest_embedding` |
| `latest_embedding` | Standard | Python fallback cosine scan |
| `person_id` | Standard | Unique lookup |

### Key operations
- `store_face()` — creates new document or merges into existing (dedup)
- `update_face()` — quality-gated embedding update (only overwrites if new quality > stored)
- `vector_search()` — Atlas `$vectorSearch` with Python numpy fallback

## Collection: `events`

Every finalized track is logged here. One document per person appearance.

```json
{
  "track_id": "cam_01_1782040060_3",
  "camera_id": "cam_01",
  "timestamp": "2026-07-09T10:30:00Z",
  "status": "known",
  "alert_level": "low",
  "person_id": "cam_01_1782040060_3",
  "name": "John Doe",
  "is_masked": false,
  "similarity_score": 0.723,
  "image_url": "https://res.cloudinary.com/...",
  "reason": "Track finalized: known",
  "alerted": false
}
```

### Indexes
| Index | Purpose |
|-------|---------|
| `track_id` | Lookup by track |
| `timestamp` | Time-range queries |
| `status` | Filter by status |

## Collection: `visit_memory`

Tracks visit history per person. One document per person_id (upserted atomically).

```json
{
  "person_id": "cam_01_1782040060_3",
  "visit_count": 218,
  "first_seen": "2026-07-01T09:00:00Z",
  "last_seen": "2026-07-09T10:30:00Z",
  "last_camera": "cam_01",
  "last_status": "known",
  "typical_hours": [9, 10, 14, 15, 9, 10, ...],   // Last 20 visit hours
  "typical_cameras": ["cam_01"],
  "avg_similarity": 0.71,
  "similarity_history": [0.72, 0.68, 0.75, ...],   // Last 10 similarities
  "status_history": [{"status": "known", "timestamp": "..."}],  // Last 10
  "created_at": "2026-07-01T09:00:00Z",
  "updated_at": "2026-07-09T10:30:00Z"
}
```

### Indexes
| Index | Purpose |
|-------|---------|
| `person_id` (unique) | One doc per person |
| `last_seen` | Recent visitor queries |

### Atomic update pattern
Uses `find_one_and_update` with `$inc`, `$push/$slice`, `$addToSet`, `$set` — single round-trip, no TOCTOU race.

## Data relationships

```
faces.person_id ──► events.person_id     (one face → many events)
faces.person_id ──► visit_memory.person_id (one face → one memory)
```

## See also
- [[Database Utils]] — all MongoDB operations
- [[Vector Search & Matching]] — how vector search works
- [[Memory Boost]] — how visit history affects recognition
