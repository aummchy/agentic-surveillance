# Database Utils

> All MongoDB operations: vector search, face CRUD, event logging, visit memory, and Atlas index management.

**File**: `utils/db_utils.py` (845 lines)

## Connection management

Thread-safe lazy singleton:
```python
_client: Optional[MongoClient] = None
_client_lock = threading.Lock()

def get_client():
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = MongoClient(settings.MONGODB_URI)
    return _client
```

Per-collection locks for connection creation:
```python
_collection_locks = {
    "faces": threading.Lock(),
    "events": threading.Lock(),
    "memory": threading.Lock(),
}
```

## Collections

| Collection | Getter | Purpose |
|-----------|--------|---------|
| `faces` | `get_faces_collection()` | Person DB + embeddings + vector index |
| `events` | `get_events_collection()` | Track event log |
| `visit_memory` | `get_memory_collection()` | Visit history |

## Vector search: `vector_search(embedding, filter_role, limit)`

### Atlas path (preferred)
```
1. Build $vectorSearch pipeline:
   - index: "vector_index"
   - path: "latest_embedding"
   - queryVector: 512-dim embedding
   - numCandidates: 150
   - limit: 5
2. Execute with maxTimeMS=5000
3. Convert scores: raw_cosine = (atlas_score × 2) - 1
4. Filter: keep matches where raw_cosine >= MATCH_THRESHOLD
5. Return {matches, top2, margin, all_candidates}
```

### Python fallback (on Atlas failure)
```
1. Load up to SCAN_LIMIT (500) face documents
2. For each: L2-normalize, compute dot product
3. Sort by similarity descending
4. Return top matches above threshold
```

## Face operations

### `store_face(person_id, name, role, embedding, image_url, ...)`
```
1. IF NOT skip_search:
   - vector_search(embedding, limit=3) for dedup
   - For each match with similarity >= DEDUP_SIMILARITY_THRESHOLD (0.40):
     a. IF existing face is verified → SKIP
     b. ELSE → merge via update_face()
2. Create new document with embeddings[], latest_embedding, mean_embedding, images[]
```

### `update_face(person_id, image_url, embedding, quality_score, ...)`
```
1. $set: name, tags, verified, alert_level, updated_at
2. $push: images, embeddings (with $slice -25 for cap)
3. Recompute mean_embedding from last 25 embeddings
4. Quality-gated latest_embedding update:
   IF new quality > stored quality → overwrite
   ELSE → skip
```

### `find_similar_unknowns(embedding, threshold)`
```
1. Load up to 500 unknowns (role="unknown")
2. Compute cosine similarity for each
3. Return matches where similarity >= DEDUP_SIMILARITY_THRESHOLD (0.40)
```

## Visit memory operations

### `get_or_create_memory(person_id)` — atomic upsert
```
find_one_and_update(
    {person_id},
    {$setOnInsert: {visit_count: 0, first_seen: now, ...}},
    upsert=True,
    return_document=ReturnDocument.AFTER
)
```

### `update_visit_memory(person_id, camera_id, status, similarity)` — atomic update
```
find_one_and_update(
    {person_id},
    {
        $inc:  { visit_count: 1 },
        $push: {
            similarity_history: { $each: [sim], $slice: -10 },
            status_history: { $each: [{status, timestamp}], $slice: -10 },
            typical_hours: { $each: [hour], $slice: -20 },
        },
        $addToSet: { typical_cameras: camera_id },
        $set: { last_seen, last_camera, last_status, updated_at },
    },
    upsert=True,
    return_document=ReturnDocument.AFTER
)
```

Single round-trip, no TOCTOU race.

## Event logging: `log_event(...)`

Inserts one document per finalized track into `events` collection.

## Atlas index management

### `check_atlas_search_index()`
Checks if `vector_index` exists. Warns if missing (system falls back to Python scan).

### `backfill_missing_embeddings()`
Fixes faces with empty `latest_embedding` by copying from `embeddings[]` array. Runs once per process.

## See also
- [[MongoDB Schema]] — collection structures
- [[Vector Search & Matching]] — detailed formulas
- [[Memory Boost]] — how visit data affects recognition
- [[Atlas Score Conversion]] — score conversion math
