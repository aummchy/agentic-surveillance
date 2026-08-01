# Memory Agent

> Tracks visit history and provides context for recognition. Answers: Has this person been here before? How often? Is this a typical visit time? Should we be more confident?

**File**: `agents/memory.py` (273 lines)

## Role in system

```
Matching Agent → person_id, similarity
                    │
                    ▼
            Memory Agent
            (queries visit_memory collection)
                    │
                    ▼
            memory_context
            {visit_count, is_known, confidence_boost, typical_hours, ...}
                    │
                    ▼
            Recognition Agent (uses confidence_boost)
```

## Class: `MemoryAgent`

### Input to `run()`
```python
{
    "person_id": "cam_01_1782040060_3",
    "camera_id": "cam_01",
    "similarity": 0.723,
    "status": "known"
}
```

### Processing

```
1. get_or_create_memory(person_id)  ← atomic MongoDB upsert
2. Calculate days_since_last_visit
3. Check is_typical_time (within 2 hours of top 3 visit hours)
4. Check is_typical_camera
5. Calculate confidence_boost (-10 to +20)
6. Determine is_known (visit_count > 0 AND last_status in [known, verified, authorized])
7. Build reason string
```

### Output
```python
{
    "person_id": "cam_01_1782040060_3",
    "is_known": True,
    "visit_count": 218,
    "first_seen": "2026-07-01T09:00:00Z",
    "last_seen": "2026-07-09T10:30:00Z",
    "days_since_last_visit": 0,
    "avg_similarity": 0.71,
    "typical_hours": [9, 10, 14, 15],
    "typical_cameras": ["cam_01"],
    "is_typical_time": True,
    "is_typical_camera": True,
    "last_status": "known",
    "confidence_boost": 18,
    "reason": "Returning visitor seen 218 times. visited today. at a typical time."
}
```

## Confidence boost formula

**File**: `agents/memory.py:147-182`

```
boost = 0

# Returning visitor bonus
IF visit_count > 0:
    boost += min(10, visit_count × 2)       # +2 per visit, cap at +10

# Recency bonus
IF days_since_last <= 7:    boost += 5
ELIF days_since_last <= 30: boost += 2

# Consistency bonus
IF avg_similarity > 0.8:    boost += 3
ELIF avg_similarity > 0.6:  boost += 1

# Pattern bonuses
IF is_typical_time:          boost += 2
IF is_typical_camera:        boost += 1

# Penalty: current match much worse than usual
IF avg_similarity > 0 AND current_similarity < avg_similarity × 0.7:
    boost -= 5

RETURN clamp(boost, -10, +20)
```

### Range breakdown

| Component | Min | Max |
|-----------|-----|-----|
| Visit count | 0 | +10 |
| Recency | 0 | +5 |
| Consistency | 0 | +3 |
| Typical time | 0 | +2 |
| Typical camera | 0 | +1 |
| Penalty | -5 | 0 |
| **Total (clamped)** | **-10** | **+20** |

## `record_visit()` method

Called after recognition is complete for matched persons:

```python
def record_visit(person_id, camera_id, status, similarity, is_masked, visit_action):
```

Uses atomic MongoDB update:
```
$inc:  { visit_count: 1 }
$push: { similarity_history, status_history, typical_hours }  (with $slice for bounds)
$addToSet: { typical_cameras }
$set: { last_seen, last_camera, last_status, updated_at }
```

## is_typical_time check

```
hour_counts = count occurrences of each hour in typical_hours
common_hours = top 3 most frequent hours
is_typical_time = any(|current_hour - h| <= 2 for h in common_hours)
```

## is_known check

```
is_known = (visit_count > 0) AND (last_status in ["known", "verified", "authorized", "known_visitor"])
```

## See also
- [[Memory Boost]] — detailed boost formula with examples
- [[Database Utils]] — get_or_create_memory(), update_visit_memory()
- [[MongoDB Schema]] — visit_memory collection structure
- [[Recognition Agent]] — uses confidence_boost
