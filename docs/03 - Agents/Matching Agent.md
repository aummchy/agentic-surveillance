# Matching Agent

> Searches MongoDB for the closest known face to a given embedding. Returns a MatchResult with similarity score, person details, and margin.

**File**: `agents/matching_agent.py` (51 lines)

## Function: `run_matching_from_embedding(embedding, track_id)`

```python
def run_matching_from_embedding(embedding: list, track_id: str = "unknown") -> MatchResult:
```

### Input
- `embedding`: 512-dim L2-normalized list of floats
- `track_id`: for logging

### Processing
```
1. Validate embedding (non-null, 1-d, non-empty)
2. Call vector_search(embedding)  → db_utils.py
3. If no matches → return MatchResult(matched=False)
4. Extract best match (highest similarity)
5. Extract top2 (second-best) and margin (top1 - top2)
6. Log match_found event
7. Return MatchResult with all fields
```

### Output: `MatchResult`

```python
MatchResult(
    person_id="cam_01_1782040060_3",
    name="John Doe",
    role="visitor",
    tags=["auto_registered", "verified"],
    similarity_score=0.723,        # raw cosine (0-1 for non-opposite)
    image_url="https://...",
    matched=True,                   # similarity >= MATCH_THRESHOLD (0.45)
    verified=True,                  # "verified" in tags
    alert_level="low",
    second_best_similarity=0.499,  # top2
    margin=0.224,                   # top1 - top2
    candidate_count=3,
    all_candidates=[{"name": "John", "similarity": 0.723}, ...]
)
```

## How vector search works

**File**: `utils/db_utils.py:79-161`

### Atlas path (preferred)
```
1. MongoDB Atlas $vectorSearch pipeline:
   - index: "vector_index"
   - path: "latest_embedding"
   - queryVector: 512-dim embedding
   - numCandidates: 150
   - limit: 5

2. Atlas returns vectorSearchScore = (1 + raw_cosine) / 2
3. Convert: raw_cosine = (atlas_score × 2) - 1
4. Filter: keep matches where raw_cosine >= MATCH_THRESHOLD (0.45)
```

### Python fallback (when Atlas fails)
```
1. Load up to 500 face documents from MongoDB
2. For each: compute cosine similarity (dot product of normalized vectors)
3. Sort by similarity descending
4. Return top matches above threshold
```

## Match threshold

```
raw_cosine >= 0.45  →  matched = True
raw_cosine <  0.45  →  matched = False
```

The threshold is deliberately low (0.45) because:
- Indoor lighting causes same-person similarity to range 0.40-0.85
- Higher thresholds reject genuine matches under variable conditions
- The confidence scoring formula (5 components) provides additional discrimination

## Margin analysis

The margin (top1 - top2) indicates match specificity:
- **High margin (>0.15)**: Clear best match, high confidence
- **Medium margin (0.05-0.15)**: Good match but some ambiguity
- **Low margin (<0.05)**: Multiple similar faces, less certain

## See also
- [[Vector Search & Matching]] — detailed formulas and examples
- [[Atlas Score Conversion]] — score conversion math
- [[Database Utils]] — vector_search() implementation
- [[Confidence Scoring]] — how similarity feeds into confidence
