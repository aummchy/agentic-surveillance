# Atlas Score Conversion

> How MongoDB Atlas Vector Search scores map to raw cosine similarity.

## The conversion

Atlas `$vectorSearch` returns `vectorSearchScore` in [0, 1]:
```
vectorSearchScore = (1 + raw_cosine) / 2
```

To recover raw cosine:
```
raw_cosine = (atlas_score × 2) - 1
```

**File**: `utils/embedding_utils.py:132-133`

## Examples

| Atlas score | Raw cosine | Meaning |
|:-----------:|:----------:|---------|
| 1.0000 | 1.000 | Identical vectors |
| 0.9000 | 0.800 | Very similar |
| 0.8615 | 0.723 | Strong match |
| 0.7250 | 0.450 | Match threshold |
| 0.6250 | 0.250 | Weak similarity |
| 0.5000 | 0.000 | Orthogonal (no similarity) |
| 0.0000 | -1.000 | Opposite vectors |

## Why this matters

- **Match threshold** is defined in raw cosine space: `MATCH_THRESHOLD = 0.45`
- **Atlas returns** in normalized [0,1] space
- **Always convert** before comparing: `raw_cosine = (atlas_score × 2) - 1`
- Then compare: `raw_cosine >= 0.45`

## In code

```python
# utils/embedding_utils.py
def atlas_score_to_cosine(atlas_score: float) -> float:
    return (atlas_score * 2) - 1

# utils/db_utils.py — vector_search()
for r in results:
    raw_cosine = atlas_score_to_cosine(r.get("score", 0))
    all_results.append({
        "similarity_score": raw_cosine,
        ...
    })
```

## Defensive re-normalization

Python fallback paths re-normalize embeddings before computing cosine:
```python
query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-6)
stored_emb = stored_emb / (np.linalg.norm(stored_emb) + 1e-6)
similarity = float(np.dot(query_emb, stored_emb))
```

This is defensive — the canonical pipeline stores L2-normalized vectors, but fallback paths don't assume that.

## See also
- [[Vector Search & Matching]] — full search flow
- [[Matching Agent]] — uses converted scores
- [[Confidence Scoring]] — similarity feeds into confidence
