# Vector Search & Matching

> How face embeddings are searched against MongoDB and matched to known persons.

## End-to-end flow

```
Camera frame
  → SCRFD detect face
  → CLAHE contrast enhancement
  → ArcFace (buffalo_l): raw 512-dim embedding e
  → L2 normalize: ê = e / ‖e‖
  → Atlas $vectorSearch(ê, index="vector_index", metric=cosine)
  → Atlas returns top match + vectorSearchScore
  → raw_cosine = 2 × score − 1
  → compare_similarity(raw_cosine ≥ 0.45) → MatchResult
```

## Atlas score conversion

Atlas `$vectorSearch` returns `vectorSearchScore` in [0, 1]:
```
vectorSearchScore = (1 + raw_cosine) / 2
```

To recover raw cosine:
```
raw_cosine = (atlas_score × 2) - 1
```

Example: Atlas score 0.8615 → raw_cosine = 2(0.8615) - 1 = 0.723

## Match threshold

```
raw_cosine >= 0.45  →  matched = True
raw_cosine <  0.45  →  matched = False
```

Keep threshold ≤ 0.45. Higher rejects genuine same-person matches under indoor lighting.

## 3-person example

Stored vectors: ê_a (Aum), ê_r (Rahul), ê_p (Priya)

New query ê_q compared:

| vs | Dot product | Atlas score | Raw cosine |
|:--:|:-----------:|:-----------:|:----------:|
| ê_a | 0.723 | 0.8615 | 0.723 |
| ê_r | 0.281 | 0.6405 | 0.281 |
| ê_p | 0.119 | 0.5595 | 0.119 |

Best match = Aum (0.723 ≥ 0.45) → `MatchResult(matched=True, similarity=0.723)`

## Why same-person isn't 1.0

ArcFace embeddings shift between frames due to:
- Head rotation: ±0.05
- Lighting: ±0.03–0.10
- Expression: ±0.02–0.08
- Crop alignment: ±0.02–0.05

| Comparison | Typical cosine range |
|-----------|:--------------------:|
| Same person | 0.40 – 0.85 |
| Different person | −0.20 – 0.25 |

## Python cosine fallback

When Atlas fails (index missing, timeout):
```
1. Load up to 500 face documents
2. For each: L2-normalize, dot product
3. Sort descending
4. Return top matches above threshold
```

## Defensive re-normalization

Although embeddings are stored L2-normalized, Python fallback paths defensively re-normalize:
```python
query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-6)
stored_emb = stored_emb / (np.linalg.norm(stored_emb) + 1e-6)
```

## Similarity ranges

| Raw cosine | Meaning |
|:----------:|---------|
| 0.80 – 0.95 | Extremely strong same-person match |
| 0.65 – 0.80 | Strong same-person match |
| 0.45 – 0.65 | Possible / moderate same-person match |
| 0.25 – 0.45 | Weak similarity / often different people |
| < 0.25 | Usually different people |

## See also
- [[Atlas Score Conversion]] — score math
- [[Confidence Scoring]] — how similarity feeds into confidence (65% weight)
- [[Matching Agent]] — the matching function
- [[Database Utils]] — vector_search() implementation
