# Embedding & Matching — How It Works

## 1. What buffalo_l returns

For each detected face, InsightFace returns a **512-d raw embedding vector**:

```
e = [e₁, e₂, ..., e₅₁₂]
```

This is just 512 floats — a numeric fingerprint of the face, not a name.

## 2. L2 normalization

Before storing or searching, the raw embedding is L2-normalized:

```
|e| = √(e₁² + e₂² + ... + e₅₁₂²)

ê = e / |e|      → ‖ê‖ = 1
```

**What Atlas stores:** the L2-normalized vector `ê` (512 floats).

```json
{
  "person_id": "cam_01_1782040060_1",
  "name": "aum",
  "latest_embedding": [0.013, -0.082, ..., 0.031]
}
```

## 3. Cosine similarity = dot product

Since all stored and query vectors have norm = 1:

```
cos(θ) = ê₁ · ê₂ = a₁b₁ + a₂b₂ + ... + a₅₁₂b₅₁₂
```

This single number (range -1 to 1) is the **raw cosine similarity**. Higher = more similar.

## 4. What Atlas returns

Atlas $vectorSearch returns a normalized `vectorSearchScore` (range 0-1). Convert back:

```
raw_cosine = 2 × vectorSearchScore − 1
```

Example: `Atlas score 0.8615 → raw_cosine = 2(0.8615)−1 = 0.723`

## 5. End-to-end flow

```
Camera frame
  → SCRFD detect face
  → CLAHE contrast enhancement
  → ArcFace (buffalo_l): raw 512-d embedding e
  → L2 normalize: ê = e/|e|
  → Atlas $vectorSearch(ê, index="vector_index", metric=cosine)
  → Atlas returns top match + vectorSearchScore
  → raw_cosine = 2 × score − 1
  → compare_similarity(raw_cosine ≥ 0.45) → MatchResult
```

## 6. 3-person example

Stored vectors: êₐ (Aum), êᵣ (Rahul), êₚ (Priya)

New query êq is compared:

| vs | Dot product | Atlas score | Raw cosine |
|----|:-----------:|:-----------:|:----------:|
| êₐ | 0.723 | (1+0.723)/2 = 0.8615 | 0.723 |
| êᵣ | 0.281 | (1+0.281)/2 = 0.6405 | 0.281 |
| êₚ | 0.119 | (1+0.119)/2 = 0.5595 | 0.119 |

Best match = Aum (0.723 ≥ 0.45 threshold) → `MatchResult(matched=True, similarity=0.723)`

## 7. Why same-person isn't 1.0

ArcFace embeddings shift between frames due to: head rotation (±0.05), lighting (±0.03–0.10), expression (±0.02–0.08), crop alignment (±0.02–0.05).

| Comparison | Typical cosine range |
|-----------|:--------------------:|
| Same person | 0.40 – 0.85 |
| Different person | −0.20 – 0.25 |

# 2nd 

| Value                 | Symbol                    |                                    Range | Meaning                               |
| --------------------- | ------------------------- | ---------------------------------------: | ------------------------------------- |
| Raw embedding         | (e)                       |                           no fixed range | 512-d face vector from `buffalo_l`    |
| Normalized embedding  | (\hat e)                  | components usually in `[-1,1]`, norm = 1 | stored/search vector                  |
| L2 norm               | (|e|)                     |                                     (>0) | magnitude of raw embedding            |
| Raw cosine similarity | (\hat e_1 \cdot \hat e_2) |                                 `[-1,1]` | actual face similarity                |
| Atlas score           | `vectorSearchScore`       |                                  `[0,1]` | Atlas normalized search score         |
| Detection score       | `det_score`               |                                  `[0,1]` | face detector confidence              |
| Final confidence      | `confidence`              |                                `[0,100]` | your app-level recognition confidence |

| Raw cosine      | Meaning                                  |
| --------------- | ---------------------------------------- |
| **0.80 – 0.95** | extremely strong same-person match       |
| **0.65 – 0.80** | strong same-person match                 |
| **0.45 – 0.65** | possible / moderate same-person match    |
| **0.25 – 0.45** | weak similarity / often different people |
| **< 0.25**      | usually different people                 |


# new in current system 
Implementation Note — Defensive Re-normalization in Python Fallback Paths

Although the canonical pipeline stores and queries L2-normalized face embeddings, the Python fallback comparison paths (_python_cosine_scan, find_similar_unknowns, find_similar_faces) defensively re-normalize embeddings with np.linalg.norm(...) before cosine computation.

// “Treat raw cosine 0.25 as confidence 0, raw cosine 0.80 as confidence 100, and linearly scale everything in between.”
// confidence = clip(((raw_cosine - 0.25) / 0.55) * 100, 0, 100)