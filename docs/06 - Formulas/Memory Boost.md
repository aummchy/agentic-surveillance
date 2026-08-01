# Memory Boost

> How visit history affects recognition confidence through a boost value (-10 to +20).

**File**: `agents/memory.py:147-182`

## Formula

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

# Penalty
IF avg_similarity > 0 AND current_similarity < avg_similarity × 0.7:
    boost -= 5

RETURN clamp(boost, -10, +20)
```

## Range breakdown

| Component | Min | Max | Condition |
|-----------|-----|-----|-----------|
| Visit count | 0 | +10 | +2 per visit, cap at 10 visits |
| Recency | 0 | +5 | +5 if within 7 days, +2 if within 30 |
| Consistency | 0 | +3 | +3 if avg similarity > 0.8, +1 if > 0.6 |
| Typical time | 0 | +2 | Within 2 hours of top 3 visit hours |
| Typical camera | 0 | +1 | Same camera as usual |
| Penalty | -5 | 0 | Current similarity much lower than average |
| **Total** | **-10** | **+20** | Clamped |

## Impact on confidence

Memory contributes 5% to [[Confidence Scoring]]:
```
memory_norm = clip(boost, 0, 20) / 20
memory_contribution = 0.05 × memory_norm → 0 to 0.05 of final confidence
```

Example: boost=18 → memory_norm=0.9 → contribution=0.045 (4.5% of confidence)

## is_typical_time check

```
hour_counts = count occurrences of each hour in typical_hours
common_hours = top 3 most frequent hours
is_typical_time = any(|current_hour - h| <= 2 for h in common_hours)
```

## Worked example

```
Person has:
  visit_count = 10 visits
  days_since_last = 3 days
  avg_similarity = 0.72
  current_similarity = 0.70
  is_typical_time = True
  is_typical_camera = True

boost = 0
  + min(10, 10×2) = 10     (returning visitor)
  + 5                       (visited 3 days ago, within 7)
  + 1                       (avg_sim 0.72 > 0.6)
  + 2                       (typical time)
  + 1                       (typical camera)
  = 19

  Check penalty: 0.70 < 0.72×0.7 = 0.504? No → no penalty

  Final boost = 19 (within [-10, +20] range)
```

## See also
- [[Memory Agent]] — where the boost is computed
- [[Confidence Scoring]] — how boost affects confidence
- [[Database Utils]] — visit_memory operations
