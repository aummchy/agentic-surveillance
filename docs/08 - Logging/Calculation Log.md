# Calculation Log

> Full tabular breakdown of every confidence computation, written to `logs/calculation.log`.

**File**: `agents/scoring.py:29-137`

## When it's written

Every call to `compute_confidence()` logs a full breakdown when `ENABLE_CALC_LOG=True` (default: False).

## Format

```
═══════════════════════════════════════════════════════════════════════
                      CONFIDENCE SCORING FORMULA
═══════════════════════════════════════════════════════════════════════
  base = 0.65×sim + 0.15×quality + 0.10×track + 0.05×memory + 0.05×margin
  adjusted = base × (1 - 0.15×mask)
  confidence = int(round(1 + 99 × clip(adjusted, 0, 1)))
═══════════════════════════════════════════════════════════════════════
───────────────────────────────────────────────────────────────────────
  track=cam_01_1782040060_3  │  2026-07-09 10:30:15
───────────────────────────────────────────────────────────────────────
  Component              Raw   Normalized              Weighted
  ──────────────────── ────── ──────────── ────────────────────────────
  Similarity           0.704      0.825   0.65×0.825 = 0.536
  Face Quality         0.512      0.512   0.15×0.512 = 0.077
  Track Duration       10.5s      1.000   0.10×1.000 = 0.100
  Memory Boost         18.0       0.900   0.05×0.900 = 0.045
  Margin               0.202      0.675   0.05×0.675 = 0.034
  ──────────────────── ────── ──────────── ────────────────────────────
  BASE = 0.536 + 0.077 + 0.100 + 0.045 + 0.034 = 0.792
  MASK = 0.0 (unmasked) → no penalty
  ADJUSTED = 0.792 × 1.0000 = 0.792
  ──────────────────────────────────────────────────────────────────────
  CONFIDENCE = 1 + 99 × 0.792 = 79
  STATUS = known  (matched=true, threshold=0.45)
═══════════════════════════════════════════════════════════════════════
```

## What's logged

For each recognition:
- All 5 raw input values
- All 5 normalized values
- All 5 weighted contributions
- BASE sum
- MASK penalty effect
- ADJUSTED value
- Final CONFIDENCE
- STATUS + matched gate

## Configuration

| Setting | Default | Purpose |
|---------|---------|---------|
| `ENABLE_CALC_LOG` | False | Enable/disable calculation logging |
| `CALC_LOG_MAX_SIZE_MB` | 10 | Max file size before rotation (truncation) |

## Enabling

Set in `.env` or `config/config.jsonc`:
```
ENABLE_CALC_LOG=True
```

## File location

```
logs/calculation.log
```

Header is written once at startup (via `log_formula_header()`). Each subsequent call appends a block.

## See also
- [[Confidence Scoring]] — the formula being logged
- [[Scoring Module]] — implementation
- [[3-Tier Logging]] — other log outputs
