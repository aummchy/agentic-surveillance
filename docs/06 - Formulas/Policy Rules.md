# Policy Rules

> The 9-rule priority tree that determines alerts, registration, and status assignments.

**File**: `agents/policy.py`

## Rule priority diagram

```
RULE 1: BLACKLIST ──────────────────── "blacklist" in tags
        │                              → critical alert, always fires
        ▼
RULE 2: AUTHORIZED ─────────────────── "authorized" in tags
        │                              → no alert
        ▼
RULE 3: VERIFIED ───────────────────── verified=True AND status="known"
        │                              → no alert
        ▼
RULE 4a: AUTO-REG SELF-MATCH ──────── "auto_registered" in tags AND sim > 0.65
        │                              → known_visitor, low alert
        ▼
RULE 4b: KNOWN (MEMORY) ───────────── matched AND is_known_from_memory
        │                              → known_visitor, low alert
        ▼
RULE 5: MATCHED ────────────────────── similarity >= 0.85 OR confidence >= 80
        │   5a: high match             → known_visitor, low alert
        │   5b: mid match (≥0.45)      → unknown, low alert
        ▼
RULE 6: HIDDEN ─────────────────────── visibility == "hidden"
        │                              → high alert (avoiding detection)
        ▼
RULE 7: MASKED ─────────────────────── is_masked OR visibility == "partial"
        │   loiter > 30s → high        → masked_unknown
        │   else → medium
        ▼
RULE 8: AFTER-HOURS ────────────────── NOT office_hours OR NOT weekday
        │                              → high alert
        ▼
RULE 9: OFFICE-HOURS DEFAULT ───────── unknown during business hours
                                       → medium alert
```

## DecisionResult

```python
DecisionResult(
    status,           # what to call this person
    alert_level,      # how urgent
    should_alert,     # send notification?
    should_register,  # store in MongoDB?
    reason            # human-readable explanation
)
```

## Status values

| Status | Meaning | should_alert | should_register |
|--------|---------|:------------:|:---------------:|
| `blacklist` | Known threat | Yes (critical) | No |
| `authorized` | Pre-approved | No | No |
| `verified` | Operator-confirmed | No | No |
| `known_visitor` | Recognized returning person | No | No |
| `unknown` | Unidentified | Yes | Yes |
| `masked_unknown` | Masked/partial face | Yes | Yes |
| `intentionally_hidden` | Avoided detection | Yes (high) | No |

## Alert levels

| Level | When | Channels |
|-------|------|----------|
| `none` | Authorized/verified | None |
| `low` | Known visitor | Console (if configured) |
| `medium` | Unknown during office hours | Console + email/SMS/webhook |
| `high` | After-hours, masked loitering, hidden | Console + email/SMS/webhook |
| `critical` | Blacklisted person | All channels, immediate |

## See also
- `agents/policy.py` — implementation (contains `decide()`)
- `agents/alert_agent.py` — dispatches alerts based on decision
- [[CURRENT_ARCHITECTURE]] section 6 — policy-related thresholds
