# Policy Agent

> Centralizes ALL business rules in one place. Evaluates 9 prioritized rules to make a final decision: should we alert? should we register this person? what status should we assign?

**File**: `agents/policy.py` (317 lines)

## Role in system

```
Recognition Agent → status + confidence
Memory Agent      → visit_count, is_known
Matching Agent    → matched, tags, similarity, verified
Track             → visibility, is_masked, track_lifetime
                    │
                    ▼
            Policy Agent (9 rules, priority order)
                    │
                    ▼
            DecisionResult(status, alert_level, should_alert, should_register)
```

## Class: `PolicyAgent`

### Input to `run()`
```python
{
    "recognition_result": {"status": "known", "confidence": 79, "is_masked": False},
    "memory_context": {"visit_count": 5, "is_known": True},
    "match_result": {
        "matched": True,
        "person_id": "...",
        "name": "John Doe",
        "tags": ["auto_registered", "verified"],
        "similarity_score": 0.723,
        "verified": True,
        "alert_level": "low"
    },
    "track": Track object
}
```

### Time context (computed at call time)
```
current_hour = datetime.now().hour
is_office_hours = OFFICE_HOURS_START (9) <= hour <= OFFICE_HOURS_END (17)
is_weekday = weekday in OFFICE_DAYS ([0,1,2,3,4])  → Mon-Fri
```

## The 9 Rules (priority order, first match wins)

### RULE 1: Blacklist (highest priority)
```
IF "blacklist" in tags:
    status = "blacklist"
    alert_level = "critical"
    should_alert = True
    should_register = False
```
**Always fires.** Even if also authorized. Blacklist overrides everything.

### RULE 2: Authorized
```
IF "authorized" in tags:
    status = "authorized"
    alert_level = "none"
    should_alert = False
    should_register = False
```

### RULE 3: Verified
```
IF verified == True AND recognition_status == "known":
    status = "verified"
    alert_level = "none"
    should_alert = False
    should_register = False
```

### RULE 4a: Auto-registered self-match
```
IF "auto_registered" in tags AND similarity > 0.65:
    status = "known_visitor"
    alert_level = "low"
    should_alert = False
    should_register = False
```
Person was auto-registered as unknown, now matching themselves with decent similarity.

### RULE 4b: Known visitor (memory confirms)
```
IF matched AND is_known_from_memory:
    status = "known_visitor"
    alert_level = "low"
    should_alert = False
    should_register = False
```
Memory says this person has been here before and was previously known/verified.

### RULE 5: Matched but not verified
```
IF matched:
    5a: similarity >= 0.85 OR confidence >= 80:
        → status = "known_visitor", alert = "low"
    5b: similarity >= 0.45 (but not 5a):
        → status = "unknown", alert = "low", should_register = False
    5c: else (unreachable — vector_search filters by threshold):
        → status = "unknown", alert = "low"
```

### RULE 6: Intentionally hidden
```
IF visibility == "hidden":
    status = "intentionally_hidden"
    alert_level = "high"
    should_alert = True
    should_register = False
```
Person was tracked for ≥15 frames but face was never detected. They may be avoiding detection.

### RULE 7: Masked unknown
```
IF is_masked OR visibility == "partial":
    IF track_lifetime > LOITER_SECS (30s):
        alert_level = "high"
        reason = "Masked unknown person loitering"
    ELSE:
        alert_level = "medium"
        reason = "Unknown person with partial visibility or mask"
    status = "masked_unknown"
    should_alert = True
    should_register = True
```

### RULE 8: After-hours unknown
```
IF NOT is_office_hours OR NOT is_weekday:
    status = "unknown"
    alert_level = "high"
    should_alert = True
    should_register = True
```
Unknown person detected outside business hours = higher alert.

### RULE 9: Unknown during office hours (default)
```
status = "unknown"
alert_level = "medium"
should_alert = True
should_register = True
```

## DecisionResult

```python
DecisionResult(
    status="known",              # "verified" | "known_visitor" | "blacklist" | "unknown" | "masked_unknown" | "intentionally_hidden" | "authorized"
    alert_level="none",          # "none" | "low" | "medium" | "high" | "critical"
    person_id="...",
    name="John Doe",
    reason="Authorized person: John Doe",
    should_alert=False,
    should_register=False,
    nl_summary=""                # Filled by LLM later
)
```

## See also
- [[Policy Rules]] — visual rule priority diagram
- [[Alert Agent]] — dispatches based on should_alert + alert_level
- [[All Thresholds]] — all policy-related thresholds
