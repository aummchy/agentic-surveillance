# Decision Agent (Deprecated → Policy Agent)

> **Deprecated**: `agents/decision_agent.py` has been deleted. `decide()` now lives in `agents/policy.py`.

## What happened

The thin facade that delegated to `PolicyAgent` has been removed. The `decide()` convenience function is now directly in `agents/policy.py` as a lazy singleton wrapper around `PolicyAgent.run()`.

## New location

**File**: `agents/policy.py`

```python
def decide(track: Track, match_result: MatchResult,
           recognition_result: dict = None, memory_context: dict = None) -> DecisionResult:
```

## Callers

- `agents/track_processor.py` — track finalization
- `pipeline/recognition_pipeline.py` — progressive recognition (lazy import)

## See also
- [[Policy Agent]] — where the actual rules and `decide()` live
- [[Track Processor]] — calls decide() during finalization
- [[Recognition Pipeline]] — calls decide() during progressive recognition
