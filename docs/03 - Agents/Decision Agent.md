# Decision Agent

> Thin wrapper that delegates to [[Policy Agent]]. Maintains backward compatibility while using the centralized policy engine.

**File**: `agents/decision_agent.py` (66 lines)

## Function: `decide(track, match_result, recognition_result, memory_context)`

```python
def decide(track: Track, match_result: MatchResult,
           recognition_result: dict = None, memory_context: dict = None) -> DecisionResult:
```

### Processing

```
1. Get singleton PolicyAgent instance
2. Call policy.run({...}) with:
   - recognition_result
   - memory_context
   - match_result (converted to dict)
   - track object
3. Convert policy output to DecisionResult
4. Return DecisionResult
```

### Why this exists

The Decision Agent is a **backward-compatible shim**. Originally, decision logic was in this file. It was refactored into `PolicyAgent` (Phase 2.3) for centralized rule management. The `decide()` function signature is preserved so callers don't need to change.

## Callers

- `agents/track_processor.py:118` — track finalization
- `pipeline/recognition_pipeline.py:131` — progressive recognition

## See also
- [[Policy Agent]] — where the actual rules live
- [[Track Processor]] — calls decide() during finalization
- [[Recognition Pipeline]] — calls decide() during progressive recognition
