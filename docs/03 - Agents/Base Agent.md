# Base Agent

> Abstract base class that all agents inherit from. Defines the standard interface.

**File**: `agents/base.py` (29 lines)

## Interface

```python
class BaseAgent(ABC):
    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass
```

Every agent:
1. Accepts a **structured dictionary** as input
2. Processes it
3. Returns a **structured dictionary** as output

This makes agents:
- **Testable** — call `agent.run({...})` with known inputs, assert on outputs
- **Swappable** — replace one agent implementation with another
- **Composable** — agents can be chained (Recognition Agent uses Memory Agent output)

## Agents that inherit from BaseAgent

| Agent | Input | Output |
|-------|-------|--------|
| [[Memory Agent]] | `{person_id, camera_id, similarity, status}` | `{visit_count, is_known, confidence_boost, ...}` |
| [[Recognition Agent]] | `{similarity, is_masked, face_quality, track_duration, memory_context}` | `{status, confidence, reason}` |
| [[Policy Agent]] | `{recognition_result, memory_context, match_result, track}` | `{status, alert_level, should_alert, should_register}` |

## Agents that don't inherit (utility modules)

| Module | Why |
|--------|-----|
| [[Matching Agent]] | Pure function, not a class (just `run_matching_from_embedding()`) |
| [[Alert Agent]] | Pure functions (`dispatch()`, `should_send_alert()`) |
| [[Camera Agent]] | Not an agent in the BaseAgent sense — it's the camera loop orchestrator |
| [[Track Processor]] | Not a BaseAgent — it's a queue consumer + finalizer |

## See also
- [[System Overview]] — where agents fit in the architecture
- [[Data Flow]] — how agents are called in sequence
