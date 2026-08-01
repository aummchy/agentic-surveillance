# LLM Client

> HTTP client for Ollama local LLM. Generates NL alert summaries and powers dashboard chat. Optional — system works without it.

**File**: `utils/llm_client.py`

## Connection

```python
# httpx connection-pooled client
client = httpx.Client(
    base_url=settings.OLLAMA_URL,
    timeout=settings.OLLAMA_TIMEOUT  # 30s
)
```

## Functions

### `is_available() → bool`
```
GET /api/tags
Cached for 10 seconds (avoids hammering Ollama)
Returns True if status_code == 200
```

### `generate_nl_summary(alert_payload) → str`
```
POST /api/generate
System: "You are a surveillance alert system..."
Prompt: status, alert_level, person, camera, masked, reason, visit_history
Temperature: 0.2, Max tokens: 150
Retry: 3 attempts
Fallback: None (caller uses template string)
```

### `chat_completion(messages) → str`
```
POST /api/chat
Messages: [system_prompt, user_message]
Temperature: 0.3, Max tokens: 1024
Retry: 3 attempts
Fallback: None
```

## LLM models

| Model | VRAM | Best for |
|-------|------|----------|
| `gemma3:4b` (default) | ~2.7GB Q4 | General NL generation |
| `qwen3.5:4b` | ~2.7GB Q4 | Better reasoning |

Swap via `OLLAMA_MODEL` env var.

## Graceful degradation

When Ollama is unavailable:
- `is_available()` returns False at startup
- `generate_nl_summary()` returns None → alert uses template string
- `chat_completion()` returns None → dashboard shows raw data
- System continues fully functional without LLM

## See also
- [[Alert Agent]] — calls generate_nl_summary()
- [[Backend API]] — calls chat_completion() for dashboard chat
- [[System Overview]] — LLM is optional
