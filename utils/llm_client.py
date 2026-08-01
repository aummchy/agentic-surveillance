"""
LLM Client — Ollama HTTP integration for the surveillance system.

Provides synchronous and asynchronous interfaces to Ollama's API.
All calls have timeout + retry with graceful fallback on failure.

Usage:
    from utils.llm_client import generate_nl_summary, chat_completion

    summary = generate_nl_summary("You are a surveillance system...", context_dict)
    reply = chat_completion("What events happened today?", system_prompt="...")
"""

import json
import structlog
import httpx
import time
import threading
from typing import Optional

from config import settings

_avail_cache_ts = 0.0
_avail_cache_val = False

logger = structlog.get_logger(__name__)

# Module-level client for connection pooling
_client: Optional[httpx.Client] = None
_async_client: Optional[httpx.AsyncClient] = None
_client_lock = threading.Lock()
_async_client_lock = threading.Lock()
_shut_down = False


def _get_client() -> httpx.Client:
    global _client, _shut_down
    if _shut_down:
        raise RuntimeError("llm_client has been shut down")
    if _client is None or _client.is_closed:
        with _client_lock:
            if _shut_down:
                raise RuntimeError("llm_client has been shut down")
            if _client is None or _client.is_closed:
                _client = httpx.Client(
                    base_url=settings.OLLAMA_URL,
                    timeout=httpx.Timeout(settings.OLLAMA_TIMEOUT, connect=5.0),
                )
    return _client


def _get_async_client() -> httpx.AsyncClient:
    global _async_client, _shut_down
    if _shut_down:
        raise RuntimeError("llm_client has been shut down")
    if _async_client is None or _async_client.is_closed:
        with _async_client_lock:
            if _shut_down:
                raise RuntimeError("llm_client has been shut down")
            if _async_client is None or _async_client.is_closed:
                _async_client = httpx.AsyncClient(
                    base_url=settings.OLLAMA_URL,
                    timeout=httpx.Timeout(settings.OLLAMA_TIMEOUT, connect=5.0),
                )
    return _async_client


def generate(
    prompt: str,
    system: str = "",
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
    timeout: int = None,
) -> Optional[str]:
    """Generate text from Ollama (synchronous, one-shot).

    Args:
        prompt: User prompt text.
        system: System instruction (optional).
        model: Override model name (default: settings.OLLAMA_MODEL).
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        timeout: Override timeout in seconds.

    Returns:
        Generated text or None on failure.
    """
    model = model or settings.OLLAMA_MODEL
    client = _get_client()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    for attempt in range(3):
        try:
            resp = client.post("/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("response", "")
            logger.debug("llm_generate_done", model=model, tokens=data.get("eval_count"),
                         duration_ms=data.get("eval_duration", 0) // 1_000_000)
            return response_text.strip()
        except httpx.TimeoutException:
            logger.warning("llm_timeout", attempt=attempt + 1, model=model)
        except httpx.ConnectError:
            logger.warning("llm_connect_failed", attempt=attempt + 1,
                           model=model, url=settings.OLLAMA_URL)
            break  # No point retrying if Ollama isn't running
        except Exception as e:
            logger.error("llm_generate_error", error=str(e), model=model)
            break

    return None


async def generate_async(
    prompt: str,
    system: str = "",
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> Optional[str]:
    """Async version of generate()."""
    model = model or settings.OLLAMA_MODEL
    client = _get_async_client()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    for attempt in range(3):
        try:
            resp = await client.post("/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except httpx.TimeoutException:
            logger.warning("llm_async_timeout", attempt=attempt + 1, model=model)
        except httpx.ConnectError:
            logger.warning("llm_async_connect_failed", attempt=attempt + 1)
            break
        except Exception as e:
            logger.error("llm_async_error", error=str(e), model=model)
            break

    return None


def chat_completion(
    message: str,
    system: str = "",
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Optional[str]:
    """Chat completion (synchronous). Maintains no conversation history.

    Args:
        message: User message.
        system: System instruction.
        model: Override model.
        temperature: Sampling temperature.
        max_tokens: Max response tokens.

    Returns:
        Assistant reply or None.
    """
    model = model or settings.OLLAMA_MODEL
    client = _get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    for attempt in range(3):
        try:
            resp = client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        except httpx.TimeoutException:
            logger.warning("llm_chat_timeout", attempt=attempt + 1, model=model)
        except httpx.ConnectError:
            logger.warning("llm_chat_connect_failed", attempt=attempt + 1)
            break
        except Exception as e:
            logger.error("llm_chat_error", error=str(e), model=model)
            break

    return None


async def chat_completion_async(
    message: str,
    system: str = "",
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Optional[str]:
    """Async chat completion."""
    model = model or settings.OLLAMA_MODEL
    client = _get_async_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    for attempt in range(3):
        try:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        except httpx.TimeoutException:
            logger.warning("llm_chat_async_timeout", attempt=attempt + 1)
        except httpx.ConnectError:
            logger.warning("llm_chat_async_connect_failed")
            break
        except Exception as e:
            logger.error("llm_chat_async_error", error=str(e))
            break

    return None


# ══════════════════════════════════════════════════════════════
# Convenience functions for surveillance-specific tasks
# ══════════════════════════════════════════════════════════════

def generate_nl_summary(alert_payload: dict) -> Optional[str]:
    """Generate a natural language alert summary from an alert payload.

    Returns 1-2 sentence description suitable for console/email/SMS.
    Falls back to None if LLM unavailable (caller should use template).
    """
    system = (
        "You are a surveillance alert system. "
        "Generate a concise, factual alert description in 1-2 sentences. "
        "Be specific about what happened and why it matters. "
        "Do not use emojis or markdown. Use plain English only."
    )

    prompt = f"""Generate an alert description for this surveillance event:

Status: {alert_payload.get('status', 'unknown')}
Alert Level: {alert_payload.get('alert_level', 'medium')}
Person: {alert_payload.get('name') or 'Unknown'}
Camera: {alert_payload.get('camera_id', 'unknown')}
Masked: {alert_payload.get('is_masked', False)}
Reason: {alert_payload.get('reason', 'N/A')}
Visit History: {json.dumps(alert_payload.get('memory_context', {}), default=str)}

Alert description:"""

    return generate(prompt, system=system, temperature=0.2, max_tokens=150)


def generate_incident_summary(incident_data: dict) -> Optional[str]:
    """Generate a natural language incident report summary and recommendation."""
    system = (
        "You are a surveillance incident analyst. "
        "Write a concise 2-3 sentence incident summary followed by a recommended action. "
        "Be factual and professional. Do not use emojis."
    )

    prompt = f"""Write an incident report for this surveillance event:

Status: {incident_data.get('status', 'unknown')}
Alert Level: {incident_data.get('alert_level', 'medium')}
Person: {incident_data.get('name') or 'Unknown'}
Camera: {incident_data.get('camera_id', 'unknown')}
Reason: {incident_data.get('reason', 'N/A')}
Visit Count: {incident_data.get('visit_count', 0)}
Timestamp: {incident_data.get('timestamp', 'N/A')}

Write:
Summary:
Recommendation:"""

    return generate(prompt, system=system, temperature=0.2, max_tokens=250)


def generate_executive_summary(stats: dict, recent_events: list) -> Optional[str]:
    """Generate a natural language executive summary for a period."""
    system = (
        "You are a surveillance analyst writing an executive summary. "
        "Highlight key patterns, concerns, and notable events. "
        "Be concise and professional. 3-5 sentences."
    )

    events_text = ""
    for e in recent_events[:5]:
        events_text += f"  - {e.get('status', 'unknown')} at {e.get('camera_id', '?')} "
        events_text += f"(level: {e.get('alert_level', '?')})\n"

    prompt = f"""Surveillance Statistics:
- Total unknown persons: {stats.get('total_unknown', 0)}
- Total verified persons: {stats.get('total_verified', 0)}
- Events today: {stats.get('events_today', 0)}
- Unknown persons today: {stats.get('unknown_today', 0)}

Recent events:
{events_text or '  No recent events.'}

Executive summary:"""

    return generate(prompt, system=system, temperature=0.3, max_tokens=300)


def is_available() -> bool:
    """Check if Ollama is reachable (cached for 10s)."""
    global _avail_cache_ts, _avail_cache_val
    if not settings.LLM_ENABLED:
        return False
    now = time.monotonic()
    if now - _avail_cache_ts < 10.0:
        return _avail_cache_val
    try:
        client = _get_client()
        resp = client.get("/api/tags", timeout=3.0)
        _avail_cache_val = resp.status_code == 200
    except Exception:
        _avail_cache_val = False
    _avail_cache_ts = now
    return _avail_cache_val


_shutdown_lock = threading.Lock()


def shutdown():
    """Close HTTP clients on process exit."""
    global _client, _async_client, _shut_down
    with _shutdown_lock:
        if _shut_down:
            return
        _shut_down = True
    # Capture references before clearing globals
    client = _client
    async_client = _async_client
    _client = None
    _async_client = None
    try:
        if client and not client.is_closed:
            client.close()
    finally:
        try:
            if async_client and not async_client.is_closed:
                try:
                    import asyncio
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    import asyncio as _aio
                    try:
                        _aio.run(async_client.aclose())
                    except RuntimeError:
                        pass
                else:
                    loop.create_task(async_client.aclose())
        except Exception as e:
            logger.warning("async_client_close_failed", error=str(e))
