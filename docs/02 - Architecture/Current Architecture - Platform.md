# Current Architecture — Platform (Sections 21–24)

> Part of [`CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md). Section numbers follow the shared scheme: this file holds sections 21–24.
>
> **Source of truth:** current source code and tests. Facts that could not be proven from the source are marked `UNKNOWN — NEEDS VERIFICATION`.
>
> **Last verified:** 2026-10-03

**Navigation:** [Hub](../CURRENT_ARCHITECTURE.md) · [Components](Current%20Architecture%20-%20Components.md) · [Recognition](Current%20Architecture%20-%20Recognition.md) · [Decision Flow](Current%20Architecture%20-%20Decision%20Flow.md) · [Platform](Current%20Architecture%20-%20Platform.md) · [Code State](Current%20Architecture%20-%20Code%20State.md)

---

# 21. Dashboard

## Backend — `dashboard/backend/`

### Responsibility

FastAPI REST + WebSocket API (Uvicorn, port 8000, started from `runtime/api_server.py` via `main.py` phase 4).

### Current structure (verified from `dashboard/backend/main.py:25-29`)

| Router | Prefix | File |
|--------|--------|------|
| faces CRUD | `/api/faces` | `routes/faces.py` |
| events/stats | `/api/events` | `routes/events.py` |
| reports | `/api` (→ `/api/reports/stats`, `/summary`, `/person/{id}`, `/incidents`) | `routes/reports.py` |
| chat | `/api/chat` | `routes/chat.py` |
| WebSocket | `/ws/live` | `routes/live.py` |

### Security posture (as it exists today)

* **No authentication anywhere.**
* CORS: origins from `settings.ALLOWED_ORIGINS`, methods `["GET", "POST", "PUT", "DELETE"]`, headers `["Content-Type", "Authorization"]` (`dashboard/backend/main.py:19-22`).
* `/ws/live` validates the `Origin` header against `ALLOWED_ORIGINS` and closes with code 4003 if it does not match (`live.py:82-85`).
* The API binds per `settings`/uvicorn config — the Phase 3 review flagged binding to `0.0.0.0` without auth as finding **H3** (pending decision: localhost-bind vs token auth; Wave A item 3).

### Threading

* The API thread owns the asyncio loop; producers from other threads post coroutines onto it (`run_coroutine_threadsafe`) — only while the loop is alive.
* Long-running handlers hand work to threads: chat uses `asyncio.to_thread(_handle_query, ...)` (`chat.py:140`); reports use `asyncio.to_thread(...)` in every route.
* The `incidents` route (`reports.py:46-84`) batch-fetches events + visit histories inside `_fetch()` and passes them to `ReportAgent` (the H2 N+1 was fixed 2026-10-03 — the batch is now consumed).

### Known characteristics (observed, not changed)

* `/api/reports/incidents` took **15.5 s for 3 events** on 2026-10-03: the events query takes ~4.6 s and each incident report attempts an LLM call (~4.1 s while Ollama is down) before falling back to templates.
* The route only catches `RuntimeError` → a MongoDB/DNS outage surfaces as a raw HTTP 500 (observed once during a transient Atlas DNS timeout).
* `chat.py:132` calls `llm_client.is_available()` **synchronously inside the async handler** (M2 parked: should be `to_thread`/cached).

## Frontend — `dashboard/frontend/`

React + Vite dev server on port 5173. Consumes the WebSocket (`/ws/live`: feed frames, events, alerts) and the REST endpoints.

### Responsibility boundary

The frontend is a **consumer** of surveillance state/events. It does not own surveillance decisions.

---

# 22. LLM

## `utils/llm_client.py` (282 lines)

### Responsibility

Communication with the local Ollama server (pooled `httpx` client).

### Configuration (verified)

| Setting | Effective value | Source |
|---------|-----------------|--------|
| `OLLAMA_URL` | `http://localhost:11434` | `config.jsonc:229` |
| `OLLAMA_MODEL` | `gemma3:4b` (swap via `.env`, e.g. `qwen3.5:4b`) | `config.jsonc:230` |
| `OLLAMA_TIMEOUT` | 30 s | `config.jsonc:231` |
| `OLLAMA_CONNECT_TIMEOUT` | 5.0 s | `config.jsonc:274` |
| `LLM_MAX_RETRIES` | 3 (but `ConnectError` breaks immediately — `llm_client.py:97-100`) | `config.jsonc:272` |
| `LLM_ENABLED` | **`false`** in `config.jsonc:228` (settings default `True`) | see concern below |

### Functions and call sites (verified — the LLM is NOT in the detection pipeline)

| Function | Called from |
|----------|-------------|
| `generate_nl_summary` | `alert_agent.py:115` (alert summaries) |
| `generate_incident_summary` | `report.py:101` (per-incident reports) |
| `generate_executive_summary` | `report.py:181` (summary reports) |
| `chat_completion` | `chat.py:103, 119` (dashboard chat) |
| `is_available` | `chat.py:132, 150` (availability check; cached 10 s) |

### Important boundary

The LLM does **not** participate in:

```text
YOLO detection · ByteTrack tracking · face recognition · identity matching · policy evaluation
```

Those are deterministic/model-based surveillance components. The LLM only produces natural-language text; every consumer has a template fallback when it is unavailable.

### Verified current quirks

* **`LLM_ENABLED` is only consulted by `is_available()`** (`llm_client.py:250`). `generate()` posts to Ollama regardless of the flag. With `config.jsonc` set to `false`, chat health always reports unavailable, while alert/report LLM calls still attempt (and fail) HTTP — costing ~4 s per attempt (observed 2026-10-03). Gating consistently is parked as **M3** ("confirm intent of `LLM_ENABLED: false` first").
* Fallback behavior: `generate()` returns `None` on failure → callers use template strings. The system runs fully without Ollama.

---

# 23. Configuration

## `config/settings.py` (327 lines)

### Loading chain (verified precedence)

```text
.env            (secrets — API keys, MONGODB_URI, passwords)
    ↓  highest
config/config.jsonc   (tunables — detection, quality, recognition, LLM)
    ↓
hardcoded defaults in settings.py   (lowest)
```

* `_get(env_key, config_key, default, cast)` implements the precedence; list defaults cast element-wise (e.g. `OFFICE_DAYS=0,1,2,3,4` → `[0,1,2,3,4]`).
* `validate_config()` runs at startup; effective settings are logged.
* **Rule:** edit `config/config.jsonc` for tunables, `.env` for secrets. Never edit `settings.py` defaults.

### Current concern — duplicated defaults with mismatched values (verified examples)

The same key exists as a hardcoded default in `settings.py` **and** as a value in `config.jsonc`. Because jsonc wins, behavior today is correct — but deleting the jsonc entry would silently change behavior:

| Key | `settings.py` default | `config.jsonc` value | Effect if jsonc entry removed |
|-----|----------------------|----------------------|-------------------------------|
| `TRACK_TIMEOUT_SECS` | **3.0** (`settings.py:141,224`) | **15.0** (`config.jsonc:47`) | tracks would expire 5× faster |
| `JPEG_QUALITY_BROADCAST` | **65** (`settings.py:257`) | **85** (`config.jsonc:113`) | preview quality would drop |

Deduplicating these without changing any effective value is a parked Phase 4 item (`plan.md` Phase 4 #5). Whether **other** keys diverge the same way: `UNKNOWN — NEEDS VERIFICATION` (a full settings-vs-jsonc diff has not been done).

### Related config files

* `config/bytetrack_surveillance.yaml` — ByteTrack tracker parameters.
* `config/status.py` — status enums (Section 24).
* `config/logging_setup.py` — the 3-tier logging setup (terminal allowlist / JSON lines / debug file).

---

# 24. Status Model

## `config/status.py`

Centralizes identity status values as `Status(IntEnum)` — the single source of truth.

```text
UNKNOWN = 1        Unrecognized person
UNCERTAIN = 2      Weak match, low confidence
KNOWN = 3          Matched identity, confidence above threshold
KNOWN_VISITOR = 4  Returning visitor (auto-registered self-match or memory-confirmed)
VERIFIED = 5       Verified visitor (manual or high-similarity)
AUTHORIZED = 6     Employee / authorized person
BLACKLIST = 7      Blacklisted person (highest priority)
MASKED_UNKNOWN = 8 Masked / partial-visibility unknown
HIDDEN = 9         Intentionally hidden
```

### Semantics (verified)

* **Higher number = more trusted.** `is_known` = `status >= Status.KNOWN` (3), formalized as `IS_KNOWN_THRESHOLD = Status.KNOWN` (`config/status.py`).
* Numeric everywhere: MongoDB, API, logs, frontend.
* Label maps: `STATUS_LABELS` (display), `LABEL_TO_STATUS` (reverse).
* Helper frozensets: `RESOLVED_STATUSES = {VERIFIED, KNOWN_VISITOR, AUTHORIZED}`; `UNVERIFIED_STATUSES = {UNKNOWN, MASKED_UNKNOWN, UNCERTAIN, HIDDEN}` (used by alert cooldown keys).

### Separate, non-interchangeable concepts

| Concept | Type | Values |
|---------|------|--------|
| Status | `Status` IntEnum | 1–9 (above) |
| Alert level | `AlertLevel` StrEnum | `none / low / medium / high / critical` |
| Visibility | `Visibility` StrEnum | `unknown / visible / partial / hidden` |
| Skip reason | `SkipReason` StrEnum | e.g. `high_confidence`, `no_face`, … |
| Track ID | string | `{camera}_{epoch}_{bt}_{gen}` — transient |
| Person ID | string | database identity — persistent |

Do not treat these as interchangeable. Migration tooling for older string statuses: `scripts/migrate_status_ints.py`.
