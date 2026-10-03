# Loggers — What They Are and What This Project Uses

> Written 2026-10-03 from the source of truth: `config/logging_setup.py`,
> call-site counts from `git grep` across `*.py`, `requirements.txt`.
> Companion doc: [`docs/08 - Logging/Terminal Output Reference.md`](../08%20-%20Logging/Terminal%20Output%20Reference.md)
> (exact terminal line formats). This file explains the *concept* and the *API*.

---

## 1. What a logger is

A **logger** is a component that records what a program did while it ran —
but unlike a `print()`, it records it in a way you can later **filter, sort,
route, and parse**.

Every log entry (an "event" or "record") carries three things:

1. **A severity level** — how serious it is
2. **A message / event name** — what happened
3. **Context fields** — the details (which track, what score, which camera)

### Why not just use `print()`?

| Problem with `print()` | What the logger gives you |
|---|---|
| Everything mixed together, cannot turn off | **Levels** — show only WARNING+ when things are noisy |
| One destination only | **Multiple destinations at once** — terminal + files, each with its own format and level |
| Free text, hard to search | **Structured fields** — `similarity=0.573` becomes a queryable JSON key |
| Lost when terminal closes | **Files with rotation** — kept, capped in size, auto-archived |
| Third-party libraries spam you | **Per-library mute** — silence PyMongo/InsightFace noise without touching app code |

### Severity levels (standard Python ladder)

```
DEBUG    →  fine-grained diagnostics, only for debugging (52 uses here)
INFO     →  normal lifecycle: started, decision made, visit recorded (43 uses)
WARNING  →  something unexpected but handled (44 uses)
ERROR    →  an operation failed (40 uses)
CRITICAL →  the program cannot continue (0 uses here)
```

Each destination has a **minimum level**: the terminal shows `INFO+`, the
files record `DEBUG+`.

---

## 2. What this project uses: structlog

The project uses **[structlog](https://www.structlog.org/)** — a *structured
logging* library (`requirements.txt`: `structlog>=24.1.0`) — bridged onto
Python's built-in `logging` module.

**Structured logging** means every event is a *name plus key/value fields*,
not a formatted sentence:

```python
# Text logging (old style) — hard to grep, hard to parse
logger.info(f"track 2 matched with similarity 0.573")   # ← NOT this project's style

# Structured logging (this project)
logger.info("match_found", track_id="cam_01_..._2", similarity=0.573, verified=False)
# → JSON file:  {"event": "match_found", "similarity": 0.573, "track_id": "cam_01_...", ...}
# → terminal:   12:27:54 MATCH   trk=2 Unknown sim=0.573 top2=0.150 gap=0.423 ...
```

The **event name** is a stable, snake_case identifier
(`policy_decision`, `track_finalized`, `alert_dispatched`). Fields are
separate keyword arguments. That is why one terminal line and one JSON line
come from the *same* call.

### How a module gets its logger

Every module creates a module-level logger once:

```python
import structlog
logger = structlog.get_logger(__name__)     # 33 call sites in this repo
```

- `__name__` becomes the logger's name (`agents.policy`, `utils.db_faces`, …)
- Created **at module top level**, never inside functions (it is cached and cheap)
- `config/settings.py` uses `structlog.get_logger("config")`

---

## 3. Where logging is configured

`config/logging_setup.py` holds the entire logging system. The function
`setup_logging()` runs **automatically when `config/settings.py` is imported**
— and nearly every module imports settings — so logging is active from the
first lines of `main.py`. It is **idempotent** (guarded by an internal flag),
so calling it again does nothing.

### The 3 tiers

| Tier | Target | Min level | Format | Rotation | Audience |
|---|---|---|---|---|---|
| 1. Terminal | stderr | `INFO` | one colored line per event (`CompactTerminalRenderer`) | — | the operator watching it run |
| 2. JSON file | `logs/surveillance.jsonl` | `DEBUG` | one JSON object per line (`JSONFileRenderer`) | 5 MB × 5 backups | machines & forensics (grep, jq, scripts) |
| 3. Debug file | `logs/surveillance.debug.log` | `DEBUG` | full human-readable context (`structlog.dev.ConsoleRenderer`) | 10 MB × 3 backups | deep debugging sessions |

**Same event, three outputs.** A `DEBUG` event exists only in tiers 2–3;
`INFO` reaches all three.

### Terminal filtering (two gates)

1. **`TERMINAL_ALLOWLIST`** — a frozen set of event names allowed on the
   terminal (22 names: `match_found`, `policy_decision`, `alert`, …).
   Events not on the list go to the files only.
2. **`_BlankFilter`** — drops empty lines and non-allowlisted events below
   `WARNING`.

**Escape hatch:** `warning` / `error` / `critical` always reach the terminal,
allowlist or not — failures must never be invisible.

### Noisy third-party libraries muted

`setup_logging()` sets these to `WARNING` (their `DEBUG`/`INFO` spam is
suppressed, their warnings kept):

- `pymongo`, `pymongo.topology`, `pymongo.pool`, `pymongo.command`, `pymongo.server`
- `insightface`, `insightface.utils`, `insightface.utils.face_align`
- `cloudinary`
- `ultralytics` — handled in `pipeline/tracker.py` (its logger is grabbed and
  down-leveled there, because the tracker owns the YOLO model)

---

## 4. Common functions and what we use them for

### 4.1 Creating a logger

```python
import structlog
logger = structlog.get_logger(__name__)
```

**Use:** one per module, at the top. This is the only "construction" API you
need — 33 sites, all identical in shape.

### 4.2 Emitting an event — `logger.debug / info / warning / error`

```python
logger.debug("recognition_cycle", track_id=..., pass_=2)          # files only
logger.info("policy_decision", track_id=..., status=..., alert_level=...)
logger.warning("decide_called_without_track")
logger.error("track_processing_failed", track_id=..., error=str(e), exc_info=True)
```

| Call | Count in repo | Where it lands |
|---|---|---|
| `logger.debug(...)` | 52 | JSON + debug file only |
| `logger.info(...)` | 43 | all 3 tiers (if allowlisted) |
| `logger.warning(...)` | 44 | always all 3 tiers |
| `logger.error(...)` | 40 | always all 3 tiers |
| `logger.critical(...)` | 0 | (unused) |
| `logger.exception(...)` | 0 | (unused — see `exc_info=True` below) |

**Convention:** the first argument is the event name; everything else is a
keyword field. Values should be JSON-friendly (str, int, float, bool);
the JSON renderer falls back to `str()` for anything exotic.

### 4.3 Attaching tracebacks — `exc_info=True`

```python
logger.error("track_processing_failed", track_id=track.track_id,
             error=str(e), exc_info=True)
```

Instead of `logger.exception(...)` (0 uses), this project passes
`exc_info=True` to `logger.error`. The `format_exc_info` processor renders
the traceback; it appears in the two files, while the terminal keeps its
one-line summary. Used on every "…_failed" catch block (8 occurrences).

### 4.4 The processor pipeline (what happens between emit and file)

`setup_logging()` configures structlog with this chain — each step enriches
the event dict:

| Processor | Job |
|---|---|
| `merge_contextvars` | merges any context-local fields |
| `add_log_level` | adds `level=info` etc. |
| `TimeStamper(fmt="iso")` | adds the ISO-8601 `timestamp` |
| `add_byte_track_id` (custom) | derives short `byte_track_id=2` from composite IDs like `cam_01_1785391472_2` — so terminals can print `trk=2` |
| `StackInfoRenderer` | renders stack info if requested |
| `format_exc_info` | turns `exc_info=True` into a traceback string |
| `wrap_for_formatter` | hands the dict to the stdlib handler's renderer |

Then each handler's **renderer** formats it:
`CompactTerminalRenderer` (colored one-liner), `JSONFileRenderer`
(`json.dumps` per line), or `ConsoleRenderer` (verbose debug file).

### 4.5 Making an event pretty on the terminal

Three pieces in `CompactTerminalRenderer`, all optional — an event without
them uses a generic fallback line:

1. **`TERMINAL_ALLOWLIST`** — decides *whether* it shows at all
2. **`PREFIX_MAP`** — event → 3–7 letter prefix with a color
   (`match_found` → `MATCH` in blue, `policy_decision` → `POLICY` in green)
3. **`FORMATS`** — a `str.format` template with 24 named events
   (`"{_pfx} {_trk} {name:<10} sim={_sim} …"`); missing fields get safe
   placeholders (`---`, `trk=---`)

Status values are colored via `Colors.STATUS` keyed by the `Status` enum
(known/verified → green, uncertain → yellow, unknown/blacklist → red).

### 4.6 Rotation — files that manage themselves

`RotatingFileHandler` on both files: when `surveillance.jsonl` passes 5 MB it
is renamed `.1` (…up to `.5`) and a fresh one starts. Nothing to do
manually; disk usage is capped.

---

## 5. Practical recipes

**Watch the terminal** — just run `python main.py`; tier 1 does the work.

**Search the forensic file for an event:**

```powershell
Select-String -Path logs\surveillance.jsonl -Pattern '"event": "alert_dispatched"'
```

**Read one line as objects:**

```powershell
Get-Content logs\surveillance.jsonl -Tail 50 |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object event -eq "policy_decision"
```

**Add a new log event (checklist):**

1. `logger.info("my_new_event", some_field=...)` in the owning module
2. Decide the tier: leave it off `TERMINAL_ALLOWLIST` → files only; add it →
   it appears on the terminal
3. For a pretty terminal line: add a `PREFIX_MAP` entry and a `FORMATS`
   template in `config/logging_setup.py`
4. Keep field names snake_case and values JSON-friendly

**Common pitfalls:**

- Passing a pre-formatted f-string as the message defeats the structure —
  emit an event name + fields instead
- Forgetting the event lands files-only: the terminal gate is the
  allowlist, not the level (for `INFO`/`DEBUG`)
- Logging inside the camera per-frame hot path at `INFO` — chatty events
  should be `DEBUG` (files only) or allowlist-gated
- Logging secrets (`MONGODB_URI`, API keys) — never; the JSON file is
  world-readable on disk

---

## 6. Quick API summary

| You want to… | Use |
|---|---|
| Get a module logger | `logger = structlog.get_logger(__name__)` |
| Record a normal event | `logger.info("event_name", field=value)` |
| Record diagnostics | `logger.debug("event_name", field=value)` |
| Record a failure | `logger.error("event_name", error=str(e), exc_info=True)` |
| Show on terminal | add event name to `TERMINAL_ALLOWLIST` |
| Pretty terminal line | add entry to `PREFIX_MAP` + `FORMATS` |
| Find past events | `logs/surveillance.jsonl` (JSON, grep-able) |
| Deep-dive a session | `logs/surveillance.debug.log` |
| Change logging behavior | `config/logging_setup.py` only (setup code) |

---

## See also

- [`docs/08 - Logging/Terminal Output Reference.md`](../08%20-%20Logging/Terminal%20Output%20Reference.md)
  — exact terminal line formats, field reference, example session
- [`config/logging_setup.py`](../../config/logging_setup.py) — the entire
  implementation (318 lines: colors, renderers, allowlist, `setup_logging()`)
- `AGENTS.md` → "Logging system" — project conventions
