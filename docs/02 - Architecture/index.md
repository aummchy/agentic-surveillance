# 02 - Architecture — Index

> Folder entry point for the architecture documentation. The **authoritative
> hub** is [`../CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md) (§1–3,
> §25–26, §31 + Appendices A–D); this file is the folder's table of contents
> plus a single-screen text breakdown. Last verified 2026-10-04.

---

## 1. The architecture in one text stack

```
┌──────────────────────────────────────────────────────────────────────────┐
│ L6  CLIENTS                                                              │
│     Browser dashboard — React 18 + Vite (dashboard/frontend, :5173)      │
│     REST (axios) · WebSocket /ws/live · images from /captures            │
└───────────────┬──────────────────────────────────▲───────────────────────┘
                │ HTTP request/response            │ WS push: frame/event/alert
┌───────────────▼──────────────────────────────────┴───────────────────────┐
│ L5  API LAYER — dashboard/backend (FastAPI + Uvicorn, :8000)             │
│     routers: faces · events · reports · chat · live(WS) · CORS · static  │
│     lifecycle owned by runtime/api_server.py (daemon thread)             │
└───────────────▲──────────────────────────────────┬───────────────────────┘
                │ run_coroutine_threadsafe         │ queue.put (finished tracks)
┌───────────────┴──────────────────────────────────▼───────────────────────┐
│ L4  ORCHESTRATION — agents/ + runtime/ + main.py (composition root)      │
│     camera_agent (capture loop)      TrackProcessor (finalization,       │
│     policy · recognition · matching · memory · alert · report · scoring  │
│     2 queue workers)                 runtime/ (startup tasks, API, queue)│
└───────────────▲──────────────────────────────────┬───────────────────────┘
                │ frames / tracks                  │ embeddings / queries
┌───────────────┴──────────────────────────────────▼───────────────────────┐
│ L3  CV PIPELINE — pipeline/                                              │
│     tracker: YOLOv8 (OpenVINO or CPU) + ByteTrack (single model)         │
│     quality gates → InsightFace embed (buffalo_l, 512-d) → match         │
└───────────────┬───────────────────────────────────────────┬──────────────┘
                │                                           │
┌───────────────▼─────────────────────────┐ ┌───────────────▼──────────────┐
│ L2a STORAGE — utils/db_*                │ │ L2b INTELLIGENCE              │
│     MongoDB Atlas                       │ │     Ollama local LLM          │
│     faces · events · visit_memory       │ │     (Gemma 3 4B / Qwen 3.5)   │
│     vector_index (512-d, cosine)        │ │     NL summaries + chat       │
│     Cloudinary (image archive)          │ │     alerts (email/SMS/webhook)│
└─────────────────────────────────────────┘ └──────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────┐
│ L1  CROSS-CUTTING                                                        │
│     config/ (settings.py + config.jsonc + status enum + 3-tier logging)  │
│     tests/ (97 pytest) · scripts/ (session query tools)                  │
└──────────────────────────────────────────────────────────────────────────┘
```

Frame path: camera → tracker → track state → (progressive) recognition →
track ends → queue → TrackProcessor → match/decide/alert → DB + broadcast.
Detail: [`Data Flow.md`](Data%20Flow.md).

---

## 2. Layer by layer

| Layer | Folder / entry | Responsibility |
|-------|----------------|----------------|
| L6 Clients | `dashboard/frontend/` | React dashboard: live preview, events, alerts, chat |
| L5 API | `dashboard/backend/` + `runtime/api_server.py` | REST + WebSocket, CORS, static files |
| L4 Orchestration | `main.py`, `agents/`, `runtime/` | capture loop, one-decision-per-track, policy, alerts, startup/shutdown |
| L3 CV pipeline | `pipeline/` | YOLOv8 + ByteTrack, quality gates, face embedding, matching |
| L2a Storage | `utils/db_*.py` | MongoDB Atlas CRUD + vector search; Cloudinary images |
| L2b Intelligence | `utils/llm_client.py` | Ollama for NL summaries and operator chat |
| L1 Cross-cutting | `config/`, `tests/` | configuration chain, status enum, logging, test suite |

---

## 3. Tech stack (text)

| Concern | Choice | Where / notes |
|---------|--------|---------------|
| Language | Python 3.11 | InsightFace/onnxruntime wheels need it exactly |
| Detection | YOLOv8 (`ultralytics>=8.1.0`) | `pipeline/tracker.py`, single shared model |
| Acceleration | OpenVINO (`>=2024.0.0`) | optional; CPU works too |
| Tracking | ByteTrack | built into ultralytics, tuned by `config/bytetrack_surveillance.yaml` |
| Face recognition | InsightFace (`>=0.7.3`) `buffalo_l` | SCRFD detect + ArcFace 512-d embed, singleton |
| Database | MongoDB Atlas + PyMongo (`>=4.6.0`) | vector search index `vector_index` (512-d cosine) |
| Image archive | Cloudinary (`>=1.36.0`) | optional; local captures also served |
| API | FastAPI (`>=0.110.0`) + Uvicorn (`>=0.27.0`) | REST + `/ws/live`, port 8000 |
| Realtime | WebSocket (FastAPI endpoint) | server-push frames/events/alerts |
| Frontend | React 18 + Vite 5, axios `^1.7.9` | `dashboard/frontend/`, port 5173 |
| LLM | Ollama — Gemma 3 4B / Qwen 3.5 4B | via `httpx`, fallback to template strings |
| Logging | structlog (`>=24.1.0`) | 3-tier: terminal / JSONL / debug file |
| Config | `python-dotenv` + `config/config.jsonc` | `.env` > `.jsonc` > defaults |
| Tests | pytest | `tests/` — 97 tests, 8 files |
| Alerts | `requests` webhook, `twilio` SMS, SMTP email | `agents/alert_agent.py` |

---

## 4. Reading order — documents in this folder

Start at the hub, then pick a part:

| # | Document | Covers | Answers |
|---|----------|--------|---------|
| 0 | [`../CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md) | §1–3, §25–26, §31 + Appendices A–D | How is it wired? Threads? Thresholds? |
| 1 | [`Current Architecture - Components.md`](Current%20Architecture%20-%20Components.md) | §4–7 | What are the major components? How do detection/tracking/state work? |
| 2 | [`Current Architecture - Recognition.md`](Current%20Architecture%20-%20Recognition.md) | §8–14 | How does recognition work (both paths), quality, matching, confidence? |
| 3 | [`Current Architecture - Decision Flow.md`](Current%20Architecture%20-%20Decision%20Flow.md) | §15–20 | How does a track become a decision, alert, memory update? |
| 4 | [`Current Architecture - Platform.md`](Current%20Architecture%20-%20Platform.md) | §21–24 | Dashboard, LLM, configuration, status model? |
| 5 | [`Current Architecture - Code State.md`](Current%20Architecture%20-%20Code%20State.md) | §27–30 | What is duplicated/suspect/fixed? What do tests cover? |
| 6 | [`Data Flow.md`](Data%20Flow.md) | frame walkthrough | What happens to one frame, step by step? |
| 7 | [`Thread Architecture.md`](Thread%20Architecture.md) | thread inventory | What runs concurrently, and what protects shared state? |
| 8 | [`MongoDB Schema.md`](MongoDB%20Schema.md) | collections | What do the collections/indexes look like? |

---

## 5. Related documents

- [`../00 - Home.md`](../00%20-%20Home.md) — top-level map of all docs
- [`../CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md) — authoritative hub
- [`../ARCHITECTURE_RULES.md`](../ARCHITECTURE_RULES.md) — semantic boundaries (what must not change)
- [`../plan.md`](../plan.md) — phased roadmap (Phase 4 candidates)
- [`../../AGENTS.md`](../../AGENTS.md) — operating contract; source-of-truth rules
