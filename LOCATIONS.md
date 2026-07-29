# LOCATIONS.md — Quick Navigation

## Pipeline
- Tracking → pipeline/tracker.py
- Face detection → pipeline/face.py
- Quality scoring → pipeline/face.py:compute_quality()
- Models → pipeline/models.py
- Track state → pipeline/track_state.py

## Agents
- Camera loop → agents/camera_agent.py
- Recognition → agents/recognition.py
- Matching → agents/matching_agent.py
- Memory → agents/memory.py
- Scoring → agents/scoring.py
- Policy → agents/policy.py
- Alerts → agents/alert_agent.py
- Reports → agents/report.py

## Data
- Database CRUD → utils/db_utils.py
- Vector search → utils/db_utils.py:vector_search()
- Embeddings → utils/embedding_utils.py
- LLM client → utils/llm_client.py

## Dashboard
- API → dashboard/backend/main.py
- Routes → dashboard/backend/routes/ (faces, events, reports, chat, live)
- Frontend → dashboard/frontend/src/

## Config
- Settings → config/settings.py
- Tunables → config/config.jsonc
- ByteTrack → config/bytetrack_surveillance.yaml

## Tests
- Test suite → tests/ (43 tests)
- Run: python -m pytest tests/ -v
