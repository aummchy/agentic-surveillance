# Common Gotchas

> Known issues, workarounds, and things that will trip you up.

## Camera

| Gotcha | Fix |
|--------|-----|
| MSMF drops frames (error -1072875772) | Set `CAMERA_BACKEND=dshow` in `.env` |
| Camera shows black screen | Check `CAMERA_INDEX=0` or set `CAMERA_SOURCE` |
| RTSP stream freezes | Set `CAMERA_SOURCE` with full RTSP URL, check network |
| Wrong resolution | Camera may not support 1280×720; check actual resolution in logs |

## Face Recognition

| Gotcha | Fix |
|--------|-----|
| `atlas_search_index_missing` warning | Create Atlas Vector Search index named `vector_index` on `faces.latest_embedding` |
| Blurry faces matched to wrong person | Quality gates prevent this — check `QUALITY_VALID_BLUR_MIN=40` |
| Low confidence for known person | Check similarity score; if 0.45-0.65, it's a moderate match. Confidence formula adds quality + memory signals |
| Face not detected | Person may be too far, too dark, or facing away. Check `DET_SCORE_RELAXED=0.20` |
| Recognition never runs | Check `RECOGNITION_INTERVAL_FRAMES=20` and `FRAME_SKIP=2` — recognition runs every 40 actual frames |

## MongoDB

| Gotcha | Fix |
|--------|-----|
| `MONGODB_URI is required` | Set in `.env`, not `config.jsonc` |
| Vector search slow | Atlas free tier has limits; check `VECTOR_SEARCH_CANDIDATES=150` |
| Python fallback triggers | Atlas index missing or timeout; system still works but slower |
| Embedding quality degrades | Quality-gated updates prevent this; check `latest_embedding_quality` field |

## LLM (Ollama)

| Gotcha | Fix |
|--------|-----|
| `llm_unavailable` at startup | Ollama not running — system works, alerts use template strings |
| Slow LLM responses | Check `OLLAMA_TIMEOUT=30`; Gemma 3 4B fits in 4GB VRAM |
| Wrong model | Set `OLLAMA_MODEL` in `.env`; swap between `gemma3:4b` and `qwen3.5:4b` |

## Dashboard

| Gotcha | Fix |
|--------|-----|
| `axios` version error | Pin `axios@1.7.9` — versions ≥1.7.10 break Vite's esbuild |
| WebSocket disconnects | Check browser console; auto-reconnect handled client-side |
| CORS error | Backend CORS configured for `localhost:5173`; check port |

## Performance

| Gotcha | Fix |
|--------|-----|
| Low FPS | Check `PERFORMANCE_STATS=True` for timing; YOLO on CPU is ~128ms/frame |
| Camera loop blocks | Should never happen — all I/O is async. Check worker thread count |
| High memory usage | `EMBEDDING_HISTORY_CAP=25` limits per-person embeddings; JPEG encoding uses memory |

## Thread Safety

| Gotcha | Fix |
|--------|-----|
| Race condition on track | `TrackState._lock` and `Track._lock` protect all mutations |
| Recognition writes to expired track | `begin_recognition()` / `end_recognition()` prevent premature removal |
| Alert sent twice | `track.mark_alerted_once()` is atomic; `should_send_alert()` has cooldown |

## Data Integrity

| Gotcha | Fix |
|--------|-----|
| Auto-registered unknowns multiply | `find_similar_unknowns()` deduplicates before creating new records |
| Verified person overwritten | `store_face()` skips verified faces during dedup |
| Embedding quality downgraded | `update_face()` only overwrites if new quality > stored quality |
| Confidence flickers | Max-confidence gate prevents downgrades; only upgrades |

## Windows Specific

| Gotcha | Fix |
|--------|-----|
| MSMF backend issues | Set `CAMERA_BACKEND=dshow` |
| Path separator issues | Use forward slashes in URLs; `os.path` handles Windows paths |
| FutureWarning from insightface | Harmless; deprecated `estimate` method in 0.26 |

## See also
- [[All Thresholds]] — every threshold value
- [[All Config Settings]] — every configuration option
- [[Quick Start]] — getting started guide
