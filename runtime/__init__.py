"""Runtime package — process lifecycle concerns extracted from main.py.

One-shot startup tasks (model prewarming, database checks, LLM probe),
the API server lifecycle, and the track queue consumers live here.
Composition/wiring stays in main.py.
"""
