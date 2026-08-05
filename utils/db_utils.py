"""Re-export facade — all public names live in domain modules.

All 12+ consumers import from here; nothing breaks.
"""
from __future__ import annotations

from datetime import datetime, timedelta  # noqa: F401
from pymongo import ReturnDocument  # noqa: F401
from config import settings  # noqa: F401
from pipeline.models import DedupResult, DedupStatus  # noqa: F401

# Client (connection + collection singletons)
from utils.db_client import get_client, close_client  # noqa: F401
from utils.db_client import get_faces_collection, get_events_collection, get_memory_collection  # noqa: F401

# Face CRUD, dedup, embedding history
from utils.db_faces import store_face, update_face, verify_person  # noqa: F401
from utils.db_faces import get_unknown_faces, get_face_by_id, delete_face  # noqa: F401
from utils.db_faces import deduplicate_identity  # noqa: F401

# Event logging + stats
from utils.db_events import log_event, get_events_with_faces, get_stats  # noqa: F401

# Visit memory CRUD
from utils.db_memory import get_or_create_memory, update_visit_memory  # noqa: F401
from utils.db_memory import get_visit_history, get_memory_stats  # noqa: F401

# Vector search + index maintenance
from utils.db_search import vector_search, find_similar_unknowns  # noqa: F401
from utils.db_search import check_atlas_search_index, backfill_missing_embeddings  # noqa: F401
