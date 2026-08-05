"""MongoDB connection singleton and collection accessors."""

import structlog
import threading
from typing import Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from config import settings

logger = structlog.get_logger(__name__)

_client: Optional[MongoClient] = None
_faces_collection: Optional[Collection] = None
_events_collection: Optional[Collection] = None
_memory_collection: Optional[Collection] = None
_client_lock = threading.Lock()
_collection_locks = {
    "faces": threading.Lock(),
    "events": threading.Lock(),
    "memory": threading.Lock(),
}


def get_client() -> MongoClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = MongoClient(settings.MONGODB_URI)
    return _client


def close_client():
    """Close MongoDB client on shutdown."""
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


def get_faces_collection() -> Collection:
    global _faces_collection
    if _faces_collection is None:
        with _collection_locks["faces"]:
            if _faces_collection is None:
                db = get_client()[settings.MONGODB_DATABASE]
                _faces_collection = db[settings.MONGODB_COLLECTION]
    return _faces_collection


def get_events_collection() -> Collection:
    global _events_collection
    if _events_collection is None:
        with _collection_locks["events"]:
            if _events_collection is None:
                db = get_client()[settings.MONGODB_DATABASE]
                _events_collection = db[settings.MONGODB_EVENTS_COLLECTION]
                _events_collection.create_index("track_id")
                _events_collection.create_index("timestamp")
                _events_collection.create_index("status")
    return _events_collection


def get_memory_collection() -> Collection:
    """Get the memory collection for visit history tracking."""
    global _memory_collection
    if _memory_collection is None:
        with _collection_locks["memory"]:
            if _memory_collection is None:
                db = get_client()[settings.MONGODB_DATABASE]
                _memory_collection = db["visit_memory"]
                _memory_collection.create_index("person_id", unique=True)
                _memory_collection.create_index("last_seen")
    return _memory_collection
