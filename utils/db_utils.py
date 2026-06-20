import logging
from datetime import datetime
from typing import Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from config import settings
from utils.embedding_utils import compare_similarity, atlas_score_to_cosine

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None
_faces_collection: Optional[Collection] = None
_events_collection: Optional[Collection] = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGODB_URI)
    return _client


def get_faces_collection() -> Collection:
    global _faces_collection
    if _faces_collection is None:
        db = get_client()[settings.MONGODB_DATABASE]
        _faces_collection = db[settings.MONGODB_COLLECTION]
    return _faces_collection


def get_events_collection() -> Collection:
    global _events_collection
    if _events_collection is None:
        db = get_client()[settings.MONGODB_DATABASE]
        _events_collection = db[settings.MONGODB_EVENTS_COLLECTION]
    return _events_collection


def vector_search(embedding: list, filter_role: str = None, limit: int = 5) -> list:
    collection = get_faces_collection()

    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "face_vector_index",
                    "path": "latest_embedding",
                    "queryVector": embedding,
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        if filter_role:
            pipeline.insert(0, {"$match": {"role": filter_role}})

        results = list(collection.aggregate(pipeline))

        matches = []
        for r in results:
            raw_cosine = atlas_score_to_cosine(r.get("score", 0))
            if compare_similarity(raw_cosine):
                matches.append({
                    "person_id": r.get("person_id"),
                    "name": r.get("name"),
                    "role": r.get("role"),
                    "tags": r.get("tags", []),
                    "similarity_score": raw_cosine,
                    "image_url": r.get("images", [{}])[0].get("url") if r.get("images") else None
                })

        return matches

    except Exception as e:
        logger.warning(f"Atlas vector search failed, falling back to Python scan: {e}")
        return _python_cosine_scan(embedding, filter_role, limit)


def _python_cosine_scan(embedding: list, filter_role: str = None, limit: int = 5) -> list:
    import numpy as np
    collection = get_faces_collection()

    query = {}
    if filter_role:
        query["role"] = filter_role

    all_faces = list(collection.find(query, {"latest_embedding": 1, "person_id": 1,
                                              "name": 1, "role": 1, "tags": 1,
                                              "images": 1}))

    if not all_faces:
        return []

    query_emb = np.array(embedding, dtype=np.float32)
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-6)

    scored = []
    for face in all_faces:
        stored_emb = np.array(face.get("latest_embedding", []), dtype=np.float32)
        if len(stored_emb) == 0:
            continue
        stored_emb = stored_emb / (np.linalg.norm(stored_emb) + 1e-6)
        similarity = float(np.dot(query_emb, stored_emb))
        scored.append({
            "person_id": face.get("person_id"),
            "name": face.get("name"),
            "role": face.get("role"),
            "tags": face.get("tags", []),
            "similarity_score": similarity,
            "image_url": face.get("images", [{}])[0].get("url") if face.get("images") else None
        })

    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    return [s for s in scored[:limit] if compare_similarity(s["similarity_score"])]


def find_similar_unknowns(embedding: list, threshold: float = None) -> list:
    if threshold is None:
        threshold = settings.DEDUP_SIMILARITY_THRESHOLD

    import numpy as np
    collection = get_faces_collection()

    unknowns = list(collection.find({"role": "unknown"}, {"latest_embedding": 1, "person_id": 1}))

    if not unknowns:
        return []

    query_emb = np.array(embedding, dtype=np.float32)
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-6)

    similar = []
    for u in unknowns:
        stored_emb = np.array(u.get("latest_embedding", []), dtype=np.float32)
        if len(stored_emb) == 0:
            continue
        stored_emb = stored_emb / (np.linalg.norm(stored_emb) + 1e-6)
        similarity = float(np.dot(query_emb, stored_emb))
        if similarity >= threshold:
            similar.append({
                "person_id": u.get("person_id"),
                "similarity_score": similarity
            })

    return similar


def store_face(person_id: str, name: str, role: str, embedding: list,
               image_url: str, tags: list = None, quality_scores: dict = None,
               camera_id: str = None) -> str:
    collection = get_faces_collection()

    existing = find_similar_unknowns(embedding)
    if existing:
        update_face(existing[0]["person_id"], image_url, embedding)
        return existing[0]["person_id"]

    doc = {
        "person_id": person_id,
        "name": name,
        "role": role,
        "embeddings": [embedding],
        "latest_embedding": embedding,
        "embedding_model": "arcface",
        "images": [{"url": image_url, "captured_at": datetime.utcnow()}],
        "source": {"camera_id": camera_id, "captured_at": datetime.utcnow()},
        "quality_scores": quality_scores or {},
        "tags": tags or [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

    collection.insert_one(doc)
    return person_id


def update_face(person_id: str, image_url: str = None, embedding: list = None):
    collection = get_faces_collection()

    update_ops = {"$set": {"updated_at": datetime.utcnow()}}

    if image_url:
        update_ops["$push"] = {"images": {"url": image_url, "captured_at": datetime.utcnow()}}

    if embedding:
        update_ops["$push"] = {"embeddings": embedding}
        update_ops["$set"]["latest_embedding"] = embedding

    collection.update_one({"person_id": person_id}, update_ops)


def log_event(track_id: str, camera_id: str, status: str, alert_level: str,
              person_id: str = None, name: str = None, is_masked: bool = False,
              similarity_score: float = 0.0, image_url: str = None,
              reason: str = "", alerted: bool = False):
    collection = get_events_collection()

    doc = {
        "track_id": track_id,
        "camera_id": camera_id,
        "timestamp": datetime.utcnow(),
        "status": status,
        "alert_level": alert_level,
        "person_id": person_id,
        "name": name,
        "is_masked": is_masked,
        "similarity_score": similarity_score,
        "image_url": image_url,
        "reason": reason,
        "alerted": alerted
    }

    collection.insert_one(doc)
