import structlog
import uuid
from datetime import datetime
from typing import Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from config import settings
from utils.embedding_utils import compare_similarity, atlas_score_to_cosine

logger = structlog.get_logger(__name__)

_client: Optional[MongoClient] = None
_faces_collection: Optional[Collection] = None
_events_collection: Optional[Collection] = None
_memory_collection: Optional[Collection] = None


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


def get_memory_collection() -> Collection:
    """Get the memory collection for visit history tracking."""
    global _memory_collection
    if _memory_collection is None:
        db = get_client()[settings.MONGODB_DATABASE]
        _memory_collection = db["visit_memory"]
        # Create indexes for efficient queries
        _memory_collection.create_index("person_id", unique=True)
        _memory_collection.create_index("last_seen")
    return _memory_collection


def vector_search(embedding: list, filter_role: str = None, limit: int = 5) -> list:
    collection = get_faces_collection()

    logger.info("vector_search_started", embedding_len=len(embedding), filter_role=filter_role)

    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
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

        results = list(collection.aggregate(pipeline, maxTimeMS=5000))

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
                    "image_url": r.get("images", [{}])[0].get("url") if r.get("images") else None,
                    "verified": r.get("verified", False),
                    "alert_level": r.get("alert_level", "low")
                })

        logger.info("atlas_search_result", match_count=len(matches),
                     scores=[round(m["similarity_score"], 4) for m in matches])
        return matches

    except Exception as e:
        logger.warning("atlas_vector_search_failed", error=str(e))
        results = _python_cosine_scan(embedding, filter_role, limit)
        logger.info("python_scan_fallback_result", match_count=len(results),
                     scores=[round(m["similarity_score"], 4) for m in results])
        return results


def _python_cosine_scan(embedding: list, filter_role: str = None, limit: int = 5) -> list:
    import numpy as np
    collection = get_faces_collection()

    query = {}
    if filter_role:
        query["role"] = filter_role

    SCAN_LIMIT = 500
    all_faces = list(collection.find(query, {"latest_embedding": 1, "person_id": 1,
                                               "name": 1, "role": 1, "tags": 1,
                                               "images": 1, "verified": 1,
                                               "alert_level": 1}).limit(SCAN_LIMIT))

    logger.debug("python_cosine_scan", total_faces=len(all_faces), filter_role=filter_role)

    if not all_faces:
        return []

    query_emb = np.array(embedding, dtype=np.float32)
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-6)

    scored = []
    skipped_empty = 0
    for face in all_faces:
        stored_emb = face.get("latest_embedding", [])
        if not stored_emb or len(stored_emb) == 0:
            skipped_empty += 1
            continue
        stored_emb = np.array(stored_emb, dtype=np.float32)
        if stored_emb.ndim != 1 or len(stored_emb) != len(query_emb):
            skipped_empty += 1
            continue
        stored_emb = stored_emb / (np.linalg.norm(stored_emb) + 1e-6)
        similarity = float(np.dot(query_emb, stored_emb))
        scored.append({
            "person_id": face.get("person_id"),
            "name": face.get("name"),
            "role": face.get("role"),
            "tags": face.get("tags", []),
            "similarity_score": similarity,
            "image_url": face.get("images", [{}])[0].get("url") if face.get("images") else None,
            "verified": face.get("verified", False),
            "alert_level": face.get("alert_level", "low")
        })

    if skipped_empty > 0:
        logger.warning("cosine_scan_skipped_faces", count=skipped_empty,
                        reason="empty or invalid latest_embedding")

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


def find_similar_faces(embedding: list, threshold: float = None) -> list:
    """Search ALL faces (not just unknowns) by embedding similarity."""
    if threshold is None:
        threshold = settings.DEDUP_SIMILARITY_THRESHOLD

    import numpy as np
    collection = get_faces_collection()

    all_faces = list(collection.find(
        {},
        {"latest_embedding": 1, "person_id": 1, "role": 1, "verified": 1, "name": 1}
    ))

    if not all_faces:
        return []

    query_emb = np.array(embedding, dtype=np.float32)
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-6)

    similar = []
    for face in all_faces:
        stored_emb = face.get("latest_embedding", [])
        if not stored_emb or len(stored_emb) == 0:
            continue
        stored_emb = np.array(stored_emb, dtype=np.float32)
        if stored_emb.ndim != 1 or len(stored_emb) != len(query_emb):
            continue
        stored_emb = stored_emb / (np.linalg.norm(stored_emb) + 1e-6)
        similarity = float(np.dot(query_emb, stored_emb))
        if similarity >= threshold:
            similar.append({
                "person_id": face.get("person_id"),
                "similarity_score": similarity,
                "verified": face.get("verified", False),
                "role": face.get("role", "unknown"),
                "name": face.get("name", "Unknown")
            })

    similar.sort(key=lambda x: x["similarity_score"], reverse=True)
    return similar


def store_face(person_id: str, name: str, role: str, embedding: list,
               image_url: str, tags: list = None, quality_scores: dict = None,
               camera_id: str = None) -> str:
    collection = get_faces_collection()

    # Try vector search first (fast, index-backed), fall back to scan
    try:
        matches = vector_search(embedding, limit=3)
    except Exception:
        matches = []

    # Check for similar unknowns
    for m in matches:
        if m["similarity_score"] >= settings.DEDUP_SIMILARITY_THRESHOLD:
            existing_id = m["person_id"]
            existing = collection.find_one({"person_id": existing_id}, {"role": 1})
            if existing and existing.get("role") == "unknown":
                update_face(existing_id, image_url, embedding)
                logger.info("store_face_merged_unknown", person_id=existing_id,
                             similarity=m["similarity_score"])
                return existing_id

    # Check for similar verified/known faces
    for m in matches:
        if m["similarity_score"] >= 0.35:
            existing_id = m["person_id"]
            existing = collection.find_one({"person_id": existing_id}, {"role": 1, "verified": 1})
            if existing:
                update_face(existing_id, image_url, embedding)
                logger.info("store_face_merged_existing", person_id=existing_id,
                             similarity=m["similarity_score"], verified=existing.get("verified", False))
                return existing_id

    doc = {
        "person_id": person_id,
        "name": name,
        "role": role,
        "embeddings": [embedding],
        "latest_embedding": embedding,
        "embedding_model": "arcface",
        "images": [{"id": str(uuid.uuid4()), "url": image_url, "captured_at": datetime.utcnow()}],
        "source": {"camera_id": camera_id, "captured_at": datetime.utcnow()},
        "quality_scores": quality_scores or {},
        "tags": tags or [],
        "verified": False,
        "verified_at": None,
        "verified_by": None,
        "alert_level": "low",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

    collection.insert_one(doc)
    logger.info("store_face_new", person_id=person_id, role=role)
    return person_id


def update_face(person_id: str, image_url: str = None, embedding: list = None,
                name: str = None, tags: list = None, verified: bool = None,
                alert_level: str = None, verified_by: str = None):
    collection = get_faces_collection()

    update_ops = {"$set": {"updated_at": datetime.utcnow()}}

    if name is not None:
        update_ops["$set"]["name"] = name
    if tags is not None:
        update_ops["$set"]["tags"] = tags
    if verified is not None:
        update_ops["$set"]["verified"] = verified
        if verified:
            update_ops["$set"]["verified_at"] = datetime.utcnow()
    if alert_level is not None:
        update_ops["$set"]["alert_level"] = alert_level
    if verified_by is not None:
        update_ops["$set"]["verified_by"] = verified_by

    push_ops = {}
    if image_url:
        push_ops["images"] = {"id": str(uuid.uuid4()), "url": image_url, "captured_at": datetime.utcnow()}
    if embedding:
        push_ops["embeddings"] = embedding
        update_ops["$set"]["latest_embedding"] = embedding

    if push_ops:
        update_ops["$push"] = push_ops
        if "embeddings" in push_ops:
            update_ops["$push"]["embeddings"] = {
                "$each": [push_ops["embeddings"]],
                "$slice": -10
            }

    collection.update_one({"person_id": person_id}, update_ops)


def verify_person(person_id: str, name: str, alert_level: str = "low",
                  verified_by: str = "operator") -> bool:
    collection = get_faces_collection()

    existing = collection.find_one({"person_id": person_id})
    if not existing:
        return False

    tags = existing.get("tags", [])
    if "verified" not in tags:
        tags.append("verified")

    update_ops = {
        "$set": {
            "name": name,
            "verified": True,
            "verified_at": datetime.utcnow(),
            "verified_by": verified_by,
            "alert_level": alert_level,
            "role": "visitor",
            "tags": tags,
            "updated_at": datetime.utcnow()
        }
    }

    result = collection.update_one({"person_id": person_id}, update_ops)
    return result.modified_count > 0


def get_unknown_faces(limit: int = 50, offset: int = 0) -> list:
    collection = get_faces_collection()

    query = {"verified": {"$ne": True}}
    sort_order = [("created_at", -1)]

    total = collection.count_documents(query)
    faces = list(collection.find(query, {"latest_embedding": 0})
                 .sort(sort_order)
                 .skip(offset)
                 .limit(limit))

    for face in faces:
        face["_id"] = str(face["_id"])

    return {"faces": faces, "total": total, "limit": limit, "offset": offset}


def get_face_by_id(person_id: str) -> dict:
    collection = get_faces_collection()
    face = collection.find_one({"person_id": person_id})
    if face:
        face["_id"] = str(face["_id"])
    return face


def update_alert_level(person_id: str, alert_level: str) -> bool:
    collection = get_faces_collection()
    result = collection.update_one(
        {"person_id": person_id},
        {"$set": {"alert_level": alert_level, "updated_at": datetime.utcnow()}}
    )
    return result.modified_count > 0


def delete_face(person_id: str) -> bool:
    collection = get_faces_collection()
    result = collection.delete_one({"person_id": person_id})
    return result.deleted_count > 0


def get_events_with_faces(limit: int = 50, offset: int = 0,
                          status_filter: str = None) -> list:
    collection = get_events_collection()
    faces_collection = get_faces_collection()

    query = {}
    if status_filter:
        query["status"] = status_filter

    total = collection.count_documents(query)
    events = list(collection.find(query)
                  .sort("timestamp", -1)
                  .skip(offset)
                  .limit(limit))

    # Batch face lookups instead of N+1
    person_ids = [e.get("person_id") for e in events if e.get("person_id")]
    face_map = {}
    if person_ids:
        faces = faces_collection.find(
            {"person_id": {"$in": person_ids}},
            {"person_id": 1, "name": 1, "verified": 1, "images": 1}
        )
        for face in faces:
            face_map[face["person_id"]] = face

    for event in events:
        event["_id"] = str(event["_id"])
        pid = event.get("person_id")
        if pid and pid in face_map:
            face = face_map[pid]
            event["person_name"] = face.get("name")
            event["person_verified"] = face.get("verified", False)
            event["person_image"] = (face.get("images", [{}])[0].get("url")
                                    if face.get("images") else None)

    return {"events": events, "total": total, "limit": limit, "offset": offset}


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


def get_stats() -> dict:
    faces_col = get_faces_collection()
    events_col = get_events_collection()

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    return {
        "total_unknown": faces_col.count_documents({"verified": {"$ne": True}}),
        "total_verified": faces_col.count_documents({"verified": True}),
        "events_today": events_col.count_documents({"timestamp": {"$gte": today_start}}),
        "unknown_today": events_col.count_documents({
            "timestamp": {"$gte": today_start},
            "status": {"$in": ["unknown", "masked_unknown"]}
        }),
    }


# ══════════════════════════════════════════════════════════════
# Memory Collection Functions (Phase 2.2)
# ══════════════════════════════════════════════════════════════

def get_or_create_memory(person_id: str) -> dict:
    """Get existing memory or create a new one for a person."""
    collection = get_memory_collection()
    
    memory = collection.find_one({"person_id": person_id})
    if memory:
        return memory
    
    # Create new memory document
    new_memory = {
        "person_id": person_id,
        "visit_count": 0,
        "first_seen": datetime.utcnow(),
        "last_seen": datetime.utcnow(),
        "last_camera": None,
        "last_status": None,
        "typical_hours": [],        # hours when person is usually seen
        "typical_cameras": [],      # cameras where person is usually seen
        "avg_similarity": 0.0,      # average similarity across visits
        "similarity_history": [],   # last N similarity scores
        "status_history": [],       # last N statuses
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    collection.insert_one(new_memory)
    return new_memory


def update_visit_memory(person_id: str, camera_id: str, status: str,
                        similarity: float, is_masked: bool = False) -> dict:
    """Update memory after a visit. Returns updated memory."""
    collection = get_memory_collection()
    now = datetime.utcnow()
    
    memory = get_or_create_memory(person_id)
    
    # Update visit count
    new_count = memory.get("visit_count", 0) + 1
    
    # Update similarity history (keep last 10)
    sim_history = memory.get("similarity_history", [])
    sim_history.append(similarity)
    if len(sim_history) > 10:
        sim_history = sim_history[-10:]
    avg_sim = sum(sim_history) / len(sim_history) if sim_history else 0.0
    
    # Update status history (keep last 10)
    status_history = memory.get("status_history", [])
    status_history.append({"status": status, "timestamp": now})
    if len(status_history) > 10:
        status_history = status_history[-10:]
    
    # Update typical hours (extract hour from last 20 visits)
    hour = now.hour
    typical_hours = memory.get("typical_hours", [])
    typical_hours.append(hour)
    if len(typical_hours) > 20:
        typical_hours = typical_hours[-20:]
    
    # Update typical cameras
    typical_cameras = memory.get("typical_cameras", [])
    if camera_id not in typical_cameras:
        typical_cameras.append(camera_id)
    if len(typical_cameras) > 5:
        typical_cameras = typical_cameras[-5:]
    
    update_ops = {
        "$set": {
            "visit_count": new_count,
            "last_seen": now,
            "last_camera": camera_id,
            "last_status": status,
            "avg_similarity": avg_sim,
            "similarity_history": sim_history,
            "status_history": status_history,
            "typical_hours": typical_hours,
            "typical_cameras": typical_cameras,
            "updated_at": now
        }
    }
    
    # Update first_seen only if it's the first visit
    if new_count == 1:
        update_ops["$set"]["first_seen"] = now
    
    collection.update_one({"person_id": person_id}, update_ops, upsert=True)
    
    # Return updated memory directly instead of re-querying
    return {
        "person_id": person_id,
        "visit_count": new_count,
        "last_seen": now,
        "last_camera": camera_id,
        "last_status": status,
        "avg_similarity": avg_sim,
        "similarity_history": sim_history,
        "status_history": status_history,
        "typical_hours": typical_hours,
        "typical_cameras": typical_cameras,
        "updated_at": now
    }


def get_visit_history(person_id: str) -> dict:
    """Get visit history for a person."""
    collection = get_memory_collection()
    return collection.find_one({"person_id": person_id}) or {}


def get_recent_unknowns(hours: int = 24, limit: int = 50) -> list:
    """Get recently seen unknown persons."""
    collection = get_memory_collection()
    from datetime import timedelta
    
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    return list(collection.find({
        "last_seen": {"$gte": cutoff},
        "last_status": {"$in": ["unknown", "masked_unknown"]}
    }).sort("last_seen", -1).limit(limit))


def get_memory_stats() -> dict:
    """Get memory collection statistics."""
    collection = get_memory_collection()
    
    return {
        "total_persons": collection.count_documents({}),
        "known_persons": collection.count_documents({"visit_count": {"$gt": 1}}),
        "recent_unknowns": collection.count_documents({
            "last_status": {"$in": ["unknown", "masked_unknown"]}
        }),
    }


def check_atlas_search_index():
    """Check if Atlas Vector Search index exists and warn if not."""
    try:
        collection = get_faces_collection()
        indexes = list(collection.list_search_indexes())
        index_names = [idx.get("name") for idx in indexes]
        if "vector_index" not in index_names:
            logger.warning("atlas_search_index_missing",
                           found_indexes=index_names,
                           expected="vector_index",
                           fallback="python_cosine_scan")
        else:
            logger.info("atlas_search_index_found", name="vector_index")
    except Exception as e:
        logger.warning("atlas_search_index_check_failed", error=str(e))


def backfill_missing_embeddings():
    """Fix faces that have embeddings array but empty latest_embedding."""
    collection = get_faces_collection()
    count = 0
    for face in collection.find({"latest_embedding": {"$exists": False}}):
        embs = face.get("embeddings", [])
        if embs:
            last_emb = embs[-1]
            collection.update_one(
                {"person_id": face["person_id"]},
                {"$set": {"latest_embedding": last_emb}}
            )
            count += 1
            logger.info("backfill_embedding", person_id=face["person_id"])
    for face in collection.find({"latest_embedding": []}):
        embs = face.get("embeddings", [])
        if embs:
            last_emb = embs[-1]
            collection.update_one(
                {"person_id": face["person_id"]},
                {"$set": {"latest_embedding": last_emb}}
            )
            count += 1
            logger.info("backfill_empty_embedding", person_id=face["person_id"])
    if count > 0:
        logger.info("backfill_complete", fixed_count=count)
    else:
        logger.info("backfill_not_needed")
