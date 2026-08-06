"""Face record CRUD, deduplication, and embedding history."""

import structlog
import uuid
import numpy as np
from datetime import datetime
from pipeline.models import DedupResult, DedupStatus
from utils.db_client import get_faces_collection
from utils.db_search import vector_search, find_similar_unknowns
from config import settings

logger = structlog.get_logger(__name__)


def _compute_mean_embedding(embeddings: list) -> list:
    """Compute the mean of a list of embeddings (L2-normalized)."""
    if not embeddings:
        return []
    arr = np.array(embeddings, dtype=np.float32)
    mean = arr.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-6:
        return embeddings[-1] if embeddings else []
    return (mean / norm).tolist()


def store_face(person_id: str, name: str, role: str, embedding: list,
               image_url: str, tags: list = None, quality_scores: dict = None,
               camera_id: str = None,
               quality_score: float = None,
               person_crop_url: str = None) -> str:
    """Insert a new face record. Dedup must be handled by the caller
    via deduplicate_identity() — this function is a pure insert."""
    collection = get_faces_collection()

    doc = {
        "person_id": person_id,
        "name": name,
        "role": role,
        "embeddings": [embedding],
        "latest_embedding": embedding,
        "mean_embedding": embedding,
        "latest_embedding_quality": quality_score if quality_score is not None else 0.0,
        "embedding_model": "arcface",
        "images": [{"id": str(uuid.uuid4()), "url": image_url, "captured_at": datetime.utcnow()}],
        "person_crop_url": person_crop_url,
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
                alert_level: str = None, verified_by: str = None,
                quality_score: float = None):
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
    if embedding is not None:
        push_ops["embeddings"] = embedding

    if push_ops:
        update_ops["$push"] = push_ops
        if "embeddings" in push_ops:
            emb_cap = getattr(settings, "EMBEDDING_HISTORY_CAP", 25)
            update_ops["$push"]["embeddings"] = {
                "$each": [push_ops["embeddings"]],
                "$slice": -emb_cap
            }
            existing = collection.find_one(
                {"person_id": person_id},
                {"embeddings": 1}
            )
            if existing:
                all_embs = existing.get("embeddings", [])
                all_embs.append(embedding)
                all_embs = all_embs[-emb_cap:]
                update_ops["$set"]["mean_embedding"] = _compute_mean_embedding(all_embs)

    result = collection.update_one({"person_id": person_id}, update_ops)

    if quality_score is not None and embedding is not None:
        collection.update_one(
            {
                "person_id": person_id,
                "$or": [
                    {"latest_embedding_quality": {"$exists": False}},
                    {"latest_embedding_quality": {"$lt": quality_score}},
                ],
            },
            {"$set": {
                "latest_embedding": embedding,
                "latest_embedding_quality": quality_score,
            }},
        )

    return result.modified_count > 0


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

    query = {"role": "unknown", "verified": {"$ne": True}}
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


def delete_face(person_id: str) -> bool:
    collection = get_faces_collection()
    result = collection.delete_one({"person_id": person_id})
    return result.deleted_count > 0


def deduplicate_identity(embedding: list, threshold: float = None) -> DedupResult:
    """Check if this embedding already exists in the database.

    Pure decision function — does NOT mutate the database.
    Atlas vector search is the primary engine. Python cosine scan
    runs only when Atlas throws an exception.
    """
    if threshold is None:
        threshold = settings.DEDUP_SIMILARITY_THRESHOLD

    try:
        result = vector_search(embedding, limit=5)
        matches = result["matches"]
    except Exception:
        logger.warning("dedup_atlas_failed",
                       msg="Atlas vector search unavailable — falling back to Python scan",
                       exc_info=True)
        matches = None

    if matches is not None:
        for m in matches:
            if m["similarity_score"] < threshold:
                continue
            existing_id = m["person_id"]
            existing = get_faces_collection().find_one(
                {"person_id": existing_id}, {"role": 1, "verified": 1}
            )
            if not existing:
                continue
            if existing.get("verified"):
                logger.warning("dedup_verified_conflict",
                               existing_person_id=existing_id,
                               similarity=m["similarity_score"])
                continue
            return DedupResult(
                status=DedupStatus.MERGED,
                person_id=existing_id,
                similarity=m["similarity_score"],
            )
        return DedupResult(status=DedupStatus.NEW)

    try:
        similar = find_similar_unknowns(embedding, threshold=threshold)
        if similar:
            best = similar[0]
            return DedupResult(
                status=DedupStatus.MERGED,
                person_id=best["person_id"],
                similarity=best["similarity_score"],
            )
        return DedupResult(status=DedupStatus.NEW)
    except Exception:
        logger.error("dedup_python_failed",
                     msg="Both Atlas and Python dedup failed — aborting registration",
                     exc_info=True)
        return DedupResult(
            status=DedupStatus.FAILED,
            reason="python_unavailable",
        )
