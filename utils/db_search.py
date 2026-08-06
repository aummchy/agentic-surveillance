"""Vector search (Atlas + Python fallback) and index maintenance."""

import structlog
import numpy as np
from utils.db_client import get_faces_collection
from utils.embedding_utils import compare_similarity, atlas_score_to_cosine
from config import settings

logger = structlog.get_logger(__name__)


def vector_search(embedding: list, filter_role: str = None, limit: int = 5) -> dict:
    collection = get_faces_collection()

    logger.debug("vector_search_started", embedding_len=len(embedding), filter_role=filter_role)

    try:
        vector_stage = {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "latest_embedding",
                "queryVector": embedding,
                "numCandidates": settings.VECTOR_SEARCH_CANDIDATES,
                "limit": limit
            }
        }
        if filter_role:
            vector_stage["$vectorSearch"]["filter"] = {"role": filter_role}

        pipeline = [
            vector_stage,
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(collection.aggregate(pipeline, maxTimeMS=5000))

        all_results = []
        for r in results:
            raw_cosine = atlas_score_to_cosine(r.get("score", 0))
            all_results.append({
                "person_id": r.get("person_id"),
                "name": r.get("name"),
                "role": r.get("role"),
                "tags": r.get("tags", []),
                "similarity_score": raw_cosine,
                "image_url": r.get("images", [{}])[0].get("url") if r.get("images") else None,
                "verified": r.get("verified", False),
                "alert_level": r.get("alert_level", "low")
            })

        top2 = None
        margin = None
        if len(all_results) >= 2:
            top2 = all_results[1]["similarity_score"]
            margin = all_results[0]["similarity_score"] - top2

        matches = [r for r in all_results if compare_similarity(r["similarity_score"])]

        logger.debug("atlas_search_result", match_count=len(matches), total_count=len(all_results),
                     scores=[round(m["similarity_score"], 4) for m in matches])
        return {
            "matches": matches,
            "top2": top2,
            "margin": margin,
            "all_candidates": [
                {"name": r.get("name", "?"), "similarity": round(r["similarity_score"], 4)}
                for r in all_results
            ]
        }

    except Exception as e:
        logger.warning("atlas_vector_search_failed", error=str(e))
        all_results = _python_cosine_scan(embedding, filter_role, limit)
        top2 = None
        margin = None
        if len(all_results) >= 2:
            top2 = all_results[1]["similarity_score"]
            margin = all_results[0]["similarity_score"] - top2
        matches = [r for r in all_results if compare_similarity(r["similarity_score"])]
        logger.info("python_scan_fallback_result", match_count=len(matches), total_count=len(all_results),
                     scores=[round(m["similarity_score"], 4) for m in matches])
        return {
            "matches": matches,
            "top2": top2,
            "margin": margin,
            "all_candidates": [
                {"name": r.get("name", "?"), "similarity": round(r["similarity_score"], 4)}
                for r in all_results
            ]
        }


def _python_cosine_scan(embedding: list, filter_role: str = None, limit: int = 5) -> list:
    collection = get_faces_collection()

    query = {}
    if filter_role:
        query["role"] = filter_role

    SCAN_LIMIT = settings.SCAN_LIMIT
    total_count = collection.count_documents(query)
    all_faces = list(collection.find(query, {"latest_embedding": 1, "person_id": 1,
                                               "name": 1, "role": 1, "tags": 1,
                                               "images": 1, "verified": 1,
                                               "alert_level": 1}).limit(SCAN_LIMIT))

    if total_count > SCAN_LIMIT:
        logger.warning("python_cosine_scan_truncated",
                       total_faces=total_count,
                       scanned=SCAN_LIMIT,
                       skipped=total_count - SCAN_LIMIT,
                       filter_role=filter_role,
                       note="Results may miss best match. Consider enabling Atlas Vector Search.")

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
        stored_emb = np.asarray(stored_emb, dtype=np.float32)
        if stored_emb.ndim != 1 or stored_emb.size == 0 or stored_emb.size != len(query_emb):
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
    return scored[:limit]


def find_similar_unknowns(embedding: list, threshold: float = None) -> list:
    if threshold is None:
        threshold = settings.DEDUP_SIMILARITY_THRESHOLD

    collection = get_faces_collection()

    unknowns = list(collection.find(
        {"role": "unknown"},
        {"latest_embedding": 1, "person_id": 1}
    ).sort("created_at", -1).limit(500))

    if not unknowns:
        return []

    query_emb = np.asarray(embedding, dtype=np.float32)
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-6)

    similar = []
    for u in unknowns:
        stored_emb = np.asarray(u.get("latest_embedding", []), dtype=np.float32)
        if stored_emb.ndim != 1 or stored_emb.size == 0 or stored_emb.size != query_emb.size:
            continue
        stored_emb = stored_emb / (np.linalg.norm(stored_emb) + 1e-6)
        similarity = float(np.dot(query_emb, stored_emb))
        if similarity >= threshold:
            similar.append({
                "person_id": u.get("person_id"),
                "similarity_score": similarity
            })

    similar.sort(key=lambda x: x["similarity_score"], reverse=True)
    return similar


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


_backfill_done = False


def backfill_missing_embeddings():
    """Fix faces that have embeddings array but empty latest_embedding.

    Uses quick count checks to avoid full collection scans when unnecessary.
    Runs at most once per process lifetime.
    """
    global _backfill_done
    if _backfill_done:
        logger.debug("backfill_already_run_this_session")
        return

    collection = get_faces_collection()

    missing_count = collection.count_documents(
        {"$or": [
            {"latest_embedding": {"$exists": False}},
            {"latest_embedding": []}
        ]}
    )

    if missing_count == 0:
        logger.info("backfill_not_needed")
        _backfill_done = True
        return

    logger.info("backfill_started", faces_to_fix=missing_count)
    collection.create_index("latest_embedding")

    count = 0
    for face in collection.find(
        {"$or": [
            {"latest_embedding": {"$exists": False}},
            {"latest_embedding": []}
        ]},
        {"person_id": 1, "embeddings": 1}
    ):
        embs = face.get("embeddings", [])
        if embs:
            last_emb = embs[-1]
            collection.update_one(
                {"person_id": face["person_id"]},
                {"$set": {"latest_embedding": last_emb}}
            )
            count += 1
            logger.info("backfill_embedding", person_id=face["person_id"])

    _backfill_done = True
    if count > 0:
        logger.info("backfill_complete", fixed_count=count)
    else:
        logger.info("backfill_complete_no_fixes_needed")
