from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from dashboard.backend.models import (
    VerifyRequest, UpdateFaceRequest, FaceResponse,
    UnknownFacesResponse, VerifyResponse, AlertLevel
)
from utils.db_utils import (
    get_unknown_faces, get_face_by_id, verify_person,
    update_face, delete_face
)
import asyncio
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("", response_model=UnknownFacesResponse)
async def list_faces(
    status: Optional[str] = Query(None, description="Filter: unknown, verified, all"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    try:
        if status == "unknown":
            result = await asyncio.to_thread(get_unknown_faces, limit=limit, offset=offset)
            return result
        elif status == "verified":
            from utils.db_utils import get_faces_collection

            def _fetch():
                collection = get_faces_collection()
                query = {"verified": True}
                total = collection.count_documents(query)
                faces = list(collection.find(query, {"latest_embedding": 0})
                            .sort("verified_at", -1)
                            .skip(offset)
                            .limit(limit))
                for face in faces:
                    face["_id"] = str(face["_id"])
                return {"faces": faces, "total": total, "limit": limit, "offset": offset}

            return await asyncio.to_thread(_fetch)
        else:
            result = await asyncio.to_thread(get_unknown_faces, limit=limit, offset=offset)
            return result
    except Exception as e:
        logger.error("list_faces_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch faces")


@router.get("/{person_id}", response_model=FaceResponse)
async def get_face(person_id: str):
    try:
        face = await asyncio.to_thread(get_face_by_id, person_id)
        if not face:
            raise HTTPException(status_code=404, detail="Person not found")
        return face
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_face_failed", person_id=person_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch face")


@router.post("/{person_id}/verify", response_model=VerifyResponse)
async def verify(person_id: str, body: VerifyRequest):
    try:
        success = await asyncio.to_thread(
            verify_person,
            person_id=person_id,
            name=body.name,
            alert_level=body.alert_level.value,
            verified_by="operator"
        )
        if not success:
            raise HTTPException(status_code=404, detail="Person not found")

        return VerifyResponse(
            status="verified",
            person_id=person_id,
            name=body.name,
            alert_level=body.alert_level.value
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("verify_failed", person_id=person_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to verify person")


@router.put("/{person_id}")
async def update(person_id: str, body: UpdateFaceRequest):
    try:
        face = await asyncio.to_thread(get_face_by_id, person_id)
        if not face:
            raise HTTPException(status_code=404, detail="Person not found")

        await asyncio.to_thread(
            update_face,
            person_id=person_id,
            name=body.name,
            tags=body.tags,
            alert_level=body.alert_level.value if body.alert_level else None
        )

        return {"status": "updated", "person_id": person_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("update_face_failed", person_id=person_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update face")


@router.delete("/{person_id}")
async def delete(person_id: str):
    try:
        success = await asyncio.to_thread(delete_face, person_id)
        if not success:
            raise HTTPException(status_code=404, detail="Person not found")

        return {"status": "deleted", "person_id": person_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_face_failed", person_id=person_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete face")
