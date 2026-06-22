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

router = APIRouter()


@router.get("", response_model=UnknownFacesResponse)
async def list_faces(
    status: Optional[str] = Query(None, description="Filter: unknown, verified, all"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    if status == "unknown":
        result = get_unknown_faces(limit=limit, offset=offset)
        return result
    elif status == "verified":
        from utils.db_utils import get_faces_collection
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
    else:
        result = get_unknown_faces(limit=limit, offset=offset)
        return result


@router.get("/{person_id}", response_model=FaceResponse)
async def get_face(person_id: str):
    face = get_face_by_id(person_id)
    if not face:
        raise HTTPException(status_code=404, detail="Person not found")
    return face


@router.post("/{person_id}/verify", response_model=VerifyResponse)
async def verify(person_id: str, body: VerifyRequest):
    success = verify_person(
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


@router.put("/{person_id}")
async def update(person_id: str, body: UpdateFaceRequest):
    face = get_face_by_id(person_id)
    if not face:
        raise HTTPException(status_code=404, detail="Person not found")

    update_face(
        person_id=person_id,
        name=body.name,
        tags=body.tags,
        alert_level=body.alert_level.value if body.alert_level else None
    )

    return {"status": "updated", "person_id": person_id}


@router.delete("/{person_id}")
async def delete(person_id: str):
    success = delete_face(person_id)
    if not success:
        raise HTTPException(status_code=404, detail="Person not found")

    return {"status": "deleted", "person_id": person_id}
