from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class AlertLevel(str, Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class VerifyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Person's name")
    alert_level: AlertLevel = Field(default=AlertLevel.low, description="Alert level for this person")


class UpdateFaceRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    alert_level: Optional[AlertLevel] = None
    tags: Optional[List[str]] = None


class FaceResponse(BaseModel):
    person_id: str
    name: Optional[str] = None
    role: str
    tags: List[str] = []
    verified: bool = False
    verified_at: Optional[datetime] = None
    alert_level: str = "low"
    images: List[dict] = []
    source: Optional[dict] = None
    created_at: datetime
    updated_at: datetime
    person_crop_url: Optional[str] = None


class UnknownFacesResponse(BaseModel):
    faces: List[FaceResponse]
    total: int
    limit: int
    offset: int


class EventResponse(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    track_id: str
    camera_id: str
    timestamp: datetime
    status: int
    alert_level: str
    person_id: Optional[str] = None
    name: Optional[str] = None
    person_name: Optional[str] = None
    person_verified: bool = False
    person_image: Optional[str] = None
    is_masked: bool = False
    similarity_score: Optional[float] = 0.0
    image_url: Optional[str] = None
    reason: str = ""
    alerted: bool = False
    person_crop_url: Optional[str] = None

    model_config = {"populate_by_name": True}


class EventsResponse(BaseModel):
    events: List[EventResponse]
    total: int
    limit: int
    offset: int


class VerifyResponse(BaseModel):
    status: str
    person_id: str
    name: str
    alert_level: str
