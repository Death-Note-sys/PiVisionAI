from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class UpdateSettingsRequest(BaseModel):
    """Payload for POST /api/v1/object-detection/settings."""
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    iou: Optional[float] = Field(None, ge=0.0, le=1.0)
    show_labels: Optional[bool] = None
    show_conf: Optional[bool] = None
    thickness: Optional[int] = Field(None, ge=1, le=10)
    model_id: Optional[str] = None

    def to_update_dict(self) -> Dict[str, Any]:
        """Only include fields the client actually set."""
        return self.model_dump(exclude_none=True)


class SwitchModelRequest(BaseModel):
    """Payload for POST /api/v1/object-detection/model."""
    model_id: str


class DetectionBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionItem(BaseModel):
    box: DetectionBox
    confidence: float
    class_id: int
    label: str


class DetectionResultResponse(BaseModel):
    """API-facing shape returned by GET /api/v1/object-detection/last-result."""
    detections: List[DetectionItem] = Field(default_factory=list)
    objects_count: int = 0
    average_confidence: float = 0.0
    latency_ms: float = 0.0
    model_name: Optional[str] = None
    timestamp: float = 0.0


class ModuleStatusResponse(BaseModel):
    active: bool
    paused: bool
    settings: Dict[str, Any] = Field(default_factory=dict)
