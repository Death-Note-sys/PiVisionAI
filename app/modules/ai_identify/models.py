from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class TeachRegionRequest(BaseModel):
    """Payload for POST /api/v1/ai-identify/teach-good and /teach-bad.
    x/y/w/h are pixel coordinates of a box drawn on the frozen frame."""
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)

class UpdateAIIdentifySettingsRequest(BaseModel):
    min_match_count: Optional[int] = Field(None, ge=4)
    match_ratio_threshold: Optional[float] = Field(None, gt=0.0, le=1.0)
    classification_margin: Optional[float] = Field(None, ge=0.0, le=1.0)
    show_bbox: Optional[bool] = None
    show_classification: Optional[bool] = None
    thickness: Optional[int] = Field(None, ge=1, le=10)

    def to_update_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)
