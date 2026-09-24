from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class TeachReferenceRequest(BaseModel):
    """Payload for POST /api/v1/color-analysis/teach-reference. x/y/w/h are
    pixel coordinates of a box drawn around the reference color swatch."""
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)

class UpdateColorSettingsRequest(BaseModel):
    num_colors: Optional[int] = Field(None, ge=1, le=10)
    delta_e_threshold: Optional[float] = Field(None, gt=0)
    downsample_size: Optional[int] = Field(None, ge=20, le=500)
    show_dominant_colors: Optional[bool] = None
    show_reference_match: Optional[bool] = None
    thickness: Optional[int] = Field(None, ge=1, le=10)

    def to_update_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)
