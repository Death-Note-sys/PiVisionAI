from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class CalibrationRequest(BaseModel):
    """Payload for POST /api/v1/measurement/calibrate. x1/y1/x2/y2 are pixel
    coordinates of two points the operator clicked on the frame;
    real_length_cm is the true physical distance between them."""
    x1: int
    y1: int
    x2: int
    y2: int
    real_length_cm: float = Field(gt=0)

class UpdateMeasurementSettingsRequest(BaseModel):
    canny_low: Optional[int] = Field(None, ge=0, le=255)
    canny_high: Optional[int] = Field(None, ge=0, le=255)
    min_contour_area: Optional[float] = Field(None, ge=0)
    show_contours: Optional[bool] = None
    show_dimensions: Optional[bool] = None
    unit: Optional[str] = None
    thickness: Optional[int] = Field(None, ge=1, le=10)

    def to_update_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)
