from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError
from typing import Optional, Dict, Any

class CalibrationRequest(BaseModel):
    """Payload for POST /api/v1/measurement/calibrate. x1/y1/x2/y2 are pixel
    coordinates of two points the operator clicked on the frame;
    real_length_mm is the true physical distance between them."""
    x1: int
    y1: int
    x2: int
    y2: int
    real_length_mm: Optional[float] = None
    real_length_cm: Optional[float] = None

class UpdateMeasurementSettingsRequest(BaseModel):
    canny_low: Optional[int] = Field(None, ge=0, le=255)
    canny_high: Optional[int] = Field(None, ge=0, le=255)
    min_contour_area: Optional[float] = Field(None, ge=0)
    show_contours: Optional[bool] = None
    show_dimensions: Optional[bool] = None
    unit: Optional[str] = None
    thickness: Optional[int] = Field(None, ge=1, le=10)
    target_width_mm: Optional[float] = None
    target_length_mm: Optional[float] = None
    tolerance_mm: Optional[float] = None
    judgment_roi: Optional[Dict[str, int]] = None
    judge_enabled: Optional[bool] = None

    @model_validator(mode="after")
    def _validate_tolerance_when_judging(self):
        if self.tolerance_mm is not None:
            if self.tolerance_mm < 0:
                raise PydanticCustomError("invalid_tolerance", "tolerance_mm cannot be negative", {})
            if self.tolerance_mm == 0.0 and self.judge_enabled is True:
                raise PydanticCustomError("invalid_tolerance", "tolerance_mm must be > 0", {})
        return self

    def to_update_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

