from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator
from pydantic_core import PydanticCustomError


class TeachTemplateRequest(BaseModel):
    """Payload for POST /api/v1/object-counting/teach-template."""
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class UpdateObjectCountingSettingsRequest(BaseModel):
    """Payload for POST /api/v1/object-counting/settings."""
    match_threshold: Optional[float] = Field(None, ge=0.3, le=0.99)
    suppress_frac: Optional[float] = Field(None, ge=0.2, le=1.0)
    iou_threshold: Optional[float] = Field(None, ge=0.05, le=0.9)
    allow_size_variation: Optional[bool] = None
    size_variation_pct: Optional[int] = Field(None, ge=1, le=30)
    max_objects: Optional[int] = Field(None, ge=1, le=1000)
    work_scale: Optional[float] = Field(None, ge=0.25, le=1.0)
    search_roi: Optional[Dict[str, Any]] = None
    show_boxes: Optional[bool] = None
    show_numbers: Optional[bool] = None
    thickness: Optional[int] = Field(None, ge=1, le=10)
    judge_enabled: Optional[bool] = None
    count_min: Optional[int] = None
    count_max: Optional[int] = None

    @model_validator(mode='after')
    def check_counts(self):
        if self.count_min is not None and self.count_min < 0:
            raise PydanticCustomError('value_error', 'count_min must be >= 0')
        if self.count_max is not None and self.count_max < 0:
            raise PydanticCustomError('value_error', 'count_max must be >= 0')
        if self.count_min is not None and self.count_max is not None:
            if self.count_max > 0 and self.count_min > self.count_max:
                raise PydanticCustomError('value_error', 'count_min cannot be greater than count_max')
        return self

    def to_update_dict(self) -> Dict[str, Any]:
        """Only include fields the client actually set."""
        return self.model_dump(exclude_none=True)
