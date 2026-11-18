from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError

VALID_ENGINES = ("auto", "sam", "grabcut")
VALID_VIEW_MODES = ("overlay", "cutout")


class SetROIRequest(BaseModel):
    """Payload for POST /api/v1/target-extraction/set-roi."""
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class UpdateTargetExtractionSettingsRequest(BaseModel):
    """Payload for POST /api/v1/target-extraction/settings."""
    engine: Optional[str] = None
    sam_weights: Optional[str] = None
    grabcut_iterations: Optional[int] = Field(None, ge=1, le=10)
    view_mode: Optional[str] = None
    show_info: Optional[bool] = None
    thickness: Optional[int] = Field(None, ge=1, le=10)
    judge_enabled: Optional[bool] = None
    min_area_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_area_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_ENGINES:
            raise PydanticCustomError(
                "value_error",
                f"engine must be one of {VALID_ENGINES}, got '{v}'"
            )
        return v

    @field_validator("view_mode")
    @classmethod
    def validate_view_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_VIEW_MODES:
            raise PydanticCustomError(
                "value_error",
                f"view_mode must be one of {VALID_VIEW_MODES}, got '{v}'"
            )
        return v

    def to_update_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)
