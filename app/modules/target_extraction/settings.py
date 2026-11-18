import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError
from app.core.contracts import ISettingsProvider

_DEFAULT_SAM_WEIGHTS = (
    "models/mobile_sam.pt" if os.path.exists("models/mobile_sam.pt") else "mobile_sam.pt"
)

VALID_ENGINES = ("auto", "sam", "grabcut")
VALID_VIEW_MODES = ("overlay", "cutout")


class TargetExtractionSettingsModel(BaseModel):
    # Engine
    engine: str = "auto"
    sam_weights: str = _DEFAULT_SAM_WEIGHTS
    grabcut_iterations: int = Field(default=5, ge=1, le=10)
    # ROI: set by set_roi; {} = no ROI configured
    roi_box: Dict[str, Any] = Field(default_factory=dict)
    # Display
    view_mode: str = "overlay"
    show_info: bool = True
    thickness: int = Field(default=2, ge=1, le=10)
    # Judging
    judge_enabled: bool = False
    min_area_ratio: float = Field(default=0.10, ge=0.0, le=1.0)
    max_area_ratio: float = Field(default=0.95, ge=0.0, le=1.0)
    # Schema version
    version: int = 1

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: str) -> str:
        if v not in VALID_ENGINES:
            raise PydanticCustomError(
                "value_error",
                f"engine must be one of {VALID_ENGINES}, got '{v}'"
            )
        return v

    @field_validator("view_mode")
    @classmethod
    def validate_view_mode(cls, v: str) -> str:
        if v not in VALID_VIEW_MODES:
            raise PydanticCustomError(
                "value_error",
                f"view_mode must be one of {VALID_VIEW_MODES}, got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_area_ratios(self):
        if self.min_area_ratio >= self.max_area_ratio:
            raise PydanticCustomError(
                "value_error",
                "min_area_ratio must be less than max_area_ratio"
            )
        return self


class TargetExtractionSettings(ISettingsProvider):
    def __init__(self):
        self._settings = TargetExtractionSettingsModel()

    def get_settings(self) -> Dict[str, Any]:
        return self._settings.model_dump()

    def update(self, new_settings: Dict[str, Any]) -> bool:
        try:
            current = self._settings.model_dump()
            current.update({k: v for k, v in new_settings.items() if k in current})
            self._settings = TargetExtractionSettingsModel(**current)
            return True
        except Exception:
            return False
