from typing import Dict, Any
from app.core.contracts import ISettingsProvider
from pydantic import BaseModel, Field


class ObjectCountingSettingsModel(BaseModel):
    # Matching
    match_threshold: float = Field(default=0.7, ge=0.3, le=0.99)
    suppress_frac: float = Field(default=0.5, ge=0.2, le=1.0)
    iou_threshold: float = Field(default=0.3, ge=0.05, le=0.9)
    # Size variation
    allow_size_variation: bool = False
    size_variation_pct: int = Field(default=10, ge=1, le=30)
    # Caps
    max_objects: int = Field(default=200, ge=1, le=1000)
    work_scale: float = Field(default=1.0, ge=0.25, le=1.0)
    # ROI: empty dict = whole frame; set x,y,w,h keys to restrict
    search_roi: Dict[str, Any] = Field(default_factory=dict)
    # Display
    show_boxes: bool = True
    show_numbers: bool = True
    thickness: int = Field(default=2, ge=1, le=10)
    # Judging
    judge_enabled: bool = False
    count_min: int = 0
    count_max: int = 0
    # Schema version
    version: int = 1


class ObjectCountingSettings(ISettingsProvider):
    def __init__(self):
        self._settings = ObjectCountingSettingsModel()

    def get_settings(self) -> Dict[str, Any]:
        return self._settings.model_dump()

    def update(self, new_settings: Dict[str, Any]) -> bool:
        try:
            for k, v in new_settings.items():
                if hasattr(self._settings, k):
                    setattr(self._settings, k, v)
            return True
        except Exception:
            return False
