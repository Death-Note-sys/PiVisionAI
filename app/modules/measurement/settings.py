from typing import Dict, Any
from app.core.contracts import ISettingsProvider
from pydantic import BaseModel

class MeasurementSettingsModel(BaseModel):
    canny_low: int = 50
    canny_high: int = 150
    min_contour_area: float = 500.0
    show_contours: bool = True
    show_dimensions: bool = True
    unit: str = "cm"
    thickness: int = 2

class MeasurementSettings(ISettingsProvider):
    def __init__(self):
        self._settings = MeasurementSettingsModel()

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
