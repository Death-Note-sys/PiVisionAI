from typing import Dict, Any
from app.core.contracts import ISettingsProvider
from pydantic import BaseModel

class ColorAnalysisSettingsModel(BaseModel):
    num_colors: int = 5
    delta_e_threshold: float = 10.0
    downsample_size: int = 100
    show_dominant_colors: bool = True
    show_reference_match: bool = True
    thickness: int = 2

class ColorAnalysisSettings(ISettingsProvider):
    def __init__(self):
        self._settings = ColorAnalysisSettingsModel()

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
