from typing import Dict, Any
from app.core.contracts import ISettingsProvider
from pydantic import BaseModel

class AIIdentifySettingsModel(BaseModel):
    min_match_count: int = 10
    match_ratio_threshold: float = 0.75
    classification_margin: float = 0.05
    show_bbox: bool = True
    show_classification: bool = True
    thickness: int = 2

class AIIdentifySettings(ISettingsProvider):
    def __init__(self):
        self._settings = AIIdentifySettingsModel()

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
