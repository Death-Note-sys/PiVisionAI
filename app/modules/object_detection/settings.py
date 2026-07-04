from typing import Dict, Any
from app.core.contracts import ISettingsProvider
from pydantic import BaseModel

class ODSettingsModel(BaseModel):
    confidence: float = 0.5
    iou: float = 0.45
    show_labels: bool = True
    show_conf: bool = True
    thickness: int = 2
    model_id: str = "yolo11n"

class ObjectDetectionSettings(ISettingsProvider):
    def __init__(self):
        self._settings = ODSettingsModel()
        
    def get_settings(self) -> Dict[str, Any]:
        return self._settings.dict()
        
    def update(self, new_settings: Dict[str, Any]) -> bool:
        try:
            for k, v in new_settings.items():
                if hasattr(self._settings, k):
                    setattr(self._settings, k, v)
            return True
        except Exception:
            return False
