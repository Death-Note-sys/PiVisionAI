from typing import Dict, Any
from app.core.contracts import ISettingsProvider
from pydantic import BaseModel

class OCRSettingsModel(BaseModel):
    min_confidence: float = 0.3
    show_text: bool = True
    show_confidence: bool = True
    thickness: int = 2
    model_id: str = "easyocr"

class OCRSettings(ISettingsProvider):
    def __init__(self):
        self._settings = OCRSettingsModel()

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
