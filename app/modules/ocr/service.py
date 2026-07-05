import logging
from typing import Dict, Any
from app.core.contracts import IService
from .settings import OCRSettings

logger = logging.getLogger(__name__)

class OCRService(IService):
    def __init__(self, settings: OCRSettings, module_ref=None):
        self.settings = settings
        self.module_ref = module_ref
        self.is_active = False
        self.is_paused = False

    def start(self) -> bool:
        self.is_active = True
        self.is_paused = False
        return True

    def stop(self) -> bool:
        self.is_active = False
        self.is_paused = False
        return True

    def pause(self) -> bool:
        self.is_paused = True
        return True

    def resume(self) -> bool:
        self.is_paused = False
        return True

    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        return self.settings.update(new_settings)

    def get_status(self) -> Dict[str, Any]:
        status = {
            "active": self.is_active,
            "paused": self.is_paused,
            "settings": self.settings.get_settings(),
        }
        if self.module_ref and hasattr(self.module_ref, "last_result"):
            r = self.module_ref.last_result
            status["telemetry"] = {
                "text_count": len(r.texts),
                "latency_ms": r.latency_ms,
                "model_name": r.model_name,
            }
        return status
