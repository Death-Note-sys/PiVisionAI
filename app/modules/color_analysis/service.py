import logging
from typing import Dict, Any
from app.core.contracts import IService
from .settings import ColorAnalysisSettings

logger = logging.getLogger(__name__)

class ColorAnalysisService(IService):
    def __init__(self, settings: ColorAnalysisSettings, module_ref=None):
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

    def teach_reference(self, x: int, y: int, w: int, h: int) -> bool:
        if self.module_ref and hasattr(self.module_ref, "teach_reference"):
            return self.module_ref.teach_reference(x, y, w, h)
        return False

    def reset_reference(self) -> bool:
        if self.module_ref and hasattr(self.module_ref, "reset_reference"):
            return self.module_ref.reset_reference()
        return False

    def get_status(self) -> Dict[str, Any]:
        status = {
            "active": self.is_active,
            "paused": self.is_paused,
            "settings": self.settings.get_settings(),
        }
        if self.module_ref and hasattr(self.module_ref, "last_result"):
            r = self.module_ref.last_result
            status["telemetry"] = {
                "dominant_colors": r.dominant_colors,
                "reference_status": r.reference_status,
                "reference_color": r.reference_color,
                "delta_e": r.delta_e,
                "match": r.match,
                "latency_ms": r.latency_ms,
            }
        return status
