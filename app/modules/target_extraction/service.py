import logging
from typing import Dict, Any, Optional
from app.core.contracts import IService
from .settings import TargetExtractionSettings

logger = logging.getLogger(__name__)


class TargetExtractionService(IService):
    def __init__(self, settings: TargetExtractionSettings, module_ref=None):
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
        if self.module_ref and hasattr(self.module_ref, "configure"):
            return self.module_ref.configure(new_settings)
        return self.settings.update(new_settings)

    def set_roi(self, x: int, y: int, w: int, h: int) -> bool:
        if self.module_ref and hasattr(self.module_ref, "set_roi"):
            return self.module_ref.set_roi(x, y, w, h)
        return False

    def reset_roi(self) -> bool:
        if self.module_ref and hasattr(self.module_ref, "reset_roi"):
            return self.module_ref.reset_roi()
        return False

    def get_extraction_path(self) -> Optional[str]:
        if self.module_ref and hasattr(self.module_ref, "get_extraction_path"):
            return self.module_ref.get_extraction_path()
        return None

    def get_status(self) -> Dict[str, Any]:
        roi_status = "Untaught"
        if self.module_ref and hasattr(self.module_ref, "roi_status"):
            roi_status = self.module_ref.roi_status
        elif self.module_ref and hasattr(self.module_ref, "last_result"):
            roi_status = getattr(self.module_ref.last_result, "roi_status", "Untaught")

        status = {
            "active": self.is_active,
            "paused": self.is_paused,
            "settings": self.settings.get_settings(),
            "telemetry": {
                "roi_status": roi_status,
                "engine": None,
                "fallback_reason": None,
                "area_px": 0,
                "area_ratio": 0.0,
                "bbox": None,
                "centroid": None,
                "is_ok": None,
                "reason": None,
                "latency_ms": 0.0,
            },
        }
        if self.module_ref and hasattr(self.module_ref, "last_result"):
            r = self.module_ref.last_result
            status["telemetry"].update({
                "roi_status": roi_status,
                "engine": r.engine,
                "fallback_reason": r.fallback_reason,
                "area_px": r.area_px,
                "area_ratio": r.area_ratio,
                "bbox": r.bbox,
                "centroid": r.centroid,
                "is_ok": r.is_ok,
                "reason": r.reason,
                "latency_ms": r.latency_ms,
            })
        return status
