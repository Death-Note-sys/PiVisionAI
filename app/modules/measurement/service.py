import logging
from typing import Dict, Any, Optional
from app.core.contracts import IService
from .settings import MeasurementSettings

logger = logging.getLogger(__name__)

class MeasurementService(IService):
    """Business-logic service for the Measurement module (IService side of
    the plugin factory) — mirrors ObjectDetectionService's shape exactly,
    plus a calibrate() passthrough to the controller."""

    def __init__(self, settings: MeasurementSettings, module_ref=None):
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

    def calibrate(self, x1: int, y1: int, x2: int, y2: int, real_length_cm: float) -> bool:
        if self.module_ref and hasattr(self.module_ref, "calibrate"):
            return self.module_ref.calibrate(x1, y1, x2, y2, real_length_cm)
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
                "measurement_count": len(r.measurements),
                "calibration_status": r.calibration_status,
                "pixels_per_cm": r.pixels_per_cm,
                "latency_ms": r.latency_ms,
            }
        return status
