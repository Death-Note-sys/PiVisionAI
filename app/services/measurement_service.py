import logging
from typing import Dict, Any, Optional
from app.core.container import Container

logger = logging.getLogger(__name__)

MEASUREMENT_MODULE_ID = "core-measurement"


class MeasurementService:
    """Thin bridge between the Measurement API blueprint and the Container.
    Mirrors ObjectDetectionService's pattern exactly (app/services/object_detection_service.py)."""

    def __init__(self):
        self.container = Container.get_instance()

    def _ensure_active(self) -> bool:
        controller = self.container.module_controller
        if (
            controller.active_metadata
            and controller.active_metadata.id == MEASUREMENT_MODULE_ID
        ):
            return True
        return controller.switch_module(MEASUREMENT_MODULE_ID)

    def start(self) -> bool:
        if not self._ensure_active():
            return False
        service = self.container.module_controller.get_active_service()
        return service.start() if service else False

    def stop(self) -> bool:
        service = self.container.module_controller.get_active_service()
        return service.stop() if service else False

    def pause(self) -> bool:
        service = self.container.module_controller.get_active_service()
        return service.pause() if service else False

    def resume(self) -> bool:
        service = self.container.module_controller.get_active_service()
        return service.resume() if service else False

    def get_status(self) -> Optional[Dict[str, Any]]:
        controller = self.container.module_controller
        if not controller.active_metadata or controller.active_metadata.id != MEASUREMENT_MODULE_ID:
            return None
        service = controller.get_active_service()
        return service.get_status() if service else None

    def update_settings(self, settings: Dict[str, Any]) -> bool:
        if not self._ensure_active():
            return False
        service = self.container.module_controller.get_active_service()
        return service.update_settings(settings) if service else False

    def calibrate(self, x1: int, y1: int, x2: int, y2: int, real_length_cm: float) -> bool:
        if not self._ensure_active():
            return False
        service = self.container.module_controller.get_active_service()
        if service and hasattr(service, "calibrate"):
            return service.calibrate(x1, y1, x2, y2, real_length_cm)
        return False

    def get_latest_jpeg(self) -> bytes:
        return self.container.pipeline.output_manager.get_latest_jpeg()
