import logging
from typing import Dict, Any, Optional
from app.core.container import Container

logger = logging.getLogger(__name__)

COLOR_ANALYSIS_MODULE_ID = "core-color-analysis"


class ColorAnalysisService:
    """Thin bridge between the Color Analysis API blueprint and the
    Container. get_status() includes the module-identity check from the
    start (same fix retrofitted into the other services earlier tonight)."""

    def __init__(self):
        self.container = Container.get_instance()

    def _ensure_active(self) -> bool:
        controller = self.container.module_controller
        if (
            controller.active_metadata
            and controller.active_metadata.id == COLOR_ANALYSIS_MODULE_ID
        ):
            return True
        return controller.switch_module(COLOR_ANALYSIS_MODULE_ID)

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
        if not controller.active_metadata or controller.active_metadata.id != COLOR_ANALYSIS_MODULE_ID:
            return None
        service = controller.get_active_service()
        return service.get_status() if service else None

    def update_settings(self, settings: Dict[str, Any]) -> bool:
        if not self._ensure_active():
            return False
        service = self.container.module_controller.get_active_service()
        return service.update_settings(settings) if service else False

    def teach_reference(self, x: int, y: int, w: int, h: int) -> bool:
        if not self._ensure_active():
            return False
        service = self.container.module_controller.get_active_service()
        if service and hasattr(service, "teach_reference"):
            return service.teach_reference(x, y, w, h)
        return False

    def reset_reference(self) -> bool:
        service = self.container.module_controller.get_active_service()
        if service and hasattr(service, "reset_reference"):
            return service.reset_reference()
        return False

    def get_latest_jpeg(self) -> bytes:
        return self.container.pipeline.output_manager.get_latest_jpeg()
