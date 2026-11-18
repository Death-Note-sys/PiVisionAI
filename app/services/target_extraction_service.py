import logging
from typing import Dict, Any, Optional
from app.core.container import Container

logger = logging.getLogger(__name__)

TARGET_EXTRACTION_MODULE_ID = "core-target-extraction"


class TargetExtractionAppService:
    """Thin bridge between the Target Extraction API blueprint and the Container."""

    def __init__(self):
        self.container = Container.get_instance()

    def _ensure_active(self) -> bool:
        controller = self.container.module_controller
        if (
            controller.active_metadata
            and controller.active_metadata.id == TARGET_EXTRACTION_MODULE_ID
        ):
            return True
        return controller.switch_module(TARGET_EXTRACTION_MODULE_ID)

    @property
    def settings(self):
        service = self.container.module_controller.get_active_service()
        if service and hasattr(service, "settings"):
            return service.settings
        plugin = self.container.plugin_manager.plugins.get(TARGET_EXTRACTION_MODULE_ID)
        if plugin and hasattr(plugin, "get_settings_provider"):
            return plugin.get_settings_provider()
        return None

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
        if not controller.active_metadata or controller.active_metadata.id != TARGET_EXTRACTION_MODULE_ID:
            return None
        service = controller.get_active_service()
        return service.get_status() if service else None

    def update_settings(self, settings: Dict[str, Any]) -> bool:
        if not self._ensure_active():
            return False
        service = self.container.module_controller.get_active_service()
        return service.update_settings(settings) if service else False

    def set_roi(self, x: int, y: int, w: int, h: int) -> bool:
        if not self._ensure_active():
            return False
        service = self.container.module_controller.get_active_service()
        if service and hasattr(service, "set_roi"):
            return service.set_roi(x, y, w, h)
        return False

    def reset_roi(self) -> bool:
        service = self.container.module_controller.get_active_service()
        if service and hasattr(service, "reset_roi"):
            return service.reset_roi()
        return False

    def get_extraction_path(self) -> Optional[str]:
        service = self.container.module_controller.get_active_service()
        if service and hasattr(service, "get_extraction_path"):
            return service.get_extraction_path()
        return None
