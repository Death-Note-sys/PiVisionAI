import logging
from typing import Dict, Any, Optional
from app.core.container import Container

logger = logging.getLogger(__name__)

OBJECT_COUNTING_MODULE_ID = "core-object-counting"


class ObjectCountingService:
    """Thin bridge between the Object Counting API blueprint and the Container."""

    def __init__(self):
        self.container = Container.get_instance()

    def _ensure_active(self) -> bool:
        controller = self.container.module_controller
        if (
            controller.active_metadata
            and controller.active_metadata.id == OBJECT_COUNTING_MODULE_ID
        ):
            return True
        return controller.switch_module(OBJECT_COUNTING_MODULE_ID)

    @property
    def settings(self):
        """Provide access to the settings provider for API validation."""
        service = self.container.module_controller.get_active_service()
        if service and hasattr(service, "settings"):
            return service.settings
        # Fall back to plugin settings provider if available
        plugin = self.container.plugin_manager.plugins.get(OBJECT_COUNTING_MODULE_ID)
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
        if not controller.active_metadata or controller.active_metadata.id != OBJECT_COUNTING_MODULE_ID:
            return None
        service = controller.get_active_service()
        return service.get_status() if service else None

    def update_settings(self, settings: Dict[str, Any]) -> bool:
        if not self._ensure_active():
            return False
        service = self.container.module_controller.get_active_service()
        return service.update_settings(settings) if service else False

    def teach_template(self, x: int, y: int, w: int, h: int) -> tuple:
        if not self._ensure_active():
            return False, "Failed to activate Object Counting module"
        service = self.container.module_controller.get_active_service()
        if service and hasattr(service, "teach_template"):
            return service.teach_template(x, y, w, h)
        return False, "Active service does not support teach_template"

    def reset_template(self) -> bool:
        service = self.container.module_controller.get_active_service()
        if service and hasattr(service, "reset_template"):
            return service.reset_template()
        return False

    def get_template_path(self) -> Optional[str]:
        service = self.container.module_controller.get_active_service()
        if service and hasattr(service, "get_template_path"):
            return service.get_template_path()
        return None

    def get_latest_jpeg(self) -> bytes:
        return self.container.pipeline.output_manager.get_latest_jpeg()
