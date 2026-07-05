import logging
from typing import Dict, Any, Optional
from app.core.container import Container

logger = logging.getLogger(__name__)

OBJECT_DETECTION_MODULE_ID = "core-object-detection"

class ObjectDetectionService:
    def __init__(self):
        self.container = Container.get_instance()
        
    def _ensure_active(self):
        mc = self.container.module_controller
        if not mc.active_metadata or mc.active_metadata.id != "core-object-detection":
            logger.info("ObjectDetectionService: Module not active, switching to core-object-detection")
            mc.switch_module("core-object-detection")
            
    def start(self) -> bool:
        self._ensure_active()
        svc = self.container.module_controller.get_active_service()
        return svc.start() if svc else False
        
    def stop(self) -> bool:
        self._ensure_active()
        svc = self.container.module_controller.get_active_service()
        return svc.stop() if svc else False
        
    def pause(self) -> bool:
        self._ensure_active()
        svc = self.container.module_controller.get_active_service()
        return svc.pause() if svc else False
        
    def resume(self) -> bool:
        self._ensure_active()
        svc = self.container.module_controller.get_active_service()
        return svc.resume() if svc else False
        
    def get_status(self) -> Optional[Dict[str, Any]]:
        controller = self.container.module_controller
        if not controller.active_metadata or controller.active_metadata.id != OBJECT_DETECTION_MODULE_ID:
            return None
        service = controller.get_active_service()
        return service.get_status() if service else None
        
    def update_settings(self, settings: dict) -> bool:
        self._ensure_active()
        svc = self.container.module_controller.get_active_service()
        return svc.update_settings(settings) if svc else False
        
    def switch_model(self, model_id: str) -> bool:
        self._ensure_active()
        mod = self.container.module_controller.active_module_instance
        if mod and hasattr(mod, 'configure'):
            return mod.configure({"model_id": model_id})
        return False
        
    def get_latest_jpeg(self) -> bytes:
        return self.container.pipeline.output_manager.get_latest_jpeg()
