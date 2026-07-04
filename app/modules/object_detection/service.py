import logging
from typing import Dict, Any
from app.core.contracts import IService
from .settings import ObjectDetectionSettings

logger = logging.getLogger(__name__)

class ObjectDetectionService(IService):
    """API Service for external interaction."""
    
    def __init__(self, settings: ObjectDetectionSettings, module_ref=None):
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
            "settings": self.settings.get_settings()
        }
        if self.module_ref and hasattr(self.module_ref, 'last_result'):
            status["telemetry"] = {
                "objects_count": self.module_ref.last_result.objects_count,
                "latency_ms": self.module_ref.last_result.latency_ms,
                "model_name": self.module_ref.last_result.model_name,
                "timestamp": self.module_ref.last_result.timestamp
            }
        return status
