import logging
from typing import Dict, Any
from app.core.container import Container

logger = logging.getLogger(__name__)

class SystemService:
    """Business logic for system-level operations (Telemetry, Sessions, Config)."""
    
    def __init__(self):
        self.container = Container.get_instance()

    def get_status(self) -> Dict[str, Any]:
        """Aggregate system status across all core components."""
        perf_metrics = self.container.performance_monitor.get_metrics()
        cam_info = {
            "is_connected": self.container.camera_manager.is_connected,
            "index": self.container.camera_manager.camera_index,
            "fps": perf_metrics.fps
        }
        
        active_backend = self.container.ai_runtime.active_backend
        
        return {
            "status": "running",
            "camera": cam_info,
            "performance": perf_metrics.model_dump(),
            "backend": active_backend.model_dump() if active_backend else None,
            "session": self.container.session_manager.metadata.model_dump()
        }

    def reload_config(self, new_settings: Dict[str, Any]) -> bool:
        """Update system configuration dynamically."""
        try:
            self.container.config.update_settings(new_settings)
            self.container.event_bus.publish("SettingsChanged", new_settings)
            return True
        except Exception as e:
            logger.error(f"Failed to reload config: {e}")
            return False
