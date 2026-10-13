import logging
from typing import Dict, Any
from app.core.container import Container

logger = logging.getLogger(__name__)

class SystemService:
    """Business logic for system-level operations (Telemetry, Sessions, Config)."""
    
    @property
    def container(self):
        return Container.get_instance()


    def get_status(self) -> Dict[str, Any]:
        """Aggregate system status across all core components."""
        perf_metrics = self.container.performance_monitor.get_metrics()
        cam_info = {
            "is_connected": self.container.camera_manager.is_connected,
            "index": self.container.camera_manager.camera_index,
            "fps": perf_metrics.fps
        }
        
        active_backend = self.container.ai_runtime.active_backend
        
        from app.core.models.system import SystemStatus
        
        pipeline_status = self.container.pipeline.get_trigger_status()
        global_state = self.container.state_store.get_global_state()
        
        active_module_id = None
        module_state = {}
        if self.container.pipeline.active_module:
            meta = self.container.module_controller.active_metadata
            if meta:
                active_module_id = meta.id
                module_state = self.container.state_store.get_module_state(active_module_id)
                
        status = SystemStatus(
            status="running",
            camera=cam_info,
            performance=perf_metrics.model_dump(),
            backend=active_backend.model_dump() if active_backend else None,
            session=self.container.session_manager.metadata.model_dump(),
            
            is_recording=self.container.output_manager.is_recording,
            trigger_mode=pipeline_status["mode"],
            trigger_interval_seconds=pipeline_status["interval_seconds"],
            pending_trigger=pipeline_status["pending"],
            last_trigger_time=pipeline_status["last_trigger_time"],
            
            total_count=global_state["total_count"],
            ok_count=global_state["ok_count"],
            ng_count=global_state["ng_count"],
            error_state=global_state["error_state"],
            latest_ng_metadata=global_state["latest_ng_metadata"],
            
            modbus_enabled=self.container.hardware_manager.enabled,
            modbus_port=self.container.hardware_manager.port,
            
            active_module_id=active_module_id,
            module_state=module_state
        )
        return status.model_dump()

    def reload_config(self, new_settings: Dict[str, Any]) -> bool:
        """Update system configuration dynamically."""
        try:
            self.container.config.update_settings(new_settings)
            self.container.event_bus.publish("SettingsChanged", new_settings)
            return True
        except Exception as e:
            logger.error(f"Failed to reload config: {e}")
            return False
