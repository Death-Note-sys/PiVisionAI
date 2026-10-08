import logging
from typing import Dict, Any, List
from app.core.container import Container

logger = logging.getLogger(__name__)

class ModuleService:
    """Business logic for module and plugin operations."""
    
    def __init__(self):
        self.container = Container.get_instance()
        self.module_controller = self.container.module_controller
        self.plugin_manager = self.container.plugin_manager

    def list_available_modules(self) -> List[Dict[str, Any]]:
        """Return a list of all discovered plugins and modules."""
        modules = self.plugin_manager.list_plugins()
        return [m.model_dump() for m in modules]

    def switch_module(self, module_id: str) -> bool:
        """Switch the active computer vision module."""
        return self.module_controller.switch_module(module_id)

    def get_active_module_info(self) -> Dict[str, Any]:
        """Return metadata for the currently active module."""
        if self.module_controller.active_metadata:
            return self.module_controller.active_metadata.model_dump()
        return {}

    def update_module_settings(self, settings: Dict[str, Any]) -> bool:
        """Update settings of the active module."""
        return self.module_controller.update_settings(settings)
        
    def start_active(self) -> bool:
        service = self.module_controller.get_active_service()
        return service.start() if service and hasattr(service, 'start') else False

    def stop_active(self) -> bool:
        service = self.module_controller.get_active_service()
        return service.stop() if service and hasattr(service, 'stop') else False

    def get_active_status(self) -> Optional[Dict[str, Any]]:
        service = self.module_controller.get_active_service()
        return service.get_status() if service and hasattr(service, 'get_status') else None
        
    def invoke_active_method(self, method_name: str, **kwargs) -> Any:
        service = self.module_controller.get_active_service()
        if service and hasattr(service, method_name):
            method = getattr(service, method_name)
            if callable(method):
                return method(**kwargs)
        raise ValueError(f"Active module does not support '{method_name}'")
