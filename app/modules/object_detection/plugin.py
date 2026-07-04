import logging
from app.core.contracts import IPlugin, IModule, IService, ISettingsProvider
from .controller import ObjectDetectionController
from .service import ObjectDetectionService
from .settings import ObjectDetectionSettings

logger = logging.getLogger(__name__)

class ObjectDetectionPlugin(IPlugin):
    """Factory for the Object Detection module components."""
    
    def __init__(self, event_bus, ai_runtime, model_registry):
        self.event_bus = event_bus
        self.ai_runtime = ai_runtime
        self.model_registry = model_registry
        self.settings = ObjectDetectionSettings()
        self._module = None
        
    def create_module(self) -> IModule:
        self._module = ObjectDetectionController(self.event_bus, self.ai_runtime, self.settings)
        return self._module
        
    def create_service(self) -> IService:
        return ObjectDetectionService(self.settings, module_ref=self._module)
        
    def get_settings_provider(self) -> ISettingsProvider:
        return self.settings
