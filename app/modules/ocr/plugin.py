import logging
from app.core.contracts import IPlugin, IModule, IService, ISettingsProvider
from .controller import OCRController
from .service import OCRService
from .settings import OCRSettings

logger = logging.getLogger(__name__)

class OCRPlugin(IPlugin):
    def __init__(self, event_bus, ai_runtime, model_registry):
        self.event_bus = event_bus
        self.ai_runtime = ai_runtime
        self.model_registry = model_registry
        self.settings = OCRSettings()
        self._module = None

    def create_module(self) -> IModule:
        self._module = OCRController(self.event_bus, self.ai_runtime, self.settings)
        return self._module

    def create_service(self) -> IService:
        return OCRService(self.settings, module_ref=self._module)

    def get_settings_provider(self) -> ISettingsProvider:
        return self.settings
