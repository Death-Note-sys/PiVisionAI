import logging
from app.core.contracts import IPlugin, IModule, IService, ISettingsProvider
from .controller import AIIdentifyController
from .service import AIIdentifyService
from .settings import AIIdentifySettings

logger = logging.getLogger(__name__)

class AIIdentifyPlugin(IPlugin):
    """Factory for AI Identify. Classical CV — accepts ai_runtime and
    model_registry for interface compatibility with ModuleController's
    uniform instantiation call, but uses neither."""

    def __init__(self, event_bus, ai_runtime=None, model_registry=None):
        self.event_bus = event_bus
        self.settings = AIIdentifySettings()
        self._module = None

    def create_module(self) -> IModule:
        self._module = AIIdentifyController(self.event_bus, self.settings)
        return self._module

    def create_service(self) -> IService:
        return AIIdentifyService(self.settings, module_ref=self._module)

    def get_settings_provider(self) -> ISettingsProvider:
        return self.settings
