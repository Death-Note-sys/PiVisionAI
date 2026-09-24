import logging
from app.core.contracts import IPlugin, IModule, IService, ISettingsProvider
from .controller import ColorAnalysisController
from .service import ColorAnalysisService
from .settings import ColorAnalysisSettings

logger = logging.getLogger(__name__)

class ColorAnalysisPlugin(IPlugin):
    """Factory for Color Analysis. Classical CV — accepts ai_runtime and
    model_registry for interface compatibility only, uses neither."""

    def __init__(self, event_bus, ai_runtime=None, model_registry=None):
        self.event_bus = event_bus
        self.settings = ColorAnalysisSettings()
        self._module = None

    def create_module(self) -> IModule:
        self._module = ColorAnalysisController(self.event_bus, self.settings)
        return self._module

    def create_service(self) -> IService:
        return ColorAnalysisService(self.settings, module_ref=self._module)

    def get_settings_provider(self) -> ISettingsProvider:
        return self.settings
