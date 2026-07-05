import logging
from app.core.contracts import IPlugin, IModule, IService, ISettingsProvider
from .controller import MeasurementController
from .service import MeasurementService
from .settings import MeasurementSettings

logger = logging.getLogger(__name__)

class MeasurementPlugin(IPlugin):
    """Factory for the Measurement module. Classical CV — accepts ai_runtime
    and model_registry for interface compatibility with ModuleController's
    uniform instantiation call, but does not use either."""

    def __init__(self, event_bus, ai_runtime=None, model_registry=None):
        self.event_bus = event_bus
        self.settings = MeasurementSettings()
        self._module = None

    def create_module(self) -> IModule:
        self._module = MeasurementController(self.event_bus, self.settings)
        return self._module

    def create_service(self) -> IService:
        return MeasurementService(self.settings, module_ref=self._module)

    def get_settings_provider(self) -> ISettingsProvider:
        return self.settings
