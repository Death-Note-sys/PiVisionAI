import logging
from app.core.contracts import IPlugin, IModule, IService, ISettingsProvider
from .controller import ObjectCountingController
from .service import ObjectCountingService
from .settings import ObjectCountingSettings

logger = logging.getLogger(__name__)


class ObjectCountingPlugin(IPlugin):
    """Factory for Similar Object Counting (Classical CV, template matching)."""

    def __init__(self, event_bus, ai_runtime=None, model_registry=None):
        self.event_bus = event_bus
        self.settings = ObjectCountingSettings()
        self._module = None

    def create_module(self) -> IModule:
        self._module = ObjectCountingController(self.event_bus, self.settings)
        return self._module

    def create_service(self) -> IService:
        return ObjectCountingService(self.settings, module_ref=self._module)

    def get_settings_provider(self) -> ISettingsProvider:
        return self.settings
