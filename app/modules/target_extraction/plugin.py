import logging
from app.core.contracts import IPlugin, IModule, IService, ISettingsProvider
from .controller import TargetExtractionController
from .service import TargetExtractionService
from .settings import TargetExtractionSettings

logger = logging.getLogger(__name__)


class TargetExtractionPlugin(IPlugin):
    """Factory for AI Target Extraction (box-prompted SAM + GrabCut fallback)."""

    def __init__(self, event_bus, ai_runtime=None, model_registry=None):
        self.event_bus = event_bus
        self.settings = TargetExtractionSettings()
        self._module = None

    def create_module(self) -> IModule:
        self._module = TargetExtractionController(self.event_bus, self.settings)
        return self._module

    def create_service(self) -> IService:
        return TargetExtractionService(self.settings, module_ref=self._module)

    def get_settings_provider(self) -> ISettingsProvider:
        return self.settings
