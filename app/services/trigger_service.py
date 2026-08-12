import logging
from typing import Dict, Any
from app.core.container import Container

logger = logging.getLogger(__name__)


class TriggerService:
    """Thin bridge to Pipeline's trigger mechanism. Module-agnostic —
    applies to whichever module is currently active."""

    def __init__(self):
        self.container = Container.get_instance()

    def set_mode(self, mode: str) -> bool:
        return self.container.pipeline.set_trigger_mode(mode)

    def set_interval(self, seconds: float) -> bool:
        return self.container.pipeline.set_trigger_interval(seconds)

    def fire(self) -> bool:
        return self.container.pipeline.fire_trigger()

    def get_status(self) -> Dict[str, Any]:
        return self.container.pipeline.get_trigger_status()
