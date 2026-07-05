import logging
import time
from typing import Dict, Any
from app.core.contracts import IModule
from app.core.event_bus import EventBus
from app.core.ai_runtime import AIRuntimeManager
from app.core.models.results import OCRResult
from .settings import OCRSettings

logger = logging.getLogger(__name__)

class OCRController(IModule):
    def __init__(self, event_bus: EventBus, ai_runtime: AIRuntimeManager, settings: OCRSettings):
        self.event_bus = event_bus
        self.ai_runtime = ai_runtime
        self.settings = settings
        self.active_model_id = "easyocr"
        self.last_result: OCRResult = OCRResult()

    def initialize(self) -> bool:
        logger.info("OCRController initialized.")
        return True

    def configure(self, settings: Dict[str, Any]) -> bool:
        self.settings.update(settings)
        if "model_id" in settings:
            self.active_model_id = settings["model_id"]
        return True

    def process(self, context: Dict[str, Any]) -> OCRResult:
        frame = context["frame"]
        min_confidence = self.settings.get_settings().get("min_confidence", 0.3)

        try:
            adapter_result = self.ai_runtime.predict(
                self.active_model_id, {"frame": frame, "min_confidence": min_confidence}
            )
        except Exception as e:
            logger.error(f"OCR inference failed: {e}")
            return OCRResult()

        if not adapter_result:
            return OCRResult()

        result = OCRResult(
            texts=adapter_result.get("texts", []),
            latency_ms=adapter_result.get("latency_ms", 0.0),
            model_name=self.active_model_id,
            timestamp=time.time(),
        )
        self.last_result = result

        if len(result.texts) > 0:
            self.event_bus.publish("TextDetected", {"count": len(result.texts), "model": result.model_name})

        return result

    def render(self, result: OCRResult) -> Any:
        return result

    def cleanup(self) -> None:
        self.ai_runtime.unload_model(self.active_model_id)
        logger.info("OCRController cleaned up.")

    def health_check(self) -> bool:
        return True
