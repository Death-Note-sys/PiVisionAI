import logging
import time
import re
from typing import Dict, Any
from app.core.base_module import BaseModuleController
from app.core.event_bus import EventBus
from app.core.ai_runtime import AIRuntimeManager
from app.core.models.results import OCRResult
from .settings import OCRSettings

logger = logging.getLogger(__name__)

class OCRController(BaseModuleController):
    def __init__(self, event_bus: EventBus, ai_runtime: AIRuntimeManager, settings: OCRSettings):
        super().__init__("core-ocr", event_bus)
        self.ai_runtime = ai_runtime
        self.settings = settings
        
        # Load settings
        saved_settings = self.load_settings()
        if saved_settings:
            self.settings.update(saved_settings)
            
        self.active_model_id = self.settings.get_settings().get("model_id", "easyocr")
        self.last_result: OCRResult = OCRResult()

    def initialize(self) -> bool:
        logger.info("OCRController initialized.")
        return True

    def configure(self, settings: Dict[str, Any]) -> bool:
        current = self.settings.get_settings()
        merged_judge = settings["judge_enabled"] if "judge_enabled" in settings else current.get("judge_enabled", False)
        merged_pattern = settings["expected_pattern"] if "expected_pattern" in settings else current.get("expected_pattern", "")

        if merged_judge:
            if not merged_pattern or not str(merged_pattern).strip():
                logger.error("OCR configure failed: expected_pattern cannot be empty when judge_enabled is True.")
                return False
            try:
                re.compile(merged_pattern)
            except re.error as e:
                logger.error(f"OCR configure failed: invalid regex pattern '{merged_pattern}': {e}")
                return False

        if self.settings.update(settings):
            if "model_id" in settings:
                self.active_model_id = settings["model_id"]
            self.save_settings(self.settings.get_settings())
            return True
        return False

    def process(self, context: Dict[str, Any]) -> OCRResult:
        frame = context["frame"]
        settings = self.settings.get_settings()
        min_confidence = settings.get("min_confidence", 0.3)
        canvas_size = settings.get("canvas_size", 960)
        expected_pattern = settings.get("expected_pattern", "")
        try:
            adapter_result = self.ai_runtime.predict(
                self.active_model_id, {"frame": frame, "min_confidence": min_confidence, "canvas_size": canvas_size}
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
        
        is_ok = None
        judge_enabled = settings.get("judge_enabled", False)
        if judge_enabled and expected_pattern:
            is_ok = False
            try:
                regex = re.compile(expected_pattern)
                for txt in result.texts:
                    if regex.fullmatch(txt["text"]):
                        is_ok = True
                        break
            except re.error as e:
                logger.error(f"Invalid regex '{expected_pattern}': {e}")
                is_ok = False
        result.is_ok = is_ok

        
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
