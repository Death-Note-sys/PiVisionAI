import logging
import time
from typing import Dict, Any
from app.core.base_module import BaseModuleController
from app.core.event_bus import EventBus
from app.core.ai_runtime import AIRuntimeManager
from app.core.models.results import DetectionResult
from .settings import ObjectDetectionSettings

logger = logging.getLogger(__name__)

class ObjectDetectionController(BaseModuleController):
    """Business logic for Object Detection. Purely logic, no rendering."""
    
    def __init__(self, event_bus: EventBus, ai_runtime: AIRuntimeManager, settings: ObjectDetectionSettings):
        super().__init__("core-object-detection", event_bus)
        self.ai_runtime = ai_runtime
        self.settings = settings
        
        # Load settings
        saved_settings = self.load_settings()
        if saved_settings:
            self.settings.update(saved_settings)
            
        self.active_model_id = self.settings.get_settings().get("model_id", "yolo11n")
        self.last_result = DetectionResult()
        
    def initialize(self) -> bool:
        logger.info("ObjectDetectionController initialized.")
        return True
        
    def configure(self, settings: Dict[str, Any]) -> bool:
        if self.settings.update(settings):
            if "model_id" in settings:
                self.active_model_id = settings["model_id"]
            self.save_settings(self.settings.get_settings())
            return True
        return False
        
    def process(self, context: Dict[str, Any]) -> DetectionResult:
        frame = context["frame"]
        
        # Get settings
        conf = self.settings.get_settings().get("confidence", 0.5)
        count_min = self.settings.get_settings().get("count_min", 0)
        count_max = self.settings.get_settings().get("count_max", 0)
        
        # Request inference from AI Runtime Adapter
        try:
            adapter_result = self.ai_runtime.predict(self.active_model_id, {"frame": frame, "conf": conf})
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return DetectionResult()
            
        if not adapter_result:
            return DetectionResult()
            
        # Parse into standard DetectionResult
        result = DetectionResult(
            detections=adapter_result.get("detections", []),
            latency_ms=adapter_result.get("latency_ms", 0.0),
            objects_count=len(adapter_result.get("detections", [])),
            model_name=self.active_model_id,
        )
        
        is_ok = None
        if count_min > 0 or count_max > 0:
            if count_max == 0: count_max = 999999
            is_ok = (count_min <= result.objects_count <= count_max)
            result.is_ok = is_ok
        
        if result.objects_count > 0:
            self.event_bus.publish("ObjectsDetected", {"count": result.objects_count, "model": result.model_name})
            
        self.last_result = result
        return result
        
    def render(self, result: DetectionResult) -> Any:
        """Returns the result to be passed to the RendererManager."""
        # The result itself acts as rendering instructions.
        return result
        
    def cleanup(self) -> None:
        self.ai_runtime.unload_model(self.active_model_id)
        logger.info("ObjectDetectionController cleaned up.")
        
    def health_check(self) -> bool:
        return True
