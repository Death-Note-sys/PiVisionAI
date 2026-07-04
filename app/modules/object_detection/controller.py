import logging
import time
from typing import Dict, Any
from app.core.contracts import IModule
from app.core.event_bus import EventBus
from app.core.ai_runtime import AIRuntimeManager
from app.core.models.results import DetectionResult
from .settings import ObjectDetectionSettings

logger = logging.getLogger(__name__)

class ObjectDetectionController(IModule):
    """Business logic for Object Detection. Purely logic, no rendering."""
    
    def __init__(self, event_bus: EventBus, ai_runtime: AIRuntimeManager, settings: ObjectDetectionSettings):
        self.event_bus = event_bus
        self.ai_runtime = ai_runtime
        self.settings = settings
        self.active_model_id = "yolo11n" # Default, normally retrieved via Settings
        self.last_result = DetectionResult()
        
    def initialize(self) -> bool:
        logger.info("ObjectDetectionController initialized.")
        return True
        
    def configure(self, settings: Dict[str, Any]) -> bool:
        self.settings.update(settings)
        if "model_id" in settings:
            self.active_model_id = settings["model_id"]
        return True
        
    def process(self, context: Dict[str, Any]) -> DetectionResult:
        frame = context["frame"]
        
        # Get settings
        conf = self.settings.get_settings().get("confidence", 0.5)
        
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
            timestamp=time.time()
        )
        
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
