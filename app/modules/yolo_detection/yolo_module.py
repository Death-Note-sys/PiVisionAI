import logging
import cv2
from typing import Dict, Any
from ultralytics import YOLO

from app.core.event_bus import EventBus
from app.core.models.base import Detection, BoundingBox

logger = logging.getLogger(__name__)

class YOLODetectionModule:
    """YOLO Object Detection adapting to the new Pipeline Engine."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.id = "core-yolo-detection"
        
        # Load the default model directly for now.
        # Ideally, AIRuntimeManager would hand us an ONNX session, but for PyTorch YOLO 
        # ultralytics has its own loading mechanism. We can still register it with the framework.
        self.model_name = "yolo11n.pt"
        try:
            self.model = YOLO(self.model_name)
            logger.info(f"YOLO module loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

        self.confidence_threshold = 0.5
        
    def update_settings(self, settings: Dict[str, Any]):
        """Update runtime settings like confidence."""
        if "confidence" in settings:
            self.confidence_threshold = float(settings["confidence"])
            logger.info(f"YOLO confidence updated to {self.confidence_threshold}")

    def preprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Pre-process"""
        # YOLOv8/11 handles its own resizing, but we could do OpenCV CUDA prep here
        return context

    def infer(self, context: Dict[str, Any], ai_runtime: Any) -> Dict[str, Any]:
        """Stage 2: Inference"""
        if not self.model:
            return context
            
        frame = context["frame"]
        
        # We would normally ask ai_runtime for the session, but we use ultralytics directly here
        # (Though we should respect the backend priority in future integration)
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes[i].xyxy[0].cpu().numpy().astype(int)
                conf = float(boxes[i].conf[0].cpu().numpy())
                cls_id = int(boxes[i].cls[0].cpu().numpy())
                label = result.names[cls_id]
                
                # Convert to strictly typed Pydantic models
                bbox = BoundingBox(x1=int(box[0]), y1=int(box[1]), x2=int(box[2]), y2=int(box[3]))
                detection = Detection(box=bbox, label=label, confidence=conf)
                
                context["detections"].append(detection)
                
                # Publish individual event
                self.event_bus.publish("ObjectDetected", detection)
                
        return context

    def postprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Post-processing (NMS etc if not handled by model)"""
        return context

    def visualize(self, context: Dict[str, Any]) -> Any:
        """Stage 4: Drawing on the frame"""
        frame = context["frame"]
        
        for det in context["detections"]:
            if det.box:
                cv2.rectangle(frame, (det.box.x1, det.box.y1), (det.box.x2, det.box.y2), (0, 255, 0), 2)
                text = f"{det.label} {det.confidence:.2f}"
                cv2.putText(frame, text, (det.box.x1, det.box.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        return frame

    def cleanup(self):
        """Free resources on unload."""
        self.model = None
        logger.info("YOLO Module cleaned up.")
