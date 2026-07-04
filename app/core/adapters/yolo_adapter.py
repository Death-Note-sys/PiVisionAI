import logging
import time
from typing import Dict, Any
from app.core.adapters.base_adapter import BaseAdapter
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class YoloAdapter(BaseAdapter):
    """Adapter for the Ultralytics YOLO framework."""
    
    def load_model(self, model_path: str, parameters: Dict[str, Any]) -> bool:
        try:
            logger.info(f"YoloAdapter loading model from {model_path}")
            self.model = YOLO(model_path)
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"YoloAdapter failed to load model: {e}")
            self.is_loaded = False
            return False
            
    def predict(self, input_data: Any) -> Any:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded.")
            
        # Expecting input_data to be a dict containing 'frame' and 'conf'
        frame = input_data.get("frame")
        conf = input_data.get("conf", 0.5)
        
        start_time = time.perf_counter()
        results = self.model(frame, conf=conf, verbose=False)
        end_time = time.perf_counter()
        
        # Translate ultralytics Results into our agnostic format
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i].xyxy[0].cpu().numpy().astype(int)
                c = float(boxes[i].conf[0].cpu().numpy())
                cls_id = int(boxes[i].cls[0].cpu().numpy())
                label = result.names[cls_id]
                
                detections.append({
                    "box": {"x1": int(box[0]), "y1": int(box[1]), "x2": int(box[2]), "y2": int(box[3])},
                    "confidence": c,
                    "class_id": cls_id,
                    "label": label
                })
                
        return {
            "detections": detections,
            "latency_ms": (end_time - start_time) * 1000
        }
