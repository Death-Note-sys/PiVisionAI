import cv2
import numpy as np
from typing import Dict, Any
from app.core.renderers.base_renderer import BaseRenderer

class DetectionRenderer(BaseRenderer):
    """Draws bounding boxes and labels for object detection."""
    
    def render(self, frame: np.ndarray, result: Any, metadata: Dict[str, Any]) -> np.ndarray:
        if not hasattr(result, "detections"):
            return frame
            
        out_frame = frame.copy()
        
        # Settings can be passed via metadata
        settings = metadata.get("settings", {})
        show_labels = settings.get("show_labels", True)
        show_conf = settings.get("show_conf", True)
        thickness = int(settings.get("thickness", 2))
        
        for det in result.detections:
            box = det.get("box", {})
            x1, y1 = box.get("x1", 0), box.get("y1", 0)
            x2, y2 = box.get("x2", 0), box.get("y2", 0)
            label = det.get("label", "Unknown")
            conf = det.get("confidence", 0.0)
            
            # Draw box
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            
            # Draw label
            if show_labels:
                text = label
                if show_conf:
                    text += f" {conf:.2f}"
                cv2.putText(out_frame, text, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), max(1, thickness - 1))
                
        # Draw HUD (Latency, Model, FPS)
        hud_y = 30
        if getattr(result, "latency_ms", 0) > 0:
            cv2.putText(out_frame, f"Infer: {result.latency_ms:.1f}ms", (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            hud_y += 30
            
        if getattr(result, "model_name", None):
            cv2.putText(out_frame, f"Model: {result.model_name}", (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            hud_y += 30
            
        return out_frame
