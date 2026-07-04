import logging
import cv2
import numpy as np
from typing import Dict, Any

from app.core.event_bus import EventBus
from app.core.models.base import Detection, BoundingBox

logger = logging.getLogger(__name__)

class MotionDetectionModule:
    """Motion Detection Module adapting to the new Pipeline Engine."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.id = "core-motion-detection"
        
        logger.info("[MotionDetection] Initializing resources...")
        self.algorithm = "MOG2"
        self.sensitivity = 500  # min contour area
        
        self.fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.fgbg_knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)
        
        self.movement_percentage = 0.0
        self.motion_area = 0
        
    def update_settings(self, settings: Dict[str, Any]):
        if "algorithm" in settings:
            self.algorithm = settings["algorithm"]
            # reset history when switching
            if self.algorithm == "MOG2":
                self.fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
            else:
                self.fgbg_knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)
                
        if "sensitivity" in settings:
            self.sensitivity = int(settings["sensitivity"])

    def preprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Pre-process"""
        # No specific preprocessing needed, subtractor takes raw BGR or Grayscale
        return context

    def infer(self, context: Dict[str, Any], ai_runtime: Any) -> Dict[str, Any]:
        """Stage 2: Inference"""
        frame = context["frame"]
        
        # Select subtractor
        subtractor = self.fgbg_mog2 if self.algorithm == "MOG2" else self.fgbg_knn
        
        # Apply subtractor
        fgmask = subtractor.apply(frame)
        
        # Threshold to remove shadows (which are usually gray, 127)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.motion_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.sensitivity:
                continue
                
            self.motion_area += area
            
            # Draw bounding box
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # Create Strict Pydantic Models
            box = BoundingBox(x1=x, y1=y, x2=x+w, y2=y+h)
            detection = Detection(box=box, label="Motion", confidence=1.0)
            
            context["detections"].append(detection)
            
        context["fgmask"] = fgmask
        
        h, w = frame.shape[:2]
        total_pixels = h * w
        self.movement_percentage = (self.motion_area / total_pixels) * 100
        
        if self.motion_area > 0:
            self.event_bus.publish("MotionDetected", {
                "area": self.motion_area,
                "percentage": self.movement_percentage
            })
            
        return context

    def postprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context

    def visualize(self, context: Dict[str, Any]) -> Any:
        """Stage 4: Drawing on the frame"""
        frame = context["frame"]
        fgmask = context.get("fgmask")
        
        for det in context["detections"]:
            if det.box:
                cv2.rectangle(frame, (det.box.x1, det.box.y1), (det.box.x2, det.box.y2), (0, 0, 255), 2)
                cv2.putText(frame, det.label, (det.box.x1, det.box.y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
        if fgmask is not None:
            # Draw the motion mask overlay slightly transparently
            # Convert mask to BGR
            colored_mask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
            colored_mask[:, :, 0] = 0 # zero out Blue
            colored_mask[:, :, 1] = 0 # zero out Green
            # Only red remains
            
            # Blend
            cv2.addWeighted(colored_mask, 0.3, frame, 0.7, 0, frame)

        cv2.putText(frame, f"Movement: {self.movement_percentage:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Area: {int(self.motion_area)} px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
        return frame

    def cleanup(self):
        """Free resources on unload."""
        logger.info("[MotionDetection] Cleaning up resources...")
        self.fgbg_mog2 = None
        self.fgbg_knn = None
