import logging
import cv2
import numpy as np
from typing import Dict, Any

from app.core.event_bus import EventBus
from app.core.models.base import BoundingBox, Detection

logger = logging.getLogger(__name__)

class ShapeDetectionModule:
    """Shape Detection adapting to the new Pipeline Engine."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.id = "core-shape-detection"
        
        logger.info("[ShapeDetectionModule] Initializing resources...")
        self.min_contour_area = 1000
        self.shapes_detected = 0

    def update_settings(self, settings: Dict[str, Any]):
        if "min_contour_area" in settings:
            self.min_contour_area = int(settings["min_contour_area"])

    def preprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Pre-process"""
        frame = context["frame"]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        context["closed_edges"] = closed
        return context

    def infer(self, context: Dict[str, Any], ai_runtime: Any) -> Dict[str, Any]:
        """Stage 2: Inference"""
        closed = context["closed_edges"]
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.shapes_detected = 0
        detected_shapes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
                
            self.shapes_detected += 1
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            vertices = len(approx)
            
            # Identify shape
            shape_name = "Unknown"
            if vertices == 3:
                shape_name = "Triangle"
            elif vertices == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            elif vertices == 5:
                shape_name = "Pentagon"
            elif vertices == 6:
                shape_name = "Hexagon"
            else:
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    (center, axes, orientation) = ellipse
                    major_axis, minor_axis = axes
                    if minor_axis != 0:
                        ratio = major_axis / minor_axis
                        shape_name = "Circle" if 0.95 <= ratio <= 1.05 else "Ellipse"
                    else:
                        shape_name = "Circle"
                else:
                    shape_name = "Circle"
                    
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
            
            detected_shapes.append({
                "contour": approx,
                "center": (cX, cY),
                "name": shape_name,
                "vertices": vertices,
                "area": area,
                "perimeter": perimeter
            })
            
            # Use standard Detection format
            (x, y, w, h) = cv2.boundingRect(approx)
            box = BoundingBox(x1=x, y1=y, x2=x+w, y2=y+h)
            det = Detection(box=box, label=shape_name, confidence=1.0)
            context["detections"].append(det)
            
        context["detected_shapes"] = detected_shapes
        return context

    def postprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context

    def visualize(self, context: Dict[str, Any]) -> Any:
        """Stage 4: Return visual layout"""
        frame = context["frame"]
        shapes = context.get("detected_shapes", [])
        
        for shape in shapes:
            approx = shape["contour"]
            cX, cY = shape["center"]
            
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 4, (255, 0, 0), -1)
            
            text_lines = [
                f"{shape['name']} ({shape['vertices']}v)",
                f"A: {int(shape['area'])}",
                f"P: {int(shape['perimeter'])}"
            ]
            
            y_offset = cY - 20
            for line in text_lines:
                cv2.putText(frame, line, (cX - 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, line, (cX - 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y_offset += 20
                
        cv2.putText(frame, f"Shapes: {self.shapes_detected}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame

    def cleanup(self):
        logger.info("[ShapeDetectionModule] Cleaning up...")
