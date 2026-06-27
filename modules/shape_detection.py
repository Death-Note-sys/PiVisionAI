import cv2
import time
import numpy as np
from core.base_module import BaseVisionModule

class ShapeDetectionModule(BaseVisionModule):
    def initialize(self, config=None):
        print("[ShapeDetection] Initializing resources...")
        self.min_contour_area = 1000
        self.inference_time_ms = 0.0
        self.shapes_detected = 0

    def process(self, frame):
        start_time = time.time()
        
        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.shapes_detected = 0
        
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
                # Differentiate between square and rectangle
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            elif vertices == 5:
                shape_name = "Pentagon"
            elif vertices == 6:
                shape_name = "Hexagon"
            else:
                # Could be circle or ellipse
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    (center, axes, orientation) = ellipse
                    major_axis, minor_axis = axes
                    if minor_axis == 0:
                        continue
                    ratio = major_axis / minor_axis
                    shape_name = "Circle" if 0.95 <= ratio <= 1.05 else "Ellipse"
                else:
                    shape_name = "Circle"
                    
            # Compute center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
                
            # Draw
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 4, (255, 0, 0), -1)
            
            # Text layout
            text_lines = [
                f"{shape_name} ({vertices}v)",
                f"A: {int(area)}",
                f"P: {int(perimeter)}"
            ]
            
            y_offset = cY - 20
            for line in text_lines:
                cv2.putText(frame, line, (cX - 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, line, (cX - 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y_offset += 20
        
        end_time = time.time()
        self.inference_time_ms = (end_time - start_time) * 1000
        
        cv2.putText(frame, f"Shapes: {self.shapes_detected}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Infer: {self.inference_time_ms:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame

    def update_settings(self, settings_dict):
        if "min_contour_area" in settings_dict:
            self.min_contour_area = int(settings_dict["min_contour_area"])

    def cleanup(self):
        print("[ShapeDetection] Cleaning up...")

    def metadata(self):
        min_area = getattr(self, 'min_contour_area', 1000)
        shapes_ct = getattr(self, 'shapes_detected', 0)
        infer = getattr(self, 'inference_time_ms', 0.0)
        
        return {
            "id": "shape-detection",
            "name": "Shape Detection",
            "version": "1.0",
            "description": "Detects geometric shapes and calculates area/perimeter.",
            "settings": {
                "min_contour_area": {
                    "type": "slider",
                    "min": 100,
                    "max": 10000,
                    "step": 100,
                    "default": min_area
                }
            },
            "module_data": {
                "shapes_detected": shapes_ct,
                "inference_time_ms": f"{infer:.1f}"
            }
        }
