import cv2
import time
import numpy as np
from core.base_module import BaseVisionModule

class MotionDetectionModule(BaseVisionModule):
    def initialize(self, config=None):
        print("[MotionDetection] Initializing resources...")
        self.algorithm = "MOG2"
        self.sensitivity = 500  # min contour area
        
        self.fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.fgbg_knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)
        
        self.inference_time_ms = 0.0
        self.movement_percentage = 0.0
        self.motion_area = 0

    def process(self, frame):
        start_time = time.time()
        
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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"Motion", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw the motion mask overlay slightly transparently
        # Convert mask to BGR
        colored_mask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        colored_mask[:, :, 0] = 0 # zero out Blue
        colored_mask[:, :, 1] = 0 # zero out Green
        # Only red remains
        
        # Blend
        cv2.addWeighted(colored_mask, 0.3, frame, 0.7, 0, frame)

        h, w = frame.shape[:2]
        total_pixels = h * w
        self.movement_percentage = (self.motion_area / total_pixels) * 100
        
        end_time = time.time()
        self.inference_time_ms = (end_time - start_time) * 1000
        
        cv2.putText(frame, f"Movement: {self.movement_percentage:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Area: {int(self.motion_area)} px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Infer: {self.inference_time_ms:.1f}ms", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def update_settings(self, settings_dict):
        if "algorithm" in settings_dict:
            self.algorithm = settings_dict["algorithm"]
            # reset history when switching
            if self.algorithm == "MOG2":
                self.fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
            else:
                self.fgbg_knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)
                
        if "sensitivity" in settings_dict:
            self.sensitivity = int(settings_dict["sensitivity"])

    def cleanup(self):
        print("[MotionDetection] Cleaning up...")

    def metadata(self):
        algo = getattr(self, 'algorithm', 'MOG2')
        sens = getattr(self, 'sensitivity', 500)
        perc = getattr(self, 'movement_percentage', 0.0)
        infer = getattr(self, 'inference_time_ms', 0.0)
        
        return {
            "id": "motion-detection",
            "name": "Motion Detection",
            "version": "1.0",
            "description": "Background subtraction based motion tracking.",
            "settings": {
                "algorithm": {
                    "type": "select",
                    "options": ["MOG2", "KNN"],
                    "default": algo
                },
                "sensitivity": {
                    "type": "slider",
                    "min": 100,
                    "max": 5000,
                    "step": 100,
                    "default": sens
                }
            },
            "module_data": {
                "movement_percent": f"{perc:.1f}%",
                "inference_time_ms": f"{infer:.1f}"
            }
        }
