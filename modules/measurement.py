import cv2
import numpy as np
import time
import os
import json
import logging
from typing import Dict, Any
from core.base_module import BaseVisionModule

logger = logging.getLogger(__name__)

class MeasurementModule(BaseVisionModule):
    """
    Measurement module using A4 paper as a reference.
    Calculates pixels/cm and allows point-to-point and object measurement.
    """

    def initialize(self):
        logger.info("[Measurement] Initializing resources...")
        self.points = []
        self.pixels_per_cm = None
        self.a4_real_width = 21.0
        self.a4_real_height = 29.7
        self.latest_measurements = []
        self.transform_matrix = None
        
        # We need to save exports
        self.exports_dir = os.path.join(os.getcwd(), 'exports')
        os.makedirs(self.exports_dir, exist_ok=True)

    def process(self, frame: np.ndarray) -> np.ndarray:
        # 1. Detect A4 paper
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        # morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        paper_contour = None
        
        if contours:
            # Sort by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for c in contours:
                area = cv2.contourArea(c)
                if area < 5000: # Min area for a paper
                    continue
                
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                
                # If we have 4 points, we assume it's the paper
                if len(approx) == 4:
                    paper_contour = approx
                    break
                    
        # 2. Compute pixels per cm if paper found
        if paper_contour is not None:
            cv2.drawContours(frame, [paper_contour], -1, (0, 255, 0), 2)
            cv2.putText(frame, "A4 Reference", (paper_contour[0][0][0], paper_contour[0][0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Order points: top-left, top-right, bottom-right, bottom-left
            pts = paper_contour.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            (tl, tr, br, bl) = rect
            
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            # Destination points
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")
            
            # Compute perspective transform
            self.transform_matrix = cv2.getPerspectiveTransform(rect, dst)
            
            # Assume paper is portrait or landscape based on max dims
            if maxWidth < maxHeight:
                self.pixels_per_cm = maxWidth / self.a4_real_width
            else:
                self.pixels_per_cm = maxWidth / self.a4_real_height

        # 3. Handle distance measurement
        if len(self.points) == 2 and self.pixels_per_cm:
            pt1 = tuple(self.points[0])
            pt2 = tuple(self.points[1])
            
            cv2.circle(frame, pt1, 5, (0, 0, 255), -1)
            cv2.circle(frame, pt2, 5, (0, 0, 255), -1)
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
            
            pixel_dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            cm_dist = pixel_dist / self.pixels_per_cm
            
            mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.putText(frame, f"{cm_dist:.2f} cm", (mid[0], mid[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
            # Store in measurements for export
            self.latest_measurements = [{"type": "distance", "cm": cm_dist}]
            
        elif len(self.points) == 1:
            cv2.circle(frame, tuple(self.points[0]), 5, (0, 0, 255), -1)
            
        # Draw status
        status = f"Pixels/CM: {self.pixels_per_cm:.2f}" if self.pixels_per_cm else "Pixels/CM: Calibrating (Need A4)"
        cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        self.last_frame = frame.copy() # Store for export
        return frame

    def cleanup(self):
        logger.info("[Measurement] Cleaning up...")

    def handle_interaction(self, data: Dict[str, Any]):
        action = data.get("action")
        
        if action == "click":
            # Extract x, y (which come as ratios 0.0-1.0 from frontend)
            if self.last_frame is not None:
                h, w = self.last_frame.shape[:2]
                rx = data.get("x", 0)
                ry = data.get("y", 0)
                px = int(rx * w)
                py = int(ry * h)
                
                if len(self.points) >= 2:
                    self.points = [] # reset
                self.points.append((px, py))
                logger.info(f"[Measurement] Point added: {px}, {py}")
                
        elif action == "export":
            logger.info("[Measurement] Exporting data...")
            if hasattr(self, 'last_frame') and self.last_frame is not None:
                timestamp = int(time.time())
                
                # Save Image
                img_path = os.path.join(self.exports_dir, f"measurement_{timestamp}.jpg")
                cv2.imwrite(img_path, self.last_frame)
                
                # Save JSON
                json_path = os.path.join(self.exports_dir, f"measurement_{timestamp}.json")
                with open(json_path, 'w') as f:
                    json.dump({
                        "timestamp": timestamp,
                        "pixels_per_cm": self.pixels_per_cm,
                        "measurements": self.latest_measurements
                    }, f)
                logger.info(f"[Measurement] Exported to {img_path}")

    def update_settings(self, settings: Dict[str, Any]):
        pass

    def metadata(self) -> dict:
        return {
            "id": "measurement",
            "name": "Measurement",
            "description": "Measures distance between two points using an A4 paper reference.",
            "version": "1.0",
            "settings": {}
        }
