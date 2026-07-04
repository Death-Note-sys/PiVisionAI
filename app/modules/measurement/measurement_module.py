import logging
import cv2
import numpy as np
import os
import json
import time
from typing import Dict, Any

from app.core.event_bus import EventBus

logger = logging.getLogger(__name__)

class MeasurementModule:
    """Measurement module adapting to the new Pipeline Engine."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.id = "core-measurement"
        
        logger.info("[MeasurementModule] Initializing resources...")
        self.points = []
        self.pixels_per_cm = None
        self.a4_real_width = 21.0
        self.a4_real_height = 29.7
        self.latest_measurements = []
        
        self.exports_dir = os.path.join(os.getcwd(), 'exports')
        os.makedirs(self.exports_dir, exist_ok=True)
        
        self.event_bus.subscribe("UserInteraction", self._on_interaction)

    def update_settings(self, settings: Dict[str, Any]):
        pass

    def _on_interaction(self, event_type: str, data: Dict[str, Any]):
        if data.get("module_id") != self.id and data.get("module_id") != "measurement":
            return
            
        action = data.get("action")
        if action == "click":
            # Points will be collected here, but we need resolution. We just store ratios and let visualize handle it.
            rx = data.get("x", 0.0)
            ry = data.get("y", 0.0)
            if len(self.points) >= 2:
                self.points = []
            self.points.append({"rx": rx, "ry": ry})
            logger.info(f"[Measurement] Point added: ratio ({rx}, {ry})")
            
        elif action == "export":
            logger.info("[Measurement] Export requested.")
            self.event_bus.publish("MeasurementExport", {"measurements": self.latest_measurements})

    def preprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context

    def infer(self, context: Dict[str, Any], ai_runtime: Any) -> Dict[str, Any]:
        """Detect A4 paper reference."""
        frame = context["frame"]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        paper_contour = None
        
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for c in contours:
                if cv2.contourArea(c) < 5000:
                    continue
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    paper_contour = approx
                    break
                    
        context["paper_contour"] = paper_contour
        return context

    def postprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context

    def visualize(self, context: Dict[str, Any]) -> Any:
        frame = context["frame"]
        paper_contour = context.get("paper_contour")
        
        if paper_contour is not None:
            cv2.drawContours(frame, [paper_contour], -1, (0, 255, 0), 2)
            cv2.putText(frame, "A4 Reference", (paper_contour[0][0][0], paper_contour[0][0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
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
            
            if maxWidth < maxHeight:
                self.pixels_per_cm = maxWidth / self.a4_real_width
            else:
                self.pixels_per_cm = maxWidth / self.a4_real_height

        h, w = frame.shape[:2]
        pixel_points = []
        for p in self.points:
            px, py = int(p["rx"] * w), int(p["ry"] * h)
            pixel_points.append((px, py))
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

        if len(pixel_points) == 2 and self.pixels_per_cm:
            pt1, pt2 = pixel_points[0], pixel_points[1]
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
            pixel_dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            cm_dist = pixel_dist / self.pixels_per_cm
            
            mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.putText(frame, f"{cm_dist:.2f} cm", (mid[0], mid[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            self.latest_measurements = [{"type": "distance", "cm": cm_dist}]

        status = f"Pixels/CM: {self.pixels_per_cm:.2f}" if self.pixels_per_cm else "Pixels/CM: Calibrating (Need A4)"
        cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame

    def cleanup(self):
        logger.info("[MeasurementModule] Cleaning up...")
