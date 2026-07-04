import logging
import cv2
import numpy as np
import colorsys
import io
import csv
from typing import Dict, Any

from app.core.event_bus import EventBus
from app.core.models.base import Detection, BoundingBox

logger = logging.getLogger(__name__)

class ColorDetectorModule:
    """Color Detector Module adapting to the new Pipeline Engine."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.id = "core-color-detector"
        
        logger.info("[ColorDetector] Initializing resources...")
        try:
            import webcolors
            self.webcolors = webcolors
        except ImportError:
            logger.error("Failed to load webcolors.")
            self.webcolors = None
            
        self.color_history = []
        self.dominant_color = None
        self.dominant_hex = "#000000"
        self.latest_frame = None
        
    def update_settings(self, settings: Dict[str, Any]):
        pass

    def preprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Pre-process"""
        return context

    def infer(self, context: Dict[str, Any], ai_runtime: Any) -> Dict[str, Any]:
        """Stage 2: Inference"""
        frame = context["frame"]
        self.latest_frame = frame.copy()
        
        # Calculate dominant color (downsample for speed)
        small = cv2.resize(frame, (32, 24))
        pixels = small.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        counts = np.bincount(labels.flatten())
        dominant_bgr = centers[np.argmax(counts)]
        
        self.dominant_color = (int(dominant_bgr[2]), int(dominant_bgr[1]), int(dominant_bgr[0])) # RGB
        self.dominant_hex = "#{:02x}{:02x}{:02x}".format(*self.dominant_color).upper()
        
        context["dominant_bgr"] = dominant_bgr
        
        return context

    def postprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context

    def visualize(self, context: Dict[str, Any]) -> Any:
        """Stage 4: Drawing on the frame"""
        frame = context["frame"]
        dominant_bgr = context.get("dominant_bgr")
        
        if dominant_bgr is not None:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (10, h-60), (60, h-10), tuple(map(int, dominant_bgr)), -1)
            cv2.rectangle(frame, (10, h-60), (60, h-10), (255, 255, 255), 2)
            cv2.putText(frame, f"Dom: {self.dominant_hex}", (70, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.color_history:
            latest = self.color_history[0]
            if "pos" in latest:
                px, py = latest["pos"]
                cv2.rectangle(frame, (px-5, py-5), (px+5, py+5), (255, 255, 255), 1)
                cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)
                cv2.putText(frame, latest["hex"], (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, latest["hex"], (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
        return frame

    def handle_interaction(self, action: str, x: float, y: float) -> Any:
        if action == "click" and self.latest_frame is not None:
            h, w = self.latest_frame.shape[:2]
            
            # Map frontend coordinates (0.0 to 1.0) to pixel coordinates
            px = int(x * w)
            py = int(y * h)
            
            # 5x5 region
            x1 = max(0, px - 2)
            y1 = max(0, py - 2)
            x2 = min(w, px + 3)
            y2 = min(h, py + 3)
            
            region = self.latest_frame[y1:y2, x1:x2]
            if region.size == 0:
                return {}
                
            avg_bgr = np.mean(region, axis=(0, 1))
            r, g, b = int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0])
            
            h_v, s_v, v_v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            hsv_str = f"hsv({int(h_v*360)}, {int(s_v*100)}%, {int(v_v*100)}%)"
            hex_str = "#{:02x}{:02x}{:02x}".format(r, g, b).upper()
            
            css_name = "Unknown"
            if self.webcolors:
                try:
                    css_name = self.closest_colour((r, g, b))
                except Exception:
                    pass
                
            color_data = {
                "id": len(self.color_history) + 1,
                "rgb": f"rgb({r}, {g}, {b})",
                "hsv": hsv_str,
                "hex": hex_str,
                "css": css_name,
                "pos": (px, py) 
            }
            
            self.color_history.insert(0, color_data)
            if len(self.color_history) > 10:
                self.color_history.pop()
                
            self.event_bus.publish("ColorSampled", {"hex": hex_str, "css": css_name})
            return {"status": "success", "sampled": hex_str}

        elif action == "export":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["ID", "RGB", "HSV", "HEX", "CSS Name"])
            for c in self.color_history:
                writer.writerow([c["id"], c["rgb"], c["hsv"], c["hex"], c["css"]])
                
            return {
                "type": "download",
                "filename": "color_history.csv",
                "mimetype": "text/csv",
                "data": output.getvalue()
            }
        return {}

    def closest_colour(self, requested_colour):
        min_colours = {}
        for key, name in self.webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = self.webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    def cleanup(self):
        logger.info("[ColorDetector] Cleaning up...")
        self.color_history = []
        self.latest_frame = None
