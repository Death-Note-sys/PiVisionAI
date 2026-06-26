import cv2
import numpy as np
import colorsys
import webcolors
from core.base_module import BaseVisionModule

class ColorDetector(BaseVisionModule):
    def initialize(self, config=None):
        self.color_history = [] # list of {rgb, hsv, hex, css_name}
        self.dominant_color = None
        self.dominant_hex = "#000000"
        
        # We need a reference to the latest frame to extract pixel data on click
        self.latest_frame = None

    def process(self, frame):
        self.latest_frame = frame.copy()
        
        # Calculate dominant color (downsample for speed)
        small = cv2.resize(frame, (32, 24))
        pixels = small.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # The most frequent label is dominant color
        counts = np.bincount(labels.flatten())
        dominant_bgr = centers[np.argmax(counts)]
        
        self.dominant_color = (int(dominant_bgr[2]), int(dominant_bgr[1]), int(dominant_bgr[0])) # RGB
        self.dominant_hex = "#{:02x}{:02x}{:02x}".format(*self.dominant_color)
        
        # Overlay dominant color
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (10, h-60), (60, h-10), tuple(map(int, dominant_bgr)), -1)
        cv2.rectangle(frame, (10, h-60), (60, h-10), (255, 255, 255), 2)
        cv2.putText(frame, f"Dom: {self.dominant_hex}", (70, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw reticles on latest selected regions
        if self.color_history:
            latest = self.color_history[0]
            if "pos" in latest:
                px, py = latest["pos"]
                cv2.rectangle(frame, (px-5, py-5), (px+5, py+5), (255, 255, 255), 1)
                cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)
                cv2.putText(frame, latest["hex"], (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, latest["hex"], (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
        return frame

    def handle_interaction(self, action, x, y):
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
                return
                
            avg_bgr = np.mean(region, axis=(0, 1))
            r, g, b = int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0])
            
            # HSV
            h_v, s_v, v_v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            hsv_str = f"hsv({int(h_v*360)}, {int(s_v*100)}%, {int(v_v*100)}%)"
            
            hex_str = "#{:02x}{:02x}{:02x}".format(r, g, b).upper()
            
            # Nearest CSS
            css_name = "Unknown"
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
                "pos": (px, py) # used internally for drawing reticle
            }
            
            self.color_history.insert(0, color_data)
            if len(self.color_history) > 10:
                self.color_history.pop()

        elif action == "export":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["ID", "RGB", "HSV", "HEX", "CSS Name"])
            for c in self.color_history:
                writer.writerow([c["id"], c["rgb"], c["hsv"], c["hex"], c["css"]])
                
            return {
                "type": "download",
                "filename": "color_history.csv",
                "data": output.getvalue()
            }

    def closest_colour(self, requested_colour):
        min_colours = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    def cleanup(self):
        self.color_history = []
        self.latest_frame = None

    def metadata(self):
        history = getattr(self, 'color_history', [])
        dominant = getattr(self, 'dominant_hex', "#000000")
        
        export_history = [
            {k: v for k, v in item.items() if k != "pos"} 
            for item in history
        ]
        return {
            "id": "color-detector",
            "name": "Color Detector",
            "version": "1.0",
            "description": "Click to sample colors. Calculates RGB, HSV, HEX, and nearest CSS color.",
            "settings": {},
            "module_data": {
                "type": "color_history",
                "history": export_history,
                "dominant_hex": dominant
            }
        }
