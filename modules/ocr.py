import cv2
import time
import json
import gc
from core.base_module import BaseVisionModule

class OCRDetector(BaseVisionModule):
    def initialize(self, config=None):
        print("[OCRDetector] Initializing resources...")
        import easyocr
        # Load EasyOCR model into memory only when module is active
        # gpu=True since we are now optimized for Desktop
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.history = []
        self.export_format = "CSV"

    def process(self, frame):
        # Run deep learning OCR on every frame (will drop framerate on weak CPUs but perfectly sticks to text)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray)
        
        for (bbox, text, prob) in results:
            if prob > 0.3:  # Only save reasonably confident detections
                # check if not already recently added
                if not any(h['text'] == text for h in self.history[-10:]):
                    self.history.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "text": text,
                        "confidence": round(float(prob), 3)
                    })
                    
                # Draw bounding box and text
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                br = (int(br[0]), int(br[1]))
                
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
                display_text = f"{text} ({prob:.2f})"
                cv2.putText(frame, display_text, (tl[0], tl[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def handle_interaction(self, action, x, y):
        if action == "export":
            if self.export_format == "CSV":
                import csv
                import io
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["Timestamp", "Text", "Confidence"])
                for item in self.history:
                    writer.writerow([item["timestamp"], item["text"], item["confidence"]])
                return {
                    "type": "download",
                    "filename": "ocr_history.csv",
                    "mimetype": "text/csv",
                    "data": output.getvalue()
                }
            elif self.export_format == "JSON":
                return {
                    "type": "download",
                    "filename": "ocr_history.json",
                    "mimetype": "application/json",
                    "data": json.dumps(self.history, indent=4)
                }
            elif self.export_format == "TXT":
                lines = ["--- OCR Detections ---"]
                for item in self.history:
                    lines.append(f"[{item['timestamp']}] {item['text']} (Conf: {item['confidence']})")
                return {
                    "type": "download",
                    "filename": "ocr_history.txt",
                    "mimetype": "text/plain",
                    "data": "\n".join(lines)
                }
        return {}

    def update_settings(self, settings_dict):
        if "export_format" in settings_dict:
            self.export_format = settings_dict["export_format"]

    def cleanup(self):
        print("[OCRDetector] Cleaning up resources. Unloading model...")
        self.reader = None
        self.history = []
        # Force garbage collection to free up memory from PyTorch models
        gc.collect()

    def metadata(self):
        current_format = getattr(self, 'export_format', 'CSV')
        history_list = getattr(self, 'history', [])
        return {
            "id": "ocr",
            "name": "OCR Text Scanner",
            "version": "1.0",
            "description": "Scans for text, numbers, and license plates using EasyOCR. Runs every 2 seconds to optimize for Raspberry Pi.",
            "settings": {
                "export_format": {
                    "type": "select",
                    "options": ["CSV", "JSON", "TXT"],
                    "default": current_format
                }
            },
            "module_data": {
                "detections_count": len(history_list)
            }
        }
