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
        # gpu=False to optimize for Raspberry Pi constraints (though it may still be slow)
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.history = []
        self.export_format = "CSV"
        
        self.last_ocr_time = 0
        self.ocr_interval = 2.0  # Run OCR once every 2 seconds to maintain frame rate
        
        self.latest_results = []

    def process(self, frame):
        current_time = time.time()
        
        # Only run deep learning OCR periodically to avoid completely blocking the video feed
        if current_time - self.last_ocr_time > self.ocr_interval:
            self.last_ocr_time = current_time
            # Run inference
            # We convert to grayscale for faster processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self.reader.readtext(gray)
            
            self.latest_results = results
            
            # Store in history
            for (bbox, text, prob) in results:
                if prob > 0.3:  # Only save reasonably confident detections
                    # check if not already recently added
                    if not any(h['text'] == text for h in self.history[-10:]):
                        self.history.append({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "text": text,
                            "confidence": round(float(prob), 3)
                        })

        # Draw the latest results continuously so they don't flicker between OCR runs
        for (bbox, text, prob) in self.latest_results:
            if prob > 0.3:
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                br = (int(br[0]), int(br[1]))
                
                # Draw bounding box
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
                
                # Draw text and confidence
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
        self.latest_results = []
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
