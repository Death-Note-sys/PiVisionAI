import cv2
import time
import json
from core.base_module import BaseVisionModule

class BarcodeScannerModule(BaseVisionModule):
    def initialize(self, config=None):
        print("[BarcodeScanner] Initializing resources...")
        from pyzbar import pyzbar
        self.pyzbar = pyzbar
        
        self.history = []
        self.export_format = "CSV"
        
        # Setting: Timeout in seconds to prevent duplicate scans
        self.duplicate_timeout = 2.0 
        
        self.inference_time_ms = 0.0

    def process(self, frame):
        start_time = time.time()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        barcodes = self.pyzbar.decode(gray)
        
        current_time = time.time()
        
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            barcode_data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            
            # Check for duplicates within timeout
            is_duplicate = False
            for record in reversed(self.history):
                if record["data"] == barcode_data and record["type"] == barcode_type:
                    if current_time - record["_raw_time"] < self.duplicate_timeout:
                        is_duplicate = True
                    break
                    
            if not is_duplicate:
                timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
                self.history.append({
                    "timestamp": timestamp_str,
                    "data": barcode_data,
                    "type": barcode_type,
                    "_raw_time": current_time
                })
                
            display_text = f"{barcode_data} ({barcode_type})"
            cv2.putText(frame, display_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        end_time = time.time()
        self.inference_time_ms = (end_time - start_time) * 1000
        
        cv2.putText(frame, f"Infer: {self.inference_time_ms:.1f}ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
        return frame

    def handle_interaction(self, action, x, y):
        if action == "export":
            if self.export_format == "CSV":
                import csv
                import io
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["Timestamp", "Type", "Data"])
                for item in self.history:
                    writer.writerow([item["timestamp"], item["type"], item["data"]])
                return {
                    "type": "download",
                    "filename": "barcode_history.csv",
                    "mimetype": "text/csv",
                    "data": output.getvalue()
                }
            elif self.export_format == "JSON":
                clean_history = [ {k:v for k,v in h.items() if k != "_raw_time"} for h in self.history ]
                return {
                    "type": "download",
                    "filename": "barcode_history.json",
                    "mimetype": "application/json",
                    "data": json.dumps(clean_history, indent=4)
                }
            elif self.export_format == "TXT":
                lines = ["--- Barcode Scans ---"]
                for item in self.history:
                    lines.append(f"[{item['timestamp']}] {item['type']}: {item['data']}")
                return {
                    "type": "download",
                    "filename": "barcode_history.txt",
                    "mimetype": "text/plain",
                    "data": "\n".join(lines)
                }
        return {}

    def update_settings(self, settings_dict):
        if "export_format" in settings_dict:
            self.export_format = settings_dict["export_format"]
        if "duplicate_timeout" in settings_dict:
            self.duplicate_timeout = float(settings_dict["duplicate_timeout"])

    def cleanup(self):
        print("[BarcodeScanner] Cleaning up...")
        self.history = []

    def metadata(self):
        current_format = getattr(self, 'export_format', 'CSV')
        timeout_setting = getattr(self, 'duplicate_timeout', 2.0)
        history_list = getattr(self, 'history', [])
        infer = getattr(self, 'inference_time_ms', 0.0)
        
        return {
            "id": "barcode-scanner",
            "name": "Barcode & QR Scanner",
            "version": "1.0",
            "description": "Scans multiple 1D and 2D barcode formats using pyzbar.",
            "settings": {
                "export_format": {
                    "type": "select",
                    "options": ["CSV", "JSON", "TXT"],
                    "default": current_format
                },
                "duplicate_timeout": {
                    "type": "slider",
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "default": timeout_setting
                }
            },
            "module_data": {
                "total_scans": len(history_list),
                "inference_time_ms": f"{infer:.1f}"
            }
        }
