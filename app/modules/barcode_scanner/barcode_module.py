import logging
import cv2
import time
from typing import Dict, Any

from app.core.event_bus import EventBus
from app.core.models.base import Detection, BoundingBox

logger = logging.getLogger(__name__)

class BarcodeScannerModule:
    """Barcode and QR Code Scanner adapting to the new Pipeline Engine."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.id = "core-barcode-scanner"
        
        logger.info("[BarcodeScannerModule] Initializing resources...")
        try:
            from pyzbar import pyzbar
            self.pyzbar = pyzbar
            logger.info("pyzbar loaded successfully.")
        except ImportError:
            logger.error("Failed to load pyzbar. Is it installed?")
            self.pyzbar = None

        self.history = []
        self.export_format = "CSV"
        self.duplicate_timeout = 2.0
        
    def update_settings(self, settings: Dict[str, Any]):
        if "export_format" in settings:
            self.export_format = settings["export_format"]
        if "duplicate_timeout" in settings:
            self.duplicate_timeout = float(settings["duplicate_timeout"])
            logger.info(f"Barcode duplicate timeout updated to {self.duplicate_timeout}s")

    def preprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Pre-process (Grayscale conversion)"""
        frame = context["frame"]
        context["gray"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return context

    def infer(self, context: Dict[str, Any], ai_runtime: Any) -> Dict[str, Any]:
        """Stage 2: Inference"""
        if not self.pyzbar:
            return context
            
        gray = context["gray"]
        current_time = time.time()
        
        barcodes = self.pyzbar.decode(gray)
        
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            barcode_data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            
            # Create Strict Pydantic Models
            box = BoundingBox(x1=x, y1=y, x2=x+w, y2=y+h)
            detection = Detection(box=box, label=f"{barcode_data} ({barcode_type})", confidence=1.0)
            
            context["detections"].append(detection)
            
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
                # Publish new barcode found
                self.event_bus.publish("BarcodeScanned", {"data": barcode_data, "type": barcode_type})
                
        return context

    def postprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context

    def visualize(self, context: Dict[str, Any]) -> Any:
        """Stage 4: Drawing on the frame"""
        frame = context["frame"]
        
        for det in context["detections"]:
            if det.box:
                cv2.rectangle(frame, (det.box.x1, det.box.y1), (det.box.x2, det.box.y2), (0, 255, 255), 2)
                cv2.putText(frame, det.label, (det.box.x1, det.box.y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
        return frame

    def cleanup(self):
        """Free resources on unload."""
        logger.info("[BarcodeScannerModule] Cleaning up resources...")
        self.history = []
