import logging
import cv2
import time
import gc
from typing import Dict, Any
import easyocr

from app.core.event_bus import EventBus
from app.core.models.base import Detection, BoundingBox

logger = logging.getLogger(__name__)

class OCRScannerModule:
    """OCR Text Scanner adapting to the new Pipeline Engine."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.id = "core-ocr-scanner"
        
        logger.info("[OCRScannerModule] Initializing resources...")
        try:
            self.reader = easyocr.Reader(['en'], gpu=True)
            logger.info("EasyOCR model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR model: {e}")
            self.reader = None

        self.history = []
        self.export_format = "CSV"
        self.confidence_threshold = 0.3
        
    def update_settings(self, settings: Dict[str, Any]):
        if "export_format" in settings:
            self.export_format = settings["export_format"]
            logger.info(f"OCR export format updated to {self.export_format}")

    def preprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Pre-process (Grayscale conversion)"""
        frame = context["frame"]
        # EasyOCR prefers grayscale for speed
        context["gray"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return context

    def infer(self, context: Dict[str, Any], ai_runtime: Any) -> Dict[str, Any]:
        """Stage 2: Inference"""
        if not self.reader:
            return context
            
        gray = context["gray"]
        
        # Process every frame, but we could add frame skipping logic here if needed for low-end hardware
        results = self.reader.readtext(gray)
        
        for (bbox, text, prob) in results:
            if prob > self.confidence_threshold:
                # BBox format from EasyOCR: [(tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)]
                tl = bbox[0]
                br = bbox[2]
                
                # Create Strict Pydantic Models
                box = BoundingBox(x1=int(tl[0]), y1=int(tl[1]), x2=int(br[0]), y2=int(br[2]))
                detection = Detection(box=box, label=text, confidence=float(prob))
                
                context["detections"].append(detection)
                
                # De-duplicate history
                if not any(h['text'] == text for h in self.history[-10:]):
                    self.history.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "text": text,
                        "confidence": round(float(prob), 3)
                    })
                    # Publish new text found
                    self.event_bus.publish("OCRTextRead", {"text": text, "confidence": float(prob)})
                
        return context

    def postprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context

    def visualize(self, context: Dict[str, Any]) -> Any:
        """Stage 4: Drawing on the frame"""
        frame = context["frame"]
        
        for det in context["detections"]:
            if det.box:
                cv2.rectangle(frame, (det.box.x1, det.box.y1), (det.box.x2, det.box.y2), (0, 255, 0), 2)
                display_text = f"{det.label} ({det.confidence:.2f})"
                cv2.putText(frame, display_text, (det.box.x1, det.box.y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        return frame

    def cleanup(self):
        """Free resources on unload."""
        logger.info("[OCRScannerModule] Cleaning up resources. Unloading model...")
        self.reader = None
        self.history = []
        gc.collect()
