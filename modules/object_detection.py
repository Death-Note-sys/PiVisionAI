import cv2
import numpy as np
import time
import os
import logging
from typing import Dict, Any
from core.base_module import BaseVisionModule

logger = logging.getLogger(__name__)

class ObjectDetectionModule(BaseVisionModule):
    """
    YOLO-based Object Detection module optimized for Raspberry Pi.
    """

    def initialize(self):
        logger.info("[ObjectDetection] Initializing resources...")
        self.conf_threshold = 0.5
        self.model = None
        
        # Load weights only upon activation
        try:
            from ultralytics import YOLO
            weights_dir = os.path.join(os.getcwd(), 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            
            # Prefer YOLO11n, fallback to v8n. Ultralytics auto-downloads to current dir if not absolute.
            yolo11_path = os.path.join(weights_dir, 'yolo11n.pt')
            yolov8_path = os.path.join(weights_dir, 'yolov8n.pt')
            
            if os.path.exists(yolo11_path):
                self.model = YOLO(yolo11_path)
            elif os.path.exists(yolov8_path):
                self.model = YOLO(yolov8_path)
            else:
                logger.info("Weights not found locally. Downloading YOLO11n...")
                self.model = YOLO('yolo11n.pt') # downloads to cwd
                # Move to weights folder ideally, but for now ultralytics handles it
                
            logger.info("[ObjectDetection] YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"[ObjectDetection] Failed to load YOLO: {e}")

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.model is None:
            return frame

        start_time = time.time()

        # Run inference (optimized for Pi: half precision not supported on generic CPU, use lower imgsz)
        results = self.model(frame, conf=self.conf_threshold, verbose=False, imgsz=320)
        
        inf_time_ms = (time.time() - start_time) * 1000
        
        # Draw results manually to meet specific requirements (FPS, custom colors, etc.)
        result = results[0]
        boxes = result.boxes
        
        total_objects = len(boxes)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            
            # Draw bounding box
            color = (255, 85, 85) # Vibrant blue/purple
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label and conf
            text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw metrics overlay
        fps = 1000.0 / inf_time_ms if inf_time_ms > 0 else 0
        overlay_text = [
            f"Objects: {total_objects}",
            f"Inf Time: {inf_time_ms:.1f} ms",
            f"Module FPS: {fps:.1f}"
        ]
        
        y_offset = 30
        for text in overlay_text:
            cv2.putText(frame, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 30

        return frame

    def cleanup(self):
        logger.info("[ObjectDetection] Cleaning up. Unloading model from memory...")
        self.model = None

    def update_settings(self, settings: Dict[str, Any]):
        if 'confidence' in settings:
            try:
                self.conf_threshold = float(settings['confidence'])
                logger.info(f"[ObjectDetection] Confidence updated to {self.conf_threshold}")
            except ValueError:
                pass

    def metadata(self) -> dict:
        return {
            "id": "object-detection",
            "name": "Object Detection",
            "description": "YOLO-based real-time object detection module.",
            "version": "1.0",
            "settings": {
                "confidence": {
                    "type": "slider",
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "default": self.conf_threshold if hasattr(self, 'conf_threshold') else 0.5
                }
            }
        }
