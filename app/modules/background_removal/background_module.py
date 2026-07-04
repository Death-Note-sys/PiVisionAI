import logging
import cv2
import os
import urllib.request
import numpy as np
from typing import Dict, Any

from app.core.event_bus import EventBus

logger = logging.getLogger(__name__)

class BackgroundRemovalModule:
    """Background Removal adapting to the new Pipeline Engine."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.id = "core-background-removal"
        
        logger.info("[BackgroundRemovalModule] Initializing resources...")
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        self.mp = mp
        os.makedirs("weights", exist_ok=True)
        seg_path = "weights/selfie_segmenter.tflite"
        
        if not os.path.exists(seg_path):
            logger.info("Downloading Image Segmenter model...")
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite", seg_path)

        base_options = python.BaseOptions(model_asset_path=seg_path)
        options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
        self.segmenter = vision.ImageSegmenter.create_from_options(options)
        
        self.mode = "Blur"

    def update_settings(self, settings: Dict[str, Any]):
        if "mode" in settings:
            self.mode = settings["mode"]
        logger.info(f"Background Removal mode updated: {self.mode}")

    def preprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Pre-process"""
        frame = context["frame"]
        context["mp_image"] = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return context

    def infer(self, context: Dict[str, Any], ai_runtime: Any) -> Dict[str, Any]:
        """Stage 2: Inference"""
        segmentation_result = self.segmenter.segment(context["mp_image"])
        mask = segmentation_result.category_mask.numpy_view()
        # Squeeze the mask just in case it has an extra dimension like (H, W, 1)
        mask = np.squeeze(mask)
        # The model returns 0 for person and >0 for background
        context["condition"] = np.stack((mask,) * 3, axis=-1) == 0
        return context

    def postprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context

    def visualize(self, context: Dict[str, Any]) -> Any:
        """Stage 4: Return the segmented frame"""
        frame = context["frame"]
        condition = context.get("condition")
        
        if condition is None:
            return frame
            
        if self.mode == "Solid Color":
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            bg_image[:] = (0, 255, 0)
            output_image = np.where(condition, frame, bg_image)
        elif self.mode == "Blur":
            bg_image = cv2.GaussianBlur(frame, (55, 55), 0)
            output_image = np.where(condition, frame, bg_image)
        elif self.mode == "Transparent":
            h, w = frame.shape[:2]
            checkerboard = np.zeros((h, w, 3), dtype=np.uint8)
            checkerboard[:] = (200, 200, 200)
            square_size = 20
            for y in range(0, h, square_size):
                for x in range(0, w, square_size):
                    if (x // square_size + y // square_size) % 2 == 0:
                        cv2.rectangle(checkerboard, (x, y), (x+square_size, y+square_size), (150, 150, 150), -1)
            output_image = np.where(condition, frame, checkerboard)
        elif self.mode == "Custom Image":
            h, w = frame.shape[:2]
            bg_image = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(h):
                bg_image[i, :] = (255, int(255 * (i/h)), 0)
            output_image = np.where(condition, frame, bg_image)
        else:
            output_image = frame
            
        cv2.putText(output_image, f"Mode: {self.mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return output_image

    def cleanup(self):
        logger.info("[BackgroundRemovalModule] Cleaning up...")
        if hasattr(self, 'segmenter'):
            self.segmenter.close()
