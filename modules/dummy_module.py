import cv2
import numpy as np
from core.base_module import BaseVisionModule

class DummyModule(BaseVisionModule):
    """
    A simple dummy module for testing the plugin system.
    Draws a visual overlay on the frame to prove it is active.
    """

    def initialize(self):
        print("[DummyModule] Initializing resources...")
        self.color = (0, 255, 0) # Green
        self.thickness = 2

    def process(self, frame: np.ndarray) -> np.ndarray:
        # Draw a rectangle and text
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (50, 50), (w - 50, h - 50), self.color, self.thickness)
        
        text = "Dummy Module Active"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (60, 90), font, 1.0, self.color, 2, cv2.LINE_AA)
        
        return frame

    def cleanup(self):
        print("[DummyModule] Cleaning up resources...")

    def metadata(self) -> dict:
        return {
            "id": "dummy-module",
            "name": "Dummy Module",
            "description": "A testing module that draws a green bounding box overlay.",
            "version": "1.0"
        }
