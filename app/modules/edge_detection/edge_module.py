import logging
import cv2
from typing import Dict, Any

from app.core.event_bus import EventBus

logger = logging.getLogger(__name__)

class EdgeDetectionModule:
    """Edge Detection adapting to the new Pipeline Engine."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.id = "core-edge-detection"
        
        logger.info("[EdgeDetectionModule] Initializing resources...")
        self.algorithm = "Canny"
        self.threshold1 = 100
        self.threshold2 = 200
        self.kernel_size_idx = 1 # mapping: 1->3, 2->5, 3->7

    def update_settings(self, settings: Dict[str, Any]):
        if "algorithm" in settings:
            self.algorithm = settings["algorithm"]
        if "threshold1" in settings:
            self.threshold1 = int(settings["threshold1"])
        if "threshold2" in settings:
            self.threshold2 = int(settings["threshold2"])
        if "kernel_size" in settings:
            self.kernel_size_idx = int(settings["kernel_size"])
        logger.info(f"Edge Detection settings updated: {self.algorithm}")

    def preprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Pre-process (Grayscale and Blur)"""
        frame = context["frame"]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Map kernel index to actual size
        ksize = (self.kernel_size_idx * 2) + 1
        ksize = min(max(ksize, 1), 7)
        if ksize == 1 and self.algorithm != "Laplacian":
            ksize = 3
            
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        context["gray"] = gray
        context["blurred"] = blurred
        context["ksize"] = ksize
        return context

    def infer(self, context: Dict[str, Any], ai_runtime: Any) -> Dict[str, Any]:
        """Stage 2: Inference (Edge Detection Algorithms)"""
        blurred = context["blurred"]
        ksize = context["ksize"]
        
        if self.algorithm == "Canny":
            edges = cv2.Canny(blurred, self.threshold1, self.threshold2, apertureSize=ksize if ksize in [3,5,7] else 3)
            context["edges_result"] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
        elif self.algorithm == "Sobel":
            grad_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=ksize)
            grad_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=ksize)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            context["edges_result"] = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
            
        elif self.algorithm == "Scharr":
            grad_x = cv2.Scharr(blurred, cv2.CV_16S, 1, 0)
            grad_y = cv2.Scharr(blurred, cv2.CV_16S, 0, 1)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            scharr = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            context["edges_result"] = cv2.cvtColor(scharr, cv2.COLOR_GRAY2BGR)
            
        elif self.algorithm == "Laplacian":
            lap = cv2.Laplacian(blurred, cv2.CV_16S, ksize=ksize)
            abs_lap = cv2.convertScaleAbs(lap)
            context["edges_result"] = cv2.cvtColor(abs_lap, cv2.COLOR_GRAY2BGR)
            
        else:
            context["edges_result"] = context["frame"]
            
        return context

    def postprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context

    def visualize(self, context: Dict[str, Any]) -> Any:
        """Stage 4: Return the edge detection frame"""
        frame = context.get("edges_result", context["frame"])
        cv2.putText(frame, f"Algo: {self.algorithm}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def cleanup(self):
        logger.info("[EdgeDetectionModule] Cleaning up...")
