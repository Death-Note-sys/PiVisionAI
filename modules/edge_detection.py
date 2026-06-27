import cv2
import time
import numpy as np
from core.base_module import BaseVisionModule

class EdgeDetectionModule(BaseVisionModule):
    def initialize(self, config=None):
        print("[EdgeDetection] Initializing resources...")
        self.algorithm = "Canny"
        self.threshold1 = 100
        self.threshold2 = 200
        
        # Kernel size must be odd and <= 31
        # For Sobel and Scharr, we'll map a slider of 1-3 to 1,3,5
        self.kernel_size_idx = 1 # mapping: 1->3, 2->5, 3->7
        
        self.inference_time_ms = 0.0

    def process(self, frame):
        start_time = time.time()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Map kernel index to actual size
        ksize = (self.kernel_size_idx * 2) + 1
        # Bound it to reasonable values
        ksize = min(max(ksize, 1), 7)
        if ksize == 1 and self.algorithm != "Laplacian":
            ksize = 3
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        
        if self.algorithm == "Canny":
            edges = cv2.Canny(blurred, self.threshold1, self.threshold2, apertureSize=ksize if ksize in [3,5,7] else 3)
            result_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
        elif self.algorithm == "Sobel":
            grad_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=ksize)
            grad_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=ksize)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            result_frame = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
            
        elif self.algorithm == "Scharr":
            # Scharr only works with dx=1,dy=0 or dx=0,dy=1 and ksize is inherently 3 (OpenCV uses CV_SCHARR which is -1)
            grad_x = cv2.Scharr(blurred, cv2.CV_16S, 1, 0)
            grad_y = cv2.Scharr(blurred, cv2.CV_16S, 0, 1)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            scharr = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            result_frame = cv2.cvtColor(scharr, cv2.COLOR_GRAY2BGR)
            
        elif self.algorithm == "Laplacian":
            lap = cv2.Laplacian(blurred, cv2.CV_16S, ksize=ksize)
            abs_lap = cv2.convertScaleAbs(lap)
            result_frame = cv2.cvtColor(abs_lap, cv2.COLOR_GRAY2BGR)
            
        else:
            result_frame = frame
            
        end_time = time.time()
        self.inference_time_ms = (end_time - start_time) * 1000
        
        cv2.putText(result_frame, f"Algo: {self.algorithm}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Infer: {self.inference_time_ms:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        return result_frame

    def update_settings(self, settings_dict):
        if "algorithm" in settings_dict:
            self.algorithm = settings_dict["algorithm"]
        if "threshold1" in settings_dict:
            self.threshold1 = int(settings_dict["threshold1"])
        if "threshold2" in settings_dict:
            self.threshold2 = int(settings_dict["threshold2"])
        if "kernel_size" in settings_dict:
            self.kernel_size_idx = int(settings_dict["kernel_size"])

    def cleanup(self):
        print("[EdgeDetection] Cleaning up...")

    def metadata(self):
        algo = getattr(self, 'algorithm', 'Canny')
        t1 = getattr(self, 'threshold1', 100)
        t2 = getattr(self, 'threshold2', 200)
        k = getattr(self, 'kernel_size_idx', 1)
        infer = getattr(self, 'inference_time_ms', 0.0)
        
        return {
            "id": "edge-detection",
            "name": "Edge Detection",
            "version": "1.0",
            "description": "Applies various edge detection filters.",
            "settings": {
                "algorithm": {
                    "type": "select",
                    "options": ["Canny", "Sobel", "Laplacian", "Scharr"],
                    "default": algo
                },
                "threshold1": {
                    "type": "slider",
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "default": t1
                },
                "threshold2": {
                    "type": "slider",
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "default": t2
                },
                "kernel_size": {
                    "type": "slider",
                    "min": 1,
                    "max": 3,
                    "step": 1,
                    "default": k
                }
            },
            "module_data": {
                "inference_time_ms": f"{infer:.1f}"
            }
        }
