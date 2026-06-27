import cv2
import time
import os
import urllib.request
from core.base_module import BaseVisionModule

class FaceDetectionModule(BaseVisionModule):
    def initialize(self, config=None):
        print("[FaceDetection] Initializing resources...")
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        self.mp = mp
        
        # Download models for Tasks API
        os.makedirs("weights", exist_ok=True)
        fd_path = "weights/blaze_face_short_range.tflite"
        fm_path = "weights/face_landmarker.task"
        
        if not os.path.exists(fd_path):
            print("Downloading Face Detector model...")
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite", fd_path)
            
        if not os.path.exists(fm_path):
            print("Downloading Face Landmarker model...")
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", fm_path)

        base_options_fd = python.BaseOptions(model_asset_path=fd_path)
        options_fd = vision.FaceDetectorOptions(base_options=base_options_fd)
        self.face_detector = vision.FaceDetector.create_from_options(options_fd)
        
        base_options_fm = python.BaseOptions(model_asset_path=fm_path)
        options_fm = vision.FaceLandmarkerOptions(
            base_options=base_options_fm,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=5)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_fm)

        self.enable_face_mesh = 0
        self.inference_time_ms = 0.0
        self.face_count = 0

    def process(self, frame):
        start_time = time.time()
        
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.face_count = 0
        h, w, _ = frame.shape
        
        if self.enable_face_mesh == 1:
            detection_result = self.face_landmarker.detect(mp_image)
            self.face_count = len(detection_result.face_landmarks)
            for face_landmarks in detection_result.face_landmarks:
                for lm in face_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        else:
            detection_result = self.face_detector.detect(mp_image)
            self.face_count = len(detection_result.detections)
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 255), 2)
                
                if detection.categories:
                    conf = detection.categories[0].score
                    cv2.putText(frame, f"Conf: {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
                cx, cy = x + bw // 2, y + bh // 2
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(frame, "Center", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        end_time = time.time()
        self.inference_time_ms = (end_time - start_time) * 1000

        cv2.putText(frame, f"Faces: {self.face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Infer: {self.inference_time_ms:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame

    def update_settings(self, settings_dict):
        if "enable_face_mesh" in settings_dict:
            self.enable_face_mesh = int(settings_dict["enable_face_mesh"])

    def cleanup(self):
        print("[FaceDetection] Cleaning up...")

    def metadata(self):
        mesh_setting = getattr(self, 'enable_face_mesh', 0)
        fc = getattr(self, 'face_count', 0)
        infer = getattr(self, 'inference_time_ms', 0.0)
        return {
            "id": "face-detection",
            "name": "Face Detection",
            "version": "1.0",
            "description": "MediaPipe Face Detection & Mesh (Tasks API).",
            "settings": {
                "enable_face_mesh": {
                    "type": "slider",
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "default": mesh_setting
                }
            },
            "module_data": {
                "faces_detected": fc,
                "inference_time_ms": f"{infer:.1f}"
            }
        }
