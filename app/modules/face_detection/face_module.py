import logging
import cv2
import os
import urllib.request
from typing import Dict, Any

from app.core.event_bus import EventBus
from app.core.models.base import Detection, BoundingBox

logger = logging.getLogger(__name__)

class FaceDetectionModule:
    """Face Detection and Mesh Module adapting to the new Pipeline Engine."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.id = "core-face-detection"
        
        logger.info("[FaceDetection] Initializing resources...")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            self.mp = mp
            self.python_mp = python
            self.vision_mp = vision
            
            # Download models for Tasks API
            weights_dir = os.path.join(os.getcwd(), "sessions", "default", "weights")
            os.makedirs(weights_dir, exist_ok=True)
            
            fd_path = os.path.join(weights_dir, "blaze_face_short_range.tflite")
            fm_path = os.path.join(weights_dir, "face_landmarker.task")
            
            if not os.path.exists(fd_path):
                logger.info("Downloading Face Detector model...")
                urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite", fd_path)
                
            if not os.path.exists(fm_path):
                logger.info("Downloading Face Landmarker model...")
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
        except Exception as e:
            logger.error(f"Failed to initialize Face Detection: {e}")
            self.face_detector = None
            self.face_landmarker = None

        self.enable_face_mesh = 0
        self.face_count = 0
        
    def update_settings(self, settings: Dict[str, Any]):
        if "enable_face_mesh" in settings:
            self.enable_face_mesh = int(settings["enable_face_mesh"])

    def preprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        frame = context["frame"]
        if self.face_detector:
            # MediaPipe expects RGB format
            context["mp_image"] = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return context

    def infer(self, context: Dict[str, Any], ai_runtime: Any) -> Dict[str, Any]:
        if not self.face_detector or "mp_image" not in context:
            return context
            
        mp_image = context["mp_image"]
        h, w = context["frame"].shape[:2]
        
        self.face_count = 0
        
        if self.enable_face_mesh == 1:
            detection_result = self.face_landmarker.detect(mp_image)
            self.face_count = len(detection_result.face_landmarks)
            context["face_landmarks"] = detection_result.face_landmarks
            
            if self.face_count > 0:
                self.event_bus.publish("FacesDetected", {"count": self.face_count, "type": "mesh"})
        else:
            detection_result = self.face_detector.detect(mp_image)
            self.face_count = len(detection_result.detections)
            
            for det in detection_result.detections:
                bbox = det.bounding_box
                x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                conf = det.categories[0].score if det.categories else 1.0
                
                # Strict models
                box = BoundingBox(x1=x, y1=y, x2=x+bw, y2=y+bh)
                detection = Detection(box=box, label="Face", confidence=float(conf))
                context["detections"].append(detection)
                
            if self.face_count > 0:
                self.event_bus.publish("FacesDetected", {"count": self.face_count, "type": "box"})
                
        return context

    def postprocess(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context

    def visualize(self, context: Dict[str, Any]) -> Any:
        frame = context["frame"]
        h, w = frame.shape[:2]
        
        if self.enable_face_mesh == 1 and "face_landmarks" in context:
            for face_landmarks in context["face_landmarks"]:
                for lm in face_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        else:
            for det in context["detections"]:
                if det.box:
                    x, y = det.box.x1, det.box.y1
                    cv2.rectangle(frame, (x, y), (det.box.x2, det.box.y2), (255, 0, 255), 2)
                    cv2.putText(frame, f"Conf: {det.confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
                    cx = x + (det.box.x2 - x) // 2
                    cy = y + (det.box.y2 - y) // 2
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        cv2.putText(frame, f"Faces: {self.face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame

    def cleanup(self):
        logger.info("[FaceDetection] Cleaning up...")
        self.face_detector = None
        self.face_landmarker = None
