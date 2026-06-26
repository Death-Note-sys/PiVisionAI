import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import threading
import time
import logging
import platform
import numpy as np
from typing import List, Dict, Optional
from .module_manager import ModuleManager

logger = logging.getLogger(__name__)

class CameraManager:
    """
    Advanced CameraManager supporting auto-detection, multiple resolutions, 
    FPS calculation, dynamic switching, and graceful disconnects.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CameraManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, config=None):
        if not self._initialized:
            self.config = config
            
            # Auto-detect camera index if not explicitly provided
            self.camera_index = config.CAMERA_INDEX if config else 0
            
            # Assume camera index 0 by default to prevent hardware lockups from rapid scanning
            self.camera_index = config.CAMERA_INDEX if config else 0
                
            self.width = config.CAMERA_WIDTH if config else 640
            self.height = config.CAMERA_HEIGHT if config else 480
            self.target_fps = config.CAMERA_FPS if config else 30
            
            self.cap = None
            self.current_frame = None
            self.is_running = False
            self.thread = None
            self.lock = threading.Lock()
            
            # Telemetry
            self.actual_fps = 0.0
            self.is_connected = False
            
            # Supported Resolutions
            self.supported_resolutions = ["640x480", "1280x720", "1920x1080"]
            
            self.module_manager = ModuleManager()
            self._initialized = True
            logger.info("Advanced CameraManager initialized.")

    @staticmethod
    def scan_cameras() -> List[Dict]:
        """Scan for available cameras."""
        available_cameras = []
        # Try standard indices 0-1 to save time
        for i in range(2):
            cap = cv2.VideoCapture(i)
                
            if cap.isOpened():
                # Verify that the camera can actually read frames
                ret, _ = cap.read()
                if ret:
                    backend = cap.getBackendName()
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    available_cameras.append({
                        "index": i,
                        "name": f"Camera {i} ({backend})",
                        "resolution": f"{w}x{h}"
                    })
                cap.release()
        return available_cameras

    def switch_camera(self, index: int, width: int, height: int):
        """Safely switch the active camera or resolution."""
        logger.info(f"Switching camera to index {index} with resolution {width}x{height}")
        with self.lock:
            self.camera_index = index
            self.width = width
            self.height = height
            
            if self.cap:
                self.cap.release()
                self.is_connected = False
                
            # The _update_loop will automatically try to reconnect because is_running is still True.

    def start(self):
        """Start the background camera thread."""
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        logger.info("Camera stream thread started.")

    def stop(self):
        """Stop the background camera thread."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.is_connected = False
        logger.info("Camera stream stopped.")

    def _open_camera(self) -> bool:
        """Internal method to open the camera."""
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.camera_index)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        if self.cap.isOpened():
            # Allow the camera hardware to settle before trying to read frames
            time.sleep(1.0)
            self.is_connected = True
            logger.info(f"Successfully opened camera {self.camera_index}")
            return True
        else:
            self.is_connected = False
            return False

    def _update_loop(self):
        """Background loop with reconnect logic and FPS calculation."""
        
        # Initial connect
        self._open_camera()
        
        last_time = time.time()
        ema_alpha = 0.1 # Exponential moving average factor for FPS
        
        while self.is_running:
            if not self.is_connected:
                logger.warning(f"Camera disconnected. Attempting to reconnect to index {self.camera_index}...")
                if self._open_camera():
                    # Successfully reconnected, reset timers
                    last_time = time.time()
                else:
                    # Wait before retrying
                    time.sleep(2.0)
                    continue

            ret, frame = self.cap.read()
            
            if not ret:
                logger.error("Failed to grab frame. Camera might be disconnected.")
                self.is_connected = False
                continue

            # Calculate FPS
            current_time = time.time()
            time_diff = current_time - last_time
            if time_diff > 0:
                instant_fps = 1.0 / time_diff
                self.actual_fps = (ema_alpha * instant_fps) + ((1 - ema_alpha) * self.actual_fps)
            last_time = current_time

            # Process frame through active modules
            try:
                processed_frame = self.module_manager.process_frame(frame)
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                processed_frame = frame

            with self.lock:
                self.current_frame = processed_frame

            # Yield slightly to prevent 100% CPU on fast cameras
            time.sleep(max(0, (1.0 / self.target_fps) - (time.time() - current_time)))

    def get_info(self) -> Dict:
        """Get telemetry info for the current camera state."""
        return {
            "is_connected": self.is_connected,
            "index": self.camera_index,
            "resolution": f"{self.width}x{self.height}",
            "fps": round(self.actual_fps, 1),
            "target_fps": self.target_fps
        }

    def get_frame(self):
        """Get the latest processed frame."""
        with self.lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()

    def get_mjpeg_stream(self):
        """Generator function for MJPEG streaming."""
        while True:
            if not self.is_connected:
                time.sleep(0.5)
                continue
                
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(1.0 / self.target_fps)
