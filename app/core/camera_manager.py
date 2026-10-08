import os
# Disable MSMF HW transforms on Windows which can cause lag
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import threading
import time
import logging
from collections import deque
from typing import Dict, Any, List, Optional
from .config_service import ConfigService
from .event_bus import EventBus
from .pipeline import Pipeline
from .performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

class CameraManager:
    """
    High-performance CameraManager using a dual-thread Producer-Consumer 
    pattern and a low-latency deque buffer.
    """
    def __init__(self, config: ConfigService, event_bus: EventBus, pipeline: Pipeline, monitor: PerformanceMonitor):
        self.config = config
        self.event_bus = event_bus
        self.pipeline = pipeline
        self.monitor = monitor

        self.camera_index = int(self.config.get("camera_index", 0))
        self.target_fps = int(self.config.get("camera_fps", 30))
        
        # Deque buffer ensures we always grab the newest frame and drop stale ones
        self._raw_frames = deque(maxlen=2)
        
        self.cap = None
        self.is_connected = False
        
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._processing_thread = None
        self._cap_lock = threading.Lock()
        self.camera_settings = {}
        
    def apply_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Apply camera settings and return read-back values."""
        self.camera_settings.update(settings)
        if not self.cap or not self.is_connected:
            return {}
            
        readback = {}
        with self._cap_lock:
            # Auto Exposure
            if "auto_exposure" in settings:
                # DSHOW: 0.25 manual, 0.75 auto
                val = 0.75 if settings["auto_exposure"] else 0.25
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
                actual = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                readback["auto_exposure"] = actual > 0.5
                
            if "exposure" in settings and not settings.get("auto_exposure", True):
                self.cap.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
                readback["exposure"] = self.cap.get(cv2.CAP_PROP_EXPOSURE)
                
            if "gain" in settings:
                self.cap.set(cv2.CAP_PROP_GAIN, settings["gain"])
                readback["gain"] = self.cap.get(cv2.CAP_PROP_GAIN)
                
            if "auto_wb" in settings:
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1.0 if settings["auto_wb"] else 0.0)
                readback["auto_wb"] = self.cap.get(cv2.CAP_PROP_AUTO_WB) > 0.5
                
            if "wb_temperature" in settings and not settings.get("auto_wb", True):
                self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, settings["wb_temperature"])
                readback["wb_temperature"] = self.cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
                
            if "autofocus" in settings:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if settings["autofocus"] else 0.0)
                readback["autofocus"] = self.cap.get(cv2.CAP_PROP_AUTOFOCUS) > 0.5
                
            if "focus" in settings and not settings.get("autofocus", True):
                self.cap.set(cv2.CAP_PROP_FOCUS, settings["focus"])
                readback["focus"] = self.cap.get(cv2.CAP_PROP_FOCUS)
                
        return readback

    def start(self):
        """Start the capture and processing threads."""
        if self._capture_thread and self._capture_thread.is_alive():
            return
            
        self._stop_event.clear()
        self._open_camera()
        
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCapture")
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True, name="CameraProcess")
        
        self._capture_thread.start()
        self._processing_thread.start()
        
        self.event_bus.publish("CameraStarted", {"index": self.camera_index})
        logger.info("CameraManager threads started.")

    def stop(self):
        """Stop threads and release hardware."""
        self._stop_event.set()
        if self._capture_thread: self._capture_thread.join(timeout=2.0)
        if self._processing_thread: self._processing_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.is_connected = False
            
        self.event_bus.publish("CameraStopped", {"index": self.camera_index})
        logger.info("CameraManager stopped.")

    def _open_camera(self) -> bool:
        with self._cap_lock:
            if self.cap:
                self.cap.release()
                
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
            configured_backend = self.config.get("camera_backend", "auto")
            
            opened = False
            for backend in backends:
                self.cap = cv2.VideoCapture(self.camera_index, backend)
                if self.cap.isOpened():
                    opened = True
                    break
                    
            if not opened:
                self.is_connected = False
                logger.error(f"Failed to open Camera {self.camera_index}.")
                return False
                
            # Attempt to set high resolution if configured
            w, h = map(int, self.config.get("camera_resolution", "1280x720").split('x'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            time.sleep(1.0)
            self.is_connected = True
            logger.info(f"Camera {self.camera_index} opened.")
            
        # Apply any pending settings
        if self.camera_settings:
            self.apply_settings(self.camera_settings)
            
        return True

    def _capture_loop(self):
        """Producer: Grab frames as fast as hardware allows to clear buffers."""
        while not self._stop_event.is_set():
            if not self.is_connected:
                time.sleep(1.0)
                self._open_camera()
                continue
                
            with self._cap_lock:
                ret, frame = self.cap.read()
                
            if ret:
                self._raw_frames.append(frame)
                self.event_bus.publish("FrameCaptured")
            else:
                self.is_connected = False

    def _processing_loop(self):
        """Consumer: Process the most recent frame."""
        while not self._stop_event.is_set():
            if len(self._raw_frames) == 0:
                time.sleep(0.01)
                continue
                
            # Pop the most recent frame (LIFO for low latency)
            frame = self._raw_frames.pop()
            
            try:
                self.pipeline.process_frame(frame)
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)


