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
        self.camera_settings = {
            "auto_exposure": True,
            "exposure": -5.0,
            "gain": 0.0,
            "auto_wb": True,
            "wb_temperature": 4600.0
        }
        self.last_readback = {}
        
    def apply_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Apply camera settings and return read-back values."""
        self.camera_settings.update(settings)
        if not self.cap or not self.is_connected:
            return dict(settings)
            
        readback = {}
        with self._cap_lock:
            # Auto Exposure
            if "auto_exposure" in settings:
                is_auto = bool(settings["auto_exposure"])
                val = 0.75 if is_auto else 0.25
                try:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
                    actual = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                    if isinstance(actual, (int, float)):
                        if actual < 0:
                            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0 if is_auto else 0.0)
                            actual = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                        readback["auto_exposure"] = "unsupported" if (isinstance(actual, (int, float)) and actual < 0) else (bool(actual > 0.5) if isinstance(actual, (int, float)) else is_auto)
                    else:
                        readback["auto_exposure"] = is_auto
                except Exception as e:
                    logger.warning(f"Error setting auto_exposure: {e}")
                    readback["auto_exposure"] = is_auto
                
            is_auto_exp = self.camera_settings.get("auto_exposure", True)
            if "exposure" in settings and not is_auto_exp:
                try:
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, float(settings["exposure"]))
                    actual_exp = self.cap.get(cv2.CAP_PROP_EXPOSURE)
                    readback["exposure"] = actual_exp if isinstance(actual_exp, (int, float)) and actual_exp >= -100 else float(settings["exposure"])
                except Exception as e:
                    logger.warning(f"Error setting exposure: {e}")
                
            if "gain" in settings:
                try:
                    self.cap.set(cv2.CAP_PROP_GAIN, float(settings["gain"]))
                    actual_gain = self.cap.get(cv2.CAP_PROP_GAIN)
                    readback["gain"] = actual_gain if isinstance(actual_gain, (int, float)) and actual_gain >= 0 else float(settings["gain"])
                except Exception as e:
                    logger.warning(f"Error setting gain: {e}")
                
            if "auto_wb" in settings:
                is_auto_wb = bool(settings["auto_wb"])
                try:
                    self.cap.set(cv2.CAP_PROP_AUTO_WB, 1.0 if is_auto_wb else 0.0)
                    actual_wb = self.cap.get(cv2.CAP_PROP_AUTO_WB)
                    if isinstance(actual_wb, (int, float)):
                        readback["auto_wb"] = "unsupported" if actual_wb < 0 else bool(actual_wb > 0.5)
                    else:
                        readback["auto_wb"] = is_auto_wb
                except Exception as e:
                    logger.warning(f"Error setting auto_wb: {e}")
                    readback["auto_wb"] = is_auto_wb
                
            is_auto_wb = self.camera_settings.get("auto_wb", True)
            if "wb_temperature" in settings and not is_auto_wb:
                try:
                    self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(settings["wb_temperature"]))
                    actual_wb_t = self.cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
                    readback["wb_temperature"] = actual_wb_t if isinstance(actual_wb_t, (int, float)) and actual_wb_t > 0 else float(settings["wb_temperature"])
                except Exception as e:
                    logger.warning(f"Error setting wb_temperature: {e}")
                
            if "autofocus" in settings:
                try:
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if settings["autofocus"] else 0.0)
                    actual_af = self.cap.get(cv2.CAP_PROP_AUTOFOCUS)
                    readback["autofocus"] = bool(actual_af > 0.5) if isinstance(actual_af, (int, float)) else bool(settings["autofocus"])
                except Exception as e:
                    logger.warning(f"Error setting autofocus: {e}")
                
            is_autofocus = self.camera_settings.get("autofocus", True)
            if "focus" in settings and not is_autofocus:
                try:
                    self.cap.set(cv2.CAP_PROP_FOCUS, float(settings["focus"]))
                    actual_f = self.cap.get(cv2.CAP_PROP_FOCUS)
                    readback["focus"] = actual_f if isinstance(actual_f, (int, float)) else float(settings["focus"])
                except Exception as e:
                    logger.warning(f"Error setting focus: {e}")
                
        self.last_readback.update(readback)
        return readback

    def get_settings(self) -> Dict[str, Any]:
        """Return current camera settings and latest driver readback."""
        return {
            "settings": dict(self.camera_settings),
            "readback": dict(self.last_readback)
        }

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
                
            self.camera_index = int(self.config.get("camera_index", self.camera_index))
            raw_backend = str(self.config.get("camera_backend", "auto")).lower()
            if "dshow" in raw_backend:
                backends = [cv2.CAP_DSHOW]
            elif "msmf" in raw_backend:
                backends = [cv2.CAP_MSMF]
            elif "any" in raw_backend:
                backends = [cv2.CAP_ANY]
            elif "v4l" in raw_backend:
                backends = [getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)]
            else: # "auto"
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
            
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


