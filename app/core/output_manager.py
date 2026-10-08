import logging
import cv2
import numpy as np
import time
import threading
from typing import Dict, Any, List
from app.core.contracts import IOutputManager

logger = logging.getLogger(__name__)

class OutputManager(IOutputManager):
    """Manages streaming and recording of rendered frames."""
    
    def __init__(self):
        self.clients: Dict[str, str] = {} # id -> type
        self.latest_frame: np.ndarray = None
        self.is_recording = False
        self.video_writer = None
        self._lock = threading.Lock()
        self.latest_jpeg = b""
        self.frame_id = 0
        
        import collections
        self.ng_queue = collections.deque(maxlen=50)
        self.ng_thread = threading.Thread(target=self._ng_worker, daemon=True)
        self.ng_thread.start()
        
    def save_ng(self, frame: np.ndarray, metadata: Dict[str, Any], file_path: str = None) -> None:
        copied_frame = frame.copy()
        if len(self.ng_queue) == self.ng_queue.maxlen:
            logger.warning("NG queue is full, dropping oldest frame.")
        self.ng_queue.append((copied_frame, metadata, file_path))
        
    def _ng_worker(self):
        import os
        from datetime import datetime
        while True:
            if self.ng_queue:
                frame, meta, explicit_path = self.ng_queue.popleft()
                try:
                    from app.core.container import Container
                    container = Container.get_instance()
                    if container and getattr(container, 'session_manager', None):
                        out_dir = container.session_manager.get_data_dir("ng_images")
                        
                        if explicit_path:
                            path = explicit_path
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                        else:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            mod = meta.get("module_id", "unknown")
                            fname = f"{mod}_{ts}.jpg"
                            path = os.path.join(out_dir, fname)
                            
                        cv2.imwrite(path, frame)
                        logger.info(f"Saved NG image to {path}")
                except Exception as e:
                    logger.error(f"Error saving NG image: {e}")
            else:
                time.sleep(0.1)
        
    def add_client(self, client_id: str, client_type: str = "mjpeg") -> None:
        self.clients[client_id] = client_type
        logger.info(f"OutputManager: Added client {client_id} ({client_type})")
        
    def remove_client(self, client_id: str) -> None:
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"OutputManager: Removed client {client_id}")
            
    def broadcast(self, frame: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Called by the pipeline after rendering is complete."""
        ret, buffer = cv2.imencode('.jpg', frame)
        jpeg = buffer.tobytes() if ret else b""
        
        with self._lock:
            self.latest_frame = frame
            self.latest_jpeg = jpeg
            self.frame_id += 1
            
        # If recording to disk
        if self.is_recording and self.video_writer is not None:
            try:
                self.video_writer.write(frame)
            except Exception as e:
                logger.error(f"Failed to write frame to video: {e}")
                
    def get_latest_jpeg(self) -> bytes:
        """Returns the latest frame encoded as JPEG for streaming."""
        with self._lock:
            if not self.latest_jpeg:
                ret, buffer = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))
                return buffer.tobytes()
            return self.latest_jpeg
        
    def get_mjpeg_generator(self, target_fps: int = 30):
        """Generic MJPEG generator. Single source of truth for all streaming
        routes — serves whatever is most recently broadcast, whether that's
        a raw camera frame (no module active) or a rendered one."""
        frame_interval = 1.0 / target_fps if target_fps > 0 else 0.033
        last_yielded_id = -1
        while True:
            with self._lock:
                current_id = self.frame_id
                frame_bytes = self.latest_jpeg
                
            if current_id != last_yielded_id and frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                last_yielded_id = current_id
                
            time.sleep(frame_interval)
        
    def start_recording(self, filepath: str, fps: int = 30, resolution: tuple = (640, 480)) -> bool:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, resolution)
            self.is_recording = True
            logger.info(f"OutputManager: Started recording to {filepath}")
            return True
        except Exception as e:
            logger.error(f"OutputManager: Failed to start recording: {e}")
            return False
            
    def stop_recording(self) -> None:
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            self.is_recording = False
            self.video_writer = None
            logger.info("OutputManager: Stopped recording")
