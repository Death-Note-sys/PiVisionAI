import logging
import cv2
import numpy as np
import time
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
        
    def add_client(self, client_id: str, client_type: str = "mjpeg") -> None:
        self.clients[client_id] = client_type
        logger.info(f"OutputManager: Added client {client_id} ({client_type})")
        
    def remove_client(self, client_id: str) -> None:
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"OutputManager: Removed client {client_id}")
            
    def broadcast(self, frame: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Called by the pipeline after rendering is complete."""
        self.latest_frame = frame
        
        # If recording to disk
        if self.is_recording and self.video_writer is not None:
            try:
                self.video_writer.write(frame)
            except Exception as e:
                logger.error(f"Failed to write frame to video: {e}")
                
    def get_latest_jpeg(self) -> bytes:
        """Returns the latest frame encoded as JPEG for streaming."""
        if self.latest_frame is None:
            # Return empty black frame if nothing is available yet
            ret, buffer = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))
            return buffer.tobytes()
            
        ret, buffer = cv2.imencode('.jpg', self.latest_frame)
        if ret:
            return buffer.tobytes()
        return b""
        
    def get_mjpeg_generator(self, target_fps: int = 30):
        """Generic MJPEG generator. Single source of truth for all streaming
        routes — serves whatever is most recently broadcast, whether that's
        a raw camera frame (no module active) or a rendered one."""
        frame_interval = 1.0 / target_fps if target_fps > 0 else 0.033
        while True:
            frame_bytes = self.get_latest_jpeg()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
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
