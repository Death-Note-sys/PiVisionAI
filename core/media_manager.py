import os
import cv2
import time
import logging
import threading

logger = logging.getLogger(__name__)

class MediaManager:
    def __init__(self, data_dir, camera_manager):
        self.screenshots_dir = os.path.join(data_dir, 'screenshots')
        self.recordings_dir = os.path.join(data_dir, 'recordings')
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        self.camera_manager = camera_manager
        
        self.is_recording = False
        self.recording_thread = None
        self.video_writer = None
        self.recording_start_time = 0
        self.record_feed_type = "Processed Feed"
        
    def take_screenshot(self, format_ext="PNG", quality=100, custom_name="", feed_type="Processed Feed"):
        if feed_type == "Original Feed":
            with self.camera_manager.lock:
                frame = self.camera_manager.raw_frame.copy() if hasattr(self.camera_manager, 'raw_frame') and self.camera_manager.raw_frame is not None else None
        else:
            frame = self.camera_manager.get_frame()
            
        if frame is None:
            return {"error": "No frame available"}
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = custom_name if custom_name else f"screenshot_{timestamp}"
        ext = format_ext.lower()
        
        filepath = os.path.join(self.screenshots_dir, f"{name}.{ext}")
        
        try:
            if ext == "jpg" or ext == "jpeg":
                cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            elif ext == "webp":
                cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)])
            else:
                cv2.imwrite(filepath, frame)
            
            logger.info(f"Screenshot saved: {filepath}")
            return {"success": True, "filepath": filepath, "filename": f"{name}.{ext}"}
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return {"error": str(e)}

    def start_recording(self, format_ext="MP4", codec="H264", feed_type="Processed Feed"):
        if self.is_recording:
            return {"error": "Already recording"}
            
        self.record_feed_type = feed_type
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ext = format_ext.lower()
        filepath = os.path.join(self.recordings_dir, f"record_{timestamp}.{ext}")
        
        if codec == "H264":
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        elif codec == "XVID":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif codec == "MJPEG":
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
        info = self.camera_manager.get_info()
        w, h = map(int, info['resolution'].split('x'))
        fps = info['target_fps']
        if fps <= 0: fps = 30
        
        self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
        if not self.video_writer.isOpened():
            logger.warning(f"Codec {codec} failed, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
            if not self.video_writer.isOpened():
                return {"error": "Failed to initialize VideoWriter"}
        
        self.is_recording = True
        self.recording_start_time = time.time()
        self.recording_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.recording_thread.start()
        
        logger.info(f"Started recording: {filepath}")
        return {"success": True, "filepath": filepath}
        
    def stop_recording(self):
        if not self.is_recording:
            return {"error": "Not recording"}
            
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
            
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        dur = time.time() - self.recording_start_time
        logger.info(f"Stopped recording. Duration: {dur:.1f}s")
        return {"success": True, "duration": dur}
        
    def _record_loop(self):
        fps = self.camera_manager.target_fps
        if fps <= 0: fps = 30
        frame_time = 1.0 / fps
        
        while self.is_recording:
            start_t = time.time()
            
            if self.record_feed_type == "Original Feed":
                with self.camera_manager.lock:
                    frame = self.camera_manager.raw_frame.copy() if hasattr(self.camera_manager, 'raw_frame') and self.camera_manager.raw_frame is not None else None
            else:
                frame = self.camera_manager.get_frame()
                
            if frame is not None and self.video_writer is not None:
                try:
                    self.video_writer.write(frame)
                except Exception as e:
                    logger.error(f"Error writing frame: {e}")
            
            elapsed = time.time() - start_t
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)

    def get_status(self):
        dur = time.time() - self.recording_start_time if self.is_recording else 0
        return {
            "is_recording": self.is_recording,
            "duration": round(dur, 1)
        }
