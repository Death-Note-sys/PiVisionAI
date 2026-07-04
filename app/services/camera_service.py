import logging
from typing import Dict, Any, Generator
from app.core.container import Container

logger = logging.getLogger(__name__)

class CameraService:
    """Business logic for camera operations."""
    
    def __init__(self):
        self.container = Container.get_instance()
        self.camera_manager = self.container.camera_manager
        self.output_manager = self.container.output_manager

    def get_stream(self) -> Generator[bytes, None, None]:
        """Return the MJPEG stream generator."""
        return self.output_manager.get_mjpeg_generator(
            target_fps=self.camera_manager.target_fps
        )

    def switch_camera(self, index: int, resolution: str) -> bool:
        """Switch camera hardware or resolution."""
        try:
            # Need to restart camera manager to apply changes cleanly
            self.camera_manager.stop()
            self.container.config.update_settings({
                "camera_index": index,
                "camera_resolution": resolution
            })
            # Re-init is handled in start based on config
            self.camera_manager.camera_index = index
            self.camera_manager.start()
            return True
        except Exception as e:
            logger.error(f"Failed to switch camera: {e}")
            return False
