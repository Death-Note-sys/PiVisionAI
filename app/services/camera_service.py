import logging
import threading
from typing import Dict, Any, Generator
from app.core.container import Container

logger = logging.getLogger(__name__)

class CameraService:
    """Business logic for camera operations."""
    
    def __init__(self):
        self.container = Container.get_instance()
        self.camera_manager = self.container.camera_manager
        self.output_manager = self.container.output_manager
        self._switch_lock = threading.Lock()

    def get_stream(self) -> Generator[bytes, None, None]:
        """Return the MJPEG stream generator."""
        return self.output_manager.get_mjpeg_generator(
            target_fps=self.camera_manager.target_fps
        )

    def switch_camera(self, index: int, resolution: str, backend: str = None) -> bool:
        """Switch camera hardware or resolution and persist settings."""
        try:
            index = int(index)
            parts = str(resolution).lower().split("x")
            if len(parts) != 2 or int(parts[0]) <= 0 or int(parts[1]) <= 0:
                raise ValueError(f"Invalid resolution: {resolution}")

            # Serialize hardware teardown/open so concurrent switch requests never race on VideoCapture.
            with self._switch_lock:
                self.camera_manager.stop()
                updates = {
                    "camera_index": index,
                    "camera_resolution": resolution
                }
                if backend:
                    updates["camera_backend"] = backend
                self.container.config.update_settings(updates)
                # Re-init is handled in start based on config
                self.camera_manager.camera_index = index
                self.camera_manager.start()
                return True
        except Exception as e:
            logger.error(f"Failed to switch camera: {e}")
            return False

    def get_camera_settings(self) -> Dict[str, Any]:
        """Get current camera hardware settings and readback."""
        if hasattr(self.camera_manager, "get_settings"):
            return self.camera_manager.get_settings()
        return {
            "settings": getattr(self.camera_manager, "camera_settings", {}),
            "readback": getattr(self.camera_manager, "last_readback", {})
        }

    def update_camera_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update camera hardware settings and return applied settings & readback."""
        readback = self.camera_manager.apply_settings(settings)
        return {
            "settings": getattr(self.camera_manager, "camera_settings", {}),
            "readback": readback
        }

