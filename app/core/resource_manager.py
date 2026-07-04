import logging
import psutil
import gc
from typing import Optional
from .event_bus import EventBus

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manages system resources, monitors usage, and handles cleanup to prevent OOM."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.ram_threshold_percent = 90.0

    def get_ram_usage(self) -> float:
        """Get current RAM usage percentage."""
        return psutil.virtual_memory().percent

    def get_vram_usage(self) -> Optional[float]:
        """Attempt to get VRAM usage if pynvml is available, else return None."""
        # For a full implementation, we'd use pynvml or WMI
        return None

    def cleanup(self):
        """Force garbage collection and memory cleanup."""
        logger.info("Running manual garbage collection...")
        gc.collect()
        
        # If OpenCV is loaded, we could try to clear some internal caches if needed
        # cv2.destroyAllWindows()
        
    def check_and_recover(self):
        """Check if resources are critical and attempt recovery."""
        usage = self.get_ram_usage()
        if usage > self.ram_threshold_percent:
            logger.warning(f"CRITICAL: RAM usage at {usage}%. Triggering recovery.")
            self.cleanup()
            # Publish event so AIRuntimeManager can evict unused models
            self.event_bus.publish("ResourceCritical", {"type": "RAM", "usage": usage})

    def shutdown(self):
        """Perform final resource cleanup on exit."""
        self.cleanup()
        logger.info("ResourceManager shutdown complete.")
