import time
import logging
import psutil
from typing import Dict, Any, Deque
from collections import deque
from .event_bus import EventBus
from .models.system import PerformanceMetrics

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Tracks granular latency, FPS, and system metrics."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.metrics = PerformanceMetrics()
        
        self.start_time = time.time()
        
        # Latency tracking (moving average of last 30 frames)
        self._inference_latencies: Deque[float] = deque(maxlen=30)
        self._e2e_latencies: Deque[float] = deque(maxlen=30)
        
        # FPS Tracking
        self._frame_times: Deque[float] = deque(maxlen=30)
        
        self.event_bus.subscribe("FrameProcessed", self._on_frame_processed)
        
    def _on_frame_processed(self, payload: Dict[str, Any]):
        """Calculate FPS and record latencies when a frame completes."""
        now = time.time()
        self._frame_times.append(now)
        
        # Calculate FPS
        if len(self._frame_times) > 1:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self.metrics.fps = round(len(self._frame_times) / elapsed, 2)
                
        # Update latencies if provided in payload
        if "inference_ms" in payload:
            self._inference_latencies.append(payload["inference_ms"])
            self.metrics.inference_latency_ms = sum(self._inference_latencies) / len(self._inference_latencies)
            
        if "e2e_ms" in payload:
            self._e2e_latencies.append(payload["e2e_ms"])
            self.metrics.e2e_latency_ms = sum(self._e2e_latencies) / len(self._e2e_latencies)

    def update_system_metrics(self):
        """Update CPU, RAM, Disk, and Network stats using psutil."""
        self.metrics.cpu_usage_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        self.metrics.ram_usage_gb = round(mem.used / (1024**3), 2)
        self.metrics.ram_total_gb = round(mem.total / (1024**3), 2)
        
        disk = psutil.disk_usage('/')
        self.metrics.disk_usage_percent = disk.percent
        
        self.metrics.uptime_seconds = round(time.time() - self.start_time, 2)

    def get_metrics(self) -> PerformanceMetrics:
        """Return the latest performance snapshot."""
        self.update_system_metrics()
        return self.metrics
