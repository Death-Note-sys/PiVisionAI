import psutil
import time
import threading
import logging

try:
    import wmi
    import pythoncom
    HAS_WMI = True
except ImportError:
    HAS_WMI = False

logger = logging.getLogger(__name__)

class SystemTelemetry:
    def __init__(self):
        self.stats = {
            "cpu_usage": 0.0,
            "gpu_usage": 0.0,
            "ram_usage": 0.0,
            "ram_total": 0.0,
            "disk_usage": 0.0,
            "disk_total": 0.0,
            "net_speed_rx": 0.0, # Mbps
            "net_speed_tx": 0.0, # Mbps
            "uptime": 0
        }
        self.is_running = False
        self.thread = None
        
        self.last_net_io = psutil.net_io_counters()
        self.last_net_time = time.time()
        self.start_time = time.time()

    def start(self):
        if self.is_running: return
        self.is_running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _update_loop(self):
        if HAS_WMI:
            try:
                pythoncom.CoInitialize()
                w = wmi.WMI()
            except Exception as e:
                logger.error(f"WMI init failed for Telemetry: {e}")
                w = None
        else:
            w = None

        while self.is_running:
            try:
                self.stats["cpu_usage"] = psutil.cpu_percent(interval=0)
                
                mem = psutil.virtual_memory()
                self.stats["ram_usage"] = round(mem.used / (1024**3), 1)
                self.stats["ram_total"] = round(mem.total / (1024**3), 1)
                
                # Check root directory on Windows (usually C:\, but we can check the path of the script)
                disk = psutil.disk_usage('/')
                self.stats["disk_usage"] = round(disk.used / (1024**3), 1)
                self.stats["disk_total"] = round(disk.total / (1024**3), 1)
                
                current_net_io = psutil.net_io_counters()
                current_time = time.time()
                dt = current_time - self.last_net_time
                if dt > 0:
                    rx = (current_net_io.bytes_recv - self.last_net_io.bytes_recv) / dt
                    tx = (current_net_io.bytes_sent - self.last_net_io.bytes_sent) / dt
                    self.stats["net_speed_rx"] = round((rx * 8) / 1000000, 2)
                    self.stats["net_speed_tx"] = round((tx * 8) / 1000000, 2)
                self.last_net_io = current_net_io
                self.last_net_time = current_time
                
                self.stats["uptime"] = int(time.time() - self.start_time)
                
                if w:
                    try:
                        # Fetch GPU 3D engine usage
                        engines = w.query("SELECT UtilizationPercentage, Name FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine")
                        total_gpu = 0
                        for eng in engines:
                            if eng.Name and "engtype_3D" in eng.Name:
                                total_gpu += int(eng.UtilizationPercentage)
                        self.stats["gpu_usage"] = min(100.0, float(total_gpu))
                    except Exception:
                        self.stats["gpu_usage"] = 0.0

            except Exception as e:
                logger.debug(f"Telemetry update error: {e}")
                
            time.sleep(1.0)
            
        if HAS_WMI and w:
            pythoncom.CoUninitialize()

    def get_stats(self):
        return self.stats
