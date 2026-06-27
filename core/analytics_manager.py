import json
import os
import logging
from datetime import date

logger = logging.getLogger(__name__)

class AnalyticsManager:
    def __init__(self, data_dir):
        self.file_path = os.path.join(data_dir, 'analytics.json')
        self.stats = {}
        self.unsaved_changes = 0
        self.save_threshold = 30 # save every 30 increments to save disk I/O
        self.load()

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    self.stats = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load analytics: {e}")
                self.stats = {}
        
        self._ensure_today()
            
        if "module_usage" not in self.stats:
            self.stats["module_usage"] = {}

    def _ensure_today(self):
        today = str(date.today())
        if "daily" not in self.stats:
            self.stats["daily"] = {}
        if today not in self.stats["daily"]:
            self.stats["daily"][today] = {
                "objects_detected": 0,
                "ocr_reads": 0,
                "barcode_scans": 0,
                "inference_time_ms": 0,
                "frames_processed": 0
            }
        return today

    def save(self, force=False):
        self.unsaved_changes += 1
        if force or self.unsaved_changes >= self.save_threshold:
            try:
                with open(self.file_path, 'w') as f:
                    json.dump(self.stats, f, indent=4)
                self.unsaved_changes = 0
            except Exception as e:
                logger.error(f"Failed to save analytics: {e}")

    def increment(self, metric, amount=1, module_name=None):
        today = self._ensure_today()
            
        if metric in self.stats["daily"][today]:
            self.stats["daily"][today][metric] += amount
            
        if module_name:
            if module_name not in self.stats["module_usage"]:
                self.stats["module_usage"][module_name] = 0
            self.stats["module_usage"][module_name] += amount
            
        self.save()
        
    def log_inference(self, time_ms, active_module=None):
        today = self._ensure_today()
        self.stats["daily"][today]["frames_processed"] += 1
        self.stats["daily"][today]["inference_time_ms"] += time_ms
        
        if active_module:
            if active_module not in self.stats["module_usage"]:
                self.stats["module_usage"][active_module] = 0
            # Track frames processed per module (can convert to time later by multiplying by avg fps)
            self.stats["module_usage"][active_module] += 1
            
        self.save()

    def get_summary(self):
        today = self._ensure_today()
        day_stats = self.stats["daily"].get(today, {})
        
        frames = day_stats.get("frames_processed", 0)
        total_inf = day_stats.get("inference_time_ms", 0)
        avg_inf = (total_inf / frames) if frames > 0 else 0
        
        return {
            "objects_detected": day_stats.get("objects_detected", 0),
            "ocr_reads": day_stats.get("ocr_reads", 0),
            "barcode_scans": day_stats.get("barcode_scans", 0),
            "avg_inference_time": round(avg_inf, 2),
            "frames_processed": frames,
            "module_usage": self.stats.get("module_usage", {})
        }
