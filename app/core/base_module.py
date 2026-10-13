import os
import json
import time
import shutil
import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional
from app.core.contracts import IModule
from app.core.container import Container
from app.core.event_bus import EventBus

logger = logging.getLogger(__name__)

class BaseModuleController(IModule):
    """Abstract base class for all computer vision modules, providing isolated persistence."""
    
    def __init__(self, module_id: str, event_bus: EventBus):
        self.id = module_id
        self.event_bus = event_bus
        
        container = Container.get_instance()
        if container and getattr(container, 'session_manager', None):
            self.module_dir = container.session_manager.get_data_dir(f"modules/{self.id}")
            self.settings_file = os.path.join(self.module_dir, "settings.json")
            self.data_file = os.path.join(self.module_dir, "data.json")
        else:
            self.module_dir = None
            self.settings_file = None
            self.data_file = None

    def _safe_load_json(self, filepath: Optional[str]) -> Dict[str, Any]:
        """Safely loads JSON with corrupt-file backup."""
        if not filepath or not os.path.exists(filepath):
            return {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {filepath}: {e}")
            try:
                corrupt_backup = filepath + f".corrupt_{int(time.time())}"
                shutil.copy2(filepath, corrupt_backup)
                logger.warning(f"Backed up corrupt file {filepath} -> {corrupt_backup}")
            except Exception:
                pass
            return {}

    def _atomic_save_json(self, filepath: Optional[str], data: Dict[str, Any]):
        """Atomically saves JSON via .tmp + os.replace."""
        if not filepath:
            return
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            tmp_path = filepath + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, filepath)
        except Exception as e:
            logger.error(f"Failed to atomically save JSON to {filepath}: {e}")

    def load_settings(self) -> Dict[str, Any]:
        """Loads settings from disk if available."""
        return self._safe_load_json(self.settings_file)
            
    def save_settings(self, settings: Dict[str, Any]):
        """Saves current settings to disk atomically."""
        self._atomic_save_json(self.settings_file, settings)

    def load_data(self) -> Dict[str, Any]:
        """Loads non-settings teach/calibration data."""
        return self._safe_load_json(self.data_file)

    def save_data(self, data: Dict[str, Any]):
        """Saves non-settings teach/calibration data atomically."""
        self._atomic_save_json(self.data_file, data)
            
    def save_image(self, filename: str, image: np.ndarray):
        """Saves an image into the module's persistence directory."""
        if not self.module_dir: return
        cv2.imwrite(os.path.join(self.module_dir, filename), image)
        
    def load_image(self, filename: str) -> Optional[np.ndarray]:
        """Loads an image from the module's persistence directory."""
        if not self.module_dir: return None
        path = os.path.join(self.module_dir, filename)
        if os.path.exists(path):
            return cv2.imread(path)
        return None
