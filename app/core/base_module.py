import os
import json
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

    def load_settings(self) -> Dict[str, Any]:
        """Loads settings from disk if available."""
        if not self.settings_file or not os.path.exists(self.settings_file):
            return {}
        try:
            with open(self.settings_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load settings for module {self.id}: {e}")
            return {}
            
    def save_settings(self, settings: Dict[str, Any]):
        """Saves current settings to disk."""
        if not self.settings_file:
            return
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save settings for module {self.id}: {e}")

    def load_data(self) -> Dict[str, Any]:
        """Loads non-settings teach/calibration data."""
        if not self.data_file or not os.path.exists(self.data_file):
            return {}
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load data for module {self.id}: {e}")
            return {}

    def save_data(self, data: Dict[str, Any]):
        """Saves non-settings teach/calibration data."""
        if not self.data_file:
            return
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save data for module {self.id}: {e}")
            
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
