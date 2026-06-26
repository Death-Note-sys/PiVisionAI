import os
import sys
import threading
import logging
import importlib.util
import inspect
from typing import Dict, List, Optional, Type
from .base_module import BaseVisionModule
from config import config

logger = logging.getLogger(__name__)

class ModuleManager:
    """
    Singleton class to manage dynamic AI vision modules via importlib.
    Supports hot-swapping and guarantees only one active module at a time.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModuleManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # Map of module ID to its instantiated Class
            self.loaded_modules: Dict[str, BaseVisionModule] = {}
            
            # Map of module ID to its python module reference (for reloading)
            self._py_modules: Dict[str, object] = {}
            
            self.active_module_id: Optional[str] = None
            self.modules_dir = os.path.join(config['default'].BASE_DIR, 'modules')
            self._module_lock = threading.Lock()
            self._initialized = True
            
            self.discover_and_load_all()
            logger.info("ModuleManager initialized.")

    def discover_and_load_all(self):
        """Scans the modules directory and loads valid plugins."""
        if not os.path.exists(self.modules_dir):
            os.makedirs(self.modules_dir)
            return

        with self._module_lock:
            for filename in os.listdir(self.modules_dir):
                if filename.endswith(".py") and not filename.startswith("__"):
                    self._load_module_from_file(filename)

    def _load_module_from_file(self, filename: str):
        """Internal: Dynamically load a python file and instantiate its BaseVisionModule."""
        filepath = os.path.join(self.modules_dir, filename)
        module_name = filename[:-3] # remove .py

        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # Find classes that inherit from BaseVisionModule
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseVisionModule) and obj is not BaseVisionModule:
                        # Instantiate the module
                        instance = obj()
                        meta = instance.metadata()
                        mod_id = meta.get('id', module_name)
                        
                        self.loaded_modules[mod_id] = instance
                        self._py_modules[mod_id] = module
                        logger.info(f"Loaded module plugin: {mod_id}")
                        return
        except Exception as e:
            logger.error(f"Failed to load module {filename}: {e}")

    def get_all_metadata(self) -> List[Dict]:
        """Return metadata for all loaded modules."""
        with self._module_lock:
            meta_list = []
            for mod_id, instance in self.loaded_modules.items():
                meta = instance.metadata()
                meta['is_active'] = (mod_id == self.active_module_id)
                meta_list.append(meta)
            return meta_list

    def activate_module(self, mod_id: str) -> bool:
        """Hot swap to a new module. Only one module is active at a time."""
        with self._module_lock:
            if mod_id not in self.loaded_modules and mod_id is not None:
                logger.error(f"Cannot activate unknown module: {mod_id}")
                return False

            # If it's already the active one, do nothing
            if self.active_module_id == mod_id:
                return True

            # Cleanup current module
            if self.active_module_id:
                try:
                    self.loaded_modules[self.active_module_id].cleanup()
                    logger.info(f"Deactivated module: {self.active_module_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up module {self.active_module_id}: {e}")

            # Initialize new module
            self.active_module_id = mod_id
            if mod_id:
                try:
                    self.loaded_modules[mod_id].initialize()
                    logger.info(f"Activated module: {mod_id}")
                except Exception as e:
                    logger.error(f"Error initializing module {mod_id}: {e}")
                    self.active_module_id = None
                    return False
                    
            return True

    def reload_modules(self):
        """Force unload and reload all modules from disk."""
        # Deactivate first
        self.activate_module(None)
        
        with self._module_lock:
            self.loaded_modules.clear()
            
            # Remove from sys.modules
            for mod_id, py_mod in self._py_modules.items():
                if py_mod.__name__ in sys.modules:
                    del sys.modules[py_mod.__name__]
            self._py_modules.clear()
            
        self.discover_and_load_all()
        logger.info("All modules reloaded.")

    def process_frame(self, frame):
        """Pass the frame through the single active module."""
        # Fast lockless check
        active_id = self.active_module_id
        if not active_id:
            return frame
            
        # We process without locking the whole frame cycle to avoid blocking API,
        # relying on the GIL and atomic object swapping.
        module = self.loaded_modules.get(active_id)
        if module:
            try:
                return module.process(frame)
            except Exception as e:
                logger.error(f"Error in module {active_id} process(): {e}")
        return frame
