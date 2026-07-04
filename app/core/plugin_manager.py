import os
import json
import importlib
import importlib.util
import logging
from typing import Dict, Any, Optional, List
from .config_service import ConfigService
from .event_bus import EventBus
from .models.system import ModuleMetadata

logger = logging.getLogger(__name__)

class PluginManager:
    """Discovers, validates, and loads plugins/modules using plugin.json manifests."""
    
    def __init__(self, config: ConfigService, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.plugins_dir = os.path.join(os.getcwd(), "app", "plugins")
        self.modules_dir = os.path.join(os.getcwd(), "app", "modules")
        
        self.registry: Dict[str, ModuleMetadata] = {}
        self.loaded_classes: Dict[str, Any] = {}
        
        os.makedirs(self.plugins_dir, exist_ok=True)
        os.makedirs(self.modules_dir, exist_ok=True)
        
        self.discover_plugins()

    def discover_plugins(self):
        """Scan directories for plugin.json manifests."""
        self.registry.clear()
        
        # Scan built-in modules
        self._scan_directory(self.modules_dir)
        # Scan external plugins
        self._scan_directory(self.plugins_dir)
        
        logger.info(f"Discovered {len(self.registry)} valid plugins/modules.")

    def _scan_directory(self, directory: str):
        if not os.path.exists(directory):
            return
            
        for root, dirs, files in os.walk(directory):
            if "plugin.json" in files:
                manifest_path = os.path.join(root, "plugin.json")
                self._load_manifest(manifest_path, root)

    def _load_manifest(self, filepath: str, root_dir: str):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Add root dir to metadata for dynamic importing later
            data["source_dir"] = root_dir
            meta = ModuleMetadata(**data)
            self.registry[meta.id] = meta
        except Exception as e:
            logger.error(f"Invalid plugin manifest at {filepath}: {e}")

    def load_plugin_class(self, module_id: str) -> Optional[Any]:
        """Dynamically load and return the plugin class based on the entry_point."""
        if module_id in self.loaded_classes:
            return self.loaded_classes[module_id]
            
        meta = self.registry.get(module_id)
        if not meta:
            logger.error(f"Plugin {module_id} not found in registry.")
            return None
            
        source_dir = getattr(meta, "source_dir", "")
        entry_file, class_name = meta.entry_point.split(':')
        
        module_path = os.path.join(source_dir, entry_file)
        if not os.path.exists(module_path):
            logger.error(f"Entry point {module_path} missing for plugin {module_id}")
            return None

        try:
            # source_dir is a real subdirectory of the 'app' package tree
            # (app/modules/... or app/plugins/...), so compute its actual
            # dotted module name and import it normally instead of exec'ing
            # the file in isolation. This lets relative imports inside the
            # plugin file (from .controller import X) resolve correctly via
            # Python's normal import system.
            project_root = os.getcwd()
            rel_dir = os.path.relpath(source_dir, project_root)
            package_name = rel_dir.replace(os.sep, '.')
            module_stem = os.path.splitext(entry_file)[0]
            dotted_module_name = f"{package_name}.{module_stem}"

            module = importlib.import_module(dotted_module_name)
            plugin_class = getattr(module, class_name)

            self.loaded_classes[module_id] = plugin_class
            return plugin_class
        except Exception as e:
            logger.error(f"Failed to load plugin class {module_id}: {e}", exc_info=True)
            return None

    def list_plugins(self) -> List[ModuleMetadata]:
        return list(self.registry.values())
