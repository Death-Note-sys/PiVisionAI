import logging
from typing import Optional, Dict, Any
from .plugin_manager import PluginManager
from .pipeline import Pipeline
from .event_bus import EventBus
from .models.system import ModuleMetadata

logger = logging.getLogger(__name__)

class ModuleController:
    """Manages the active module lifecycle and hooks it into the Pipeline."""
    
    def __init__(self, plugin_manager: PluginManager, pipeline: Pipeline, event_bus: EventBus, ai_runtime: Any = None, model_registry: Any = None):
        self.plugin_manager = plugin_manager
        self.pipeline = pipeline
        self.event_bus = event_bus
        self.ai_runtime = ai_runtime
        self.model_registry = model_registry
        
        self.active_module_instance: Optional[Any] = None
        self.active_service_instance: Optional[Any] = None
        self.active_metadata: Optional[ModuleMetadata] = None

    def switch_module(self, module_id: str) -> bool:
        """Switch the active computer vision module."""
        # Clean up existing module
        if self.active_module_instance:
            self._unload_active()

        # Get metadata
        meta = self.plugin_manager.registry.get(module_id)
        if not meta:
            logger.error(f"Cannot switch to unknown module {module_id}")
            return False

        # Load class
        plugin_class = self.plugin_manager.load_plugin_class(module_id)
        if not plugin_class:
            return False

        try:
            # Instantiate the plugin factory.
            plugin_factory = plugin_class(self.event_bus, self.ai_runtime, self.model_registry)
            
            # Create module and service.
            self.active_module_instance = plugin_factory.create_module()
            if hasattr(plugin_factory, 'create_service'):
                self.active_service_instance = plugin_factory.create_service()
            else:
                self.active_service_instance = None
                
            if hasattr(self.active_module_instance, 'initialize'):
                self.active_module_instance.initialize()
            
            # If the module has an 'id' attribute, set it for the pipeline
            if not hasattr(self.active_module_instance, 'id'):
                self.active_module_instance.id = module_id

            self.active_metadata = meta
            
            # Hook into Pipeline
            self.pipeline.set_module(self.active_module_instance)
            if getattr(meta, "renderer", None):
                self.pipeline.renderer_manager.switch_renderer(meta.renderer)
            
            self.event_bus.publish("ModuleLoaded", {"module_id": module_id, "name": meta.name})
            logger.info(f"Successfully switched to module: {meta.name} ({module_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to instantiate module {module_id}: {e}", exc_info=True)
            return False

    def _unload_active(self):
        """Cleanly unload the current module."""
        if not self.active_module_instance:
            return
            
        old_id = self.active_metadata.id if self.active_metadata else "unknown"
        
        if hasattr(self.active_module_instance, 'cleanup'):
            try:
                self.active_module_instance.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up module {old_id}: {e}")
                
        self.pipeline.set_module(None)
        self.pipeline.renderer_manager.active_renderer = None
        self.active_module_instance = None
        self.active_service_instance = None
        self.active_metadata = None
        
        self.event_bus.publish("ModuleUnloaded", {"module_id": old_id})

    def get_active_service(self) -> Optional[Any]:
        return self.active_service_instance

    def update_settings(self, settings: Dict[str, Any]) -> bool:
        """Pass new settings to the active module."""
        if self.active_module_instance and hasattr(self.active_module_instance, 'configure'):
            try:
                self.active_module_instance.configure(settings)
                return True
            except Exception as e:
                logger.error(f"Error updating module settings: {e}")
        return False
