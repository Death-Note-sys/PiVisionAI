import logging
import os
from typing import List, Dict, Any, Optional
from app.core.config_service import ConfigService
from app.core.event_bus import EventBus
from app.core.resource_manager import ResourceManager
from app.core.model_registry import ModelRegistry
from app.core.models.system import AIBackend
from app.core.contracts import IAdapter
from app.core.adapters.yolo_adapter import YoloAdapter

logger = logging.getLogger(__name__)

class AIRuntimeManager:
    """Core AI inference manager using the Adapter pattern."""

    def __init__(self, config: ConfigService, event_bus: EventBus, model_registry: ModelRegistry, resource_manager: ResourceManager):
        self.config = config
        self.event_bus = event_bus
        self.model_registry = model_registry
        self.resource_manager = resource_manager
        
        self.available_backends: List[AIBackend] = []
        self.active_backend: Optional[AIBackend] = None
        
        # Cache for loaded adapters
        self._loaded_adapters: Dict[str, IAdapter] = {}
        
        self._detect_backends()
        self._select_optimal_backend()

    def _detect_backends(self):
        # Simplified for now
        self.available_backends.append(AIBackend(
            name="CPU", provider="CPUExecutionProvider", is_available=True, priority=1
        ))
        if len(self.available_backends) > 0:
            self.active_backend = self.available_backends[0]
            logger.info(f"Selected Active Backend: {self.active_backend.name}")
            self.event_bus.publish("BackendChanged", self.active_backend)

    def _select_optimal_backend(self):
        pass

    def load_model(self, model_id: str) -> bool:
        """Lazy load a model into VRAM/RAM using an Adapter."""
        if model_id in self._loaded_adapters:
            return True
            
        meta = self.model_registry.get_model(model_id)
        if not meta:
            logger.error(f"Cannot load unknown model: {model_id}")
            return False
            
        logger.info(f"Loading model {model_id} via {meta.framework} framework...")
        
        adapter = None
        if meta.framework == "ultralytics":
            adapter = YoloAdapter()
        else:
            logger.error(f"Unsupported framework: {meta.framework}")
            return False
            
        # In a real setup we'd discover the weights file path based on the model ID's directory
        # For this prototype we assume model.pt is in the registry dir
        # If meta has a 'format' and we found a file during registry scan, we pass it.
        # Let's mock the path discovery since the ModelRegistry only tracks metadata.json right now
        model_path = os.path.join(self.model_registry.models_dir, meta.task, meta.name, f"model.{meta.format}")
        if not os.path.exists(model_path):
            # Fallback for built-in models
            model_path = meta.name
            
        success = adapter.load_model(model_path, {})
        if success:
            self._loaded_adapters[model_id] = adapter
            self.event_bus.publish("ModelLoaded", {"model_id": model_id})
            return True
            
        return False

    def unload_model(self, model_id: str):
        """Unload a model to free resources."""
        if model_id in self._loaded_adapters:
            self._loaded_adapters[model_id].cleanup()
            del self._loaded_adapters[model_id]
            self.resource_manager.cleanup()
            self.event_bus.publish("ModelUnloaded", {"model_id": model_id})
            logger.info(f"Unloaded model {model_id}")

    def predict(self, model_id: str, input_data: Any) -> Any:
        """Run inference using the adapter."""
        if model_id not in self._loaded_adapters:
            if not self.load_model(model_id):
                return None
        return self._loaded_adapters[model_id].predict(input_data)

    def shutdown(self):
        """Unload all models and cleanup."""
        for m in list(self._loaded_adapters.keys()):
            self.unload_model(m)
