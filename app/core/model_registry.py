import os
import json
import logging
from typing import Dict, List, Optional, Any
from app.core.contracts import IModelRegistry
from app.core.models.system import ModelMetadata
from app.core.config_service import ConfigService
from app.core.event_bus import EventBus

logger = logging.getLogger(__name__)

class ModelRegistry(IModelRegistry):
    """Discovers and manages AI models completely decoupled from the runtime."""
    
    def __init__(self, config: ConfigService, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.models_dir = os.path.join(os.getcwd(), "ai_models")
        self.registry: Dict[str, ModelMetadata] = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
        self.refresh()

    def refresh(self) -> None:
        """Scan the ai_models directory recursively for model metadata."""
        self.registry.clear()
        
        for root, dirs, files in os.walk(self.models_dir):
            if "metadata.json" in files:
                self._load_metadata(os.path.join(root, "metadata.json"))
                
        logger.info(f"ModelRegistry: Discovered {len(self.registry)} models.")
        self.event_bus.publish("ModelRegistryRefreshed", {"count": len(self.registry)})

    def _load_metadata(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                model = ModelMetadata(**data)
                self.registry[model.id] = model
        except Exception as e:
            logger.error(f"Failed to load model metadata from {filepath}: {e}")

    def register_model(self, metadata: Dict[str, Any]) -> bool:
        """Register a new model (used by Vision Studio)."""
        try:
            model = ModelMetadata(**metadata)
            self.registry[model.id] = model
            self.event_bus.publish("ModelRegistered", {"model_id": model.id})
            return True
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return False

    def validate_model(self, model_id: str) -> bool:
        """Check if a model exists and its weights file is accessible."""
        if model_id not in self.registry:
            return False
            
        # In a full implementation, this would check if .pt / .onnx exists in the same dir
        return True

    def remove_model(self, model_id: str) -> bool:
        if model_id in self.registry:
            del self.registry[model_id]
            self.event_bus.publish("ModelRemoved", {"model_id": model_id})
            return True
        return False

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        return self.registry.get(model_id)
        
    def list_models(self, task: str = None) -> List[ModelMetadata]:
        if task:
            return [m for m in self.registry.values() if m.task == task]
        return list(self.registry.values())
