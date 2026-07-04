from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

class IModule(ABC):
    """Lifecycle contract for all computer vision modules."""
    @abstractmethod
    def initialize(self) -> bool:
        pass
        
    @abstractmethod
    def configure(self, settings: Dict[str, Any]) -> bool:
        pass
        
    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Any:
        pass
        
    @abstractmethod
    def render(self, result: Any) -> Any:
        """Returns rendering instructions, does NOT draw directly."""
        pass
        
    @abstractmethod
    def cleanup(self) -> None:
        pass
        
    @abstractmethod
    def health_check(self) -> bool:
        pass

class IRenderer(ABC):
    """Contract for visualization logic."""
    @abstractmethod
    def initialize(self) -> bool:
        pass
        
    @abstractmethod
    def render(self, frame: np.ndarray, result: Any, metadata: Dict[str, Any]) -> np.ndarray:
        pass

class IAdapter(ABC):
    """Contract for AI Framework abstractions."""
    @abstractmethod
    def initialize(self) -> bool:
        pass
        
    @abstractmethod
    def load_model(self, model_path: str, parameters: Dict[str, Any]) -> bool:
        pass
        
    @abstractmethod
    def warmup(self) -> bool:
        pass
        
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        pass
        
    @abstractmethod
    def unload(self) -> None:
        pass
        
    @abstractmethod
    def benchmark(self) -> Dict[str, float]:
        pass
        
    @abstractmethod
    def cleanup(self) -> None:
        pass

class IPlugin(ABC):
    """Contract for Plugin factories."""
    @abstractmethod
    def create_module(self) -> IModule:
        pass

class IService(ABC):
    """Contract for business logic REST services."""
    @abstractmethod
    def start(self) -> bool:
        pass
        
    @abstractmethod
    def stop(self) -> bool:
        pass
        
    @abstractmethod
    def pause(self) -> bool:
        pass
        
    @abstractmethod
    def resume(self) -> bool:
        pass
        
    @abstractmethod
    def update_settings(self, settings: Dict[str, Any]) -> bool:
        pass

class IPipelineStage(ABC):
    """Contract for Pipeline components."""
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pass

class IOutputManager(ABC):
    """Contract for handling output destinations."""
    @abstractmethod
    def broadcast(self, frame: np.ndarray, metadata: Dict[str, Any]) -> None:
        pass
        
    @abstractmethod
    def add_client(self, client_id: str, client_type: str) -> None:
        pass
        
    @abstractmethod
    def remove_client(self, client_id: str) -> None:
        pass

class IModelRegistry(ABC):
    """Contract for managing model metadata."""
    @abstractmethod
    def refresh(self) -> None:
        pass
        
    @abstractmethod
    def register_model(self, metadata: Dict[str, Any]) -> bool:
        pass
        
    @abstractmethod
    def validate_model(self, model_id: str) -> bool:
        pass
        
    @abstractmethod
    def remove_model(self, model_id: str) -> bool:
        pass
        
    @abstractmethod
    def get_model(self, model_id: str) -> Optional[Any]:
        pass

class IWorkspace(ABC):
    """Contract for Frontend UI components definition."""
    @abstractmethod
    def get_layout(self) -> Dict[str, Any]:
        pass

class ISettingsProvider(ABC):
    """Contract for strongly-typed settings access."""
    @abstractmethod
    def get_settings(self) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def update(self, new_settings: Dict[str, Any]) -> bool:
        pass
