import numpy as np
from typing import Dict, Any
from app.core.contracts import IRenderer

class BaseRenderer(IRenderer):
    """Base class for all module result renderers."""
    
    def initialize(self) -> bool:
        return True
        
    def render(self, frame: np.ndarray, result: Any, metadata: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError
