import logging
from typing import Dict, Any, Optional
from app.core.contracts import IRenderer
from app.core.renderers.detection_renderer import DetectionRenderer

logger = logging.getLogger(__name__)

class RendererManager:
    """Manages lifecycles of renderers and delegates drawing."""
    
    def __init__(self):
        self.renderers: Dict[str, IRenderer] = {}
        self.active_renderer: Optional[IRenderer] = None
        self._register_default_renderers()
        
    def _register_default_renderers(self):
        self.renderers["detection"] = DetectionRenderer()
        # Other renderers can be registered dynamically
        
    def switch_renderer(self, renderer_name: str) -> bool:
        if renderer_name in self.renderers:
            self.active_renderer = self.renderers[renderer_name]
            logger.info(f"Switched active renderer to {renderer_name}")
            return True
        logger.warning(f"Renderer {renderer_name} not found.")
        self.active_renderer = None
        return False
        
    def render(self, frame: Any, result: Any, metadata: Dict[str, Any]) -> Any:
        if self.active_renderer is None or result is None:
            return frame
        try:
            return self.active_renderer.render(frame, result, metadata)
        except Exception as e:
            logger.error(f"Renderer error: {e}")
            return frame
