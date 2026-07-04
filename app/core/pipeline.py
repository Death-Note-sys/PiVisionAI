import logging
import time
from typing import Any, Dict, Optional
from app.core.event_bus import EventBus
from app.core.contracts import IModule, IPipelineStage
from app.core.renderer_manager import RendererManager
from app.core.output_manager import OutputManager

logger = logging.getLogger(__name__)

class PreprocessingStage(IPipelineStage):
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Global preprocessing like resizing or normalization
        return context

class PostprocessingStage(IPipelineStage):
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Global postprocessing like tracking association or filtering
        return context

class Pipeline:
    """
    Decoupled processing engine.
    Stages: Preprocess -> Module -> Postprocess -> Render -> Output.
    """
    
    def __init__(self, event_bus: EventBus, renderer_manager: RendererManager, output_manager: OutputManager):
        self.event_bus = event_bus
        self.renderer_manager = renderer_manager
        self.output_manager = output_manager
        
        self.active_module: Optional[IModule] = None
        self.preprocess_stage = PreprocessingStage()
        self.postprocess_stage = PostprocessingStage()

    def set_module(self, module: IModule):
        """Set the active computer vision module."""
        self.active_module = module
        logger.info(f"Pipeline active module set.")

    def process_frame(self, frame: Any) -> None:
        """Execute the pipeline on a single frame."""
        if not self.active_module:
            self.output_manager.broadcast(frame, {})
            return

        context = {
            "original_frame": frame,
            "frame": frame.copy() if hasattr(frame, "copy") else frame,
            "metrics": {},
        }
        
        start_e2e = time.perf_counter()

        try:
            # 1. Pre-processing
            context = self.preprocess_stage.execute(context)

            # 2. Module Execution (Business Logic)
            start_mod = time.perf_counter()
            result = self.active_module.process(context)
            context["metrics"]["module_ms"] = (time.perf_counter() - start_mod) * 1000

            # 3. Post-processing
            context = self.postprocess_stage.execute(context)

            # 4. Rendering (Separated from Module)
            start_rend = time.perf_counter()
            render_instructions = self.active_module.render(result) if hasattr(self.active_module, 'render') else result
            rendered_frame = self.renderer_manager.render(context["frame"], render_instructions, {"settings": {}})
            context["metrics"]["render_ms"] = (time.perf_counter() - start_rend) * 1000

            # 5. Output Management
            context["metrics"]["e2e_ms"] = (time.perf_counter() - start_e2e) * 1000
            self.output_manager.broadcast(rendered_frame, context["metrics"])
            
            self.event_bus.publish("FrameProcessed", context["metrics"])

        except Exception as e:
            logger.error(f"Pipeline error during frame processing: {e}", exc_info=True)
            self.output_manager.broadcast(frame, {})
