import logging
import time
import threading
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

        self.trigger_mode = "continuous"  # "continuous" | "single" | "interval"
        self.trigger_interval_seconds = 5.0
        self._pending_trigger = False
        self._last_trigger_time = 0.0
        self._frozen_output_frame = None
        self._trigger_lock = threading.Lock()

    def set_module(self, module: IModule):
        """Set the active computer vision module."""
        self.active_module = module
        with self._trigger_lock:
            self.trigger_mode = "continuous"
            self._pending_trigger = False
            self._last_trigger_time = 0.0
        logger.info(f"Pipeline active module set.")

    def set_trigger_mode(self, mode: str) -> bool:
        if mode not in ("continuous", "single", "interval"):
            return False
        with self._trigger_lock:
            self.trigger_mode = mode
            self._pending_trigger = False
            self._frozen_output_frame = None
            if mode == "interval":
                self._last_trigger_time = time.time()
        logger.info(f"Trigger mode set to: {mode}")
        return True

    def set_trigger_interval(self, seconds: float) -> bool:
        if seconds <= 0:
            return False
        with self._trigger_lock:
            self.trigger_interval_seconds = seconds
        return True

    def fire_trigger(self) -> bool:
        with self._trigger_lock:
            self._pending_trigger = True
        return True

    def get_trigger_status(self) -> dict:
        with self._trigger_lock:
            return {
                "mode": self.trigger_mode,
                "interval_seconds": self.trigger_interval_seconds,
                "pending": self._pending_trigger,
                "last_trigger_time": self._last_trigger_time,
            }

    def _should_process_frame(self) -> bool:
        """Gate for whether process_frame() should run the active module's
        expensive process() this cycle, or just keep the raw video flowing."""
        with self._trigger_lock:
            if self.trigger_mode == "continuous":
                return True

            if self.trigger_mode == "single":
                if self._pending_trigger:
                    self._pending_trigger = False
                    return True
                return False

            if self.trigger_mode == "interval":
                now = time.time()
                if self._pending_trigger:
                    self._pending_trigger = False
                    self._last_trigger_time = now
                    return True
                if (now - self._last_trigger_time) >= self.trigger_interval_seconds:
                    self._last_trigger_time = now
                    return True
                return False

            return True

    def process_frame(self, frame: Any) -> None:
        """Execute the pipeline on a single frame."""
        if not self.active_module:
            self.output_manager.broadcast(frame, {})
            return

        # Give the active module a cheap, mandatory look at the raw frame,
        # independent of trigger gating. Some modules (AI Identify's teach
        # workflow) need a reasonably fresh frame available on demand even
        # when full inference is currently gated off to save compute.
        if hasattr(self.active_module, "on_raw_frame"):
            try:
                self.active_module.on_raw_frame(frame)
            except Exception as e:
                logger.error(f"on_raw_frame hook failed: {e}")

        context = {
            "original_frame": frame,
            "frame": frame.copy() if hasattr(frame, "copy") else frame,
            "metrics": {},
        }

        start_e2e = time.perf_counter()

        if not self._should_process_frame():
            with self._trigger_lock:
                frozen = self._frozen_output_frame
            if self.trigger_mode in ("single", "interval") and frozen is not None:
                # Show the last triggered result, frozen, until the next trigger.
                self.output_manager.broadcast(frozen, {})
            else:
                # No trigger has fired yet this session — keep live video playing.
                self.output_manager.broadcast(frame, {})
            return

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
            if self.trigger_mode in ("single", "interval"):
                with self._trigger_lock:
                    self._frozen_output_frame = rendered_frame.copy() if hasattr(rendered_frame, "copy") else rendered_frame

            self.event_bus.publish("FrameProcessed", context["metrics"])

        except Exception as e:
            logger.error(f"Pipeline error during frame processing: {e}", exc_info=True)
            self.output_manager.broadcast(frame, {})
