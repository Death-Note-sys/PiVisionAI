import logging
from app.core.event_bus import EventBus
from app.core.state import StateStore
from app.core.output_manager import OutputManager

logger = logging.getLogger(__name__)

class InspectionCoordinator:
    """Central orchestrator that resolves OK/NG status from module results."""
    
    def __init__(self, event_bus: EventBus, state_store: StateStore, output_manager: OutputManager):
        self.event_bus = event_bus
        self.state_store = state_store
        self.output_manager = output_manager
        
        self.event_bus.subscribe("FrameProcessed", self._on_frame_processed)
        
    def _on_frame_processed(self, payload: dict):
        result = payload.get("result")
        if not result:
            return
            
        # Determine if it's an inspection result
        is_ok = None
        if hasattr(result, "match"):
            if result.match is not None:
                is_ok = result.match
        elif hasattr(result, "is_ok"):
            is_ok = result.is_ok
            
        if is_ok is None:
            # Not an inspectable result or undetermined
            return
            
        if is_ok:
            self.state_store.increment_ok()
            logger.info("Coordinator: Inspection OK")
        else:
            module_id = payload.get("module_id", "unknown")
            
            # Serialize result attributes for metadata
            meta = {"module_id": module_id}
            if hasattr(result, "model_dump"):
                meta.update(result.model_dump())
            elif hasattr(result, "__dict__"):
                for k, v in result.__dict__.items():
                    if isinstance(v, (int, float, str, bool, list, dict)):
                        meta[k] = v
                        
            self.state_store.increment_ng(meta)
            
            # Bundle frame and save to NG queue
            frame = payload.get("frame")
            if frame is not None:
                self.output_manager.save_ng(frame, meta)
                
            logger.info("Coordinator: Inspection NG")
