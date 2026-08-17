import logging
from typing import Dict, Any
from app.core.contracts import IService
from .settings import AIIdentifySettings

logger = logging.getLogger(__name__)

class AIIdentifyService(IService):
    def __init__(self, settings: AIIdentifySettings, module_ref=None):
        self.settings = settings
        self.module_ref = module_ref
        self.is_active = False
        self.is_paused = False

    def start(self) -> bool:
        self.is_active = True
        self.is_paused = False
        return True

    def stop(self) -> bool:
        self.is_active = False
        self.is_paused = False
        return True

    def pause(self) -> bool:
        self.is_paused = True
        return True

    def resume(self) -> bool:
        self.is_paused = False
        return True

    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        return self.settings.update(new_settings)

    def teach_good(self, x: int, y: int, w: int, h: int) -> bool:
        if self.module_ref and hasattr(self.module_ref, "teach_good"):
            return self.module_ref.teach_good(x, y, w, h)
        return False

    def teach_bad(self, x: int, y: int, w: int, h: int) -> bool:
        if self.module_ref and hasattr(self.module_ref, "teach_bad"):
            return self.module_ref.teach_bad(x, y, w, h)
        return False

    def remove_good_reference(self, index: int) -> bool:
        if self.module_ref and hasattr(self.module_ref, "remove_good_reference"):
            return self.module_ref.remove_good_reference(index)
        return False

    def remove_bad_reference(self, index: int) -> bool:
        if self.module_ref and hasattr(self.module_ref, "remove_bad_reference"):
            return self.module_ref.remove_bad_reference(index)
        return False

    def reset_teaching(self) -> bool:
        if self.module_ref and hasattr(self.module_ref, "reset_teaching"):
            return self.module_ref.reset_teaching()
        return False

    def get_status(self) -> Dict[str, Any]:
        status = {
            "active": self.is_active,
            "paused": self.is_paused,
            "settings": self.settings.get_settings(),
        }
        if self.module_ref and hasattr(self.module_ref, "last_result"):
            r = self.module_ref.last_result
            status["telemetry"] = {
                "teach_status": r.teach_status,
                "located": r.located,
                "classification": r.classification,
                "good_similarity": r.good_similarity,
                "bad_similarity": r.bad_similarity,
                "match_confidence": r.match_confidence,
                "latency_ms": r.latency_ms,
                "good_reference_count": len(self.module_ref.good_references) if hasattr(self.module_ref, "good_references") else 0,
                "bad_reference_count": len(self.module_ref.bad_references) if hasattr(self.module_ref, "bad_references") else 0,
            }
        return status
