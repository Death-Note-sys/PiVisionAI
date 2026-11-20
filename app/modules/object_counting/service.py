import logging
from typing import Dict, Any, Optional
from app.core.contracts import IService
from .settings import ObjectCountingSettings

logger = logging.getLogger(__name__)


class ObjectCountingService(IService):
    def __init__(self, settings: ObjectCountingSettings, module_ref=None):
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
        if self.module_ref and hasattr(self.module_ref, "configure"):
            return self.module_ref.configure(new_settings)
        return self.settings.update(new_settings)

    def teach_template(self, x: int, y: int, w: int, h: int) -> tuple:
        if self.module_ref and hasattr(self.module_ref, "teach_template"):
            ok = self.module_ref.teach_template(x, y, w, h)
            reason = getattr(self.module_ref, "last_teach_error", "") or ""
            return ok, reason
        return False, "Module not initialized"

    def reset_template(self) -> bool:
        if self.module_ref and hasattr(self.module_ref, "reset_template"):
            return self.module_ref.reset_template()
        return False

    def get_template_path(self) -> Optional[str]:
        if self.module_ref and hasattr(self.module_ref, "get_template_path"):
            return self.module_ref.get_template_path()
        return None

    def get_status(self) -> Dict[str, Any]:
        tpl_status = "Untaught"
        if self.module_ref and hasattr(self.module_ref, "template_status"):
            tpl_status = self.module_ref.template_status
        elif self.module_ref and hasattr(self.module_ref, "last_result"):
            tpl_status = getattr(self.module_ref.last_result, "template_status", "Untaught")

        status = {
            "active": self.is_active,
            "paused": self.is_paused,
            "settings": self.settings.get_settings(),
            "telemetry": {
                "template_status": tpl_status,
                "count": 0,
                "matches": [],
                "is_ok": None,
                "latency_ms": 0.0,
            },
        }
        if self.module_ref and hasattr(self.module_ref, "last_result"):
            r = self.module_ref.last_result
            status["telemetry"].update({
                "template_status": tpl_status,
                "count": r.count,
                "matches": r.matches,
                "is_ok": r.is_ok,
                "latency_ms": r.latency_ms,
            })
        return status
