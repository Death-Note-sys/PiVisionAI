import threading
from typing import Dict, Any

class StateStore:
    def __init__(self):
        self._lock = threading.Lock()
        
        # Global Counters
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0
        
        # System State
        self.error_state = 0
        self.latest_status = 0 # 0=Idle, 1=OK, 2=NG, 3=Uncertain
        self.latest_ng_metadata: Dict[str, Any] = {}
        
        # Module Runtime States
        self.modules: Dict[str, Dict[str, Any]] = {}
        
    def increment_ok(self):
        with self._lock:
            self.ok_count += 1
            self.total_count += 1
            
    def increment_ng(self, metadata: Dict[str, Any] = None):
        with self._lock:
            self.ng_count += 1
            self.total_count += 1
            if metadata:
                self.latest_ng_metadata = metadata
                
    def reset_counters(self):
        with self._lock:
            self.total_count = 0
            self.ok_count = 0
            self.ng_count = 0
            self.latest_ng_metadata = {}
            
    def set_error_state(self, error_code: int):
        with self._lock:
            self.error_state = error_code
            
    def update_status(self, status: int):
        with self._lock:
            self.latest_status = status
            
    def update(self, module_id: str, patch: Dict[str, Any]):
        """Read-merge-write under a lock for per-module state."""
        with self._lock:
            if module_id not in self.modules:
                self.modules[module_id] = {}
            self.modules[module_id].update(patch)
            
    def get_module_state(self, module_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self.modules.get(module_id, {}))
            
    def get_global_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_count": self.total_count,
                "ok_count": self.ok_count,
                "ng_count": self.ng_count,
                "error_state": self.error_state,
                "latest_status": self.latest_status,
                "latest_ng_metadata": self.latest_ng_metadata
            }
