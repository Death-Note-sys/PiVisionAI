import os
import json
import time
import shutil
import threading
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

import sqlite3

class StateStore:
    def __init__(self, data_dir: Optional[str] = None, persist: bool = False, db_path: Optional[str] = None):
        self._lock = threading.RLock()
        self.persist = persist
        self.db_path = db_path
        
        if db_path:
            self.data_dir = os.path.dirname(db_path)
            os.makedirs(self.data_dir, exist_ok=True)
            self._init_sqlite()
        else:
            if data_dir is None:
                data_dir = os.environ.get("PI_VISION_DATA_DIR", "data")
            self.data_dir = data_dir

        self.counters_file = os.path.join(self.data_dir, "counters.json") if self.persist else None
        
        # Per-Module Counters: {module_id: {"total_count": 0, "ok_count": 0, "ng_count": 0, "uncertain_count": 0}}
        self.module_counters: Dict[str, Dict[str, int]] = {}
        self.active_module_id = "default"
        
        # System State
        self.error_state = 0
        self.latest_status = 0 # 0=Idle, 1=OK, 2=NG, 3=Uncertain
        self.latest_ng_metadata: Dict[str, Any] = {}
        
        # Module Runtime States
        self.modules: Dict[str, Dict[str, Any]] = {}
        
        self._load_counters()

    def _init_sqlite(self):
        if self.db_path:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS global_state (key TEXT PRIMARY KEY, value TEXT)")
                conn.commit()

    @property
    def total_count(self) -> int:
        with self._lock:
            return sum(m.get("total_count", 0) for m in self.module_counters.values())

    @total_count.setter
    def total_count(self, val: int):
        with self._lock:
            self._ensure_module_counters(self.active_module_id)["total_count"] = val

    @property
    def ok_count(self) -> int:
        with self._lock:
            return sum(m.get("ok_count", 0) for m in self.module_counters.values())

    @ok_count.setter
    def ok_count(self, val: int):
        with self._lock:
            self._ensure_module_counters(self.active_module_id)["ok_count"] = val

    @property
    def ng_count(self) -> int:
        with self._lock:
            return sum(m.get("ng_count", 0) for m in self.module_counters.values())

    @ng_count.setter
    def ng_count(self, val: int):
        with self._lock:
            self._ensure_module_counters(self.active_module_id)["ng_count"] = val

    def _ensure_module_counters(self, module_id: str) -> Dict[str, int]:
        m_id = module_id or "default"
        if m_id not in self.module_counters:
            self.module_counters[m_id] = {
                "total_count": 0,
                "ok_count": 0,
                "ng_count": 0,
                "uncertain_count": 0
            }
        return self.module_counters[m_id]

    def _load_counters(self):
        if self.persist and self.counters_file and os.path.exists(self.counters_file):
            try:
                with open(self.counters_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                if "modules" in data and isinstance(data["modules"], dict):
                    self.module_counters = data["modules"]
                else:
                    # Legacy flat format migration
                    self.module_counters = {
                        "default": {
                            "total_count": data.get("total_count", 0),
                            "ok_count": data.get("ok_count", 0),
                            "ng_count": data.get("ng_count", 0),
                            "uncertain_count": data.get("uncertain_count", 0)
                        }
                    }
            except Exception as e:
                logger.error(f"Failed to load counters from {self.counters_file}: {e}")
                try:
                    corrupt_backup = self.counters_file + f".corrupt_{int(time.time())}"
                    shutil.copy2(self.counters_file, corrupt_backup)
                    logger.warning(f"Backed up corrupt counters to {corrupt_backup}")
                except Exception:
                    pass

    def _save_counters(self):
        if self.persist and self.counters_file:
            try:
                os.makedirs(os.path.dirname(self.counters_file), exist_ok=True)
                tmp_file = self.counters_file + ".tmp"
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "total_count": self.total_count,
                        "ok_count": self.ok_count,
                        "ng_count": self.ng_count,
                        "modules": self.module_counters
                    }, f, indent=4)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_file, self.counters_file)
            except Exception as e:
                logger.error(f"Failed to save counters: {e}")

    def increment_ok(self, module_id: Optional[str] = None):
        with self._lock:
            m = self._ensure_module_counters(module_id or self.active_module_id)
            m["ok_count"] += 1
            m["total_count"] += 1
            self._save_counters()
            
    def increment_ng(self, metadata: Optional[Dict[str, Any]] = None, module_id: Optional[str] = None, is_uncertain: bool = False):
        with self._lock:
            target_mod = module_id or (metadata.get("module_id") if metadata else None) or self.active_module_id
            m = self._ensure_module_counters(target_mod)
            m["ng_count"] += 1
            m["total_count"] += 1
            if is_uncertain:
                m["uncertain_count"] = m.get("uncertain_count", 0) + 1
            if metadata:
                self.latest_ng_metadata = metadata
            self._save_counters()
                
    def reset_counters(self, module_id: Optional[str] = None):
        with self._lock:
            if module_id:
                m = self._ensure_module_counters(module_id)
                m["total_count"] = 0
                m["ok_count"] = 0
                m["ng_count"] = 0
                m["uncertain_count"] = 0
            else:
                self.module_counters.clear()
            self.latest_ng_metadata = {}
            self._save_counters()

    def get_counters(self, module_id: Optional[str] = None) -> Dict[str, int]:
        with self._lock:
            m = self._ensure_module_counters(module_id or self.active_module_id)
            return dict(m)
            
    def set_error_state(self, error_code: int):
        with self._lock:
            self.error_state = error_code
            
    def update_status(self, status: int):
        with self._lock:
            self.latest_status = status
            
    def update(self, module_id: str, patch: Dict[str, Any]):
        """Read-merge-write under a lock for per-module state."""
        with self._lock:
            if self.db_path:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("CREATE TABLE IF NOT EXISTS global_state (key TEXT PRIMARY KEY, value TEXT)")
                    conn.execute("INSERT OR REPLACE INTO global_state (key, value) VALUES (?, ?)", (module_id, json.dumps(patch)))
                    conn.commit()
            if module_id not in self.modules:
                self.modules[module_id] = {}
            if isinstance(patch, dict):
                self.modules[module_id].update(patch)
            else:
                self.modules[module_id] = patch

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if self.db_path and os.path.exists(self.db_path):
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.execute("SELECT value FROM global_state WHERE key = ?", (key,))
                    row = cur.fetchone()
                    if row:
                        return json.loads(row[0])
            return self.modules.get(key)
            
    def get_module_state(self, module_id: str) -> Dict[str, Any]:
        with self._lock:
            res = self.get(module_id)
            return dict(res) if isinstance(res, dict) else {}
            
    def get_global_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_count": self.total_count,
                "ok_count": self.ok_count,
                "ng_count": self.ng_count,
                "error_state": self.error_state,
                "latest_status": self.latest_status,
                "latest_ng_metadata": self.latest_ng_metadata,
                "modules": dict(self.module_counters)
            }
