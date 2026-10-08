
import sqlite3
import json
import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class StateStore:
    def __init__(self, db_path: str = "app/data/pivision.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS global_state (key TEXT PRIMARY KEY, value TEXT)")
                conn.execute("CREATE TABLE IF NOT EXISTS module_settings (module_id TEXT PRIMARY KEY, settings TEXT)")
                conn.commit()

    def update(self, key: str, value: Any):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO global_state (key, value) VALUES (?, ?)", (key, json.dumps(value)))
                conn.commit()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("SELECT value FROM global_state WHERE key = ?", (key,))
                row = cur.fetchone()
                if row:
                    return json.loads(row[0])
                return None
