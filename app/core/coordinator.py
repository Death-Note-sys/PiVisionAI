import logging
from app.core.event_bus import EventBus
from app.core.state import StateStore
from app.core.output_manager import OutputManager

import sqlite3
import time
import os
import json
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

class InspectionCoordinator:
    """Central orchestrator that resolves OK/NG status from module results."""
    
    def __init__(self, event_bus: EventBus, state_store: StateStore, output_manager: OutputManager):
        self.event_bus = event_bus
        self.state_store = state_store
        self.output_manager = output_manager
        self._db_lock = threading.Lock()
        
    def _get_db_path(self):
        from app.core.container import Container
        container = Container.get_instance()
        if container and container.config:
            data_dir = container.config.get("data_dir", "data")
        else:
            data_dir = os.environ.get("PI_VISION_DATA_DIR", "data")
        return os.path.join(data_dir, "inspections.db")

    def _init_db(self):
        db_path = self._get_db_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with self._db_lock:
            with sqlite3.connect(db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS inspections(
                        id INTEGER PRIMARY KEY, ts REAL, trigger_id INTEGER,
                        module_id TEXT, judgment TEXT, code INTEGER, reason TEXT,
                        values_json TEXT, raw_path TEXT, overlay_path TEXT
                    )
                ''')
                conn.execute('CREATE INDEX IF NOT EXISTS ix_ts ON inspections(ts)')
                
    def _enforce_cap(self, conn, cap: int):
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, raw_path, overlay_path FROM inspections 
            WHERE raw_path IS NOT NULL 
            ORDER BY id DESC LIMIT -1 OFFSET ?
        ''', (cap,))
        rows = cursor.fetchall()
        for row in rows:
            rid, raw_path, overlay_path = row
            if raw_path and os.path.exists(raw_path):
                try: os.remove(raw_path)
                except: pass
            if overlay_path and os.path.exists(overlay_path):
                try: os.remove(overlay_path)
                except: pass
            conn.execute('UPDATE inspections SET raw_path = NULL, overlay_path = NULL WHERE id = ?', (rid,))

    def record(self, result: Any, frame: Any, module_id: str, trigger_id: int = 0):
        """Record and persist an inspection judgment (called strictly on triggered inspections)."""
        if not result:
            return
            
        # Determine if it's an inspectable result
        is_ok = None
        status = 0
        is_uncertain = False

        if hasattr(result, "match"):
            if result.match is not None:
                is_ok = bool(result.match)
                status = 1 if is_ok else 2
        elif hasattr(result, "classification"):
            if result.classification in ("Good", "Bad", "Uncertain"):
                is_ok = (result.classification == "Good")
                if result.classification == "Good":
                    status = 1
                elif result.classification == "Bad":
                    status = 2
                elif result.classification == "Uncertain":
                    status = 3
                    is_uncertain = True
        elif hasattr(result, "is_ok"):
            if result.is_ok is not None:
                is_ok = bool(result.is_ok)
                status = 1 if is_ok else 2
            
        if is_ok is None:
            # Not an inspectable result or undetermined
            return
            
        self.state_store.update_status(status)
        
        # Serialize metadata
        meta = {"module_id": module_id}
        if hasattr(result, "model_dump"):
            meta.update(result.model_dump())
        elif hasattr(result, "__dict__"):
            for k, v in result.__dict__.items():
                if isinstance(v, (int, float, str, bool, list, dict)):
                    meta[k] = v

        raw_path = None
        # Schema rule: judgment column is OK or NG only; Uncertain is stored as NG with code 3.
        judgment_str = "OK" if is_ok else "NG"

        if is_ok:
            self.state_store.increment_ok(module_id=module_id)
            logger.info(f"Coordinator: Inspection OK (module={module_id}, trigger={trigger_id})")
        else:
            self.state_store.increment_ng(metadata=meta, module_id=module_id, is_uncertain=is_uncertain)
            # Save frame for NG inspections
            if frame is not None:
                try:
                    from app.core.container import Container
                    container = Container.get_instance()
                    out_dir = container.session_manager.get_data_dir("ng_images") if container and hasattr(container, "session_manager") else os.path.join(os.environ.get("PI_VISION_DATA_DIR", "data"), "ng_images")
                    from datetime import datetime
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    raw_path = os.path.join(out_dir, f"{module_id}_{ts_str}.jpg")
                    self.output_manager.save_ng(frame, meta, raw_path)
                except Exception as e:
                    logger.error(f"Failed to save NG image: {e}")
            logger.info(f"Coordinator: Inspection NG (module={module_id}, trigger={trigger_id}, code={status})")

        # Persist every triggered inspection row to SQLite (OK rows have raw_path=None)
        try:
            self._init_db()
            with self._db_lock:
                with sqlite3.connect(self._get_db_path()) as conn:
                    conn.execute('''
                        INSERT INTO inspections(ts, trigger_id, module_id, judgment, code, reason, values_json, raw_path, overlay_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        time.time(), trigger_id, module_id, 
                        judgment_str, status, meta.get("reason", ""), json.dumps(meta), raw_path, None
                    ))
                    
                    if raw_path:
                        from app.core.container import Container
                        container = Container.get_instance()
                        cap = int(container.config.get("ng_image_cap", 2000)) if container and hasattr(container, "config") else 2000
                        self._enforce_cap(conn, cap)
        except Exception as e:
            logger.error(f"Failed to log inspection to DB: {e}")


