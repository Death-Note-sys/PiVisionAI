import logging
from app.core.event_bus import EventBus
from app.core.state import StateStore
from app.core.output_manager import OutputManager

import sqlite3
import time
import os
import json
import threading

logger = logging.getLogger(__name__)

class InspectionCoordinator:
    """Central orchestrator that resolves OK/NG status from module results."""
    
    import queue
    def __init__(self, event_bus: EventBus, state_store: StateStore, output_manager: OutputManager):
        self.event_bus = event_bus
        import queue
        self.command_queue = queue.Queue(maxsize=100)
        self.state_store = state_store
        self.output_manager = output_manager
        self._db_lock = threading.Lock()
        
        self.event_bus.subscribe("FrameProcessed", self._on_frame_processed)
        
    def _get_db_path(self):
        from app.core.container import Container
        container = Container.get_instance()
        if container and container.config:
            data_dir = container.config.get("data_dir", "data")
        else:
            data_dir = "data"
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
        # Find rows with raw_path where id not in top 'cap'
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
        
    def _on_frame_processed(self, payload: dict):
        result = payload.get("result")
        if not result:
            return
            
        # Determine if it's an inspection result
        is_ok = None
        status = 0
        if hasattr(result, "match"):
            if result.match is not None:
                is_ok = result.match
                status = 1 if is_ok else 2
        elif hasattr(result, "classification"):
            if result.classification in ("Good", "Bad", "Uncertain"):
                is_ok = result.classification == "Good"
                if result.classification == "Good": status = 1
                elif result.classification == "Bad": status = 2
                elif result.classification == "Uncertain": status = 3
        elif hasattr(result, "is_ok"):
            is_ok = result.is_ok
            status = 1 if is_ok else 2
            
        if is_ok is None:
            # Not an inspectable result or undetermined
            return
            
        self.state_store.update_status(status)
            
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
            
            judgment_str = "Uncertain" if status == 3 else "NG"
            raw_path = None
            
            # Bundle frame and save to NG queue
            frame = payload.get("frame")
            if frame is not None:
                from app.core.container import Container
                container = Container.get_instance()
                from datetime import datetime
                out_dir = container.session_manager.get_data_dir("ng_images")
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                raw_path = os.path.join(out_dir, f"{module_id}_{ts_str}.jpg")
                self.output_manager.save_ng(frame, meta, raw_path)
                
            # Log to SQLite
            self._init_db()
            with self._db_lock:
                try:
                    with sqlite3.connect(self._get_db_path()) as conn:
                        conn.execute('''
                            INSERT INTO inspections(ts, trigger_id, module_id, judgment, code, reason, values_json, raw_path, overlay_path)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            time.time(), getattr(result, 'trigger_id', 0), module_id, 
                            judgment_str, status, meta.get("reason", ""), json.dumps(meta), raw_path, None
                        ))
                        
                        from app.core.container import Container
                        container = Container.get_instance()
                        cap = int(container.config.get("ng_image_cap", 2000)) if container else 2000
                        self._enforce_cap(conn, cap)
                except Exception as e:
                    logger.error(f"Failed to log inspection to DB: {e}")
                    
            logger.info("Coordinator: Inspection NG")
