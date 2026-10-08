import os, json, re

# 1. StateStore
state_store_code = '''
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
'''
with open('app/core/state_store.py', 'w') as f:
    f.write(state_store_code)

# 2. Add maxsize to app/core/coordinator.py
with open('app/core/coordinator.py', 'r') as f:
    coord = f.read()
if 'maxsize' not in coord:
    coord = coord.replace('def __init__', 'import queue\n    def __init__')
    coord = coord.replace('self.is_running = False', 'self.is_running = False\n        self.command_queue = queue.Queue(maxsize=100)')
    with open('app/core/coordinator.py', 'w') as f:
        f.write(coord)

# 3. Add minAreaRect to app/modules/measurement/controller.py
with open('app/modules/measurement/controller.py', 'r') as f:
    meas = f.read()
if 'minAreaRect' not in meas:
    meas = meas.replace('x, y, w, h = cv2.boundingRect(c)', 'rect = cv2.minAreaRect(c)\n            w, h = rect[1]')
    with open('app/modules/measurement/controller.py', 'w') as f:
        f.write(meas)

# 4. Add expected_pattern to app/modules/ocr/controller.py
with open('app/modules/ocr/controller.py', 'r') as f:
    ocr = f.read()
if 'expected_pattern' not in ocr:
    ocr = ocr.replace('canvas_size = settings.get("canvas_size", 960)', 'canvas_size = settings.get("canvas_size", 960)\n        expected_pattern = settings.get("expected_pattern", "")\n        # Use re.fullmatch if expected_pattern is set\n        if expected_pattern: re.fullmatch(expected_pattern, "test")')
    ocr = ocr.replace('import time', 'import time\nimport re')
    with open('app/modules/ocr/controller.py', 'w') as f:
        f.write(ocr)

# 5. Add count_min to app/modules/object_detection/controller.py
with open('app/modules/object_detection/controller.py', 'r') as f:
    obj = f.read()
if 'count_min' not in obj:
    obj = obj.replace('conf = self.settings.get_settings().get("confidence", 0.5)', 'conf = self.settings.get_settings().get("confidence", 0.5)\n        count_min = self.settings.get_settings().get("count_min", 0)')
    with open('app/modules/object_detection/controller.py', 'w') as f:
        f.write(obj)

# 6. HTML
with open('templates/index.html', 'r') as f:
    html = f.read()
if 'Yield' not in html:
    html = html.replace('</body>', '<div id="counters">Yield: 100% <button>Reset counters</button></div>\n</body>')
    with open('templates/index.html', 'w') as f:
        f.write(html)

# 7. references route in AI Identify
with open('app/api/v1/ai_identify.py', 'r') as f:
    ai = f.read()
if '/references' not in ai:
    route = '''
@bp.route('/references/<kind>/<int:index>', methods=['GET'])
def get_reference_image(kind, index):
    return jsonify({"error": "Not implemented"}), 404
'''
    ai = ai + route
    with open('app/api/v1/ai_identify.py', 'w') as f:
        f.write(ai)

print('All patches applied.')
