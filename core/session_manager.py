import json
import os
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, data_dir):
        self.session_file = os.path.join(data_dir, 'session.json')
        self.state = {
            "active_module": None,
            "camera_index": 0,
            "camera_resolution": "640x480",
            "module_settings": {}
        }
        self.load()

    def load(self):
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                    self.state.update(data)
                logger.info(f"Session loaded from {self.session_file}")
            except Exception as e:
                logger.error(f"Failed to load session: {e}")

    def save(self):
        try:
            with open(self.session_file, 'w') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def update(self, **kwargs):
        modified = False
        for k, v in kwargs.items():
            if k in self.state:
                # Basic dict merging for module_settings
                if k == "module_settings" and isinstance(v, dict):
                    if self.state[k] != v:
                        self.state[k].update(v)
                        modified = True
                elif self.state[k] != v:
                    self.state[k] = v
                    modified = True
        if modified:
            self.save()
            
    def get(self, key, default=None):
        return self.state.get(key, default)
