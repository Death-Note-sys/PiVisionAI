import os
import json
import logging

logger = logging.getLogger(__name__)

class LocalizationManager:
    def __init__(self, data_dir):
        self.locales_dir = os.path.join(data_dir, 'locales')
        os.makedirs(self.locales_dir, exist_ok=True)
        
        self.current_lang = 'en'
        self.translations = {}
        
        # Ensure default english exists
        self._ensure_default_en()
        self.load_language('en')

    def _ensure_default_en(self):
        en_path = os.path.join(self.locales_dir, 'en.json')
        if not os.path.exists(en_path):
            en_data = {
                "dashboard": "Dashboard",
                "recordings": "Recordings",
                "settings": "Settings",
                "media_gallery": "Media Gallery",
                "performance": "Performance",
                "analytics": "Analytics",
                "system_online": "System Online",
                "system_offline": "System Offline",
                "live_stream": "Live Stream",
                "edge_processing": "Edge processing dashboard",
                "primary_feed": "Primary Feed",
                "active_modules": "Active Modules",
                "module_settings": "Module Settings"
            }
            try:
                with open(en_path, 'w', encoding='utf-8') as f:
                    json.dump(en_data, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to write default EN locale: {e}")

    def load_language(self, lang_code):
        path = os.path.join(self.locales_dir, f'{lang_code}.json')
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.translations = json.load(f)
                self.current_lang = lang_code
                return True
            except Exception as e:
                logger.error(f"Failed to load locale {lang_code}: {e}")
        return False
        
    def get_translations(self):
        return self.translations
