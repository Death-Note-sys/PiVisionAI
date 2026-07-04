from app.core.config_service import ConfigService
from app.core.event_bus import EventBus
from app.core.plugin_manager import PluginManager

try:
    config = ConfigService()
    event_bus = EventBus()
    pm = PluginManager(config, event_bus)
    
    print("Plugins:", pm.registry.keys())
    
    cls = pm.load_plugin_class("core-ocr-scanner")
    print("Class:", cls)
    
    if cls:
        inst = cls(event_bus=event_bus)
        print("Instantiated successfully:", inst)
except Exception as e:
    import traceback
    traceback.print_exc()
