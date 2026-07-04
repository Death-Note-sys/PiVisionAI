import logging
from typing import Optional

from .config_service import ConfigService
from .event_bus import EventBus
from .session_manager import SessionManager
from .resource_manager import ResourceManager
from .performance_monitor import PerformanceMonitor
from .task_scheduler import TaskScheduler
from .camera_manager import CameraManager
from .ai_runtime import AIRuntimeManager
from .model_registry import ModelRegistry
from .pipeline import Pipeline
from .plugin_manager import PluginManager
from .module_controller import ModuleController
from .renderer_manager import RendererManager
from .output_manager import OutputManager

logger = logging.getLogger(__name__)

class Container:
    """
    Lightweight Dependency Injection Container.
    Initializes and manages lifecycle of all core singleton services.
    """
    
    _instance: Optional['Container'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Container, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        logger.info("Initializing Dependency Container...")
        
        # Initialize fundamental services first
        self.config = ConfigService()
        self.event_bus = EventBus()
        
        # Initialize Core Infrastructure
        self.resource_manager = ResourceManager(self.event_bus)
        self.performance_monitor = PerformanceMonitor(self.event_bus)
        self.session_manager = SessionManager(self.config, self.event_bus)
        self.task_scheduler = TaskScheduler(self.event_bus)
        
        # Initialize AI & Vision Engine
        self.model_registry = ModelRegistry(self.config, self.event_bus)
        self.ai_runtime = AIRuntimeManager(self.config, self.event_bus, self.model_registry, self.resource_manager)
        
        self.renderer_manager = RendererManager()
        self.output_manager = OutputManager()
        self.pipeline = Pipeline(self.event_bus, self.renderer_manager, self.output_manager)
        
        self.camera_manager = CameraManager(self.config, self.event_bus, self.pipeline, self.performance_monitor)
        
        # Initialize Plugin & Module Management
        self.plugin_manager = PluginManager(self.config, self.event_bus)
        self.module_controller = ModuleController(
            self.plugin_manager, self.pipeline, self.event_bus,
            self.ai_runtime, self.model_registry
        )

        logger.info("Dependency Container initialized successfully.")

    @classmethod
    def get_instance(cls) -> 'Container':
        if cls._instance is None:
            return Container()
        return cls._instance

    def shutdown(self):
        """Cleanly shutdown all services."""
        logger.info("Shutting down services...")
        if hasattr(self, 'task_scheduler'): self.task_scheduler.shutdown()
        if hasattr(self, 'camera_manager'): self.camera_manager.stop()
        if hasattr(self, 'ai_runtime'): self.ai_runtime.shutdown()
        if hasattr(self, 'resource_manager'): self.resource_manager.shutdown()
        logger.info("Shutdown complete.")
