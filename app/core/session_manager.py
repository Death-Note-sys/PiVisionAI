import os
import uuid
import json
import logging
from datetime import datetime
from typing import Optional
from .config_service import ConfigService
from .event_bus import EventBus
from .models.system import SessionMetadata

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages unique workspaces for every application run."""
    
    def __init__(self, config: ConfigService, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.session_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        self.base_dir = self.config.get("data_dir", "sessions")
        self.session_dir = os.path.join(self.base_dir, self.session_id)
        
        self.metadata = SessionMetadata(
            session_id=self.session_id,
            start_time=self.start_time.isoformat()
        )
        
        self._initialize_workspace()
        self.event_bus.subscribe("ApplicationShutdown", self._on_shutdown)

    def _initialize_workspace(self):
        """Create directory structure for the current session."""
        dirs = ['captures', 'recordings', 'exports', 'logs']
        for d in dirs:
            os.makedirs(os.path.join(self.session_dir, d), exist_ok=True)
            
        self._save_metadata()
        logger.info(f"Initialized new session: {self.session_id}")

    def _save_metadata(self):
        """Persist session metadata to disk."""
        meta_path = os.path.join(self.session_dir, "metadata.json")
        try:
            with open(meta_path, 'w') as f:
                f.write(self.metadata.model_dump_json(indent=4))
        except Exception as e:
            logger.error(f"Failed to save session metadata: {e}")

    def _on_shutdown(self, _):
        """Handle application shutdown event."""
        self.metadata.end_time = datetime.now().isoformat()
        self._save_metadata()
        logger.info(f"Session {self.session_id} saved and closed.")

    def get_session_dir(self, subfolder: Optional[str] = None) -> str:
        """Get the absolute path to the session directory or a specific subfolder."""
        if subfolder:
            path = os.path.join(self.session_dir, subfolder)
            os.makedirs(path, exist_ok=True)
            return path
        return self.session_dir
