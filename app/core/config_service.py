import os
import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class AppSettings(BaseModel):
    """Strongly typed application settings."""
    profile_name: str = "Default"
    data_dir: str = "sessions"
    max_log_size_mb: int = 5
    api_key_mode: str = "Local"  # Local or Secure
    api_key: Optional[str] = None
    cors_origins: str = "*"

class ConfigService:
    """Centralized configuration manager avoiding direct .env reads elsewhere."""
    
    def __init__(self, env_file: str = ".env"):
        self._settings = AppSettings()
        self.env_file = env_file
        self._load_env()

    def _load_env(self) -> None:
        """Load configuration from environment variables and .env file."""
        try:
            from dotenv import load_dotenv
            load_dotenv(self.env_file)
        except ImportError:
            logger.warning("python-dotenv not installed, relying on system environment variables.")

        # Override defaults with env vars
        self._settings = AppSettings(
            profile_name=os.getenv("PI_VISION_PROFILE", "Default"),
            data_dir=os.getenv("PI_VISION_DATA_DIR", "sessions"),
            api_key_mode=os.getenv("PI_VISION_API_MODE", "Local"),
            api_key=os.getenv("PI_VISION_API_KEY"),
            cors_origins=os.getenv("PI_VISION_CORS", "*")
        )
        logger.info(f"Configuration loaded for profile: {self._settings.profile_name}")

    def get_settings(self) -> AppSettings:
        """Return the current application settings."""
        return self._settings

    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """Update settings dynamically at runtime."""
        current_dict = self._settings.model_dump()
        current_dict.update(new_settings)
        self._settings = AppSettings(**current_dict)
        logger.info("Application settings updated at runtime.")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value."""
        return getattr(self._settings, key, default)
