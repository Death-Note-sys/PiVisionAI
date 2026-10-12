import pytest
import os
import threading
from app.core.config_service import ConfigService

def test_config_service_loads_defaults(monkeypatch):
    monkeypatch.delenv("PI_VISION_PROFILE", raising=False)
    monkeypatch.delenv("PI_VISION_DATA_DIR", raising=False)
    
    config = ConfigService(env_file=".env.nonexistent")
    settings = config.get_settings()
    
    assert settings.profile_name == "Default"
    assert settings.data_dir == "data"
    assert settings.camera_index == 0
    assert settings.modbus_enabled is True
    assert settings.modbus_port == 5020

def test_config_service_loads_from_env(monkeypatch, tmp_path):
    custom_dir = str(tmp_path / "custom_test_data")
    monkeypatch.setenv("PI_VISION_PROFILE", "TestProfile")
    monkeypatch.setenv("PI_VISION_DATA_DIR", custom_dir)
    monkeypatch.setenv("PI_VISION_CAMERA_INDEX", "1")
    monkeypatch.setenv("PI_VISION_MODBUS_PORT", "502")
    
    config = ConfigService(env_file=".env.nonexistent")
    settings = config.get_settings()
    
    assert settings.profile_name == "TestProfile"
    assert settings.data_dir == custom_dir
    assert settings.camera_index == 1
    assert settings.modbus_port == 502
