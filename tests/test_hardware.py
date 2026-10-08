import pytest
from unittest.mock import MagicMock
import time
from app.core.hardware import HardwareManager
from app.core.state import StateStore
from app.core.config_service import ConfigService

def test_hardware_manager_initialization():
    config = MagicMock(spec=ConfigService)
    settings_mock = MagicMock()
    settings_mock.modbus_enabled = True
    settings_mock.modbus_port = 50202  # Use high port for testing
    config.get_settings.return_value = settings_mock
    
    event_bus = MagicMock()
    state_store = StateStore()
    
    hw = HardwareManager(config, event_bus, state_store)
    
    assert hw.enabled is True
    assert hw.port == 50202
    
    # Just initialize and wait a bit for thread to start
    hw.initialize()
    time.sleep(0.2)
    assert hw._running is True
    assert hw._modbus_thread.is_alive()
    assert hw._sync_thread.is_alive()
    
    # We don't shut it down properly because there's no shutdown method yet, 
    # but the threads are daemon threads so it's fine for the test runner.
    hw._running = False
