import pytest
from unittest.mock import MagicMock, patch
import time
import socket
from app.core.hardware import HardwareManager, MODULE_CODE_MAP
from app.core.state import StateStore
from app.core.config_service import ConfigService


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_hardware_manager_initialization():
    port = _find_free_port()
    config = MagicMock(spec=ConfigService)
    settings_mock = MagicMock()
    settings_mock.modbus_enabled = True
    settings_mock.modbus_port = port
    settings_mock.modbus_host = "127.0.0.1"
    config.get_settings.return_value = settings_mock
    
    event_bus = MagicMock()
    state_store = StateStore()
    
    hw = HardwareManager(config, event_bus, state_store)
    
    assert hw.enabled is True
    assert hw.port == port
    
    hw.initialize()
    time.sleep(0.2)
    assert hw._running is True
    assert hw._modbus_thread.is_alive()
    assert hw._sync_thread.is_alive()
    
    hw.stop()
    assert hw._running is False


def test_hardware_manager_disabled():
    config = MagicMock(spec=ConfigService)
    settings_mock = MagicMock()
    settings_mock.modbus_enabled = False
    config.get_settings.return_value = settings_mock
    
    event_bus = MagicMock()
    state_store = StateStore()
    
    hw = HardwareManager(config, event_bus, state_store)
    assert hw.initialize() is True
    assert hw._running is False
    assert hw._modbus_thread is None
    assert hw._sync_thread is None


def test_hardware_frame_processed_event_handling():
    config = MagicMock(spec=ConfigService)
    settings_mock = MagicMock()
    settings_mock.modbus_enabled = False
    config.get_settings.return_value = settings_mock
    
    event_bus = MagicMock()
    state_store = StateStore()
    
    hw = HardwareManager(config, event_bus, state_store)
    
    # 1. Classification result (AI Identify) Good -> status 1
    res_good = MagicMock()
    res_good.classification = "Good"
    del res_good.match
    del res_good.is_ok
    hw._on_frame_processed({
        "module_id": "core-ai-identify",
        "triggered": True,
        "trigger_id": 101,
        "result": res_good
    })
    assert hw._module_last_status.get("core-ai-identify") == 1
    assert hw._last_trigger_id == 101
    assert hw._last_status_ts > 0

    # 2. Match result (Color Analysis) NG -> status 2
    res_ng = MagicMock()
    res_ng.match = False
    del res_ng.classification
    del res_ng.is_ok
    hw._on_frame_processed({
        "module_id": "core-color-analysis",
        "triggered": True,
        "trigger_id": 102,
        "result": res_ng
    })
    assert hw._module_last_status.get("core-color-analysis") == 2
    assert hw._last_trigger_id == 102
    # Ensure per-module isolation: AI Identify status unchanged
    assert hw._module_last_status.get("core-ai-identify") == 1

    # 3. Non-triggered frames do not update trigger ID or status
    hw._on_frame_processed({
        "module_id": "core-ai-identify",
        "triggered": False,
        "trigger_id": 0,
        "result": res_good
    })
    assert hw._last_trigger_id == 102

    # 4. is_ok True (Object Counting) -> status 1
    res_counting = MagicMock()
    res_counting.is_ok = True
    del res_counting.match
    del res_counting.classification
    hw._on_frame_processed({
        "module_id": "core-object-counting",
        "triggered": True,
        "trigger_id": 103,
        "result": res_counting
    })
    assert hw._module_last_status.get("core-object-counting") == 1
    assert hw._last_trigger_id == 103

    # 5. is_ok False (Target Extraction) -> status 2
    res_extraction = MagicMock()
    res_extraction.is_ok = False
    del res_extraction.match
    del res_extraction.classification
    hw._on_frame_processed({
        "module_id": "core-target-extraction",
        "triggered": True,
        "trigger_id": 104,
        "result": res_extraction
    })
    assert hw._module_last_status.get("core-target-extraction") == 2
    assert hw._last_trigger_id == 104
