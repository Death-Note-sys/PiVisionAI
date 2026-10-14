"""
Automated Modbus integration test.

Starts a real Modbus TCP server on a free port and uses a real
ModbusTcpClient to verify:
  - HR9 write -> HardwareTrigger published, HR9 auto-cleared
  - Coil 0 write -> HardwareTrigger published, coil auto-cleared
  - Heartbeat advances
  - Counter words are big-endian
  - HR9 is never overwritten by the sync loop (50 consecutive triggers, none lost)
  - Module codes match plan
  - Per-module status (HR1) starts at 0 (Idle) and updates after inspection
  - Last Trigger ID (HR11) updates on triggered inspections
"""

import pytest
import time
import threading
from unittest.mock import MagicMock, patch

from pymodbus.client import ModbusTcpClient

from app.core.hardware import HardwareManager, MODULE_CODE_MAP
from app.core.state import StateStore
from app.core.event_bus import EventBus
from app.core.config_service import ConfigService


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def modbus_fixture():
    """Start a real HardwareManager with Modbus on a free port."""
    port = _find_free_port()

    config = MagicMock(spec=ConfigService)
    settings_mock = MagicMock()
    settings_mock.modbus_enabled = True
    settings_mock.modbus_port = port
    settings_mock.modbus_host = "127.0.0.1"   # must be a real string; MagicMock attr breaks bind
    config.get_settings.return_value = settings_mock

    event_bus = EventBus()
    state_store = StateStore()

    hw = HardwareManager(config, event_bus, state_store)

    # Patch Container.get_instance so the sync loop doesn't crash.
    # Container is lazily imported inside _sync_loop from app.core.container,
    # so we patch it at its origin module.
    mock_container = MagicMock()
    mock_container.module_controller = None
    with patch("app.core.container.Container.get_instance", return_value=mock_container):
        hw.initialize()
        # Wait for server to be ready
        time.sleep(0.5)

        client = ModbusTcpClient("127.0.0.1", port=port)
        assert client.connect(), f"Could not connect to Modbus on port {port}"

        yield {
            "hw": hw,
            "client": client,
            "event_bus": event_bus,
            "state_store": state_store,
            "mock_container": mock_container,
            "port": port,
        }

        client.close()
        hw.stop()


class TestModbusHR9Trigger:
    """HR9 write -> HardwareTrigger published and HR9 cleared."""

    def test_hr9_trigger(self, modbus_fixture):
        client = modbus_fixture["client"]
        event_bus = modbus_fixture["event_bus"]

        triggers_received = []
        event_bus.subscribe("HardwareTrigger", lambda _: triggers_received.append(1))

        # Write 1 to HR9
        result = client.write_register(9, 1)
        assert not result.isError(), f"HR9 write failed: {result}"

        # Wait for sync loop to consume it
        time.sleep(0.3)

        assert len(triggers_received) == 1, f"Expected 1 trigger, got {len(triggers_received)}"

        # HR9 should be auto-reset to 0
        rr = client.read_holding_registers(9, 1)
        assert not rr.isError()
        assert rr.registers[0] == 0, f"HR9 not reset: {rr.registers[0]}"


class TestModbusCoilTrigger:
    """Coil 0 write -> HardwareTrigger published and coil 0 cleared."""

    def test_coil_trigger(self, modbus_fixture):
        client = modbus_fixture["client"]
        event_bus = modbus_fixture["event_bus"]

        triggers_received = []
        event_bus.subscribe("HardwareTrigger", lambda _: triggers_received.append(1))

        # Write True to Coil 0
        result = client.write_coil(0, True)
        assert not result.isError(), f"Coil write failed: {result}"

        time.sleep(0.3)

        assert len(triggers_received) == 1
        cr = client.read_coils(0, 1)
        assert not cr.isError()
        assert cr.bits[0] == False, "Coil 0 not cleared"


class TestHeartbeat:
    """Heartbeat (HR8) should advance over time."""

    def test_heartbeat_advances(self, modbus_fixture):
        client = modbus_fixture["client"]

        rr1 = client.read_holding_registers(8, 1)
        assert not rr1.isError()
        hb1 = rr1.registers[0]

        # Heartbeat increments every ~1 second (10 * 100ms loops)
        time.sleep(1.5)

        rr2 = client.read_holding_registers(8, 1)
        assert not rr2.isError()
        hb2 = rr2.registers[0]

        assert hb2 > hb1, f"Heartbeat did not advance: {hb1} -> {hb2}"


class TestCountersBigEndian:
    """Counter words are stored big-endian (HR2=high, HR3=low)."""

    def test_counters_big_endian(self, modbus_fixture):
        state_store = modbus_fixture["state_store"]
        client = modbus_fixture["client"]

        # Set a known value: 0x0001_0002 = 65538
        state_store.module_counters["test-mod"] = {
            "total_count": 65538, "ok_count": 65537, "ng_count": 1, "uncertain_count": 0
        }
        state_store.active_module_id = "test-mod"

        # Need to set active module on the hardware manager too
        mock_meta = MagicMock()
        mock_meta.id = "test-mod"
        modbus_fixture["mock_container"].module_controller = MagicMock()
        modbus_fixture["mock_container"].module_controller.active_metadata = mock_meta

        time.sleep(0.3)  # Wait for sync

        rr = client.read_holding_registers(2, 6)
        assert not rr.isError()
        total_high, total_low = rr.registers[0], rr.registers[1]
        ok_high, ok_low = rr.registers[2], rr.registers[3]
        ng_high, ng_low = rr.registers[4], rr.registers[5]

        total = (total_high << 16) | total_low
        ok = (ok_high << 16) | ok_low
        ng = (ng_high << 16) | ng_low

        assert total == 65538, f"Total: expected 65538, got {total}"
        assert ok == 65537, f"OK: expected 65537, got {ok}"
        assert ng == 1, f"NG: expected 1, got {ng}"


class TestHR9NotOverwritten:
    """50 consecutive triggers via HR9: none should be lost.
    
    The sync loop writes HR0-HR8 and HR10-HR11 but skips HR9,
    so a PLC write to HR9 between sync iterations is preserved.
    """

    def test_50_triggers_none_lost(self, modbus_fixture):
        client = modbus_fixture["client"]
        event_bus = modbus_fixture["event_bus"]

        triggers_received = []
        event_bus.subscribe("HardwareTrigger", lambda _: triggers_received.append(1))

        for i in range(50):
            result = client.write_register(9, 1)
            assert not result.isError(), f"HR9 write #{i} failed"
            # Wait enough for sync loop to consume (100ms poll + margin)
            time.sleep(0.15)

        # Final wait for stragglers
        time.sleep(0.5)

        assert len(triggers_received) == 50, \
            f"Lost triggers: expected 50, got {len(triggers_received)}"


class TestModuleCodes:
    """Module codes match the documented table in docs/modbus_map.md.

    This test uses fixed literals (the documented truth), NOT the code's own
    constants, so that a drift between the doc table and the server will fail here.
    """

    # Fixed literals from the 'Module Codes' table in docs/modbus_map.md
    DOC_MODULE_CODES = {
        "core-object-detection": 1,
        "core-measurement":      2,
        "core-ai-identify":      3,
        "core-ocr":              4,
        "core-color-analysis":   5,
    }

    def test_module_code_literals(self):
        """Server MODULE_CODE_MAP must match every code in the documented table."""
        for module_id, expected_code in self.DOC_MODULE_CODES.items():
            assert MODULE_CODE_MAP.get(module_id) == expected_code, (
                f"Module {module_id!r}: doc says {expected_code}, "
                f"server has {MODULE_CODE_MAP.get(module_id)!r}"
            )

    def test_module_codes_match_doc(self):
        """Parse docs/modbus_map.md Module Codes table and verify server agrees."""
        import re, pathlib
        doc_path = pathlib.Path(__file__).parent.parent / "docs" / "modbus_map.md"
        text = doc_path.read_text(encoding="utf-8")

        # Extract the Module Codes table rows: | code | name |
        # Rows look like:  | 1 | Object Detection |
        rows = re.findall(r"^\|\s*(\d+)\s*\|\s*(.+?)\s*\|", text, re.MULTILINE)
        # Filter only numeric-code rows (not header rows)
        doc_codes = {}
        name_to_id = {
            "Object Detection": "core-object-detection",
            "Measurement":      "core-measurement",
            "AI Identify":      "core-ai-identify",
            "OCR":              "core-ocr",
            "Color Analysis":   "core-color-analysis",
        }
        for code_str, name in rows:
            mid = name_to_id.get(name.strip())
            if mid:
                doc_codes[mid] = int(code_str)

        assert doc_codes, "Could not parse Module Codes table from modbus_map.md"

        for module_id, doc_code in doc_codes.items():
            server_code = MODULE_CODE_MAP.get(module_id)
            assert server_code == doc_code, (
                f"Module {module_id!r}: modbus_map.md says {doc_code}, "
                f"MODULE_CODE_MAP has {server_code!r}"
            )


    def test_active_module_register(self, modbus_fixture):
        client = modbus_fixture["client"]

        # Set active module to AI Identify (code 3)
        mock_meta = MagicMock()
        mock_meta.id = "core-ai-identify"
        modbus_fixture["mock_container"].module_controller = MagicMock()
        modbus_fixture["mock_container"].module_controller.active_metadata = mock_meta

        time.sleep(0.3)

        rr = client.read_holding_registers(0, 1)
        assert not rr.isError()
        assert rr.registers[0] == 3, f"Active Module ID: expected 3, got {rr.registers[0]}"


class TestPerModuleStatus:
    """HR1 should be 0 (Idle) until the active module has an inspection."""

    def test_status_idle_until_inspection(self, modbus_fixture):
        client = modbus_fixture["client"]
        hw = modbus_fixture["hw"]

        # Set active module
        mock_meta = MagicMock()
        mock_meta.id = "core-color-analysis"
        modbus_fixture["mock_container"].module_controller = MagicMock()
        modbus_fixture["mock_container"].module_controller.active_metadata = mock_meta

        time.sleep(0.3)

        # Status should be 0 (Idle) - no inspection yet
        rr = client.read_holding_registers(1, 1)
        assert not rr.isError()
        assert rr.registers[0] == 0, f"Expected Idle (0), got {rr.registers[0]}"

        # Simulate a FrameProcessed event with OK result for color
        result_mock = MagicMock()
        result_mock.match = True  # OK
        del result_mock.classification
        del result_mock.is_ok
        hw._on_frame_processed({
            "module_id": "core-color-analysis",
            "triggered": True,
            "trigger_id": 1,
            "result": result_mock,
        })

        time.sleep(0.3)

        rr = client.read_holding_registers(1, 1)
        assert not rr.isError()
        assert rr.registers[0] == 1, f"Expected OK (1), got {rr.registers[0]}"

    def test_status_per_module_isolation(self, modbus_fixture):
        """Switching modules should show each module's own last status."""
        client = modbus_fixture["client"]
        hw = modbus_fixture["hw"]

        # Color got OK
        result_ok = MagicMock()
        result_ok.match = True
        del result_ok.classification
        del result_ok.is_ok
        hw._on_frame_processed({
            "module_id": "core-color-analysis",
            "triggered": True,
            "trigger_id": 1,
            "result": result_ok,
        })

        # Measurement has NO inspections yet
        mock_meta = MagicMock()
        mock_meta.id = "core-measurement"
        modbus_fixture["mock_container"].module_controller = MagicMock()
        modbus_fixture["mock_container"].module_controller.active_metadata = mock_meta

        time.sleep(0.3)

        rr = client.read_holding_registers(1, 1)
        assert not rr.isError()
        assert rr.registers[0] == 0, f"Measurement should be Idle (0), got {rr.registers[0]}"

        # Switch back to color
        mock_meta.id = "core-color-analysis"
        time.sleep(0.3)

        rr = client.read_holding_registers(1, 1)
        assert not rr.isError()
        assert rr.registers[0] == 1, f"Color should still be OK (1), got {rr.registers[0]}"


class TestLastTriggerID:
    """HR11 updates on triggered inspections."""

    def test_last_trigger_id(self, modbus_fixture):
        client = modbus_fixture["client"]
        hw = modbus_fixture["hw"]

        # Initial should be 0
        rr = client.read_holding_registers(11, 1)
        assert not rr.isError()
        assert rr.registers[0] == 0, f"Initial trigger ID should be 0, got {rr.registers[0]}"

        # Simulate triggered inspection with trigger_id=42
        result_mock = MagicMock()
        result_mock.match = True
        del result_mock.classification
        del result_mock.is_ok
        hw._on_frame_processed({
            "module_id": "core-ai-identify",
            "triggered": True,
            "trigger_id": 42,
            "result": result_mock,
        })

        time.sleep(0.3)

        rr = client.read_holding_registers(11, 1)
        assert not rr.isError()
        assert rr.registers[0] == 42, f"Last trigger ID: expected 42, got {rr.registers[0]}"


class TestStatusAgeRegister:
    """HR12 = Status Age ms: 65535 before any inspection, small value after."""

    def test_status_age_starts_at_never(self, modbus_fixture):
        """HR12 should be 65535 (never) immediately after server start."""
        client = modbus_fixture["client"]
        time.sleep(0.3)  # one sync cycle
        rr = client.read_holding_registers(12, 1)
        assert not rr.isError()
        assert rr.registers[0] == 65535, (
            f"HR12 should be 65535 (never updated), got {rr.registers[0]}"
        )

    def test_status_age_updates_after_inspection(self, modbus_fixture):
        """HR12 should be a small value (< 2000 ms) just after a triggered inspection."""
        client = modbus_fixture["client"]
        hw = modbus_fixture["hw"]

        result_mock = MagicMock()
        result_mock.match = True
        del result_mock.classification
        del result_mock.is_ok
        hw._on_frame_processed({
            "module_id": "core-color-analysis",
            "triggered": True,
            "trigger_id": 7,
            "result": result_mock,
        })

        time.sleep(0.3)  # let sync loop write HR12

        rr = client.read_holding_registers(12, 1)
        assert not rr.isError()
        age_ms = rr.registers[0]
        assert age_ms < 2000, (
            f"HR12 status age should be < 2000 ms after fresh inspection, got {age_ms}"
        )
        assert age_ms != 65535, "HR12 should not be 65535 after an inspection"
