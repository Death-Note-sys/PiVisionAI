import pytest
from unittest.mock import MagicMock
from app.core.coordinator import InspectionCoordinator
from app.core.state import StateStore
from app.core.output_manager import OutputManager

def test_coordinator_handles_ok_result():
    event_bus = MagicMock()
    state_store = StateStore()
    output_manager = MagicMock(spec=OutputManager)
    
    coordinator = InspectionCoordinator(event_bus, state_store, output_manager)
    
    class MockResult:
        match = True
        
    payload = {
        "result": MockResult(),
        "frame": "mock_frame",
        "module_id": "test_module"
    }
    
    coordinator.record(payload["result"], payload["frame"], payload["module_id"], trigger_id=1)
    
    assert state_store.ok_count == 1
    assert state_store.ng_count == 0
    assert state_store.total_count == 1
    
    # Should not save NG frame
    output_manager.save_ng.assert_not_called()

def test_coordinator_handles_ng_result():
    event_bus = MagicMock()
    state_store = StateStore()
    output_manager = MagicMock(spec=OutputManager)
    
    coordinator = InspectionCoordinator(event_bus, state_store, output_manager)
    
    class MockResult:
        def __init__(self):
            self.match = False
            self.foo = "bar"
        
    payload = {
        "result": MockResult(),
        "frame": "mock_frame",
        "module_id": "test_module"
    }
    
    coordinator.record(payload["result"], payload["frame"], payload["module_id"], trigger_id=1)
    
    assert state_store.ok_count == 0
    assert state_store.ng_count == 1
    assert state_store.total_count == 1
    assert state_store.latest_ng_metadata["foo"] == "bar"
    assert state_store.latest_ng_metadata["module_id"] == "test_module"
    
    # Should save NG frame
    from unittest.mock import ANY
    output_manager.save_ng.assert_called_once_with("mock_frame", state_store.latest_ng_metadata, ANY)
    
def test_coordinator_ignores_non_inspection_results():
    event_bus = MagicMock()
    state_store = StateStore()
    output_manager = MagicMock(spec=OutputManager)
    
    coordinator = InspectionCoordinator(event_bus, state_store, output_manager)
    
    class MockResult:
        unrelated = True
        
    coordinator.record(MockResult(), "mock_frame", "test_module", trigger_id=1)
    
    assert state_store.total_count == 0
    output_manager.save_ng.assert_not_called()
