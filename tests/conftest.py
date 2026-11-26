import pytest
import os
import shutil
import tempfile
import atexit

# CRITICAL: Isolate data directory and disable modbus immediately upon conftest load,
# BEFORE test collection imports any app/api blueprints or Container singletons.
_GLOBAL_TEST_DATA = tempfile.mkdtemp(prefix="pivision_test_data_")
os.environ["PI_VISION_DATA_DIR"] = _GLOBAL_TEST_DATA
os.environ["PI_VISION_MODBUS_ENABLED"] = "false"

def _cleanup():
    shutil.rmtree(_GLOBAL_TEST_DATA, ignore_errors=True)

atexit.register(_cleanup)

# Patch default StateStore data directory
from app.core.state import StateStore
_orig_statestore_init = StateStore.__init__
def _mock_statestore_init(self, *args, **kwargs):
    if not args and "data_dir" not in kwargs and "db_path" not in kwargs:
        kwargs["data_dir"] = os.environ.get("PI_VISION_DATA_DIR", _GLOBAL_TEST_DATA)
    _orig_statestore_init(self, *args, **kwargs)
StateStore.__init__ = _mock_statestore_init

# Patch inspection db path
try:
    from app.api.v1 import inspection
    inspection.get_db_path = lambda: os.path.join(os.environ.get("PI_VISION_DATA_DIR", _GLOBAL_TEST_DATA), "inspections.db")
except ImportError:
    pass

@pytest.fixture(scope="session")
def isolate_data_dir():
    """Session-level fixture for tests that explicitly request isolate_data_dir."""
    return _GLOBAL_TEST_DATA

@pytest.fixture(autouse=True)
def fake_cv2_videocapture(monkeypatch):
    """Autouse fixture ensuring tests never open real cameras or start hardware capture threads."""
    import cv2
    class FakeVideoCapture:
        def __init__(self, *args, **kwargs):
            self._opened = False
        def isOpened(self):
            return False
        def read(self):
            return False, None
        def release(self):
            self._opened = False
        def set(self, *args, **kwargs):
            return False
        def get(self, *args, **kwargs):
            return -1.0
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)

@pytest.fixture(autouse=True)
def clear_container(monkeypatch):
    """Ensure clean container instance and clean thread/camera/output/modbus teardown."""
    from app.core.container import Container
    from app.core.output_manager import OutputManager

    created_outputs = []
    created_containers = []
    orig_om_init = OutputManager.__init__
    orig_container_init = Container.__init__

    def _tracked_om_init(self, *args, **kwargs):
        orig_om_init(self, *args, **kwargs)
        created_outputs.append(self)

    def _tracked_container_init(self, *args, **kwargs):
        orig_container_init(self, *args, **kwargs)
        created_containers.append(self)

    monkeypatch.setattr(OutputManager, "__init__", _tracked_om_init)
    monkeypatch.setattr(Container, "__init__", _tracked_container_init)

    if Container._instance is not None:
        try:
            if hasattr(Container._instance, "camera_manager") and Container._instance.camera_manager:
                Container._instance.camera_manager.stop()
            if hasattr(Container._instance, "output_manager") and Container._instance.output_manager:
                Container._instance.output_manager.shutdown()
            Container._instance.shutdown()
        except Exception:
            pass
    Container._instance = None
    yield
    if Container._instance is not None:
        try:
            if hasattr(Container._instance, "camera_manager") and Container._instance.camera_manager:
                Container._instance.camera_manager.stop()
            if hasattr(Container._instance, "output_manager") and Container._instance.output_manager:
                Container._instance.output_manager.shutdown()
            Container._instance.shutdown()
        except Exception:
            pass
    for om in dict.fromkeys(created_outputs):
        try:
            om.shutdown()
        except Exception:
            pass
    seen_containers = set()
    for c in created_containers:
        if c is not Container._instance and id(c) not in seen_containers:
            seen_containers.add(id(c))
            try:
                c.shutdown()
            except Exception:
                pass
    Container._instance = None

