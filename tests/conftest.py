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
def clear_container():
    """Ensure clean container instance and clean thread/camera/modbus teardown."""
    from app.core.container import Container
    if Container._instance is not None:
        try:
            if hasattr(Container._instance, "camera_manager") and Container._instance.camera_manager:
                Container._instance.camera_manager.stop()
            Container._instance.shutdown()
        except Exception:
            pass
    Container._instance = None
    yield
    if Container._instance is not None:
        try:
            if hasattr(Container._instance, "camera_manager") and Container._instance.camera_manager:
                Container._instance.camera_manager.stop()
            Container._instance.shutdown()
        except Exception:
            pass
    Container._instance = None

