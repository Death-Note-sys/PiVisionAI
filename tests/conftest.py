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

# Patch default StateStore path
from app.core.state_store import StateStore
_orig_statestore_init = StateStore.__init__
def _mock_statestore_init(self, db_path=None):
    if db_path is None or db_path == "app/data/pivision.db":
        db_path = os.path.join(os.environ.get("PI_VISION_DATA_DIR", _GLOBAL_TEST_DATA), "pivision.db")
    _orig_statestore_init(self, db_path)
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
def clear_container():
    """Ensure clean container instance and clean thread/modbus teardown."""
    from app.core.container import Container
    if Container._instance is not None:
        try:
            Container._instance.shutdown()
        except Exception:
            pass
    Container._instance = None
    yield
    if Container._instance is not None:
        try:
            Container._instance.shutdown()
        except Exception:
            pass
    Container._instance = None
