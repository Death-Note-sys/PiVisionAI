import re
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

from app.core.camera_manager import CameraManager
from app.services.camera_service import CameraService


def test_index_html_has_single_camera_change_listener():
    """Camera index and resolution dropdowns must each attach only one change listener."""
    html = Path("templates/index.html").read_text(encoding="utf-8")
    assert "switchCameraHardware" not in html

    idx_listeners = re.findall(
        r"cameraIndexSelect['\"]?\)?\.addEventListener\(\s*['\"]change['\"]",
        html,
    )
    res_listeners = re.findall(
        r"cameraResolutionSelect['\"]?\)?\.addEventListener\(\s*['\"]change['\"]",
        html,
    )
    assert len(idx_listeners) == 1
    assert len(res_listeners) == 1


def test_camera_manager_stop_is_idempotent_and_clears_cap():
    """Calling stop() multiple times releases VideoCapture at most once and clears self.cap."""
    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: d
    cm = CameraManager(cfg, MagicMock(), MagicMock(), MagicMock())
    mock_cap = MagicMock()
    cm.cap = mock_cap
    cm.is_connected = True

    cm.stop()
    cm.stop()

    assert mock_cap.release.call_count == 1
    assert cm.cap is None
    assert cm.is_connected is False


def test_camera_manager_stop_waits_for_cap_lock_before_releasing():
    """stop() must acquire _cap_lock so it never releases cap concurrently with cap.read()."""
    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: d
    cm = CameraManager(cfg, MagicMock(), MagicMock(), MagicMock())
    mock_cap = MagicMock()
    cm.cap = mock_cap
    cm.is_connected = True

    stop_done = threading.Event()
    cm._cap_lock.acquire()
    try:
        t = threading.Thread(target=lambda: (cm.stop(), stop_done.set()))
        t.start()
        time.sleep(0.05)
        # While _cap_lock is held (e.g. inside read()), stop() must not call release().
        assert mock_cap.release.call_count == 0
        assert not stop_done.is_set()
    finally:
        cm._cap_lock.release()

    assert stop_done.wait(timeout=2.0)
    assert mock_cap.release.call_count == 1
    assert cm.cap is None


def test_camera_manager_apply_settings_handles_concurrent_stop():
    """apply_settings() must not crash if stop() clears self.cap before _cap_lock is acquired."""
    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: d
    cm = CameraManager(cfg, MagicMock(), MagicMock(), MagicMock())
    mock_cap = MagicMock()
    cm.cap = mock_cap
    cm.is_connected = True

    result = {}
    errors = []

    def worker():
        try:
            result["out"] = cm.apply_settings({"gain": 12.0})
        except Exception as exc:
            errors.append(exc)

    cm._cap_lock.acquire()
    try:
        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.03)
        cm.cap = None
        cm.is_connected = False
    finally:
        cm._cap_lock.release()

    t.join(timeout=2.0)
    assert not errors
    assert result["out"] == {"gain": 12.0}
    assert mock_cap.set.call_count == 0


def test_camera_service_serializes_concurrent_switches():
    """Concurrent switch_camera() calls must execute sequentially without overlapping."""
    svc = CameraService()
    mock_mgr = MagicMock()
    active_switches = 0
    max_concurrent = 0
    counter_lock = threading.Lock()
    entered_first = threading.Event()

    def slow_stop():
        nonlocal active_switches, max_concurrent
        with counter_lock:
            active_switches += 1
            max_concurrent = max(max_concurrent, active_switches)
        entered_first.set()
        time.sleep(0.06)

    def slow_start():
        nonlocal active_switches
        time.sleep(0.02)
        with counter_lock:
            active_switches -= 1

    mock_mgr.stop.side_effect = slow_stop
    mock_mgr.start.side_effect = slow_start
    svc.camera_manager = mock_mgr
    svc.container = MagicMock()

    results = []

    def run_switch(res):
        results.append(svc.switch_camera(0, res))

    t1 = threading.Thread(target=run_switch, args=("1280x720",))
    t1.start()
    assert entered_first.wait(timeout=2.0)

    t2 = threading.Thread(target=run_switch, args=("640x480",))
    t2.start()

    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    assert results == [True, True]
    assert max_concurrent == 1


def test_camera_service_rejects_invalid_resolution_without_stopping():
    """Invalid resolution should fail before stopping the active camera."""
    svc = CameraService()
    mock_mgr = MagicMock()
    svc.camera_manager = mock_mgr

    assert svc.switch_camera(0, "invalid") is False
    mock_mgr.stop.assert_not_called()
    mock_mgr.start.assert_not_called()
