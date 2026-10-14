import os
import json
import numpy as np
import pytest
import cv2
from unittest.mock import MagicMock

from app.core.event_bus import EventBus
from app.core.config_service import ConfigService
from app.core.state import StateStore
from app.core.camera_manager import CameraManager

from app.modules.ocr.controller import OCRController
from app.modules.ocr.settings import OCRSettings
from app.modules.color_analysis.controller import ColorAnalysisController
from app.modules.color_analysis.settings import ColorAnalysisSettings
from app.modules.ai_identify.controller import AIIdentifyController
from app.modules.ai_identify.settings import AIIdentifySettings
from app.modules.object_detection.controller import ObjectDetectionController
from app.modules.object_detection.settings import ObjectDetectionSettings
from app.modules.measurement.controller import MeasurementController
from app.modules.measurement.settings import MeasurementSettings

def make_texture(size=200, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (size, size), dtype=np.uint8)
    blurred = cv2.GaussianBlur(noise, (5, 5), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

def test_per_module_settings_survive_new_controller_instance():
    bus = EventBus()
    ai_runtime = MagicMock()

    # 1. OCR
    c_ocr1 = OCRController(bus, ai_runtime, OCRSettings())
    c_ocr1.configure({"min_confidence": 0.85, "expected_pattern": "^ID-[0-9]{4}$"})
    c_ocr2 = OCRController(bus, ai_runtime, OCRSettings())
    assert c_ocr2.settings.get_settings()["min_confidence"] == pytest.approx(0.85)
    assert c_ocr2.settings.get_settings()["expected_pattern"] == "^ID-[0-9]{4}$"

    # 2. Color Analysis
    c_col1 = ColorAnalysisController(bus, ColorAnalysisSettings())
    c_col1.configure({"delta_e_threshold": 18.5, "num_colors": 4})
    c_col2 = ColorAnalysisController(bus, ColorAnalysisSettings())
    assert c_col2.settings.get_settings()["delta_e_threshold"] == pytest.approx(18.5)
    assert c_col2.settings.get_settings()["num_colors"] == 4

    # 3. AI Identify
    c_ai1 = AIIdentifyController(bus, AIIdentifySettings())
    c_ai1.configure({"min_match_count": 15, "match_ratio_threshold": 0.68})
    c_ai2 = AIIdentifyController(bus, AIIdentifySettings())
    assert c_ai2.settings.get_settings()["min_match_count"] == 15
    assert c_ai2.settings.get_settings()["match_ratio_threshold"] == pytest.approx(0.68)

    # 4. Object Detection
    c_od1 = ObjectDetectionController(bus, ai_runtime, ObjectDetectionSettings())
    c_od1.configure({"confidence": 0.72, "count_min": 2, "count_max": 5})
    c_od2 = ObjectDetectionController(bus, ai_runtime, ObjectDetectionSettings())
    assert c_od2.settings.get_settings()["confidence"] == pytest.approx(0.72)
    assert c_od2.settings.get_settings()["count_min"] == 2
    assert c_od2.settings.get_settings()["count_max"] == 5

    # 5. Measurement
    c_ms1 = MeasurementController(bus, MeasurementSettings())
    c_ms1.configure({"canny_low": 42, "tolerance_mm": 1.5})
    c_ms2 = MeasurementController(bus, MeasurementSettings())
    assert c_ms2.settings.get_settings()["canny_low"] == 42
    assert c_ms2.settings.get_settings()["tolerance_mm"] == pytest.approx(1.5)

def test_measurement_calibration_persists_and_resets():
    # --- isolation: remove any leftover calibration data from other tests ---
    from app.core.container import Container
    _container = Container.get_instance()
    _meas_dir = _container.session_manager.get_data_dir("modules/core-measurement")
    _data_json = os.path.join(_meas_dir, "data.json")
    if os.path.exists(_data_json):
        os.remove(_data_json)
    # -----------------------------------------------------------------------

    bus = EventBus()
    c1 = MeasurementController(bus, MeasurementSettings())
    assert c1.calibrate(x1=0, y1=0, x2=200, y2=0, real_length_mm=40.0) is True
    assert c1.pixels_per_mm == pytest.approx(5.0)
    assert c1.calibration_status == "Calibrated"

    # Verify persistence on new controller instance
    c2 = MeasurementController(bus, MeasurementSettings())
    assert c2.calibration_status == "Calibrated"
    assert c2.pixels_per_mm == pytest.approx(5.0)

    # Test reset path
    assert c2.reset_calibration() is True
    assert c2.calibration_status == "Uncalibrated"
    assert c2.pixels_per_mm is None

    # Verify reset persists on third controller instance
    c3 = MeasurementController(bus, MeasurementSettings())
    assert c3.calibration_status == "Uncalibrated"
    assert c3.pixels_per_mm is None


def test_measurement_legacy_pixels_per_cm_migration():
    bus = EventBus()
    # Write a legacy data.json with pixels_per_cm
    from app.core.container import Container
    container = Container.get_instance()
    meas_dir = container.session_manager.get_data_dir("modules/core-measurement")
    os.makedirs(meas_dir, exist_ok=True)
    with open(os.path.join(meas_dir, "data.json"), "w") as f:
        json.dump({"pixels_per_cm": 25.0}, f)

    # Controller should migrate 25.0 px/cm to 2.5 px/mm
    c = MeasurementController(bus, MeasurementSettings())
    assert c.calibration_status == "Calibrated"
    assert c.pixels_per_mm == pytest.approx(2.5)

    # Verify migrated file content
    with open(os.path.join(meas_dir, "data.json"), "r") as f:
        migrated = json.load(f)
    assert migrated.get("pixels_per_mm") == pytest.approx(2.5)

def test_color_reference_persists_and_recomputes_true_float32_lab():
    bus = EventBus()
    c1 = ColorAnalysisController(bus, ColorAnalysisSettings())

    # Create solid test frame with known color BGR = (60, 120, 240)
    frame = np.full((100, 100, 3), (60, 120, 240), dtype=np.uint8)
    c1.last_frame = frame
    assert c1.teach_reference(10, 10, 40, 40) is True

    lab1 = c1.reference_lab.copy()
    rgb1 = c1.reference_rgb

    # Ground truth float32 LAB conversion
    expected_px = (np.asarray((60, 120, 240), np.float32) / 255.0).reshape(1, 1, 3)
    expected_lab = cv2.cvtColor(expected_px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)
    assert np.allclose(lab1, expected_lab)

    # Reload new controller instance
    c2 = ColorAnalysisController(bus, ColorAnalysisSettings())
    assert c2.reference_status == "Taught"
    assert c2.reference_rgb == rgb1
    assert np.allclose(c2.reference_lab, expected_lab)

def test_ai_identify_references_persist_and_classify_after_reload():
    bus = EventBus()
    c1 = AIIdentifyController(bus, AIIdentifySettings())

    good_frame = make_texture(200, seed=10)
    bad_frame = make_texture(200, seed=20)

    c1.last_frame = good_frame
    assert c1.teach_good(0, 0, 200, 200) is True

    c1.last_frame = bad_frame
    assert c1.teach_bad(0, 0, 200, 200) is True
    assert c1.teach_status == "Taught"

    # Reload into a fresh controller instance
    c2 = AIIdentifyController(bus, AIIdentifySettings())
    assert c2.teach_status == "Taught"
    assert len(c2.good_references) == 1
    assert len(c2.bad_references) == 1

    # Classify good_frame on the reloaded controller
    res = c2.process({"frame": good_frame})
    assert res.classification == "Good"

def test_counters_persist_across_new_instance():
    store1 = StateStore(persist=True)
    store1.reset_counters()

    for _ in range(7):
        store1.increment_ok()
    for _ in range(3):
        store1.increment_ng({"reason": "defect"})

    assert store1.ok_count == 7
    assert store1.ng_count == 3
    assert store1.total_count == 10

    # New StateStore in same data dir loads persisted counts
    store2 = StateStore(persist=True)
    assert store2.ok_count == 7
    assert store2.ng_count == 3
    assert store2.total_count == 10

def test_camera_index_persists_across_restart():
    config = ConfigService()
    config.update_settings({"camera_index": 3, "camera_resolution": "640x480", "camera_backend": "dshow"})

    # Emulate restart: instantiate fresh ConfigService and CameraManager
    new_config = ConfigService()
    assert new_config.get("camera_index") == 3
    assert new_config.get("camera_resolution") == "640x480"
    assert new_config.get("camera_backend") == "dshow"

    bus = EventBus()
    cam_mgr = CameraManager(new_config, bus, MagicMock(), MagicMock())
    assert cam_mgr.camera_index == 3

def test_camera_devices_api_and_switch_restart():
    from app.api import create_app
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        # 1. GET /api/v1/camera/devices
        res = client.get("/api/v1/camera/devices")
        assert res.status_code == 200
        data = res.get_json()
        assert "devices" in data
        assert len(data["devices"]) == 5
        for dev in data["devices"]:
            assert "index" in dev
            assert "opened" in dev
            assert "width" in dev
            assert "height" in dev

        # 2. POST /api/v1/camera/switch
        from unittest.mock import patch
        with patch.object(CameraManager, "start", return_value=True):
            switch_res = client.post("/api/v1/camera/switch", json={
                "index": 2,
                "resolution": "1920x1080",
                "backend": "dshow"
            })
            assert switch_res.status_code == 200
            assert switch_res.get_json()["success"] is True

        # 3. Verify persistence across fresh ConfigService / CameraManager
        fresh_config = ConfigService()
        assert fresh_config.get("camera_index") == 2
        assert fresh_config.get("camera_resolution") == "1920x1080"
        assert fresh_config.get("camera_backend") == "dshow"

def test_ai_identify_reset_teaching_clears_references_without_touching_counters(tmp_path, monkeypatch):
    data_dir = str(tmp_path / "data")
    monkeypatch.setenv("PI_VISION_DATA_DIR", data_dir)
    os.makedirs(data_dir, exist_ok=True)

    from app.core.container import Container
    Container._instance = None
    container = Container.get_instance()
    container.session_manager.data_dir = data_dir

    bus = EventBus()
    controller = AIIdentifyController(bus, AIIdentifySettings())
    controller.module_dir = container.session_manager.get_data_dir("modules/core-ai-identify")
    os.makedirs(controller.module_dir, exist_ok=True)
    
    # Global state store counters
    store = container.state_store
    store.data_dir = data_dir
    store.counters_file = os.path.join(data_dir, "counters.json")
    store.persist = True
    store.reset_counters()
    for _ in range(5):
        store.increment_ok()
    for _ in range(2):
        store.increment_ng({"reason": "defect"})
    assert store.ok_count == 5
    assert store.ng_count == 2
    assert store.total_count == 7

    # Teach references
    g = make_texture(200, seed=1)
    b = make_texture(200, seed=2)
    controller.last_frame = g
    assert controller.teach_good(0, 0, 200, 200) is True
    controller.last_frame = b
    assert controller.teach_bad(0, 0, 200, 200) is True
    assert len(controller.good_references) >= 1
    assert len(controller.bad_references) >= 1

    # Reset teaching: clears references, MUST NOT touch counters
    assert controller.reset_teaching() is True
    assert len(controller.good_references) == 0
    assert len(controller.bad_references) == 0
    assert controller.teach_status == "Untaught"

    # Counters remain unchanged
    assert store.ok_count == 5
    assert store.ng_count == 2
    assert store.total_count == 7
    Container._instance = None

def test_object_detection_judge_enabled_and_expect_none():
    bus = EventBus()
    runtime = MagicMock()
    controller = ObjectDetectionController(bus, runtime, ObjectDetectionSettings())

    # judge_enabled=False: unjudged
    runtime.predict.return_value = {"detections": [{"id": 1}], "latency_ms": 1.0}
    controller.configure({"judge_enabled": False, "count_min": 0, "count_max": 0})
    res_disabled = controller.process({"frame": np.zeros((10, 10, 3))})
    assert res_disabled.is_ok is None

    # judge_enabled=True, count_min=0, count_max=0: expect none
    controller.configure({"judge_enabled": True, "count_min": 0, "count_max": 0})
    # 0 objects -> OK
    runtime.predict.return_value = {"detections": [], "latency_ms": 1.0}
    res_ok = controller.process({"frame": np.zeros((10, 10, 3))})
    assert res_ok.is_ok is True

    # 1 object -> NG
    runtime.predict.return_value = {"detections": [{"id": 1}], "latency_ms": 1.0}
    res_ng = controller.process({"frame": np.zeros((10, 10, 3))})
    assert res_ng.is_ok is False

def test_measurement_judge_enabled_length_width_and_tolerance():
    bus = EventBus()
    controller = MeasurementController(bus, MeasurementSettings())
    controller.pixels_per_mm = 10.0

    # configure rejects tolerance_mm <= 0 when judge_enabled is True
    assert controller.configure({"judge_enabled": True, "tolerance_mm": 0.0}) is False
    assert controller.configure({"judge_enabled": True, "tolerance_mm": -1.0}) is False

    # configure succeeds with tolerance_mm > 0
    assert controller.configure({
        "judge_enabled": True,
        "target_length_mm": 20.0,
        "target_width_mm": 10.0,
        "tolerance_mm": 1.0
    }) is True

    # Frame with rectangle: long side = 200px (20mm), short side = 100px (10mm)
    frame_ok = np.zeros((300, 300, 3), dtype=np.uint8)
    frame_ok[50:150, 50:250] = 255
    res = controller.process({"frame": frame_ok})
    assert res.is_ok is True

    # Frame with rectangle: long side = 250px (25mm) -> out of tolerance length
    frame_ng = np.zeros((300, 300, 3), dtype=np.uint8)
    frame_ng[50:150, 50:300] = 255
    res2 = controller.process({"frame": frame_ng})
    assert res2.is_ok is False

    # Switch off judge_enabled: pattern stays, but judgment unjudged (is_ok is None)
    controller.configure({"judge_enabled": False})
    res_off = controller.process({"frame": frame_ok})
    assert res_off.is_ok is None
    assert controller.settings.get_settings()["target_length_mm"] == 20.0

def test_ocr_judge_enabled_keeps_pattern_while_switched_off():
    bus = EventBus()
    runtime = MagicMock()
    controller = OCRController(bus, runtime, OCRSettings())

    # Configure pattern while judge_enabled is False
    assert controller.configure({"expected_pattern": "^PASS-[0-9]{3}$", "judge_enabled": False}) is True
    runtime.predict.return_value = {"texts": [{"text": "PASS-123"}], "latency_ms": 1.0}
    res1 = controller.process({"frame": np.zeros((10, 10, 3))})
    assert res1.is_ok is None
    # Pattern is preserved
    assert controller.settings.get_settings()["expected_pattern"] == "^PASS-[0-9]{3}$"

    # Turn judge_enabled on
    assert controller.configure({"judge_enabled": True}) is True
    res2 = controller.process({"frame": np.zeros((10, 10, 3))})
    assert res2.is_ok is True

    runtime.predict.return_value = {"texts": [{"text": "FAIL-999"}], "latency_ms": 1.0}
    res3 = controller.process({"frame": np.zeros((10, 10, 3))})
    assert res3.is_ok is False

def test_real_app_restart_persists_counters(tmp_path, monkeypatch):
    data_dir = str(tmp_path / "data")
    monkeypatch.setenv("PI_VISION_DATA_DIR", data_dir)
    os.makedirs(data_dir, exist_ok=True)

    from app.core.container import Container
    Container._instance = None

    # 1. Run app instance 1
    from app.api import create_app
    app1 = create_app()
    app1.config["TESTING"] = True
    with app1.test_client() as client1:
        # Increment counters through state_store
        container1 = Container.get_instance()
        container1.state_store.data_dir = data_dir
        container1.state_store.counters_file = os.path.join(data_dir, "counters.json")
        container1.state_store.persist = True
        container1.state_store.reset_counters()
        for _ in range(4):
            container1.state_store.increment_ok()
        for _ in range(2):
            container1.state_store.increment_ng({"reason": "defect"})

        res1 = client1.get("/api/v1/system/status")
        assert res1.status_code == 200
        d1 = res1.get_json()
        assert d1["ok_count"] == 4
        assert d1["ng_count"] == 2
        assert d1["total_count"] == 6

    # 2. Simulate complete restart: reset container singleton and create fresh app
    Container._instance = None
    app2 = create_app()
    app2.config["TESTING"] = True
    with app2.test_client() as client2:
        res2 = client2.get("/api/v1/system/status")
        assert res2.status_code == 200
        d2 = res2.get_json()
        assert d2["ok_count"] == 4
        assert d2["ng_count"] == 2
        assert d2["total_count"] == 6
    Container._instance = None

def test_camera_manager_opens_persisted_index_on_restart_without_switch(tmp_path, monkeypatch):
    data_dir = str(tmp_path / "data")
    monkeypatch.setenv("PI_VISION_DATA_DIR", data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Persist camera_index = 0 and backend = dshow in settings.json
    settings_file = os.path.join(data_dir, "settings.json")
    with open(settings_file, "w") as f:
        json.dump({"camera_index": 0, "camera_backend": "dshow"}, f)

    from app.core.container import Container
    Container._instance = None
    container = Container.get_instance()
    # Confirm CameraManager reloaded persisted index and backend without switch_camera call
    assert container.camera_manager.camera_index == 0
    assert container.config.get("camera_backend") == "dshow"

    Container._instance = None

def test_measurement_tolerance_validator_zero_permitted_when_judge_off():
    from app.modules.measurement.models import UpdateMeasurementSettingsRequest
    # tolerance_mm == 0.0 with judge_enabled False is permitted
    req_off = UpdateMeasurementSettingsRequest(tolerance_mm=0.0, judge_enabled=False)
    assert req_off.tolerance_mm == 0.0

    # tolerance_mm == 0.0 with judge_enabled True must raise error
    with pytest.raises(Exception) as excinfo:
        UpdateMeasurementSettingsRequest(tolerance_mm=0.0, judge_enabled=True)
    assert "tolerance_mm must be > 0" in str(excinfo.value)



