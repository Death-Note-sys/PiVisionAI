import pytest
import numpy as np
import cv2
import json
import os
import sqlite3
from unittest.mock import MagicMock
from app.modules.measurement.controller import MeasurementController
from app.modules.measurement.settings import MeasurementSettings
from app.modules.object_detection.controller import ObjectDetectionController
from app.modules.object_detection.settings import ObjectDetectionSettings
from app.modules.ocr.controller import OCRController
from app.modules.ocr.settings import OCRSettings
from app.core.state_store import StateStore
from app.core.config_service import ConfigService
from app.core.models.results import DetectionResult

# 1. Measurement 0 vs 45 deg rotated rectangle test
def test_measurement_0_vs_45_deg_rotated_rectangle():
    event_bus = MagicMock()
    settings = MeasurementSettings()
    controller = MeasurementController(event_bus, settings)
    controller.pixels_per_mm = 1.0
    
    frame_0 = np.zeros((200, 200, 3), dtype=np.uint8)
    frame_0[80:120, 50:150] = 255 # w=100, h=40
    
    frame_45 = np.zeros((200, 200, 3), dtype=np.uint8)
    rect = ((100, 100), (100, 40), 45)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(frame_45, [box], 0, (255, 255, 255), -1)
    
    res_0 = controller.process({"frame": frame_0})
    res_45 = controller.process({"frame": frame_45})
    
    m0 = res_0.measurements[0]
    m45 = res_45.measurements[0]
    
    max0 = max(m0["width_px"], m0["height_px"])
    min0 = min(m0["width_px"], m0["height_px"])
    max45 = max(m45["width_px"], m45["height_px"])
    min45 = min(m45["width_px"], m45["height_px"])
    
    assert max0 == pytest.approx(max45, abs=3.0)
    assert min0 == pytest.approx(min45, abs=3.0)

# 2. Measurement judgment boundaries
def test_measurement_judgment_boundaries():
    event_bus = MagicMock()
    settings = MeasurementSettings()
    settings.update({"target_width_mm": 10.0, "tolerance_mm": 1.0, "judgment_roi": {"x": 0, "y": 0, "w": 100, "h": 100}})
    controller = MeasurementController(event_bus, settings)
    controller.pixels_per_mm = 10.0 # so w=100px => 10mm
    
    # OK frame: 100px width (10mm) inside ROI
    frame_ok = np.zeros((200, 200, 3), dtype=np.uint8)
    frame_ok[40:60, 40:140] = 255
    res_ok = controller.process({"frame": frame_ok})
    assert res_ok.is_ok is True
    
    # NG frame: 115px width (11.5mm) - out of tolerance
    frame_ng_tol = np.zeros((200, 200, 3), dtype=np.uint8)
    frame_ng_tol[40:60, 40:155] = 255
    res_ng_tol = controller.process({"frame": frame_ng_tol})
    assert res_ng_tol.is_ok is False
    
    # NG frame: 100px width but outside ROI (cy > 100)
    frame_ng_roi = np.zeros((200, 200, 3), dtype=np.uint8)
    frame_ng_roi[120:140, 40:140] = 255
    res_ng_roi = controller.process({"frame": frame_ng_roi})
    assert res_ng_roi.is_ok is False

# 3. Object Detection count boundaries
def test_object_detection_count_boundaries():
    event_bus = MagicMock()
    ai_runtime = MagicMock()
    ai_runtime.predict.return_value = {"detections": [{"id":1}, {"id":2}, {"id":3}], "latency_ms": 1.0}
    
    settings = ObjectDetectionSettings()
    settings.update({"count_min": 2, "count_max": 4})
    controller = ObjectDetectionController(event_bus, ai_runtime, settings)
    
    res = controller.process({"frame": np.zeros((10,10,3))})
    assert res.is_ok is True
    
    settings.update({"count_min": 4, "count_max": 5})
    res2 = controller.process({"frame": np.zeros((10,10,3))})
    assert res2.is_ok is False

# 4. OCR regex match, no-match, invalid-regex
def test_ocr_regex_boundaries():
    event_bus = MagicMock()
    ai_runtime = MagicMock()
    
    settings = OCRSettings()
    settings.update({"expected_pattern": "^PASS-\\d{3}$"})
    controller = OCRController(event_bus, ai_runtime, settings)
    
    # Match
    ai_runtime.predict.return_value = {"texts": [{"text": "PASS-123"}], "latency_ms": 1.0}
    res = controller.process({"frame": np.zeros((10,10,3))})
    assert res.is_ok is True
    
    # No match
    ai_runtime.predict.return_value = {"texts": [{"text": "OK"}], "latency_ms": 1.0}
    res2 = controller.process({"frame": np.zeros((10,10,3))})
    assert res2.is_ok is False
    
    # Invalid regex
    settings.update({"expected_pattern": "["})
    ai_runtime.predict.return_value = {"texts": [{"text": "OK"}], "latency_ms": 1.0}
    res3 = controller.process({"frame": np.zeros((10,10,3))})
    assert res3.is_ok is False

# 5. per-module persistence round trip
def test_per_module_persistence_round_trip(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = StateStore(db_path=db_path)
    store.update("core-ai-identify", {"some_key": "value123"})
    
    store2 = StateStore(db_path=db_path)
    val = store2.get("core-ai-identify")
    assert val == {"some_key": "value123"}

# 6. camera backend fallback and read-back
def test_camera_backend_fallback(monkeypatch):
    monkeypatch.setenv("PI_VISION_CAMERA_BACKEND", "v4l2")
    service = ConfigService()
    assert service.get("camera_backend") == "v4l2"
    
    monkeypatch.delenv("PI_VISION_CAMERA_BACKEND", raising=False)
    service2 = ConfigService()
    assert service2.get("camera_backend", "opencv") in ("opencv", "auto")
