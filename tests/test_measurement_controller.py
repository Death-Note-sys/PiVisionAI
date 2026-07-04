import pytest
import numpy as np
from unittest.mock import MagicMock
from app.modules.measurement.controller import MeasurementController
from app.modules.measurement.settings import MeasurementSettings
from app.core.models.results import MeasurementResult


@pytest.fixture
def controller():
    event_bus = MagicMock()
    settings = MeasurementSettings()
    return MeasurementController(event_bus, settings)


def make_test_frame():
    """A simple frame with one white square on black, for contour detection."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[50:150, 50:150] = 255
    return frame


def test_calibrate_computes_correct_pixels_per_cm(controller):
    result = controller.calibrate(x1=0, y1=0, x2=100, y2=0, real_length_cm=10.0)

    assert result is True
    assert controller.pixels_per_cm == pytest.approx(10.0)
    assert controller.calibration_status == "Calibrated"


def test_calibrate_rejects_zero_length_line(controller):
    result = controller.calibrate(x1=50, y1=50, x2=50, y2=50, real_length_cm=10.0)

    assert result is False
    assert controller.pixels_per_cm is None
    assert controller.calibration_status == "Uncalibrated"


def test_calibrate_rejects_zero_real_length(controller):
    result = controller.calibrate(x1=0, y1=0, x2=100, y2=0, real_length_cm=0)

    assert result is False


def test_calibrate_publishes_event(controller):
    controller.calibrate(x1=0, y1=0, x2=100, y2=0, real_length_cm=10.0)

    controller.event_bus.publish.assert_called_once()
    args = controller.event_bus.publish.call_args[0]
    assert args[0] == "MeasurementCalibrated"


def test_process_detects_contour_before_calibration(controller):
    context = {"frame": make_test_frame()}

    result = controller.process(context)

    assert isinstance(result, MeasurementResult)
    assert result.calibration_status == "Uncalibrated"
    assert len(result.measurements) >= 1
    assert result.measurements[0]["width_cm"] is None


def test_process_converts_to_cm_after_calibration(controller):
    controller.calibrate(x1=0, y1=0, x2=100, y2=0, real_length_cm=10.0)
    context = {"frame": make_test_frame()}

    result = controller.process(context)

    assert result.calibration_status == "Calibrated"
    assert result.pixels_per_cm == pytest.approx(10.0)
    measured = result.measurements[0]
    assert measured["width_cm"] is not None
    assert measured["width_cm"] == pytest.approx(measured["width_px"] / 10.0, rel=0.01)


def test_process_ignores_contours_below_min_area(controller):
    controller.settings.update({"min_contour_area": 999999})
    context = {"frame": make_test_frame()}

    result = controller.process(context)

    assert len(result.measurements) == 0


def test_process_stores_last_result(controller):
    context = {"frame": make_test_frame()}

    result = controller.process(context)

    assert controller.last_result is result


def test_configure_updates_settings(controller):
    result = controller.configure({"canny_low": 30, "unit": "mm"})

    assert result is True
    assert controller.settings.get_settings()["canny_low"] == 30
