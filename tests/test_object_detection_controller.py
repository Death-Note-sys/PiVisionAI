import time
import pytest
from unittest.mock import MagicMock
from app.modules.object_detection.controller import ObjectDetectionController
from app.modules.object_detection.settings import ObjectDetectionSettings
from app.core.models.results import DetectionResult


@pytest.fixture
def controller():
    event_bus = MagicMock()
    ai_runtime = MagicMock()
    settings = ObjectDetectionSettings()
    return ObjectDetectionController(event_bus, ai_runtime, settings)


def test_process_returns_detection_result_with_real_detections(controller):
    controller.ai_runtime.predict.return_value = {
        "detections": [
            {"box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "confidence": 0.9, "class_id": 0, "label": "person"}
        ],
        "latency_ms": 12.5,
    }
    context = {"frame": MagicMock()}

    result = controller.process(context)

    assert isinstance(result, DetectionResult)
    assert result.objects_count == 1
    assert result.latency_ms == 12.5
    assert result.model_name == controller.active_model_id
    assert controller.last_result is result


def test_process_returns_empty_result_when_adapter_returns_none(controller):
    controller.ai_runtime.predict.return_value = None
    context = {"frame": MagicMock()}

    result = controller.process(context)

    assert isinstance(result, DetectionResult)
    assert result.objects_count == 0
    assert result.detections == []


def test_process_handles_inference_exception_gracefully(controller):
    controller.ai_runtime.predict.side_effect = RuntimeError("adapter exploded")
    context = {"frame": MagicMock()}

    result = controller.process(context)

    assert isinstance(result, DetectionResult)
    assert result.objects_count == 0


def test_process_publishes_objects_detected_event_when_objects_found(controller):
    controller.ai_runtime.predict.return_value = {
        "detections": [{"box": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}, "confidence": 0.8, "class_id": 1, "label": "car"}],
        "latency_ms": 5.0,
    }
    context = {"frame": MagicMock()}

    controller.process(context)

    controller.event_bus.publish.assert_called_with(
        "ObjectsDetected", {"count": 1, "model": controller.active_model_id}
    )


def test_process_does_not_publish_event_when_no_objects_found(controller):
    controller.ai_runtime.predict.return_value = {"detections": [], "latency_ms": 3.0}
    context = {"frame": MagicMock()}

    controller.process(context)

    controller.event_bus.publish.assert_not_called()


def test_configure_updates_settings_and_model_id(controller):
    result = controller.configure({"confidence": 0.7, "model_id": "yolo11s"})

    assert result is True
    assert controller.active_model_id == "yolo11s"
    assert controller.settings.get_settings()["confidence"] == 0.7


def test_cleanup_unloads_active_model(controller):
    controller.cleanup()
    controller.ai_runtime.unload_model.assert_called_once_with(controller.active_model_id)


def test_health_check_returns_true(controller):
    assert controller.health_check() is True
