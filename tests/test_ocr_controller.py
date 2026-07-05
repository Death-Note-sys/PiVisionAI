import pytest
from unittest.mock import MagicMock
from app.modules.ocr.controller import OCRController
from app.modules.ocr.settings import OCRSettings
from app.core.models.results import OCRResult


@pytest.fixture
def controller():
    event_bus = MagicMock()
    ai_runtime = MagicMock()
    settings = OCRSettings()
    return OCRController(event_bus, ai_runtime, settings)


def test_process_returns_ocr_result_with_real_text(controller):
    controller.ai_runtime.predict.return_value = {
        "texts": [{"text": "HELLO", "confidence": 0.95, "points": [[0,0],[10,0],[10,10],[0,10]]}],
        "latency_ms": 320.0,
    }
    context = {"frame": MagicMock()}

    result = controller.process(context)

    assert isinstance(result, OCRResult)
    assert len(result.texts) == 1
    assert result.texts[0]["text"] == "HELLO"
    assert result.latency_ms == 320.0
    assert controller.last_result is result


def test_process_returns_empty_result_when_adapter_returns_none(controller):
    controller.ai_runtime.predict.return_value = None
    context = {"frame": MagicMock()}

    result = controller.process(context)

    assert isinstance(result, OCRResult)
    assert result.texts == []


def test_process_handles_inference_exception_gracefully(controller):
    controller.ai_runtime.predict.side_effect = RuntimeError("EasyOCR exploded")
    context = {"frame": MagicMock()}

    result = controller.process(context)

    assert isinstance(result, OCRResult)
    assert result.texts == []


def test_process_publishes_event_when_text_found(controller):
    controller.ai_runtime.predict.return_value = {
        "texts": [{"text": "X", "confidence": 0.9, "points": [[0,0],[5,0],[5,5],[0,5]]}],
        "latency_ms": 100.0,
    }
    context = {"frame": MagicMock()}

    controller.process(context)

    controller.event_bus.publish.assert_called_with(
        "TextDetected", {"count": 1, "model": controller.active_model_id}
    )


def test_process_does_not_publish_when_no_text_found(controller):
    controller.ai_runtime.predict.return_value = {"texts": [], "latency_ms": 50.0}
    context = {"frame": MagicMock()}

    controller.process(context)

    controller.event_bus.publish.assert_not_called()


def test_configure_updates_settings_and_model_id(controller):
    result = controller.configure({"min_confidence": 0.6, "model_id": "easyocr-v2"})

    assert result is True
    assert controller.active_model_id == "easyocr-v2"
    assert controller.settings.get_settings()["min_confidence"] == 0.6


def test_cleanup_unloads_active_model(controller):
    controller.cleanup()
    controller.ai_runtime.unload_model.assert_called_once_with(controller.active_model_id)
