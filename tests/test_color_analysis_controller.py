import pytest
import numpy as np
from unittest.mock import MagicMock
from app.modules.color_analysis.controller import ColorAnalysisController
from app.modules.color_analysis.settings import ColorAnalysisSettings
from app.core.models.results import ColorResult


@pytest.fixture
def controller():
    event_bus = MagicMock()
    settings = ColorAnalysisSettings()
    return ColorAnalysisController(event_bus, settings)


def make_solid_color_frame(bgr, size=200):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def make_two_tone_frame(bgr1, bgr2, size=200):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :size // 2] = bgr1
    frame[:, size // 2:] = bgr2
    return frame


def test_process_extracts_dominant_color_from_solid_frame(controller):
    red_bgr = (0, 0, 255)
    frame = make_solid_color_frame(red_bgr)

    result = controller.process({"frame": frame})

    assert isinstance(result, ColorResult)
    assert len(result.dominant_colors) >= 1
    top = result.dominant_colors[0]
    assert top["percentage"] > 90
    assert top["rgb"][0] > 200  # red channel dominant
    assert top["rgb"][2] < 50   # blue channel low


def test_process_reports_reference_status_untaught_by_default(controller):
    frame = make_solid_color_frame((0, 255, 0))
    result = controller.process({"frame": frame})

    assert result.reference_status == "Untaught"
    assert result.delta_e is None
    assert result.match is None


def test_teach_reference_fails_with_no_frame_yet(controller):
    result = controller.teach_reference(0, 0, 50, 50)
    assert result is False


def test_teach_reference_succeeds_and_stores_color(controller):
    frame = make_solid_color_frame((0, 0, 255))  # pure red in BGR
    controller.on_raw_frame(frame)

    result = controller.teach_reference(0, 0, 200, 200)

    assert result is True
    assert controller.reference_status == "Taught"
    assert controller.reference_rgb[0] > 200  # R channel


def test_process_matches_identical_color_within_tolerance(controller):
    red_frame = make_solid_color_frame((0, 0, 255))
    controller.on_raw_frame(red_frame)
    controller.teach_reference(0, 0, 200, 200)

    result = controller.process({"frame": red_frame})

    assert result.reference_status == "Taught"
    assert result.delta_e is not None
    assert result.delta_e < 5.0
    assert result.match is True


def test_process_rejects_clearly_different_color(controller):
    red_frame = make_solid_color_frame((0, 0, 255))
    controller.on_raw_frame(red_frame)
    controller.teach_reference(0, 0, 200, 200)

    blue_frame = make_solid_color_frame((255, 0, 0))
    result = controller.process({"frame": blue_frame})

    assert result.match is False
    assert result.delta_e > 10.0


def test_reset_reference_clears_state(controller):
    frame = make_solid_color_frame((0, 0, 255))
    controller.on_raw_frame(frame)
    controller.teach_reference(0, 0, 200, 200)

    result = controller.reset_reference()

    assert result is True
    assert controller.reference_status == "Untaught"
    assert controller.reference_rgb is None


def test_configure_updates_settings(controller):
    result = controller.configure({"num_colors": 3})
    assert result is True
    assert controller.settings.get_settings()["num_colors"] == 3


def test_process_reports_multiple_colors_in_two_tone_frame(controller):
    frame = make_two_tone_frame((0, 0, 255), (255, 0, 0))  # red + blue halves
    result = controller.process({"frame": frame})

    assert len(result.dominant_colors) >= 2
    total_pct = sum(c["percentage"] for c in result.dominant_colors)
    assert 90 <= total_pct <= 100
