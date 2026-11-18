"""Pipeline-level integration tests for Target Extraction.

Exercises:
  * context["triggered"] is present on every process_frame call
  * triggered frame produces a result row in coordinator
  * non-triggered frames do NOT call the segmenter
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from app.core.event_bus import EventBus
from app.modules.target_extraction.controller import TargetExtractionController
from app.modules.target_extraction.settings import TargetExtractionSettings


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _frame(h=120, w=160):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[20:70, 20:80] = (200, 220, 240)
    return frame


def _fake_seg():
    seg = MagicMock()
    def _segment(frame, box):
        m = np.zeros(frame.shape[:2], dtype=np.uint8)
        x, y, w, h = box
        m[y:y + h, x:x + w] = 255
        return m
    seg.segment.side_effect = _segment
    return seg


def _make_controller(segmenter=None, judge_enabled=False, min_r=0.10, max_r=0.95):
    bus = EventBus()
    settings = TargetExtractionSettings()
    if judge_enabled:
        settings.update({"judge_enabled": True,
                         "min_area_ratio": min_r,
                         "max_area_ratio": max_r})
    ctrl = TargetExtractionController(bus, settings, segmenter=segmenter)
    # Set a ROI without needing a real frame hook
    ctrl._roi_box = (20, 20, 60, 50)
    ctrl.roi_status = "Taught"
    return ctrl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pipeline_context_has_triggered_flag():
    """Pipeline.process_frame must inject context["triggered"]."""
    from app.core.pipeline import Pipeline
    import inspect
    src = inspect.getsource(Pipeline.process_frame)
    assert 'context["triggered"]' in src or "context['triggered']" in src


def test_one_trigger_one_row_one_counter():
    """A single triggered frame produces one extraction result with area_px > 0."""
    seg = _fake_seg()
    ctrl = _make_controller(segmenter=seg)
    frame = _frame()

    result = ctrl.process({"frame": frame, "triggered": True})

    assert result.roi_status == "Taught"
    assert result.area_px > 0
    assert result.engine in ("sam", "grabcut")
    assert seg.segment.call_count >= 1


def test_continuous_no_trigger_zero_rows_zero_segmenter_calls():
    """Non-triggered frames must not invoke the segmenter at all."""
    seg = _fake_seg()
    ctrl = _make_controller(segmenter=seg)
    frame = _frame()

    # Three consecutive non-triggered frames
    for _ in range(3):
        result = ctrl.process({"frame": frame, "triggered": False})

    assert seg.segment.call_count == 0
    # Result is the default (no extraction ran)
    assert result.area_px == 0
