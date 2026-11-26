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


def test_renderer_cutout_paints_background_outside_mask():
    """In cutout mode, pixels outside the mask equal background color (24, 24, 24)."""
    from app.core.renderers.target_extraction_renderer import TargetExtractionRenderer
    from app.core.models.results import TargetExtractionResult

    renderer = TargetExtractionRenderer()
    h, w = 120, 160
    frame = np.full((h, w, 3), 200, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[40:80, 50:120] = 255

    result = TargetExtractionResult(
        roi_status="Taught",
        engine="grabcut",
        area_px=int((mask > 0).sum()),
        area_ratio=0.5,
        roi={"x": 45, "y": 35, "w": 80, "h": 50},
    )
    result._mask = mask
    result._view_mode = "cutout"

    # Full background check when show_info is disabled
    out_clean = renderer.render(frame, result, {"settings": {"view_mode": "cutout", "show_info": False}})
    assert np.all(out_clean[mask == 0] == (24, 24, 24))
    assert np.all(out_clean[mask > 0] == (200, 200, 200))

    # Default settings via _view_mode PrivateAttr also render cutout background
    out_default = renderer.render(frame, result, {"settings": {}})
    assert tuple(out_default[0, 0]) == (24, 24, 24)
    assert tuple(out_default[60, 80]) == (200, 200, 200)
    # PrivateAttr must not leak into serialized JSON dict
    assert "_mask" not in result.model_dump()


def test_controller_render_attaches_mask_for_triggered_and_rerendered_cutout(tmp_path, monkeypatch):
    """Controller stores _last_mask and attaches it on both triggered and non-triggered renders."""
    from app.core.renderers.target_extraction_renderer import TargetExtractionRenderer

    monkeypatch.setenv("PI_VISION_DATA_DIR", str(tmp_path))
    seg = _fake_seg()
    ctrl = _make_controller(segmenter=seg)
    ctrl.configure({"view_mode": "cutout"})
    frame = np.full((120, 160, 3), 180, dtype=np.uint8)
    renderer = TargetExtractionRenderer()

    # 1. Triggered frame runs segmentation once and renders cutout
    res1 = ctrl.process({"frame": frame, "triggered": True})
    instr1 = ctrl.render(res1)
    assert instr1._mask is not None
    out1 = renderer.render(frame, instr1, {"settings": {}})
    assert tuple(out1[0, 0]) == (24, 24, 24)
    # Interior foreground point below the info text baseline (y=45) and inside ROI (20..80, 20..70)
    assert tuple(out1[58, 50]) == (180, 180, 180)
    assert seg.segment.call_count == 1

    # 2. Non-triggered frame reuses last_result and stored mask without re-running segmenter
    res2 = ctrl.process({"frame": frame, "triggered": False})
    instr2 = ctrl.render(res2)
    assert instr2._mask is not None
    out2 = renderer.render(frame, instr2, {"settings": {}})
    assert tuple(out2[0, 0]) == (24, 24, 24)
    assert tuple(out2[58, 50]) == (180, 180, 180)
    assert seg.segment.call_count == 1
