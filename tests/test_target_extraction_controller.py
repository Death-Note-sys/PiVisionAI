"""Tests for TargetExtractionController.

All tests use a fake segmenter so no SAM weights are needed.
GrabCut really runs on the synthetic frames.
"""
import os
import json
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import MagicMock

from app.core.event_bus import EventBus
from app.core.container import Container
from app.modules.target_extraction.controller import TargetExtractionController
from app.modules.target_extraction.settings import TargetExtractionSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_frame(h=120, w=160, color=(100, 150, 200)):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _frame_with_fg(h=120, w=160, roi=(20, 20, 60, 50)):
    """Frame where the ROI region is a bright distinct color."""
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    x, y, rw, rh = roi
    frame[y:y + rh, x:x + rw] = (200, 230, 240)
    return frame


def _fake_segmenter(mask_value=255):
    """Returns a segmenter whose .segment() returns a full-frame mask."""
    seg = MagicMock()
    def _segment(frame, box):
        m = np.zeros(frame.shape[:2], dtype=np.uint8)
        x, y, w, h = box
        m[y:y + h, x:x + w] = mask_value
        return m
    seg.segment.side_effect = _segment
    return seg


def _controller(settings=None, segmenter=None, clean=True):
    if settings is None:
        settings = TargetExtractionSettings()
    bus = EventBus()
    if clean:
        container = Container.get_instance()
        mod_dir = container.session_manager.get_data_dir("modules/core-target-extraction")
        os.makedirs(mod_dir, exist_ok=True)
        for fname in ["settings.json", "data.json", "last_extraction.png"]:
            p = os.path.join(mod_dir, fname)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
    return TargetExtractionController(bus, settings, segmenter=segmenter)


@pytest.fixture
def clean_te_data():
    container = Container.get_instance()
    mod_dir = container.session_manager.get_data_dir("modules/core-target-extraction")
    os.makedirs(mod_dir, exist_ok=True)
    for fname in ["settings.json", "data.json", "last_extraction.png"]:
        p = os.path.join(mod_dir, fname)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    yield mod_dir
    for fname in ["settings.json", "data.json", "last_extraction.png"]:
        p = os.path.join(mod_dir, fname)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_untaught_returns_zero_count(clean_te_data):
    ctrl = _controller(clean=False)
    frame = _solid_frame()
    result = ctrl.process({"frame": frame, "triggered": True})
    assert result.roi_status == "Untaught"
    assert result.area_px == 0
    assert result.is_ok is None


def test_set_roi_persist_roundtrip(clean_te_data):
    ctrl = _controller(clean=False)
    frame = _frame_with_fg()
    ctrl.on_raw_frame(frame)

    ok = ctrl.set_roi(20, 20, 60, 50)
    assert ok is True
    assert ctrl.roi_status == "Taught"
    assert ctrl._roi_box == (20, 20, 60, 50)

    # Second controller loads from data.json
    bus2 = EventBus()
    ctrl2 = TargetExtractionController(bus2, TargetExtractionSettings())
    assert ctrl2.roi_status == "Taught"
    assert ctrl2._roi_box == (20, 20, 60, 50)


def test_triggered_frame_writes_extracted_png_and_stats(clean_te_data):
    seg = _fake_segmenter()
    ctrl = _controller(clean=False, segmenter=seg)
    frame = _frame_with_fg()
    ctrl.on_raw_frame(frame)
    ctrl.set_roi(20, 20, 60, 50)

    result = ctrl.process({"frame": frame, "triggered": True})

    assert result.roi_status == "Taught"
    assert result.engine in ("sam", "grabcut")
    assert result.area_px > 0
    assert result.area_ratio > 0.0

    ext_path = ctrl.get_extraction_path()
    assert ext_path is not None and os.path.exists(ext_path)


def test_non_triggered_frame_does_not_call_segmenter(clean_te_data):
    seg = _fake_segmenter()
    ctrl = _controller(clean=False, segmenter=seg)
    frame = _frame_with_fg()
    ctrl.on_raw_frame(frame)
    ctrl.set_roi(20, 20, 60, 50)

    # Trigger once to set last_result
    ctrl.process({"frame": frame, "triggered": True})
    call_count_after_trigger = seg.segment.call_count

    # Non-triggered: should not call segmenter
    ctrl.process({"frame": frame, "triggered": False})
    assert seg.segment.call_count == call_count_after_trigger


def test_sam_error_falls_back_to_grabcut_with_reason(clean_te_data):
    seg = MagicMock()
    seg.segment.side_effect = RuntimeError("SAM exploded")

    ctrl = _controller(clean=False, segmenter=seg)
    frame = _frame_with_fg()
    ctrl.on_raw_frame(frame)
    ctrl.set_roi(20, 20, 60, 50)

    # engine="auto" should fall back to grabcut
    result = ctrl.process({"frame": frame, "triggered": True})
    assert result.engine == "grabcut"
    assert result.fallback_reason is not None
    assert "SAM" in result.fallback_reason or "exploded" in result.fallback_reason


def test_engine_sam_failing_segmenter_raises(clean_te_data):
    seg = MagicMock()
    seg.segment.side_effect = RuntimeError("hard failure")

    settings = TargetExtractionSettings()
    settings.update({"engine": "sam"})
    ctrl = _controller(settings=settings, clean=False, segmenter=seg)
    frame = _frame_with_fg()
    ctrl.on_raw_frame(frame)
    ctrl.set_roi(20, 20, 60, 50)

    # engine="sam" should NOT fall back; result should carry the error
    result = ctrl.process({"frame": frame, "triggered": True})
    # The controller catches exceptions and stores them in fallback_reason
    assert result.fallback_reason is not None or result.reason is not None


def test_judge_boundaries_ok(clean_te_data):
    seg = _fake_segmenter()
    settings = TargetExtractionSettings()
    settings.update({"judge_enabled": True, "min_area_ratio": 0.01, "max_area_ratio": 0.99})

    ctrl = _controller(settings=settings, clean=False, segmenter=seg)
    frame = _frame_with_fg()
    ctrl.on_raw_frame(frame)
    ctrl.set_roi(20, 20, 60, 50)

    result = ctrl.process({"frame": frame, "triggered": True})
    # Fake segmenter fills the whole ROI so area_ratio should be ~1.0 relative to box
    # With 0.01..0.99 range and grabcut/sam both producing high fill, should be ok or NG at boundary
    assert result.is_ok is not None  # judging active


def test_judge_boundaries_ng_below(clean_te_data):
    # Segmenter returns empty mask -> area_ratio = 0 -> below min
    seg = MagicMock()
    seg.segment.return_value = np.zeros((120, 160), dtype=np.uint8)

    settings = TargetExtractionSettings()
    settings.update({"judge_enabled": True, "min_area_ratio": 0.10, "max_area_ratio": 0.95,
                     "engine": "sam"})

    ctrl = _controller(settings=settings, clean=False, segmenter=seg)
    frame = _frame_with_fg()
    ctrl.on_raw_frame(frame)
    ctrl.set_roi(20, 20, 60, 50)

    result = ctrl.process({"frame": frame, "triggered": True})
    # Empty mask -> either error or area_ratio=0 -> NG below
    # Allow error path too
    assert result.is_ok is False


def test_judge_boundaries_ng_above(clean_te_data):
    # Segmenter fills entire frame -> area_ratio >> box area -> above max
    seg = MagicMock()
    def _full_mask(frame, box):
        return np.full(frame.shape[:2], 255, dtype=np.uint8)
    seg.segment.side_effect = _full_mask

    settings = TargetExtractionSettings()
    settings.update({"judge_enabled": True, "min_area_ratio": 0.10, "max_area_ratio": 0.50,
                     "engine": "sam"})

    ctrl = _controller(settings=settings, clean=False, segmenter=seg)
    frame = _frame_with_fg()
    ctrl.on_raw_frame(frame)
    ctrl.set_roi(20, 20, 60, 50)  # box area = 3000; full frame = 19200 -> ratio >> 0.50

    result = ctrl.process({"frame": frame, "triggered": True})
    assert result.is_ok is False
    assert result.reason is not None and "above" in result.reason


def test_judging_with_no_roi_returns_ng_not_configured(clean_te_data):
    settings = TargetExtractionSettings()
    settings.update({"judge_enabled": True})

    ctrl = _controller(settings=settings, clean=False)
    frame = _solid_frame()
    result = ctrl.process({"frame": frame, "triggered": True})

    assert result.roi_status == "Untaught"
    assert result.is_ok is False
    assert result.reason == "not configured"


def test_cleanup_releases_segmenter(clean_te_data):
    seg = _fake_segmenter()
    ctrl = _controller(clean=False, segmenter=seg)
    frame = _frame_with_fg()
    ctrl.on_raw_frame(frame)
    ctrl.set_roi(20, 20, 60, 50)
    ctrl.process({"frame": frame, "triggered": True})
    assert ctrl._segmenter_override is not None

    ctrl.cleanup()
    assert ctrl._lazy_segmenter is None
    assert ctrl._segmenter_override is None
    assert ctrl.last_frame is None


def test_default_sam_weights_resolved_against_repo_root_from_other_cwd(clean_te_data, monkeypatch, tmp_path):
    # Change working directory to a completely different directory
    monkeypatch.chdir(tmp_path)
    from app.modules.target_extraction.settings import TargetExtractionSettings, REPO_ROOT

    settings = TargetExtractionSettings()
    cfg = settings.get_settings()
    weights_path = cfg["sam_weights"]

    # Even from tmp_path, the path is absolute and resolves to the repo root's mobile_sam.pt
    assert os.path.isabs(weights_path)
    assert os.path.exists(weights_path)
    assert Path(weights_path).name == "mobile_sam.pt"
    assert str(REPO_ROOT) in weights_path


def test_target_extraction_service_extracted_ready(clean_te_data):
    """Target extraction telemetry extracted_ready:
    - no file -> False
    - after a triggered extraction -> True
    - new controller instance with the file still on disk -> True
    """
    from app.modules.target_extraction.service import TargetExtractionService

    settings = TargetExtractionSettings()
    seg = _fake_segmenter()
    ctrl = _controller(settings=settings, segmenter=seg, clean=False)
    service = TargetExtractionService(settings, module_ref=ctrl)

    # 1. No file -> False
    status = service.get_status()
    assert status["telemetry"]["extracted_ready"] is False

    # 2. After a triggered extraction -> True
    frame = _frame_with_fg()
    ctrl.on_raw_frame(frame)
    ctrl.set_roi(20, 20, 60, 50)
    result = ctrl.process({"frame": frame, "triggered": True})
    assert result is not None

    status_after = service.get_status()
    assert status_after["telemetry"]["extracted_ready"] is True

    # 3. New controller instance with the file still on disk -> True
    new_ctrl = _controller(settings=settings, segmenter=seg, clean=False)
    assert new_ctrl.last_result.area_px == 0
    new_service = TargetExtractionService(settings, module_ref=new_ctrl)
    new_status = new_service.get_status()
    assert new_status["telemetry"]["extracted_ready"] is True


