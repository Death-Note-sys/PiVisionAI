import cv2
import numpy as np
import pytest

from app.modules.target_extraction.extractor import (
    build_cutout, clamp_box, clean_mask, composite_on_color, extract_target,
    grabcut_mask, judge_extraction, largest_component, mask_stats,
)


def make_scene(shape=(240, 320), center=(160, 120), radius=40, seed=3):
    """Cluttered grey-noise background with a solid orange disc as the target."""
    rng = np.random.default_rng(seed)
    bg = rng.integers(90, 170, shape, dtype=np.uint8)
    scene = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    cv2.circle(scene, center, radius, (30, 140, 255), -1)  # BGR orange
    truth = np.zeros(shape, np.uint8)
    cv2.circle(truth, center, radius, 255, -1)
    box = (center[0] - radius - 25, center[1] - radius - 25, 2 * radius + 50, 2 * radius + 50)
    return scene, truth, box


def iou(a, b):
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


class FakeSegmenter:
    def __init__(self, mask=None, error=None):
        self.mask, self.error, self.calls = mask, error, 0

    def segment(self, frame, box):
        self.calls += 1
        if self.error:
            raise self.error
        return self.mask


def test_grabcut_recovers_a_clear_target():
    scene, truth, box = make_scene()
    assert iou(grabcut_mask(scene, box), truth) > 0.85


def test_extract_without_segmenter_uses_grabcut_and_says_why():
    scene, truth, box = make_scene()
    result = extract_target(scene, box, segmenter=None, engine="auto")
    assert result.engine == "grabcut"
    assert result.fallback_reason == "no SAM segmenter available"
    assert iou(result.mask, truth) > 0.85


def test_extract_uses_sam_when_it_works():
    scene, truth, box = make_scene()
    seg = FakeSegmenter(mask=truth)
    result = extract_target(scene, box, segmenter=seg, engine="auto")
    assert result.engine == "sam" and result.fallback_reason is None and seg.calls == 1


def test_extract_falls_back_to_grabcut_when_sam_raises():
    scene, truth, box = make_scene()
    seg = FakeSegmenter(error=RuntimeError("weights missing"))
    result = extract_target(scene, box, segmenter=seg, engine="auto")
    assert result.engine == "grabcut"
    assert "weights missing" in result.fallback_reason
    assert iou(result.mask, truth) > 0.85


def test_engine_sam_does_not_hide_a_failure():
    scene, _, box = make_scene()
    with pytest.raises(RuntimeError):
        extract_target(scene, box, segmenter=FakeSegmenter(error=RuntimeError("boom")), engine="sam")
    with pytest.raises(ValueError):
        extract_target(scene, box, segmenter=None, engine="sam")


def test_engine_grabcut_never_calls_the_segmenter():
    scene, _, box = make_scene()
    seg = FakeSegmenter(mask=np.zeros(scene.shape[:2], np.uint8))
    result = extract_target(scene, box, segmenter=seg, engine="grabcut")
    assert result.engine == "grabcut" and seg.calls == 0


def test_unknown_engine_is_rejected():
    scene, _, box = make_scene()
    with pytest.raises(ValueError):
        extract_target(scene, box, engine="magic")


def test_box_outside_frame_is_rejected():
    scene, _, _ = make_scene()
    with pytest.raises(ValueError):
        grabcut_mask(scene, (1000, 1000, 50, 50))
    assert clamp_box((300, 220, 100, 100), scene.shape) == (300, 220, 20, 20)
    assert clamp_box((0, 0, 4, 4), scene.shape) is None


def test_largest_component_drops_speckle():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:60, 20:60] = 255
    mask[80:83, 80:83] = 255
    kept = largest_component(mask)
    assert kept[81, 81] == 0 and kept[40, 40] == 255


def test_clean_mask_closes_a_small_hole():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 255
    mask[48:51, 48:51] = 0
    assert clean_mask(mask)[49, 49] == 255


def test_mask_stats_reports_area_ratio_bbox_and_centroid():
    mask = np.zeros((100, 200), np.uint8)
    mask[20:60, 50:110] = 255            # 40 x 60 = 2400 px
    stats = mask_stats(mask, (40, 10, 80, 60))   # box area 4800
    assert stats["area_px"] == 2400
    assert stats["area_ratio"] == 0.5
    assert stats["bbox"] == {"x": 50, "y": 20, "w": 60, "h": 40}
    assert stats["centroid"] == {"x": 79.5, "y": 39.5}


def test_mask_stats_for_empty_mask():
    stats = mask_stats(np.zeros((50, 50), np.uint8), (0, 0, 10, 10))
    assert stats == {"area_px": 0, "area_ratio": 0.0, "bbox": None, "centroid": None}


def test_judge_extraction_bounds():
    assert judge_extraction(0.5, 0.1, 0.95) is True
    assert judge_extraction(0.05, 0.1, 0.95) is False   # target missing
    assert judge_extraction(0.99, 0.1, 0.95) is False   # mask leaked into the background


def test_cutout_is_transparent_outside_and_cropped():
    scene, truth, _ = make_scene()
    cut = build_cutout(scene, truth)
    assert cut.shape[2] == 4
    assert cut.shape[0] <= 82 and cut.shape[1] <= 82
    assert cut[0, 0, 3] == 0          # corner of the crop is outside the disc
    assert cut[cut.shape[0] // 2, cut.shape[1] // 2, 3] == 255
    full = build_cutout(scene, truth, crop_to_bbox=False)
    assert full.shape[:2] == scene.shape[:2]


def test_composite_paints_background_and_keeps_target():
    scene, truth, _ = make_scene()
    out = composite_on_color(scene, truth, color=(10, 10, 10))
    assert tuple(out[0, 0]) == (10, 10, 10)
    assert np.array_equal(out[120, 160], scene[120, 160])
