import cv2
import numpy as np
import pytest

from app.modules.object_counting.matcher import (
    Match, find_matches, judge_count, non_max_suppression, validate_template,
)


def make_template(size=32, seed=7):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (size, size), dtype=np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(noise, (3, 3), 0), cv2.COLOR_GRAY2BGR)


def make_scene(template, positions, shape=(240, 320), seed=1, rotate=False, scale=1.0):
    rng = np.random.default_rng(seed)
    scene = cv2.cvtColor(rng.integers(0, 256, shape, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    tmpl = template
    if rotate:
        tmpl = cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
    if scale != 1.0:
        tmpl = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = tmpl.shape[:2]
    for (x, y) in positions:
        scene[y:y + h, x:x + w] = tmpl
    return scene


POSITIONS = [(10, 10), (80, 15), (150, 20), (30, 90), (120, 100), (220, 60), (60, 170), (200, 180)]


def test_counts_all_pasted_copies():
    t = make_template()
    scene = make_scene(t, POSITIONS)
    hits = find_matches(scene, t, threshold=0.7)
    assert len(hits) == len(POSITIONS)
    assert all(m.score > 0.95 for m in hits)


def test_boxes_land_on_the_pasted_positions():
    t = make_template()
    scene = make_scene(t, POSITIONS)
    found = {(m.x, m.y) for m in find_matches(scene, t, threshold=0.7)}
    assert found == set(POSITIONS)


def test_no_objects_gives_zero():
    t = make_template()
    scene = make_scene(t, [])
    assert find_matches(scene, t, threshold=0.7) == []


def test_touching_neighbours_are_counted_separately():
    t = make_template()
    scene = make_scene(t, [(40, 40), (72, 40), (104, 40)])  # side by side, no gap
    assert len(find_matches(scene, t, threshold=0.7)) == 3


def test_max_objects_caps_the_result():
    t = make_template()
    scene = make_scene(t, POSITIONS)
    assert len(find_matches(scene, t, threshold=0.7, max_objects=3)) == 3


def test_rotated_copies_are_not_found():
    """Documented limitation: template matching needs the same orientation."""
    t = make_template()
    scene = make_scene(t, POSITIONS[:4], rotate=True)
    assert len(find_matches(scene, t, threshold=0.7)) == 0


def test_scaled_copies_need_the_scales_list():
    t = make_template()
    scene = make_scene(t, POSITIONS[:4], scale=1.25)
    assert len(find_matches(scene, t, threshold=0.8)) == 0
    hits = find_matches(scene, t, threshold=0.8, scales=(1.0, 1.25))
    assert len(hits) == 4


def test_roi_limits_the_search_and_keeps_frame_coordinates():
    t = make_template()
    scene = make_scene(t, POSITIONS)
    # Only copies that fit entirely inside the ROI can be found.
    hits = find_matches(scene, t, threshold=0.7, roi=(0, 0, 200, 140))
    assert {(m.x, m.y) for m in hits} == {(10, 10), (80, 15), (150, 20), (30, 90), (120, 100)}


def test_work_scale_still_counts_correctly():
    t = make_template(size=48)
    scene = make_scene(t, [(10, 10), (100, 20), (180, 60), (40, 150)], shape=(240, 320))
    hits = find_matches(scene, t, threshold=0.6, work_scale=0.5)
    assert len(hits) == 4


def test_template_larger_than_roi_returns_empty():
    t = make_template()
    scene = make_scene(t, POSITIONS)
    assert find_matches(scene, t, roi=(0, 0, 10, 10)) == []


def test_flat_template_is_rejected():
    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    ok, reason = validate_template(flat)
    assert ok is False and "texture" in reason


def test_tiny_template_is_rejected():
    ok, reason = validate_template(np.zeros((4, 4, 3), dtype=np.uint8))
    assert ok is False


def test_textured_template_is_accepted():
    ok, reason = validate_template(make_template())
    assert ok is True and reason == ""


def test_nms_merges_overlapping_hits_keeping_best():
    a = Match(10, 10, 30, 30, 0.9)
    b = Match(12, 11, 30, 30, 0.8)
    c = Match(100, 100, 30, 30, 0.7)
    kept = non_max_suppression([b, a, c], iou_threshold=0.3)
    assert [m.score for m in kept] == [0.9, 0.7]


@pytest.mark.parametrize("count,cmin,cmax,expected", [
    (0, 0, 0, True),    # expect none, none found
    (1, 0, 0, False),   # expect none, one found
    (5, 5, 5, True),    # exact
    (4, 5, 5, False),
    (6, 5, 5, False),
    (5, 3, 0, True),    # no upper limit
    (2, 3, 0, False),
    (9, 3, 8, False),
])
def test_judge_count(count, cmin, cmax, expected):
    assert judge_count(count, cmin, cmax) is expected
