import os
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock

from app.core.event_bus import EventBus
from app.core.container import Container
from app.modules.object_counting.controller import ObjectCountingController
from app.modules.object_counting.settings import ObjectCountingSettings


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


@pytest.fixture
def clean_counting_data():
    container = Container.get_instance()
    mod_dir = container.session_manager.get_data_dir("modules/core-object-counting")
    os.makedirs(mod_dir, exist_ok=True)
    for fname in ["settings.json", "data.json", "template.png", "template.png.tmp"]:
        p = os.path.join(mod_dir, fname)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    yield mod_dir
    for fname in ["settings.json", "data.json", "template.png", "template.png.tmp"]:
        p = os.path.join(mod_dir, fname)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def test_untaught_reports_zero_count_and_untaught_status(clean_counting_data):
    bus = EventBus()
    settings = ObjectCountingSettings()
    controller = ObjectCountingController(bus, settings)

    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    res = controller.process({"frame": dummy_frame})

    assert res.count == 0
    assert res.template_status == "Untaught"
    assert res.is_ok is None

    # When judging is enabled, untaught reports NG
    controller.configure({"judge_enabled": True})
    res_judged = controller.process({"frame": dummy_frame})
    assert res_judged.count == 0
    assert res_judged.template_status == "Untaught"
    assert res_judged.is_ok is False


def test_teach_on_synthetic_scene_finds_all_objects(clean_counting_data):
    bus = EventBus()
    settings = ObjectCountingSettings()
    controller = ObjectCountingController(bus, settings)

    template = make_template(size=32, seed=7)
    scene = make_scene(template, POSITIONS)

    # Supply frame via hook and teach the template at first position (10, 10, 32, 32)
    controller.on_raw_frame(scene)
    ok = controller.teach_template(10, 10, 32, 32)
    assert ok is True
    assert controller.template_status == "Taught"

    res = controller.process({"frame": scene})
    assert res.count == len(POSITIONS)
    assert res.template_status == "Taught"
    assert len(res.matches) == len(POSITIONS)


def test_flat_template_rejected(clean_counting_data):
    bus = EventBus()
    settings = ObjectCountingSettings()
    controller = ObjectCountingController(bus, settings)

    flat_frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    controller.on_raw_frame(flat_frame)

    ok = controller.teach_template(10, 10, 32, 32)
    assert ok is False
    assert controller.template_status == "Untaught"
    assert controller.last_teach_error is not None
    assert "texture" in controller.last_teach_error.lower()


def test_template_larger_than_frame_rejected(clean_counting_data):
    bus = EventBus()
    settings = ObjectCountingSettings()
    controller = ObjectCountingController(bus, settings)

    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    controller.on_raw_frame(frame)

    ok = controller.teach_template(0, 0, 150, 150)
    assert ok is False
    assert "larger than frame" in controller.last_teach_error


def test_judge_boundaries_including_zero_zero_expect_none(clean_counting_data):
    bus = EventBus()
    settings = ObjectCountingSettings()
    controller = ObjectCountingController(bus, settings)

    template = make_template(size=32, seed=7)
    # Scene with 4 objects
    scene_4 = make_scene(template, POSITIONS[:4])
    # Scene with 0 objects
    scene_0 = make_scene(template, [])

    controller.on_raw_frame(scene_4)
    assert controller.teach_template(10, 10, 32, 32) is True

    # 1. Expect none (0/0): OK only when count == 0
    controller.configure({"judge_enabled": True, "count_min": 0, "count_max": 0})
    res_0 = controller.process({"frame": scene_0})
    assert res_0.count == 0
    assert res_0.is_ok is True

    res_4 = controller.process({"frame": scene_4})
    assert res_4.count == 4
    assert res_4.is_ok is False

    # 2. Exact match (4/4): OK when count == 4
    controller.configure({"count_min": 4, "count_max": 4})
    res_4 = controller.process({"frame": scene_4})
    assert res_4.count == 4
    assert res_4.is_ok is True

    res_0 = controller.process({"frame": scene_0})
    assert res_0.count == 0
    assert res_0.is_ok is False

    # 3. Range (2..5)
    controller.configure({"count_min": 2, "count_max": 5})
    assert controller.process({"frame": scene_4}).is_ok is True
    assert controller.process({"frame": scene_0}).is_ok is False

    # 4. Open upper bound (count_max = 0): at least 2
    controller.configure({"count_min": 2, "count_max": 0})
    assert controller.process({"frame": scene_4}).is_ok is True
    assert controller.process({"frame": scene_0}).is_ok is False


def test_persistence_round_trip(clean_counting_data):
    bus = EventBus()
    settings = ObjectCountingSettings()
    c1 = ObjectCountingController(bus, settings)

    template = make_template(size=32, seed=7)
    scene = make_scene(template, POSITIONS)

    c1.on_raw_frame(scene)
    assert c1.teach_template(10, 10, 32, 32) is True
    assert c1.template_status == "Taught"

    res1 = c1.process({"frame": scene})
    assert res1.count == len(POSITIONS)

    # Reload into a completely fresh controller instance
    c2 = ObjectCountingController(bus, ObjectCountingSettings())
    assert c2.template_status == "Taught"
    assert c2.template_bgr is not None

    res2 = c2.process({"frame": scene})
    assert res2.count == res1.count == len(POSITIONS)


def test_reset_template_clears_template_and_keeps_counters(clean_counting_data):
    bus = EventBus()
    settings = ObjectCountingSettings()
    controller = ObjectCountingController(bus, settings)

    template = make_template(size=32, seed=7)
    scene = make_scene(template, POSITIONS)

    controller.on_raw_frame(scene)
    assert controller.teach_template(10, 10, 32, 32) is True

    # Check template file exists
    tpl_path = controller.get_template_path()
    assert tpl_path is not None and os.path.exists(tpl_path)

    # Reset template
    assert controller.reset_template() is True
    assert controller.template_status == "Untaught"
    assert controller.template_bgr is None
    assert controller.get_template_path() is None
    assert not os.path.exists(tpl_path)


def test_max_objects_cap(clean_counting_data):
    bus = EventBus()
    settings = ObjectCountingSettings()
    controller = ObjectCountingController(bus, settings)

    template = make_template(size=32, seed=7)
    scene = make_scene(template, POSITIONS)  # 8 positions

    controller.on_raw_frame(scene)
    assert controller.teach_template(10, 10, 32, 32) is True

    controller.configure({"max_objects": 3})
    res = controller.process({"frame": scene})
    assert res.count == 3


def test_resolution_safety_same_size_counts(clean_counting_data):
    bus = EventBus()
    settings = ObjectCountingSettings()
    controller = ObjectCountingController(bus, settings)

    template = make_template(size=32, seed=7)
    scene = make_scene(template, POSITIONS, shape=(240, 320))

    controller.on_raw_frame(scene)
    assert controller.teach_template(10, 10, 32, 32) is True
    assert controller.taught_frame_width == 320
    assert controller.taught_frame_height == 240

    # Same size frame counts normally
    res = controller.process({"frame": scene})
    assert res.count == len(POSITIONS)
    assert res.template_status == "Taught"


def test_resolution_safety_different_size_flags_reteach(clean_counting_data):
    bus = EventBus()
    settings = ObjectCountingSettings()
    controller = ObjectCountingController(bus, settings)

    template = make_template(size=32, seed=7)
    scene = make_scene(template, POSITIONS, shape=(240, 320))

    controller.on_raw_frame(scene)
    assert controller.teach_template(10, 10, 32, 32) is True
    assert controller.taught_frame_width == 320
    assert controller.taught_frame_height == 240

    # Live frame with different resolution (480x640)
    different_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Without judging
    res_no_judge = controller.process({"frame": different_frame})
    assert res_no_judge.count == 0
    assert res_no_judge.template_status == "Re-teach (resolution changed)"
    assert res_no_judge.is_ok is None

    # 2. With judging enabled: returns NG with reason
    controller.configure({"judge_enabled": True})
    res_judged = controller.process({"frame": different_frame})
    assert res_judged.count == 0
    assert res_judged.template_status == "Re-teach (resolution changed)"
    assert res_judged.is_ok is False
    assert res_judged.reason == "template taught at 320x240, frame is 640x480"


def test_resolution_safety_old_data_without_size_works(clean_counting_data):
    import json
    container = Container.get_instance()
    mod_dir = container.session_manager.get_data_dir("modules/core-object-counting")

    # Simulate an old data.json created before resolution tracking was added
    template = make_template(size=32, seed=7)
    cv2.imwrite(os.path.join(mod_dir, "template.png"), template)
    with open(os.path.join(mod_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump({"template_status": "Taught"}, f)

    bus = EventBus()
    controller = ObjectCountingController(bus, ObjectCountingSettings())

    assert controller.template_status == "Taught"
    assert controller.taught_frame_width is None
    assert controller.taught_frame_height is None

    scene = make_scene(template, POSITIONS, shape=(240, 320))
    res = controller.process({"frame": scene})
    assert res.count == len(POSITIONS)
    assert res.template_status == "Taught"
