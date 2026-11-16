import sqlite3
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock

from app.core.event_bus import EventBus
from app.core.pipeline import Pipeline
from app.core.renderer_manager import RendererManager
from app.core.output_manager import OutputManager
from app.core.coordinator import InspectionCoordinator
from app.core.state import StateStore
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


def test_object_counting_pipeline_single_vs_continuous_trigger():
    event_bus = EventBus()
    state_store = StateStore(persist=False)
    output_manager = MagicMock(spec=OutputManager)
    output_manager.broadcast = MagicMock()
    output_manager.save_ng = MagicMock()

    coordinator = InspectionCoordinator(event_bus, state_store, output_manager)
    renderer_manager = RendererManager()

    pipeline = Pipeline(event_bus, renderer_manager, output_manager, coordinator=coordinator)

    # Initialize Object Counting Controller
    settings = ObjectCountingSettings()
    controller = ObjectCountingController(event_bus, settings)

    template = make_template(size=32, seed=7)
    scene = make_scene(template, POSITIONS)

    # Teach template
    controller.on_raw_frame(scene)
    assert controller.teach_template(10, 10, 32, 32) is True

    # Configure judging: OK when count between 1 and 20
    controller.configure({"judge_enabled": True, "count_min": 1, "count_max": 20})

    pipeline.set_module(controller)

    # ---------------------------------------------------------
    # 1. Continuous Mode: preview only -> 0 inspection rows, 0 counter increments
    # ---------------------------------------------------------
    pipeline.set_trigger_mode("continuous")

    pipeline.process_frame(scene)
    pipeline.process_frame(scene)

    db_path = coordinator._get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS inspections(
                id INTEGER PRIMARY KEY, ts REAL, trigger_id INTEGER,
                module_id TEXT, judgment TEXT, code INTEGER, reason TEXT,
                values_json TEXT, raw_path TEXT, overlay_path TEXT
            )
        ''')
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM inspections WHERE module_id = 'core-object-counting'")
        continuous_rows = cur.fetchone()[0]

    assert continuous_rows == 0
    assert state_store.total_count == 0

    # ---------------------------------------------------------
    # 2. Single Mode: untriggered frames do not record
    # ---------------------------------------------------------
    pipeline.set_trigger_mode("single")

    pipeline.process_frame(scene)
    pipeline.process_frame(scene)

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM inspections WHERE module_id = 'core-object-counting'")
        assert cur.fetchone()[0] == 0
    assert state_store.total_count == 0

    # ---------------------------------------------------------
    # 3. Fire trigger in Single Mode -> exactly 1 inspection row & 1 counter increment
    # ---------------------------------------------------------
    pipeline.fire_trigger()
    pipeline.process_frame(scene)

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), judgment FROM inspections WHERE module_id = 'core-object-counting' GROUP BY judgment")
        row = cur.fetchone()
        assert row is not None
        single_count, judgment = row
        assert single_count == 1
        assert judgment == "OK"

    assert state_store.total_count == 1
    assert state_store.ok_count == 1
    assert state_store.ng_count == 0
    per_module = state_store.get_counters("core-object-counting")
    assert per_module["ok_count"] == 1
    assert per_module["total_count"] == 1

    # ---------------------------------------------------------
    # 4. Subsequent frames in Single Mode -> still exactly 1 inspection row
    # ---------------------------------------------------------
    pipeline.process_frame(scene)
    pipeline.process_frame(scene)

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM inspections WHERE module_id = 'core-object-counting'")
        assert cur.fetchone()[0] == 1

    assert state_store.total_count == 1
