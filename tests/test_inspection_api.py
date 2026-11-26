import types

from app.api import create_app
from app.core.container import Container
from app.core.models.results import ObjectCountingResult, TargetExtractionResult


def test_inspection_status_last_judgment_and_trigger_id(tmp_path, monkeypatch):
    """GET /api/v1/inspection/status exposes per-module last_judgment and last_trigger_id."""
    monkeypatch.setenv("PI_VISION_DATA_DIR", str(tmp_path))

    app = create_app()
    app.config["TESTING"] = True
    container = Container.get_instance()
    container.config._settings.data_dir = str(tmp_path)
    container.state_store.reset_counters()

    mod_counting = types.SimpleNamespace(id="core-object-counting")
    mod_extraction = types.SimpleNamespace(id="core-target-extraction")
    container.pipeline.active_module = mod_counting

    with app.test_client() as client:
        # 1. Idle state before any inspections
        res0 = client.get("/api/v1/inspection/status")
        assert res0.status_code == 200
        d0 = res0.get_json()
        assert d0["module_id"] == "core-object-counting"
        assert d0["ok_count"] == 0
        assert d0["ng_count"] == 0
        assert d0["total_count"] == 0
        assert d0["last_judgment"] is None
        assert d0["last_trigger_id"] is None

        # 2. Record an OK inspection for core-object-counting with trigger_id=5
        container.coordinator.record(
            ObjectCountingResult(template_status="Taught", count=7, is_ok=True, reason="count OK"),
            frame=None,
            module_id="core-object-counting",
            trigger_id=5,
        )
        res1 = client.get("/api/v1/inspection/status")
        assert res1.status_code == 200
        d1 = res1.get_json()
        assert d1["ok_count"] == 1
        assert d1["ng_count"] == 0
        assert d1["total_count"] == 1
        assert d1["last_judgment"] == "OK"
        assert d1["last_trigger_id"] == 5

        # 3. Record an NG inspection for core-object-counting with trigger_id=6
        container.coordinator.record(
            ObjectCountingResult(template_status="Taught", count=6, is_ok=False, reason="count 6 outside [7, 7]"),
            frame=None,
            module_id="core-object-counting",
            trigger_id=6,
        )
        res2 = client.get("/api/v1/inspection/status")
        assert res2.status_code == 200
        d2 = res2.get_json()
        assert d2["ok_count"] == 1
        assert d2["ng_count"] == 1
        assert d2["total_count"] == 2
        assert d2["last_judgment"] == "NG"
        assert d2["last_trigger_id"] == 6

        # 4. Switching active module returns independent state for that module
        container.pipeline.active_module = mod_extraction
        res_other = client.get("/api/v1/inspection/status")
        assert res_other.status_code == 200
        d_other = res_other.get_json()
        assert d_other["module_id"] == "core-target-extraction"
        assert d_other["last_judgment"] is None
        assert d_other["last_trigger_id"] is None

        container.coordinator.record(
            TargetExtractionResult(roi_status="Taught", engine="grabcut", area_ratio=0.35, is_ok=True, reason="area ratio OK"),
            frame=None,
            module_id="core-target-extraction",
            trigger_id=9,
        )
        res_other2 = client.get("/api/v1/inspection/status")
        d_other2 = res_other2.get_json()
        assert d_other2["last_judgment"] == "OK"
        assert d_other2["last_trigger_id"] == 9

        # 5. Reset active module counters clears last_judgment and last_trigger_id to null
        res_reset = client.post("/api/v1/inspection/reset", json={"module_id": "core-object-counting"})
        assert res_reset.status_code == 200
        container.pipeline.active_module = mod_counting
        res_after_reset = client.get("/api/v1/inspection/status")
        d_after = res_after_reset.get_json()
        assert d_after["total_count"] == 0
        assert d_after["ok_count"] == 0
        assert d_after["ng_count"] == 0
        assert d_after["last_judgment"] is None
        assert d_after["last_trigger_id"] is None
