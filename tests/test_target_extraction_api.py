"""Tests for the Target Extraction API blueprint.

No camera or SAM weights needed; the service layer is replaced by a stub.
"""
import os
import pytest
from unittest.mock import MagicMock, patch
from flask import Flask


# ---------------------------------------------------------------------------
# Minimal Flask app wrapping only the target_extraction blueprint
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    from app.api.v1.target_extraction import bp, target_extraction_service

    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True

    # Patch the app-level service singleton with a controllable mock
    mock_svc = MagicMock()
    mock_svc.settings = MagicMock()
    mock_svc.settings.get_settings.return_value = {
        "engine": "auto",
        "sam_weights": "models/mobile_sam.pt",
        "grabcut_iterations": 5,
        "roi_box": {},
        "view_mode": "overlay",
        "show_info": True,
        "thickness": 2,
        "judge_enabled": False,
        "min_area_ratio": 0.10,
        "max_area_ratio": 0.95,
        "version": 1,
    }
    mock_svc.update_settings.return_value = True
    mock_svc.set_roi.return_value = True
    mock_svc.reset_roi.return_value = True
    mock_svc.get_extraction_path.return_value = None
    mock_svc.get_status.return_value = None

    import app.api.v1.target_extraction as te_api
    original = te_api.target_extraction_service
    te_api.target_extraction_service = mock_svc

    flask_app.register_blueprint(bp)

    with flask_app.test_client() as client:
        yield client, mock_svc

    te_api.target_extraction_service = original


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------

def test_settings_validation_engine_enum_rejected(app):
    client, _ = app
    resp = client.post(
        "/api/v1/target-extraction/settings",
        json={"engine": "magic_engine"},
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_settings_validation_min_max_ratio(app):
    client, _ = app
    # min >= max must 400
    resp = client.post(
        "/api/v1/target-extraction/settings",
        json={"min_area_ratio": 0.80, "max_area_ratio": 0.50},
        content_type="application/json",
    )
    assert resp.status_code == 400

    # Valid update succeeds
    resp_ok = client.post(
        "/api/v1/target-extraction/settings",
        json={"min_area_ratio": 0.10, "max_area_ratio": 0.90},
        content_type="application/json",
    )
    assert resp_ok.status_code == 200


# ---------------------------------------------------------------------------
# set-roi
# ---------------------------------------------------------------------------

def test_set_roi_400s_on_bad_box(app):
    client, _ = app
    # w must be > 0 (Field gt=0)
    resp = client.post(
        "/api/v1/target-extraction/set-roi",
        json={"x": 0, "y": 0, "w": 0, "h": 10},
        content_type="application/json",
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# extracted.png
# ---------------------------------------------------------------------------

def test_extracted_png_404_then_200(app, tmp_path):
    client, mock_svc = app

    # 404 when no extraction exists
    mock_svc.get_extraction_path.return_value = None
    resp = client.get("/api/v1/target-extraction/extracted.png")
    assert resp.status_code == 404

    # 200 when path exists
    import cv2
    import numpy as np
    png_path = str(tmp_path / "last_extraction.png")
    cv2.imwrite(png_path, np.zeros((10, 10, 4), dtype=np.uint8))

    mock_svc.get_extraction_path.return_value = png_path
    resp2 = client.get("/api/v1/target-extraction/extracted.png")
    assert resp2.status_code == 200
    assert resp2.content_type == "image/png"
