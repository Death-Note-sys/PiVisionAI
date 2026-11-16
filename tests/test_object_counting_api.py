import os
import tempfile
import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.api import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_counting_service():
    with patch("app.api.v1.object_counting.object_counting_service") as mock_service:
        # Default mock settings
        mock_settings = MagicMock()
        mock_settings.get_settings.return_value = {
            "count_min": 0,
            "count_max": 0,
            "judge_enabled": False,
            "match_threshold": 0.7,
        }
        mock_service.settings = mock_settings
        yield mock_service


def test_status_returns_404_when_inactive(client, mock_counting_service):
    mock_counting_service.get_status.return_value = None

    response = client.get("/api/v1/object-counting/status")
    assert response.status_code == 404
    assert response.get_json()["error"] == "Object Counting is not the active module"


def test_status_returns_200_when_active(client, mock_counting_service):
    mock_counting_service.get_status.return_value = {
        "active": True,
        "paused": False,
        "settings": {"match_threshold": 0.7},
    }

    response = client.get("/api/v1/object-counting/status")
    assert response.status_code == 200
    assert response.get_json()["active"] is True


def test_teach_template_validation_400(client, mock_counting_service):
    # 1. Missing payload / empty JSON
    res1 = client.post("/api/v1/object-counting/teach-template", json={})
    assert res1.status_code == 400

    # 2. Invalid dimensions (negative x)
    res2 = client.post("/api/v1/object-counting/teach-template", json={"x": -1, "y": 0, "w": 10, "h": 10})
    assert res2.status_code == 400

    # 3. Non-positive width
    res3 = client.post("/api/v1/object-counting/teach-template", json={"x": 0, "y": 0, "w": 0, "h": 10})
    assert res3.status_code == 400

    # 4. Service rejects teach (e.g. flat texture)
    mock_counting_service.teach_template.return_value = (False, "template has almost no texture")
    res4 = client.post("/api/v1/object-counting/teach-template", json={"x": 10, "y": 10, "w": 50, "h": 50})
    assert res4.status_code == 400
    assert "texture" in res4.get_json()["error"]


def test_teach_template_success(client, mock_counting_service):
    mock_counting_service.teach_template.return_value = (True, "")

    response = client.post(
        "/api/v1/object-counting/teach-template",
        json={"x": 10, "y": 20, "w": 32, "h": 32}
    )
    assert response.status_code == 200
    assert response.get_json() == {"success": True}
    mock_counting_service.teach_template.assert_called_once_with(10, 20, 32, 32)


def test_settings_invalid_count_min_greater_than_max_both_orders(client, mock_counting_service):
    # Case A: Explicit payload contains count_min > count_max
    mock_counting_service.settings.get_settings.return_value = {"count_min": 0, "count_max": 0}
    res_both = client.post(
        "/api/v1/object-counting/settings",
        json={"count_min": 5, "count_max": 2}
    )
    assert res_both.status_code == 400
    mock_counting_service.update_settings.assert_not_called()

    # Case B: Request order 1 — current count_max=2, client posts count_min=5
    mock_counting_service.settings.get_settings.return_value = {"count_min": 0, "count_max": 2}
    res_min = client.post(
        "/api/v1/object-counting/settings",
        json={"count_min": 5}
    )
    assert res_min.status_code == 400
    mock_counting_service.update_settings.assert_not_called()

    # Case C: Request order 2 — current count_min=5, client posts count_max=2
    mock_counting_service.settings.get_settings.return_value = {"count_min": 5, "count_max": 10}
    res_max = client.post(
        "/api/v1/object-counting/settings",
        json={"count_max": 2}
    )
    assert res_max.status_code == 400
    mock_counting_service.update_settings.assert_not_called()


def test_template_png_404_then_200(client, mock_counting_service, tmp_path):
    # 1. Untaught -> 404
    mock_counting_service.get_template_path.return_value = None
    res_404 = client.get("/api/v1/object-counting/template.png")
    assert res_404.status_code == 404

    # 2. Taught -> 200 with PNG image
    test_img_path = str(tmp_path / "test_template.png")
    dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(test_img_path, dummy_img)

    mock_counting_service.get_template_path.return_value = test_img_path
    res_200 = client.get("/api/v1/object-counting/template.png")
    assert res_200.status_code == 200
    assert "image/png" in res_200.content_type
