import pytest
from unittest.mock import MagicMock, patch
from app.api import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_od_service():
    with patch("app.api.v1.object_detection.od_service") as mock_service:
        yield mock_service


def test_start_success(client, mock_od_service):
    mock_od_service.start.return_value = True

    response = client.post("/api/v1/object-detection/start")

    assert response.status_code == 200
    assert response.get_json() == {"success": True}


def test_start_failure_returns_500(client, mock_od_service):
    mock_od_service.start.return_value = False

    response = client.post("/api/v1/object-detection/start")

    assert response.status_code == 500
    assert "error" in response.get_json()


def test_status_returns_404_when_not_active(client, mock_od_service):
    mock_od_service.get_status.return_value = None

    response = client.get("/api/v1/object-detection/status")

    assert response.status_code == 404


def test_status_returns_data_when_active(client, mock_od_service):
    mock_od_service.get_status.return_value = {
        "active": True,
        "paused": False,
        "settings": {"confidence": 0.5},
        "telemetry": {"objects_count": 2, "latency_ms": 40.0},
    }

    response = client.get("/api/v1/object-detection/status")

    assert response.status_code == 200
    data = response.get_json()
    assert data["telemetry"]["objects_count"] == 2


def test_update_settings_rejects_invalid_confidence(client, mock_od_service):
    response = client.post("/api/v1/object-detection/settings", json={"confidence": 1.5})

    assert response.status_code == 400


def test_update_settings_accepts_valid_payload(client, mock_od_service):
    mock_od_service.update_settings.return_value = True

    response = client.post("/api/v1/object-detection/settings", json={"confidence": 0.7, "show_labels": False})

    assert response.status_code == 200
    mock_od_service.update_settings.assert_called_once_with({"confidence": 0.7, "show_labels": False})


def test_switch_model_missing_payload_returns_400(client, mock_od_service):
    response = client.post("/api/v1/object-detection/model", json={})

    assert response.status_code == 400


def test_switch_model_success(client, mock_od_service):
    mock_od_service.switch_model.return_value = True

    response = client.post("/api/v1/object-detection/model", json={"model_id": "yolo11s"})

    assert response.status_code == 200
    assert response.get_json()["model_id"] == "yolo11s"
