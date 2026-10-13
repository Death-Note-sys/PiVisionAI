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
def mock_measurement_service():
    with patch("app.api.v1.measurement.measurement_service") as mock_service:
        yield mock_service


def test_start_success(client, mock_measurement_service):
    mock_measurement_service.start.return_value = True

    response = client.post("/api/v1/measurement/start")

    assert response.status_code == 200
    assert response.get_json() == {"success": True}


def test_status_returns_404_when_not_active(client, mock_measurement_service):
    mock_measurement_service.get_status.return_value = None

    response = client.get("/api/v1/measurement/status")

    assert response.status_code == 404


def test_calibrate_rejects_missing_fields(client, mock_measurement_service):
    response = client.post("/api/v1/measurement/calibrate", json={"x1": 0, "y1": 0})

    assert response.status_code == 400


def test_calibrate_rejects_zero_real_length(client, mock_measurement_service):
    response = client.post(
        "/api/v1/measurement/calibrate",
        json={"x1": 0, "y1": 0, "x2": 100, "y2": 0, "real_length_mm": 0},
    )

    assert response.status_code == 400


def test_calibrate_success(client, mock_measurement_service):
    mock_measurement_service.calibrate.return_value = True

    response = client.post(
        "/api/v1/measurement/calibrate",
        json={"x1": 0, "y1": 0, "x2": 100, "y2": 0, "real_length_mm": 10.0},
    )

    assert response.status_code == 200
    mock_measurement_service.calibrate.assert_called_once_with(0, 0, 100, 0, 10.0)


def test_calibrate_failure_returns_500(client, mock_measurement_service):
    mock_measurement_service.calibrate.return_value = False

    response = client.post(
        "/api/v1/measurement/calibrate",
        json={"x1": 5, "y1": 5, "x2": 5, "y2": 5, "real_length_mm": 10.0},
    )

    assert response.status_code == 500


def test_settings_rejects_invalid_canny_threshold(client, mock_measurement_service):
    response = client.post("/api/v1/measurement/settings", json={"canny_low": 999})

    assert response.status_code == 400


def test_settings_accepts_judgment_settings(client, mock_measurement_service):
    mock_measurement_service.update_settings.return_value = True
    payload = {
        "target_width_mm": 25.4,
        "target_length_mm": 50.8,
        "tolerance_mm": 1.5,
        "judgment_roi": {"x": 10, "y": 20, "w": 200, "h": 150},
        "judge_enabled": True
    }
    response = client.post("/api/v1/measurement/settings", json=payload)
    assert response.status_code == 200
    mock_measurement_service.update_settings.assert_called_once_with(payload)


def test_settings_rejects_non_positive_tolerance(client, mock_measurement_service):
    response = client.post("/api/v1/measurement/settings", json={"tolerance_mm": 0.0, "judge_enabled": True})
    assert response.status_code == 400

    response2 = client.post("/api/v1/measurement/settings", json={"tolerance_mm": -1.5})
    assert response2.status_code == 400


def test_settings_rejects_judge_enabled_without_positive_tolerance(client, mock_measurement_service):
    mock_measurement_service.controller = None
    response = client.post("/api/v1/measurement/settings", json={"judge_enabled": True, "tolerance_mm": 0.0})
    assert response.status_code == 400


def test_settings_rejects_both_request_orders_against_merged_state(client, mock_measurement_service):
    mock_ctrl = MagicMock()
    mock_measurement_service.controller = mock_ctrl

    # Order 1: saved tolerance is 0, request enables judging -> 400
    mock_ctrl.settings.get_settings.return_value = {"tolerance_mm": 0.0, "judge_enabled": False}
    res1 = client.post("/api/v1/measurement/settings", json={"judge_enabled": True})
    assert res1.status_code == 400

    # Order 2: saved judging is True, request sets tolerance to 0.0 -> 400
    mock_ctrl.settings.get_settings.return_value = {"tolerance_mm": 2.0, "judge_enabled": True}
    res2 = client.post("/api/v1/measurement/settings", json={"tolerance_mm": 0.0})
    assert res2.status_code == 400


