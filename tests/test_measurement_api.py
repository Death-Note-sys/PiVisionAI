import pytest
from unittest.mock import patch
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
        json={"x1": 0, "y1": 0, "x2": 100, "y2": 0, "real_length_cm": 0},
    )

    assert response.status_code == 400


def test_calibrate_success(client, mock_measurement_service):
    mock_measurement_service.calibrate.return_value = True

    response = client.post(
        "/api/v1/measurement/calibrate",
        json={"x1": 0, "y1": 0, "x2": 100, "y2": 0, "real_length_cm": 10.0},
    )

    assert response.status_code == 200
    mock_measurement_service.calibrate.assert_called_once_with(0, 0, 100, 0, 10.0)


def test_calibrate_failure_returns_500(client, mock_measurement_service):
    mock_measurement_service.calibrate.return_value = False

    response = client.post(
        "/api/v1/measurement/calibrate",
        json={"x1": 5, "y1": 5, "x2": 5, "y2": 5, "real_length_cm": 10.0},
    )

    assert response.status_code == 500


def test_settings_rejects_invalid_canny_threshold(client, mock_measurement_service):
    response = client.post("/api/v1/measurement/settings", json={"canny_low": 999})

    assert response.status_code == 400
