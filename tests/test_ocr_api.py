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
def mock_ocr_service():
    with patch("app.api.v1.ocr.ocr_service") as mock_service:
        yield mock_service


def test_start_success(client, mock_ocr_service):
    mock_ocr_service.start.return_value = True

    response = client.post("/api/v1/ocr/start")

    assert response.status_code == 200
    assert response.get_json() == {"success": True}


def test_start_failure_returns_500(client, mock_ocr_service):
    mock_ocr_service.start.return_value = False

    response = client.post("/api/v1/ocr/start")

    assert response.status_code == 500


def test_status_returns_404_when_not_active(client, mock_ocr_service):
    mock_ocr_service.get_status.return_value = None

    response = client.get("/api/v1/ocr/status")

    assert response.status_code == 404


def test_settings_rejects_invalid_confidence(client, mock_ocr_service):
    response = client.post("/api/v1/ocr/settings", json={"min_confidence": 1.5})

    assert response.status_code == 400


def test_settings_accepts_valid_payload(client, mock_ocr_service):
    mock_ocr_service.update_settings.return_value = True

    response = client.post("/api/v1/ocr/settings", json={"min_confidence": 0.5})

    assert response.status_code == 200
    mock_ocr_service.update_settings.assert_called_once_with({"min_confidence": 0.5})
