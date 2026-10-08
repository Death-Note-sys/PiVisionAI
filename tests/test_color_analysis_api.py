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
def mock_color_service():
    with patch("app.api.v1.color_analysis.color_analysis_service") as mock_service:
        yield mock_service


def test_start_success(client, mock_color_service):
    mock_color_service.start.return_value = True

    response = client.post("/api/v1/color-analysis/start")

    assert response.status_code == 200
    assert response.get_json() == {"success": True}


def test_status_returns_404_when_not_active(client, mock_color_service):
    mock_color_service.get_status.return_value = None

    response = client.get("/api/v1/color-analysis/status")

    assert response.status_code == 404


def test_teach_reference_rejects_missing_fields(client, mock_color_service):
    response = client.post("/api/v1/color-analysis/teach-reference", json={"x": 0, "y": 0})

    assert response.status_code == 400


def test_teach_reference_success(client, mock_color_service):
    mock_color_service.teach_reference.return_value = True

    response = client.post(
        "/api/v1/color-analysis/teach-reference", json={"x": 10, "y": 10, "w": 50, "h": 50}
    )

    assert response.status_code == 200
    mock_color_service.teach_reference.assert_called_once_with(10, 10, 50, 50)


def test_settings_rejects_invalid_num_colors(client, mock_color_service):
    response = client.post("/api/v1/color-analysis/settings", json={"num_colors": 50})

    assert response.status_code == 400


def test_reset_reference_success(client, mock_color_service):
    mock_color_service.reset_reference.return_value = True

    response = client.post("/api/v1/color-analysis/reset-reference")

    assert response.status_code == 200
