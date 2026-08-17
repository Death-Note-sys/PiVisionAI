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
def mock_ai_identify_service():
    with patch("app.api.v1.ai_identify.ai_identify_service") as mock_service:
        yield mock_service


def test_start_success(client, mock_ai_identify_service):
    mock_ai_identify_service.start.return_value = True

    response = client.post("/api/v1/ai-identify/start")

    assert response.status_code == 200
    assert response.get_json() == {"success": True}


def test_status_returns_404_when_not_active(client, mock_ai_identify_service):
    mock_ai_identify_service.get_status.return_value = None

    response = client.get("/api/v1/ai-identify/status")

    assert response.status_code == 404


def test_teach_good_rejects_missing_fields(client, mock_ai_identify_service):
    response = client.post("/api/v1/ai-identify/teach-good", json={"x": 0, "y": 0})

    assert response.status_code == 400


def test_teach_good_rejects_zero_width(client, mock_ai_identify_service):
    response = client.post(
        "/api/v1/ai-identify/teach-good", json={"x": 0, "y": 0, "w": 0, "h": 50}
    )

    assert response.status_code == 400


def test_teach_good_success(client, mock_ai_identify_service):
    mock_ai_identify_service.teach_good.return_value = True

    response = client.post(
        "/api/v1/ai-identify/teach-good", json={"x": 10, "y": 10, "w": 100, "h": 100}
    )

    assert response.status_code == 200
    mock_ai_identify_service.teach_good.assert_called_once_with(10, 10, 100, 100)


def test_teach_bad_failure_returns_500(client, mock_ai_identify_service):
    mock_ai_identify_service.teach_bad.return_value = False

    response = client.post(
        "/api/v1/ai-identify/teach-bad", json={"x": 10, "y": 10, "w": 100, "h": 100}
    )

    assert response.status_code == 500


def test_reset_teaching_success(client, mock_ai_identify_service):
    mock_ai_identify_service.reset_teaching.return_value = True

    response = client.post("/api/v1/ai-identify/reset-teaching")

    assert response.status_code == 200


def test_remove_good_reference_success(client, mock_ai_identify_service):
    mock_ai_identify_service.remove_good_reference.return_value = True

    response = client.delete("/api/v1/ai-identify/good-reference/0")

    assert response.status_code == 200
    mock_ai_identify_service.remove_good_reference.assert_called_once_with(0)


def test_remove_good_reference_failure_returns_500(client, mock_ai_identify_service):
    mock_ai_identify_service.remove_good_reference.return_value = False

    response = client.delete("/api/v1/ai-identify/good-reference/3")

    assert response.status_code == 500


def test_remove_bad_reference_success(client, mock_ai_identify_service):
    mock_ai_identify_service.remove_bad_reference.return_value = True

    response = client.delete("/api/v1/ai-identify/bad-reference/1")

    assert response.status_code == 200
    mock_ai_identify_service.remove_bad_reference.assert_called_once_with(1)
