import pytest
from unittest.mock import MagicMock, patch
import cv2
from app.api import create_app
from app.core.container import Container
from app.core.camera_manager import CameraManager


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_camera_settings_api_get(client):
    """GET /api/v1/camera/settings returns settings and readback dictionaries."""
    response = client.get("/api/v1/camera/settings")
    assert response.status_code == 200
    data = response.get_json()
    assert "settings" in data
    assert "readback" in data
    assert "auto_exposure" in data["settings"]
    assert "exposure" in data["settings"]
    assert "gain" in data["settings"]
    assert "auto_wb" in data["settings"]
    assert "wb_temperature" in data["settings"]


def test_camera_settings_api_post_validation(client):
    """POST /api/v1/camera/settings rejects empty body."""
    response = client.post("/api/v1/camera/settings", json={})
    assert response.status_code == 400


def test_camera_settings_api_post_update(client):
    """POST /api/v1/camera/settings updates settings and returns success."""
    payload = {
        "auto_exposure": False,
        "exposure": -6.0,
        "gain": 12.0,
        "auto_wb": False,
        "wb_temperature": 5200.0
    }
    response = client.post("/api/v1/camera/settings", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["settings"]["auto_exposure"] is False
    assert data["settings"]["exposure"] == -6.0
    assert data["settings"]["gain"] == 12.0
    assert data["settings"]["auto_wb"] is False
    assert data["settings"]["wb_temperature"] == 5200.0


def test_camera_manager_apply_settings_disconnected():
    """CameraManager retains settings when camera is not opened."""
    container = Container()
    cm = container.camera_manager
    cm.cap = None
    cm.is_connected = False
    
    res = cm.apply_settings({"gain": 15.0, "auto_exposure": False, "exposure": -4.0})
    assert cm.camera_settings["gain"] == 15.0
    assert cm.camera_settings["auto_exposure"] is False
    assert cm.camera_settings["exposure"] == -4.0
    assert res["gain"] == 15.0


def test_camera_manager_apply_settings_connected():
    """CameraManager sets cv2 properties and queries readback when connected."""
    container = Container()
    cm = container.camera_manager
    mock_cap = MagicMock()
    # Mock cap.get for cv2 props
    def mock_get(prop):
        if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
            return 0.25
        if prop == cv2.CAP_PROP_EXPOSURE:
            return -5.0
        if prop == cv2.CAP_PROP_GAIN:
            return 20.0
        if prop == cv2.CAP_PROP_AUTO_WB:
            return 0.0
        if prop == cv2.CAP_PROP_WB_TEMPERATURE:
            return 4800.0
        return -1.0
    
    mock_cap.get.side_effect = mock_get
    cm.cap = mock_cap
    cm.is_connected = True

    readback = cm.apply_settings({
        "auto_exposure": False,
        "exposure": -5.0,
        "gain": 20.0,
        "auto_wb": False,
        "wb_temperature": 4800.0
    })

    assert mock_cap.set.called
    assert readback["auto_exposure"] is False
    assert readback["exposure"] == -5.0
    assert readback["gain"] == 20.0
    assert readback["auto_wb"] is False
    assert readback["wb_temperature"] == 4800.0
