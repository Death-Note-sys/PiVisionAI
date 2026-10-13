from flask import Blueprint, jsonify, request, Response
from app.services.camera_service import CameraService
import cv2
import os

bp = Blueprint('camera_v1', __name__, url_prefix='/api/v1/camera')
camera_service = CameraService()

def normalize_camera_backend(backend: str) -> str:
    if not backend:
        return "auto"
    b = str(backend).lower().strip()
    if "dshow" in b:
        return "dshow"
    if "msmf" in b:
        return "msmf"
    if "any" in b:
        return "any"
    if "v4l" in b:
        return "v4l2"
    return "auto"

@bp.route('/stream', methods=['GET'])
def get_stream():
    """Stream MJPEG video feed."""
    return Response(
        camera_service.get_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@bp.route('/devices', methods=['GET'])
def get_camera_devices():
    """Probe camera devices (indices 0-4).
    Reports the active streaming camera as 'in use' without disrupting CameraManager,
    and probes other indices cleanly."""
    mgr = camera_service.camera_manager
    streaming_active = mgr.is_connected or (mgr._capture_thread and mgr._capture_thread.is_alive())
    active_idx = mgr.camera_index if streaming_active else None

    devices = []
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY

    for idx in range(5):
        if streaming_active and idx == active_idx:
            # CameraManager holds this device; report as in use
            width = 640
            height = 480
            if mgr.cap and mgr.cap.isOpened():
                try:
                    width = int(mgr.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(mgr.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                except Exception:
                    pass
            devices.append({
                "index": idx,
                "opened": True,
                "in_use": True,
                "status": "in use",
                "width": width,
                "height": height
            })
        else:
            # Probe device safely
            cap = cv2.VideoCapture(idx, backend)
            opened = cap.isOpened()
            width = 0
            height = 0
            if opened:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            devices.append({
                "index": idx,
                "opened": opened,
                "in_use": False,
                "status": "available" if opened else "unavailable",
                "width": width,
                "height": height
            })
    return jsonify({"devices": devices})

@bp.route('/switch', methods=['POST'])
def switch_camera():
    """Switch the active camera hardware or resolution."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
        
    index = data.get("index", 0)
    resolution = data.get("resolution", "1280x720")
    backend = normalize_camera_backend(data.get("backend"))
    
    if camera_service.switch_camera(index, resolution, backend=backend):
        return jsonify({"success": True, "backend": backend})
    return jsonify({"error": "Failed to switch camera"}), 500

@bp.route('/settings', methods=['GET', 'POST'])
def handle_camera_settings():
    """Get or update fine-grained camera hardware settings."""
    if request.method == 'GET':
        return jsonify(camera_service.get_camera_settings())
    else:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400
        updated = camera_service.update_camera_settings(data)
        return jsonify({"success": True, "settings": updated})
