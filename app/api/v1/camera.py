from flask import Blueprint, jsonify, request, Response
from app.services.camera_service import CameraService

bp = Blueprint('camera_v1', __name__, url_prefix='/api/v1/camera')
camera_service = CameraService()

@bp.route('/stream', methods=['GET'])
def get_stream():
    """Stream MJPEG video feed."""
    return Response(
        camera_service.get_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@bp.route('/switch', methods=['POST'])
def switch_camera():
    """Switch the active camera hardware or resolution."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
        
    index = data.get("index", 0)
    resolution = data.get("resolution", "1280x720")
    
    if camera_service.switch_camera(index, resolution):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to switch camera"}), 500
