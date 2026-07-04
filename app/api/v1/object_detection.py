from flask import Blueprint, jsonify, request, Response
from pydantic import ValidationError
from app.services.object_detection_service import ObjectDetectionService
from app.modules.object_detection.models import UpdateSettingsRequest, SwitchModelRequest

bp = Blueprint('object_detection_v1', __name__, url_prefix='/api/v1/object-detection')
od_service = ObjectDetectionService()


@bp.route('/start', methods=['POST'])
def start():
    """Activate the Object Detection module and start processing."""
    try:
        if od_service.start():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to start Object Detection"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/stop', methods=['POST'])
def stop():
    """Stop Object Detection processing."""
    try:
        if od_service.stop():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to stop Object Detection"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/pause', methods=['POST'])
def pause():
    try:
        if od_service.pause():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to pause Object Detection"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/resume', methods=['POST'])
def resume():
    try:
        if od_service.resume():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to resume Object Detection"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/status', methods=['GET'])
def status():
    """Get current active/paused state and settings."""
    try:
        result = od_service.get_status()
        if result is None:
            return jsonify({"error": "Object Detection is not the active module"}), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/settings', methods=['POST'])
def update_settings():
    """Update confidence/iou/rendering settings for Object Detection."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    try:
        validated = UpdateSettingsRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    if od_service.update_settings(validated.to_update_dict()):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to update settings"}), 500


@bp.route('/model', methods=['POST'])
def switch_model():
    """Switch the active YOLO / detection model."""
    data = request.get_json()
    if not data or "model_id" not in data:
        return jsonify({"error": "Missing model_id payload"}), 400

    try:
        validated = SwitchModelRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    if od_service.switch_model(validated.model_id):
        return jsonify({"success": True, "model_id": validated.model_id})
    return jsonify({"error": f"Failed to switch to model {validated.model_id}"}), 500


@bp.route('/stream')
def stream():
    """MJPEG stream of the rendered Object Detection output.

    Delegates to OutputManager's shared generator (see the MJPEG
    consolidation fix) rather than looping locally, so this route and
    /api/v1/camera/stream always serve frames from the exact same
    single source of truth.
    """
    from app.core.container import Container
    output_manager = Container.get_instance().pipeline.output_manager
    return Response(
        output_manager.get_mjpeg_generator(target_fps=30),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
