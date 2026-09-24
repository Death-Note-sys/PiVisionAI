from flask import Blueprint, jsonify, request, Response
from pydantic import ValidationError
from app.services.color_analysis_service import ColorAnalysisService
from app.modules.color_analysis.models import TeachReferenceRequest, UpdateColorSettingsRequest

bp = Blueprint('color_analysis_v1', __name__, url_prefix='/api/v1/color-analysis')
color_analysis_service = ColorAnalysisService()


@bp.route('/start', methods=['POST'])
def start():
    try:
        if color_analysis_service.start():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to start Color Analysis"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/stop', methods=['POST'])
def stop():
    try:
        if color_analysis_service.stop():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to stop Color Analysis"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/pause', methods=['POST'])
def pause():
    try:
        if color_analysis_service.pause():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to pause Color Analysis"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/resume', methods=['POST'])
def resume():
    try:
        if color_analysis_service.resume():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to resume Color Analysis"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/status', methods=['GET'])
def status():
    try:
        result = color_analysis_service.get_status()
        if result is None:
            return jsonify({"error": "Color Analysis is not the active module"}), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    try:
        validated = UpdateColorSettingsRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    if color_analysis_service.update_settings(validated.to_update_dict()):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to update settings"}), 500


@bp.route('/teach-reference', methods=['POST'])
def teach_reference():
    """Capture the current frozen frame region as the reference color."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    try:
        validated = TeachReferenceRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    if color_analysis_service.teach_reference(validated.x, validated.y, validated.w, validated.h):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to teach reference color"}), 500


@bp.route('/reset-reference', methods=['POST'])
def reset_reference():
    if color_analysis_service.reset_reference():
        return jsonify({"success": True})
    return jsonify({"error": "Failed to reset reference"}), 500


@bp.route('/stream')
def stream():
    """MJPEG stream of the rendered Color Analysis output — shares
    OutputManager with every other module's stream route."""
    from app.core.container import Container
    output_manager = Container.get_instance().pipeline.output_manager
    return Response(
        output_manager.get_mjpeg_generator(target_fps=30),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
