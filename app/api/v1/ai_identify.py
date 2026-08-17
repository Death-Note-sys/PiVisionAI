from flask import Blueprint, jsonify, request, Response
from pydantic import ValidationError
from app.services.ai_identify_service import AIIdentifyService
from app.modules.ai_identify.models import TeachRegionRequest, UpdateAIIdentifySettingsRequest

bp = Blueprint('ai_identify_v1', __name__, url_prefix='/api/v1/ai-identify')
ai_identify_service = AIIdentifyService()


@bp.route('/start', methods=['POST'])
def start():
    try:
        if ai_identify_service.start():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to start AI Identify"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/stop', methods=['POST'])
def stop():
    try:
        if ai_identify_service.stop():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to stop AI Identify"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/pause', methods=['POST'])
def pause():
    try:
        if ai_identify_service.pause():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to pause AI Identify"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/resume', methods=['POST'])
def resume():
    try:
        if ai_identify_service.resume():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to resume AI Identify"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/status', methods=['GET'])
def status():
    try:
        result = ai_identify_service.get_status()
        if result is None:
            return jsonify({"error": "AI Identify is not the active module"}), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    try:
        validated = UpdateAIIdentifySettingsRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    if ai_identify_service.update_settings(validated.to_update_dict()):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to update settings"}), 500


@bp.route('/teach-good', methods=['POST'])
def teach_good():
    """Capture the current frozen frame region as the Good reference."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    try:
        validated = TeachRegionRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    if ai_identify_service.teach_good(validated.x, validated.y, validated.w, validated.h):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to teach Good reference — region may lack distinctive features"}), 500


@bp.route('/teach-bad', methods=['POST'])
def teach_bad():
    """Capture the current frozen frame region as the Bad reference."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    try:
        validated = TeachRegionRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    if ai_identify_service.teach_bad(validated.x, validated.y, validated.w, validated.h):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to teach Bad reference"}), 500


@bp.route('/reset-teaching', methods=['POST'])
def reset_teaching():
    if ai_identify_service.reset_teaching():
        return jsonify({"success": True})
    return jsonify({"error": "Failed to reset teaching"}), 500


@bp.route('/good-reference/<int:index>', methods=['DELETE'])
def remove_good_reference(index):
    if ai_identify_service.remove_good_reference(index):
        return jsonify({"success": True})
    return jsonify({"error": f"Failed to remove good reference at index {index}"}), 500


@bp.route('/bad-reference/<int:index>', methods=['DELETE'])
def remove_bad_reference(index):
    if ai_identify_service.remove_bad_reference(index):
        return jsonify({"success": True})
    return jsonify({"error": f"Failed to remove bad reference at index {index}"}), 500


@bp.route('/stream')
def stream():
    """MJPEG stream of the rendered AI Identify output — shares OutputManager
    with every other module's stream route (see MJPEG consolidation fix)."""
    from app.core.container import Container
    output_manager = Container.get_instance().pipeline.output_manager
    return Response(
        output_manager.get_mjpeg_generator(target_fps=30),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
