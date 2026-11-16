import os
from flask import Blueprint, jsonify, request, Response, send_file
from pydantic import ValidationError
from app.services.object_counting_service import ObjectCountingService
from app.modules.object_counting.models import TeachTemplateRequest, UpdateObjectCountingSettingsRequest

bp = Blueprint('object_counting_v1', __name__, url_prefix='/api/v1/object-counting')
object_counting_service = ObjectCountingService()


@bp.route('/start', methods=['POST'])
def start():
    try:
        if object_counting_service.start():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to start Object Counting"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/stop', methods=['POST'])
def stop():
    try:
        if object_counting_service.stop():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to stop Object Counting"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/pause', methods=['POST'])
def pause():
    try:
        if object_counting_service.pause():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to pause Object Counting"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/resume', methods=['POST'])
def resume():
    try:
        if object_counting_service.resume():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to resume Object Counting"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/status', methods=['GET'])
def status():
    try:
        result = object_counting_service.get_status()
        if result is None:
            return jsonify({"error": "Object Counting is not the active module"}), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    try:
        validated = UpdateObjectCountingSettingsRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    update_dict = validated.to_update_dict()
    current = {}
    if hasattr(object_counting_service, "settings") and object_counting_service.settings is not None:
        if hasattr(object_counting_service.settings, "get_settings"):
            res = object_counting_service.settings.get_settings()
            if isinstance(res, dict):
                current = res

    merged_cmin = update_dict.get("count_min", current.get("count_min", 0))
    merged_cmax = update_dict.get("count_max", current.get("count_max", 0))
    if (merged_cmin is not None and merged_cmin < 0) or (merged_cmax is not None and merged_cmax < 0):
        return jsonify({"error": "count_min and count_max must be >= 0"}), 400
    if merged_cmax > 0 and merged_cmin > merged_cmax:
        return jsonify({"error": "count_min cannot be greater than count_max"}), 400

    if object_counting_service.update_settings(update_dict):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to update settings"}), 500


@bp.route('/teach-template', methods=['POST'])
def teach_template():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    try:
        validated = TeachTemplateRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    res = object_counting_service.teach_template(validated.x, validated.y, validated.w, validated.h)
    if isinstance(res, tuple):
        ok, reason = res
    else:
        ok, reason = bool(res), ""

    if ok:
        return jsonify({"success": True})
    return jsonify({"error": reason or "Failed to teach template"}), 400


@bp.route('/reset-template', methods=['POST'])
def reset_template():
    if object_counting_service.reset_template():
        return jsonify({"success": True})
    return jsonify({"error": "Failed to reset template"}), 500


@bp.route('/template.png', methods=['GET'])
def get_template_image():
    path = object_counting_service.get_template_path()
    if not path or not os.path.exists(path):
        return jsonify({"error": "No template taught"}), 404
    return send_file(path, mimetype='image/png')


@bp.route('/stream')
def stream():
    """MJPEG stream of the rendered Object Counting output."""
    from app.core.container import Container
    output_manager = Container.get_instance().pipeline.output_manager
    return Response(
        output_manager.get_mjpeg_generator(target_fps=30),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
