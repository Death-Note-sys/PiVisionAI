import os
from flask import Blueprint, jsonify, request, Response, send_file
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import PydanticCustomError
from typing import Optional, Dict, Any
from app.services.target_extraction_service import TargetExtractionAppService
from app.modules.target_extraction.models import SetROIRequest, UpdateTargetExtractionSettingsRequest

bp = Blueprint('target_extraction_v1', __name__, url_prefix='/api/v1/target-extraction')
target_extraction_service = TargetExtractionAppService()


@bp.route('/start', methods=['POST'])
def start():
    try:
        if target_extraction_service.start():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to start Target Extraction"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/stop', methods=['POST'])
def stop():
    try:
        if target_extraction_service.stop():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to stop Target Extraction"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/pause', methods=['POST'])
def pause():
    try:
        if target_extraction_service.pause():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to pause Target Extraction"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/resume', methods=['POST'])
def resume():
    try:
        if target_extraction_service.resume():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to resume Target Extraction"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/status', methods=['GET'])
def status():
    try:
        result = target_extraction_service.get_status()
        if result is None:
            return jsonify({"error": "Target Extraction is not the active module"}), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/settings', methods=['GET'])
def get_settings():
    try:
        svc = target_extraction_service
        if hasattr(svc, 'settings') and svc.settings:
            return jsonify(svc.settings.get_settings())
        return jsonify({}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    # Merged-state area ratio validation
    current = {}
    if hasattr(target_extraction_service, 'settings') and target_extraction_service.settings:
        current = target_extraction_service.settings.get_settings()
    merged_min = data.get("min_area_ratio", current.get("min_area_ratio", 0.10))
    merged_max = data.get("max_area_ratio", current.get("max_area_ratio", 0.95))
    if merged_min >= merged_max:
        return jsonify({"error": "min_area_ratio must be less than max_area_ratio"}), 400

    try:
        validated = UpdateTargetExtractionSettingsRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    if target_extraction_service.update_settings(validated.to_update_dict()):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to update settings"}), 500


@bp.route('/set-roi', methods=['POST'])
def set_roi():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    try:
        validated = SetROIRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    ok = target_extraction_service.set_roi(validated.x, validated.y, validated.w, validated.h)
    if ok:
        return jsonify({"success": True})
    return jsonify({"error": "Failed to set ROI - no frame or box out of bounds"}), 400


@bp.route('/reset-roi', methods=['POST'])
def reset_roi():
    if target_extraction_service.reset_roi():
        return jsonify({"success": True})
    return jsonify({"error": "Failed to reset ROI"}), 500


@bp.route('/extracted.png', methods=['GET'])
def get_extracted_image():
    path = target_extraction_service.get_extraction_path()
    if not path or not os.path.exists(path):
        return jsonify({"error": "No extraction available"}), 404
    return send_file(path, mimetype='image/png')


@bp.route('/stream')
def stream():
    """MJPEG stream of the rendered Target Extraction output."""
    from app.core.container import Container
    output_manager = Container.get_instance().pipeline.output_manager
    return Response(
        output_manager.get_mjpeg_generator(target_fps=30),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
