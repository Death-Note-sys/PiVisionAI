from flask import Blueprint, jsonify, request, Response
from pydantic import ValidationError
from app.services.ocr_service import OCRService
from app.modules.ocr.models import UpdateOCRSettingsRequest

bp = Blueprint('ocr_v1', __name__, url_prefix='/api/v1/ocr')
ocr_service = OCRService()


@bp.route('/start', methods=['POST'])
def start():
    try:
        if ocr_service.start():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to start OCR"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/stop', methods=['POST'])
def stop():
    try:
        if ocr_service.stop():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to stop OCR"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/pause', methods=['POST'])
def pause():
    try:
        if ocr_service.pause():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to pause OCR"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/resume', methods=['POST'])
def resume():
    try:
        if ocr_service.resume():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to resume OCR"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/status', methods=['GET'])
def status():
    try:
        result = ocr_service.get_status()
        if result is None:
            return jsonify({"error": "OCR is not the active module"}), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    try:
        validated = UpdateOCRSettingsRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    if ocr_service.update_settings(validated.to_update_dict()):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to update settings"}), 500


@bp.route('/stream')
def stream():
    """MJPEG stream of the rendered OCR output — shares OutputManager with
    every other module's stream route (see MJPEG consolidation fix)."""
    from app.core.container import Container
    output_manager = Container.get_instance().pipeline.output_manager
    return Response(
        output_manager.get_mjpeg_generator(target_fps=30),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
