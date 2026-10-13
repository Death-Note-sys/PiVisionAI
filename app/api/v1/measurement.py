from flask import Blueprint, jsonify, request, Response
from pydantic import ValidationError
from app.services.measurement_service import MeasurementService
from app.modules.measurement.models import CalibrationRequest, UpdateMeasurementSettingsRequest

bp = Blueprint('measurement_v1', __name__, url_prefix='/api/v1/measurement')
measurement_service = MeasurementService()


@bp.route('/start', methods=['POST'])
def start():
    try:
        if measurement_service.start():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to start Measurement"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/stop', methods=['POST'])
def stop():
    try:
        if measurement_service.stop():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to stop Measurement"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/pause', methods=['POST'])
def pause():
    try:
        if measurement_service.pause():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to pause Measurement"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/resume', methods=['POST'])
def resume():
    try:
        if measurement_service.resume():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to resume Measurement"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/status', methods=['GET'])
def status():
    try:
        result = measurement_service.get_status()
        if result is None:
            return jsonify({"error": "Measurement is not the active module"}), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    try:
        validated = UpdateMeasurementSettingsRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    update_dict = validated.to_update_dict()
    active_controller = getattr(measurement_service, "controller", None)
    if not active_controller and hasattr(measurement_service, "container") and measurement_service.container:
        active_service = measurement_service.container.module_controller.get_active_service()
        if active_service:
            active_controller = getattr(active_service, "controller", None)

    current_settings = active_controller.settings.get_settings() if active_controller and hasattr(active_controller, "settings") else {}
    merged_judge = update_dict["judge_enabled"] if "judge_enabled" in update_dict else current_settings.get("judge_enabled", False)
    merged_tol = update_dict["tolerance_mm"] if "tolerance_mm" in update_dict else current_settings.get("tolerance_mm", 0.0)

    if merged_judge and (merged_tol is None or merged_tol <= 0):
        return jsonify({"error": "tolerance_mm must be > 0 when judge_enabled is True"}), 400
    if measurement_service.update_settings(update_dict):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to update settings"}), 500


@bp.route('/calibrate', methods=['POST'])
def calibrate():
    """Calibrate pixels-per-mm from two clicked points and a known real-world length."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    try:
        validated = CalibrationRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
        
    mm_len = validated.real_length_mm
    if mm_len is None and validated.real_length_cm is not None:
        mm_len = validated.real_length_cm * 10.0
        
    if mm_len is None or mm_len <= 0:
        return jsonify({"error": "real_length_mm must be positive"}), 400
        
    if measurement_service.calibrate(
        validated.x1, validated.y1, validated.x2, validated.y2, mm_len
    ):
        return jsonify({"success": True})
    return jsonify({"error": "Calibration failed — check that the two points are not identical"}), 500


@bp.route('/reset-calibration', methods=['POST'])
def reset_calibration():
    """Reset calibration status to Uncalibrated."""
    try:
        if measurement_service.reset_calibration():
            return jsonify({"success": True})
        return jsonify({"error": "Failed to reset calibration"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@bp.route('/stream')
def stream():
    """MJPEG stream of the rendered Measurement output — shares OutputManager
    with every other module's stream route (see MJPEG consolidation fix)."""
    from app.core.container import Container
    output_manager = Container.get_instance().pipeline.output_manager
    return Response(
        output_manager.get_mjpeg_generator(target_fps=30),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
