from flask import Blueprint, jsonify, request
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from app.services.trigger_service import TriggerService

bp = Blueprint('trigger_v1', __name__, url_prefix='/api/v1/trigger')
trigger_service = TriggerService()


class SetModeRequest(BaseModel):
    mode: Literal["continuous", "single", "interval"]


class SetIntervalRequest(BaseModel):
    seconds: float = Field(gt=0)


@bp.route('/mode', methods=['POST'])
def set_mode():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    try:
        validated = SetModeRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    if trigger_service.set_mode(validated.mode):
        return jsonify({"success": True, "mode": validated.mode})
    return jsonify({"error": "Failed to set trigger mode"}), 500


@bp.route('/interval', methods=['POST'])
def set_interval():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    try:
        validated = SetIntervalRequest(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    if trigger_service.set_interval(validated.seconds):
        return jsonify({"success": True, "interval_seconds": validated.seconds})
    return jsonify({"error": "Failed to set interval"}), 500


@bp.route('/fire', methods=['POST'])
def fire():
    if trigger_service.fire():
        return jsonify({"success": True})
    return jsonify({"error": "Failed to fire trigger"}), 500


@bp.route('/status', methods=['GET'])
def status():
    return jsonify(trigger_service.get_status())
