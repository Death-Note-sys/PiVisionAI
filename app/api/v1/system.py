from flask import Blueprint, jsonify, request
from app.services.system_service import SystemService

bp = Blueprint('system_v1', __name__, url_prefix='/api/v1/system')
system_service = SystemService()

@bp.route('/status', methods=['GET'])
def get_status():
    """Get overall system status and performance metrics."""
    try:
        return jsonify(system_service.get_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route('/config', methods=['POST'])
def update_config():
    """Update system configuration dynamically."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
        
    if system_service.reload_config(data):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to update configuration"}), 500
