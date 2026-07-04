from flask import Blueprint, jsonify, request
from app.services.module_service import ModuleService

bp = Blueprint('modules_v1', __name__, url_prefix='/api/v1/modules')
module_service = ModuleService()

@bp.route('/', methods=['GET'])
def list_modules():
    """List all available modules."""
    try:
        modules = module_service.list_available_modules()
        return jsonify({"modules": modules})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route('/active', methods=['GET'])
def get_active_module():
    """Get metadata for the currently active module."""
    try:
        info = module_service.get_active_module_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route('/switch', methods=['POST'])
def switch_module():
    """Switch to a new active module."""
    data = request.get_json()
    if not data or "module_id" not in data:
        return jsonify({"error": "Missing module_id payload"}), 400
        
    if module_service.switch_module(data["module_id"]):
        return jsonify({"success": True, "active_module": data["module_id"]})
    return jsonify({"error": f"Failed to switch to module {data['module_id']}"}), 500

@bp.route('/settings', methods=['POST'])
def update_settings():
    """Update settings for the active module."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
        
    if module_service.update_module_settings(data):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to update module settings"}), 500
