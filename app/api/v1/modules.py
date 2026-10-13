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

@bp.route('/active/settings', methods=['POST'])
def update_settings():
    """Update settings for the active module."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
        
    if module_service.update_module_settings(data):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to update module settings"}), 500

@bp.route('/active/start', methods=['POST'])
def start_active():
    if module_service.start_active():
        return jsonify({"success": True})
    return jsonify({"error": "Failed to start active module"}), 500
    
@bp.route('/active/stop', methods=['POST'])
def stop_active():
    if module_service.stop_active():
        return jsonify({"success": True})
    return jsonify({"error": "Failed to stop active module"}), 500

@bp.route('/active/status', methods=['GET'])
def get_active_status():
    status = module_service.get_active_status()
    if status is None:
        return jsonify({"error": "No active module or module does not support status"}), 404
    return jsonify(status)

@bp.route('/active/stream')
def stream():
    """MJPEG stream of the rendered output for the active module."""
    from app.core.container import Container
    from flask import Response
    output_manager = Container.get_instance().pipeline.output_manager
    return Response(
        output_manager.get_mjpeg_generator(target_fps=30),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@bp.route('/active/<action>', methods=['POST'])
def invoke_active(action):
    # Explicit overrides for standard lifecycle
    if action == "settings":
        return update_settings()
    elif action == "start":
        return start_active()
    elif action == "stop":
        return stop_active()
        
    data = request.get_json(silent=True) or {}
    try:
        result = module_service.invoke_active_method(action.replace("-", "_"), **data)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route('/active/<path:action>', methods=['DELETE'])
def delete_active(action):
    parts = action.split('/')
    if len(parts) == 2 and parts[0] in ("good-reference", "bad-reference"):
        method_name = "remove_good_reference" if parts[0] == "good-reference" else "remove_bad_reference"
        try:
            index = int(parts[1])
            result = module_service.invoke_active_method(method_name, index=index)
            return jsonify({"success": True, "result": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
@bp.route('/active/gallery/<ref_type>/<int:index>', methods=['GET'])
def get_gallery_image(ref_type, index):

    from flask import send_file
    import os
    from app.core.container import Container
    
    container = Container.get_instance()
    if not container:
        return "Container not initialized", 500
        
    if ref_type == "color":
        module_dir = container.session_manager.get_data_dir("modules/core-color-analysis")
        path = os.path.join(module_dir, "reference.png")
    else:
        if not container.module_controller or not container.module_controller.active_metadata:
            return "No active module", 404
        meta = container.module_controller.active_metadata
        module_dir = container.session_manager.get_data_dir(f"modules/{meta.id}")
        path = os.path.join(module_dir, f"{ref_type}_{index}.png")
        
    path = os.path.abspath(path)
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    return "Not found", 404

@bp.route('/color/reference', methods=['GET'])
def get_color_reference_image():
    from flask import send_file
    import os
    from app.core.container import Container
    container = Container.get_instance()
    if not container:
        return "Container not initialized", 500
    module_dir = container.session_manager.get_data_dir("modules/core-color-analysis")
    path = os.path.abspath(os.path.join(module_dir, "reference.png"))
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    return "Not found", 404
