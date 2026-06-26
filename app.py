import os
import logging
from flask import Flask, Response, render_template, jsonify, request
from config import config
from core.camera_manager import CameraManager

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
camera_manager = None

def create_app(config_name='default'):
    """Flask application factory."""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Ensure instance directories exist
    os.makedirs(app.config['DATA_DIR'], exist_ok=True)
    os.makedirs(app.config['RECORDINGS_DIR'], exist_ok=True)
    os.makedirs(app.config['EXPORTS_DIR'], exist_ok=True)
    os.makedirs(app.config['WEIGHTS_DIR'], exist_ok=True)
    os.makedirs(app.config['LOGS_DIR'], exist_ok=True)
    os.makedirs(os.path.join(app.config['BASE_DIR'], 'modules'), exist_ok=True)

    # Initialize camera manager globally
    global camera_manager
    if camera_manager is None:
        camera_manager = CameraManager(config[config_name])
        camera_manager.start()

    # --- Core Routes ---
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/video_feed')
    def video_feed():
        """Route returning the MJPEG stream."""
        return Response(
            camera_manager.get_mjpeg_stream(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    @app.route('/api/status')
    def status():
        """API endpoint to get real-time system and camera status."""
        cam_info = camera_manager.get_info()
        
        active_module_meta = None
        active_mod_id = camera_manager.module_manager.active_module_id
        if active_mod_id:
            active_mod = camera_manager.module_manager.loaded_modules.get(active_mod_id)
            if active_mod:
                try:
                    active_module_meta = active_mod.metadata()
                except Exception as e:
                    active_module_meta = {"error": str(e)}

        return jsonify({
            'status': 'running',
            'camera_active': cam_info['is_connected'],
            'fps': cam_info['fps'],
            'resolution': cam_info['resolution'],
            'camera_index': cam_info['index'],
            'active_module': active_mod_id,
            'module_metadata': active_module_meta,
            'available_modules': list(camera_manager.module_manager.loaded_modules.keys())
        })

    # --- Camera Routes ---
    @app.route('/api/cameras', methods=['GET'])
    def list_cameras():
        """Scan and return available cameras."""
        cameras = CameraManager.scan_cameras()
        return jsonify({
            'available': cameras,
            'current': camera_manager.get_info(),
            'supported_resolutions': camera_manager.supported_resolutions
        })

    @app.route('/api/cameras/switch', methods=['POST'])
    def switch_camera():
        """Switch camera index or resolution."""
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        index = data.get('index', camera_manager.camera_index)
        resolution_str = data.get('resolution', f"{camera_manager.width}x{camera_manager.height}")
        
        try:
            w, h = map(int, resolution_str.split('x'))
            camera_manager.switch_camera(index=int(index), width=w, height=h)
            return jsonify({'success': True, 'message': f'Switched to Camera {index} at {w}x{h}'})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    # --- Plugin System Routes ---
    @app.route('/api/modules', methods=['GET'])
    def get_modules():
        """List all discovered modules and their metadata."""
        meta = camera_manager.module_manager.get_all_metadata()
        return jsonify(meta)

    @app.route('/api/modules/activate', methods=['POST'])
    def activate_module():
        """Hot swap the active module."""
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON payload provided'}), 400
            
        # None or empty string will deactivate current module
        mod_id = data.get('module') 
        if mod_id == "":
            mod_id = None
            
        success = camera_manager.module_manager.activate_module(mod_id)
        if success:
            state = f"Activated {mod_id}" if mod_id else "Deactivated all modules"
            return jsonify({'success': True, 'message': state})
        else:
            return jsonify({'error': f'Failed to activate {mod_id}'}), 400

    @app.route('/api/modules/settings', methods=['POST'])
    def update_settings():
        """Update settings for the active module."""
        data = request.get_json()
        active_id = camera_manager.module_manager.active_module_id
        if active_id:
            module = camera_manager.module_manager.loaded_modules.get(active_id)
            if module:
                module.update_settings(data)
                return jsonify({'success': True, 'message': 'Settings updated'})
        return jsonify({'error': 'No active module to apply settings to'}), 400

    @app.route('/api/modules/interact', methods=['POST'])
    def interact_module():
        """Handle interactions like clicks for the active module."""
        data = request.get_json()
        active_id = camera_manager.module_manager.active_module_id
        if active_id:
            module = camera_manager.module_manager.loaded_modules.get(active_id)
            if module:
                try:
                    result = module.handle_interaction(data.get('action'), data.get('x', 0), data.get('y', 0))
                    if isinstance(result, dict) and result.get("type") == "download":
                        return jsonify(result)
                    return jsonify({"status": "success", "result": result})
                except Exception as e:
                    return jsonify({"error": str(e)}), 400
        return jsonify({'error': 'No active module to handle interaction'}), 400

    @app.route('/api/modules/reload', methods=['POST'])
    def reload_modules():
        """Force unload and reload all modules from disk."""
        try:
            camera_manager.module_manager.reload_modules()
            return jsonify({'success': True, 'message': 'Modules reloaded from disk'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Cleanup on exit
    import atexit
    atexit.register(lambda: camera_manager.stop() if camera_manager else None)

    return app

if __name__ == '__main__':
    app = create_app('development')
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
