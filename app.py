import os
import logging
from flask import Flask, Response, render_template, jsonify, request, send_file
from config import config
from core.camera_manager import CameraManager
from core.session_manager import SessionManager
from core.media_manager import MediaManager
from core.export_manager import ExportManager
from core.telemetry import SystemTelemetry
from core.analytics_manager import AnalyticsManager
from core.localization import LocalizationManager

from logging.handlers import RotatingFileHandler

# Configure root logger
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'pivision.log')

file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
console_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)
camera_manager = None
media_manager = None
export_manager = None
telemetry = None
analytics_manager = None
localization_manager = None

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

    # Initialize core managers globally
    global camera_manager, media_manager, export_manager, telemetry, analytics_manager, localization_manager
    
    if telemetry is None:
        telemetry = SystemTelemetry()
        telemetry.start()
        
    if analytics_manager is None:
        analytics_manager = AnalyticsManager(app.config['DATA_DIR'])
        
    if localization_manager is None:
        localization_manager = LocalizationManager(app.config['DATA_DIR'])
        
    if camera_manager is None:
        camera_manager = CameraManager(config[config_name])
        camera_manager.analytics_manager = analytics_manager
        camera_manager.start()
        
    if media_manager is None:
        media_manager = MediaManager(app.config['DATA_DIR'], camera_manager)
        
    if export_manager is None:
        export_manager = ExportManager(app.config['EXPORTS_DIR'])

    # Initialize SessionManager
    session = SessionManager(app.config['DATA_DIR'])
    
    # Restore Session State
    saved_index = session.get('camera_index', 0)
    saved_res = session.get('camera_resolution', "640x480")
    try:
        w, h = map(int, saved_res.split('x'))
        camera_manager.switch_camera(saved_index, w, h)
    except:
        pass
        
    saved_mod = session.get('active_module')
    if saved_mod:
        # Restore module settings if any before activating
        mod_settings = session.get('module_settings', {}).get(saved_mod)
        if mod_settings:
            # We can't update settings directly until activated, or we can load them when it activates
            pass
        camera_manager.module_manager.activate_module(saved_mod)
        # Apply settings after activation
        if mod_settings:
            active_obj = camera_manager.module_manager.loaded_modules.get(saved_mod)
            if active_obj:
                active_obj.update_settings(mod_settings)

    # --- Core Routes ---
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/gallery')
    def gallery():
        return render_template('gallery.html')

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
            'available_modules': list(camera_manager.module_manager.loaded_modules.keys()),
            'telemetry': telemetry.get_stats(),
            'analytics': analytics_manager.get_summary()
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
            session.update(camera_index=int(index), camera_resolution=resolution_str)
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
            session.update(active_module=mod_id)
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
                session.update(module_settings={active_id: data})
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

    # --- Media Routes ---
    @app.route('/api/media/screenshot', methods=['POST'])
    def take_screenshot():
        data = request.get_json() or {}
        fmt = data.get('format', 'PNG')
        qual = data.get('quality', 100)
        feed = data.get('feed', 'Processed Feed')
        res = media_manager.take_screenshot(format_ext=fmt, quality=qual, feed_type=feed)
        return jsonify(res)

    @app.route('/api/media/record/start', methods=['POST'])
    def start_record():
        data = request.get_json() or {}
        fmt = data.get('format', 'MP4')
        codec = data.get('codec', 'H264')
        feed = data.get('feed', 'Processed Feed')
        res = media_manager.start_recording(format_ext=fmt, codec=codec, feed_type=feed)
        return jsonify(res)

    @app.route('/api/media/record/stop', methods=['POST'])
    def stop_record():
        res = media_manager.stop_recording()
        return jsonify(res)
        
    @app.route('/api/media/record/status', methods=['GET'])
    def record_status():
        return jsonify(media_manager.get_status())

    # --- Export & Gallery Routes ---
    @app.route('/api/export', methods=['POST'])
    def handle_export():
        data = request.get_json()
        module_id = data.get('module', camera_manager.module_manager.active_module_id)
        fmt = data.get('format', 'CSV')
        
        mod = camera_manager.module_manager.loaded_modules.get(module_id)
        if not mod:
            return jsonify({'error': 'Module not found'}), 404
            
        if hasattr(mod, 'get_export_data'):
            export_data = mod.get_export_data()
        elif hasattr(mod, 'history'):
            export_data = mod.history
        else:
            return jsonify({'error': 'Module does not support exports'}), 400
            
        res = export_manager.export(module_id, export_data, fmt)
        return jsonify(res)

    @app.route('/api/gallery/files', methods=['GET'])
    def get_gallery_files():
        files = []
        dirs = [
            (media_manager.screenshots_dir, 'image'), 
            (media_manager.recordings_dir, 'video'), 
            (export_manager.exports_dir, 'document')
        ]
        for d, cat in dirs:
            if os.path.exists(d):
                for f in os.listdir(d):
                    path = os.path.join(d, f)
                    if os.path.isfile(path):
                        st = os.stat(path)
                        files.append({
                            'name': f,
                            'category': cat,
                            'size': st.st_size,
                            'modified': st.st_mtime,
                            'path': f"/api/gallery/download?cat={cat}&file={f}"
                        })
        files.sort(key=lambda x: x['modified'], reverse=True)
        return jsonify({'files': files})
        
    @app.route('/api/gallery/download')
    def download_gallery_file():
        cat = request.args.get('cat')
        f = request.args.get('file')
        if not f or '..' in f or '/' in f or '\\' in f:
            return "Invalid file", 400
            
        if cat == 'image': d = media_manager.screenshots_dir
        elif cat == 'video': d = media_manager.recordings_dir
        elif cat == 'document': d = export_manager.exports_dir
        else: return "Invalid category", 400
        
        path = os.path.join(d, f)
        if os.path.exists(path):
            return send_file(path, as_attachment=True)
        return "File not found", 404
        
    @app.route('/api/gallery/delete', methods=['POST'])
    def delete_gallery_file():
        data = request.get_json()
        cat = data.get('cat')
        f = data.get('file')
        if not f or '..' in f or '/' in f or '\\' in f:
            return jsonify({'error': 'Invalid file'}), 400
            
        if cat == 'image': d = media_manager.screenshots_dir
        elif cat == 'video': d = media_manager.recordings_dir
        elif cat == 'document': d = export_manager.exports_dir
        else: return jsonify({'error': 'Invalid category'}), 400
        
        path = os.path.join(d, f)
        if os.path.exists(path):
            os.remove(path)
            return jsonify({'success': True})
        return jsonify({'error': 'File not found'}), 404

    @app.route('/api/locales', methods=['GET'])
    def get_locales():
        lang = request.args.get('lang', 'en')
        localization_manager.load_language(lang)
        return jsonify(localization_manager.get_translations())

    # Cleanup on exit
    import atexit
    atexit.register(lambda: camera_manager.stop() if camera_manager else None)

    return app

if __name__ == '__main__':
    app = create_app('development')
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
