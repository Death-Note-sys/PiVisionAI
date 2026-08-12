import logging
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException

from .v1.system import bp as system_bp
from .v1.camera import bp as camera_bp
from .v1.modules import bp as modules_bp
from .v1.gallery import bp as gallery_bp
from .v1.object_detection import bp as object_detection_bp
from .v1.measurement import bp as measurement_bp
from .v1.ocr import bp as ocr_bp
from .v1.ai_identify import bp as ai_identify_bp
from .v1.trigger import bp as trigger_bp

logger = logging.getLogger(__name__)

def create_app() -> Flask:
    """Application Factory handling Security Middleware and Routing."""
    app = Flask(__name__, static_folder='../../static', template_folder='../../templates')
    
    # 1. Security: CORS
    # In production, read from ConfigService. For now, allow all.
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # 2. Security: Rate Limiting
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://"
    )
    
    # 3. Security: Payload Limits
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
    
    # Register Blueprints
    app.register_blueprint(system_bp)
    app.register_blueprint(camera_bp)
    app.register_blueprint(modules_bp)
    app.register_blueprint(gallery_bp)
    app.register_blueprint(object_detection_bp)
    app.register_blueprint(measurement_bp)
    app.register_blueprint(ocr_bp)
    app.register_blueprint(ai_identify_bp)
    app.register_blueprint(trigger_bp)
    
    limiter.exempt(object_detection_bp)
    limiter.exempt(measurement_bp)
    limiter.exempt(ocr_bp)
    limiter.exempt(ai_identify_bp)
    limiter.exempt(trigger_bp)
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/gallery')
    def gallery():
        return render_template('gallery.html')
        
    # 4. Security: Global Exception Handling
    @app.errorhandler(Exception)
    def handle_global_error(e):
        if isinstance(e, HTTPException):
            return e
        logger.error(f"Unhandled Exception: {e}", exc_info=True)
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e) if app.debug else "An unexpected error occurred."
        }), 500
        
    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify({"error": "Rate limit exceeded", "message": str(e.description)}), 429
        
    @app.errorhandler(413)
    def payload_too_large(e):
        return jsonify({"error": "Payload too large", "message": "Request entity exceeds limit."}), 413

    return app
