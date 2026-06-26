import os

class Config:
    """Base configuration class."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'super-secret-pi-key'
    DEBUG = False
    TESTING = False
    
    # Base paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'captures')
    RECORDINGS_DIR = os.path.join(BASE_DIR, 'recordings')
    EXPORTS_DIR = os.path.join(BASE_DIR, 'exports')
    WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Camera settings
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    # In production, we might want to log to file and disable debug mode
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
