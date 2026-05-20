from flask import Flask
from flask_cors import CORS
from config import config


def create_app(config_name="default"):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    CORS(app, origins=app.config.get("CORS_ORIGINS", "*"))

    # Ensure directories exist
    import os as _os
    _os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    _os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

    # Register API blueprints
    from .api.upload import upload_bp
    from .api.translate import translate_bp
    from .api.project import project_bp
    from .api.download import download_bp

    app.register_blueprint(upload_bp, url_prefix="/api/upload")
    app.register_blueprint(translate_bp, url_prefix="/api/translate")
    app.register_blueprint(project_bp, url_prefix="/api/project")
    app.register_blueprint(download_bp, url_prefix="/api/download")

    return app
