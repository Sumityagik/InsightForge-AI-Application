"""
InsightForge AI — Flask application entry point.

Uses the application factory pattern (create_app) so the app can be
instantiated differently for testing vs. running vs. future CLI commands,
without import-time side effects.
"""

import logging

from flask import Flask, render_template, request, redirect, url_for
from flask_login import LoginManager, login_required

from src.config import get_config
from src.db import db

login_manager = LoginManager()
login_manager.login_view = "auth.login_page"  # wired up in Sprint 2
login_manager.session_protection = "strong"


def create_app(config_object=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_object or get_config())

    # --- Logging: detailed on the server, never exposed to the client ---
    logging.basicConfig(
        level=logging.DEBUG if app.config.get("DEBUG") else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("insightforge")

    config_cls = config_object or get_config()
    for warning in config_cls.validate():
        logger.warning(warning)

    # --- Extensions ---
    db.init_app(app)
    login_manager.init_app(app)

    # Ensure the instance/ and uploads/ folders exist (SQLite file + user uploads).
    import os
    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # --- Import models so SQLAlchemy metadata is registered ---
    with app.app_context():
        import models  # noqa: F401

    # --- Flask-Login user loader ---
    @login_manager.user_loader
    def load_user(user_id):
        from models import User
        return db.session.get(User, int(user_id))

    # Flask-Login's default behavior on an unauthenticated request is to
    # redirect to login_view. That's correct for page routes, but an API
    # route should return a clean JSON 401 instead of an HTML redirect.
    @login_manager.unauthorized_handler
    def unauthorized():
        from utils.responses import error as error_response
        if request.path.startswith("/api/"):
            return error_response("Authentication required. Please log in.", 401)
        return redirect(url_for("auth.login_page"))

    # --- Blueprints ---
    # Registered incrementally as each sprint adds them.
    _register_blueprints(app)

    # --- Basic public page routes (Home / About) ---
    @app.route("/")
    def home():
        return render_template("home.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    # --- Chat page (placeholder) ---
    # Full chat UI is built in Sprint 3. This exists now only so the login/
    # registration flow has a real authenticated destination to redirect to
    # and can be smoke-tested end-to-end.
    @app.route("/chat")
    @login_required
    def chat_page():
       return render_template("chat.html")

    # --- Centralized error handlers: never leak tracebacks to the client ---
    _register_error_handlers(app, logger)

    return app


def _register_blueprints(app):
    """
    Registers blueprints as they become available.
    Sprint 1: system_bp (health check).
    Sprint 2: auth_bp (login/register pages) + auth_api_bp (JSON auth API).
    Sprint 3: chat_bp (conversation/message JSON API).
    Sprint 4: analysis_bp (standalone text analysis).
    Sprint 5+ will add: document_bp
    """
    from api.system_routes import system_bp
    app.register_blueprint(system_bp, url_prefix="/api")

    from api.auth_routes import auth_bp, auth_api_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(auth_api_bp)

    from api.chat_routes import chat_bp
    app.register_blueprint(chat_bp)

    from api.analysis_routes import analysis_bp
    app.register_blueprint(analysis_bp)


def _register_error_handlers(app, logger):
    from utils.responses import error as error_response

    @app.errorhandler(404)
    def not_found(e):
        return error_response("Resource not found.", 404)

    @app.errorhandler(413)
    def too_large(e):
        return error_response("File is too large.", 413)

    @app.errorhandler(500)
    def server_error(e):
        logger.exception("Unhandled server error")
        return error_response("An unexpected error occurred. Please try again.", 500)


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=app.config.get("DEBUG", False))
