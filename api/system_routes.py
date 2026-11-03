"""System-level routes (health checks, etc.) — no auth required."""

from flask import Blueprint
from sqlalchemy import text

from src.db import db
from src.config import get_config
from utils.responses import success, error

system_bp = Blueprint("system", __name__)


@system_bp.route("/health", methods=["GET"])
def health():
    """
    Returns app + database status. Useful for deployment platforms
    (Render/Railway) to verify the service is actually up, not just listening.
    """
    db_status = "ok"
    try:
        db.session.execute(text("SELECT 1"))
    except Exception:
        db_status = "unreachable"

    config = get_config()
    groq_configured = bool(config.GROQ_API_KEY)

    payload = {
        "app": "InsightForge AI",
        "status": "running",
        "database": db_status,
        "groq_configured": groq_configured,
    }

    if db_status != "ok":
        return error("Database is unreachable.", 503)

    return success(payload, message="InsightForge AI backend is healthy.")
