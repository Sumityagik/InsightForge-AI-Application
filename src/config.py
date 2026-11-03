"""
Centralized application configuration.

All environment-dependent values are read here, once, from the .env file.
No other module should call os.getenv() directly for app-wide settings —
import from here instead, so there is a single source of truth.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root (parent of src/) and load .env from there explicitly,
# so config loading doesn't depend on the current working directory.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """Base configuration shared by all environments."""

    # --- Flask core ---
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-only-insecure-key-change-me")
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    # --- Database ---
    # Falls back to a local SQLite file under instance/ if DATABASE_URL is unset.
    # Because we use SQLAlchemy (not raw SQLite calls), switching to Postgres in
    # production only requires setting DATABASE_URL — no code changes needed.
    _sqlite_path = BASE_DIR / "instance" / "insightforge.db"
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL") or f"sqlite:///{_sqlite_path}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Groq / LLM ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # --- Uploads ---
    UPLOAD_FOLDER = str(BASE_DIR / os.getenv("UPLOAD_FOLDER", "uploads"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "20")) * 1024 * 1024
    ALLOWED_EXTENSIONS = {
        "pdf", "docx", "txt", "xlsx", "xls", "png", "jpg", "jpeg"
    }

    # --- Session / cookies ---
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "False").lower() == "true"
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 24 * 7  # 7 days

    @classmethod
    def validate(cls):
        """
        Called once at app startup. Warns (does not crash) about missing
        optional-but-important config so the app is still usable for
        non-AI development (e.g. auth-only work) without a Groq key.
        """
        warnings = []
        if not cls.GROQ_API_KEY:
            warnings.append(
                "GROQ_API_KEY is not set. AI chat/analysis endpoints will fail "
                "until it is configured in .env."
            )
        if cls.SECRET_KEY == "dev-only-insecure-key-change-me":
            warnings.append(
                "FLASK_SECRET_KEY is using the insecure default. "
                "Set a real random value in .env before deploying."
            )
        return warnings


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True


def get_config():
    env = os.getenv("FLASK_ENV", "development").lower()
    return ProductionConfig if env == "production" else DevelopmentConfig
