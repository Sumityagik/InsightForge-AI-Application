"""
Shared SQLAlchemy database instance.

Kept in its own module (separate from app.py and models/) to avoid circular
imports: models import `db` from here, and app.py imports both models and
this file without any module needing to import app.py back.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
