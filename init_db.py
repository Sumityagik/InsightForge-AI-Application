"""
Database initialization / verification script.

Run this once (and again after any model change, or use a real migration
tool like Alembic later) to create all tables defined in models/.

Usage:
    python init_db.py
"""

from app import create_app
from src.db import db


def main():
    app = create_app()
    with app.app_context():
        db.create_all()
        print("✔ Database tables created (or already exist).")
        print(f"  URI: {app.config['SQLALCHEMY_DATABASE_URI']}")

        # Verify tables actually exist
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        print(f"  Tables found: {tables}")

        expected = {"users", "conversations", "messages", "documents", "document_chunks"}
        missing = expected - set(tables)
        if missing:
            print(f"  ⚠ WARNING: missing expected tables: {missing}")
        else:
            print("  ✔ All expected tables are present.")


if __name__ == "__main__":
    main()
