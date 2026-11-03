"""
Sprint 1 smoke tests: verifies the app factory, DB, and models work
end-to-end (create tables, insert, hash/verify password, relationships,
cascade delete). Run with: python -m pytest tests/test_models.py -v
"""

import pytest

from app import create_app
from src.db import db
from models import User, Conversation, Message, Document, DocumentChunk


@pytest.fixture
def app():
    """Fresh in-memory SQLite app for each test — no dependency on dev DB."""
    test_app = create_app()
    test_app.config.update(
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        TESTING=True,
    )
    with test_app.app_context():
        db.create_all()
        yield test_app
        db.session.remove()
        db.drop_all()


def test_app_factory_creates_app(app):
    assert app is not None
    assert app.config["TESTING"] is True


def test_health_endpoint(app):
    client = app.test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["data"]["database"] == "ok"


def test_create_user_and_hash_password(app):
    with app.app_context():
        user = User(name="Ada Lovelace", email="ada@example.com")
        user.set_password("supersecret123")
        db.session.add(user)
        db.session.commit()

        fetched = db.session.query(User).filter_by(email="ada@example.com").first()
        assert fetched is not None
        assert fetched.password_hash != "supersecret123"  # never plain text
        assert fetched.check_password("supersecret123") is True
        assert fetched.check_password("wrongpassword") is False


def test_user_email_uniqueness(app):
    with app.app_context():
        u1 = User(name="A", email="dup@example.com")
        u1.set_password("x")
        db.session.add(u1)
        db.session.commit()

        u2 = User(name="B", email="dup@example.com")
        u2.set_password("y")
        db.session.add(u2)
        with pytest.raises(Exception):
            db.session.commit()
        db.session.rollback()


def test_conversation_message_relationship(app):
    with app.app_context():
        user = User(name="Grace Hopper", email="grace@example.com")
        user.set_password("pw")
        db.session.add(user)
        db.session.commit()

        convo = Conversation(user_id=user.id, title="Research Chat")
        db.session.add(convo)
        db.session.commit()

        msg1 = Message(conversation_id=convo.id, role="user", content="Summarize this.")
        msg2 = Message(conversation_id=convo.id, role="assistant", content="Here is the summary.")
        db.session.add_all([msg1, msg2])
        db.session.commit()

        fetched_convo = db.session.get(Conversation, convo.id)
        assert len(fetched_convo.messages) == 2
        assert fetched_convo.messages[0].role == "user"


def test_document_and_chunks(app):
    with app.app_context():
        user = User(name="Alan Turing", email="alan@example.com")
        user.set_password("pw")
        db.session.add(user)
        db.session.commit()

        doc = Document(
            user_id=user.id,
            filename="paper.pdf",
            file_type="pdf",
            file_path="/uploads/1/paper.pdf",
            extracted_text="Full extracted text...",
        )
        db.session.add(doc)
        db.session.commit()

        chunk = DocumentChunk(document_id=doc.id, chunk_index=0, content="First chunk of text.")
        db.session.add(chunk)
        db.session.commit()

        fetched_doc = db.session.get(Document, doc.id)
        assert len(fetched_doc.chunks) == 1
        assert fetched_doc.chunks[0].content == "First chunk of text."


def test_cascade_delete_user_removes_conversations_and_documents(app):
    with app.app_context():
        user = User(name="Delete Me", email="delete@example.com")
        user.set_password("pw")
        db.session.add(user)
        db.session.commit()
        user_id = user.id

        convo = Conversation(user_id=user.id, title="Temp")
        doc = Document(user_id=user.id, filename="f.txt", file_type="txt", file_path="/x")
        db.session.add_all([convo, doc])
        db.session.commit()

        db.session.delete(user)
        db.session.commit()

        assert db.session.get(User, user_id) is None
        assert db.session.query(Conversation).filter_by(user_id=user_id).first() is None
        assert db.session.query(Document).filter_by(user_id=user_id).first() is None
