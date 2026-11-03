"""
Sprint 2 tests: registration, login, logout, session protection, and
account-enumeration resistance. Run with: python -m pytest tests/test_auth.py -v
"""

import pytest

from app import create_app
from src.db import db


@pytest.fixture
def app():
    test_app = create_app()
    test_app.config.update(
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        TESTING=True,
        WTF_CSRF_ENABLED=False,
    )
    with test_app.app_context():
        db.create_all()
        yield test_app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def register(client, name="Ada Lovelace", email="ada@example.com",
              password="supersecret123", confirm=None):
    return client.post("/api/auth/register", json={
        "name": name,
        "email": email,
        "password": password,
        "confirm_password": confirm if confirm is not None else password,
    })


def login(client, email="ada@example.com", password="supersecret123"):
    return client.post("/api/auth/login", json={"email": email, "password": password})


# ---- Pages ----

def test_login_page_loads(client):
    resp = client.get("/login")
    assert resp.status_code == 200
    assert b"Log In" in resp.data


def test_register_page_loads(client):
    resp = client.get("/register")
    assert resp.status_code == 200
    assert b"Create Account" in resp.data or b"Create your account" in resp.data


# ---- Registration ----

def test_register_success(client):
    resp = register(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["data"]["email"] == "ada@example.com"
    assert "password" not in body["data"]
    assert "password_hash" not in body["data"]


def test_register_logs_user_in(client):
    register(client)
    me_resp = client.get("/api/auth/me")
    assert me_resp.status_code == 200
    assert me_resp.get_json()["data"]["email"] == "ada@example.com"


def test_register_duplicate_email_rejected(client):
    register(client)
    resp = register(client, name="Someone Else")
    assert resp.status_code == 409
    assert resp.get_json()["status"] == "error"


def test_register_password_mismatch_rejected(client):
    resp = register(client, confirm="different-password")
    assert resp.status_code == 400


def test_register_short_password_rejected(client):
    resp = register(client, password="short", confirm="short")
    assert resp.status_code == 400


def test_register_invalid_email_rejected(client):
    resp = register(client, email="not-an-email")
    assert resp.status_code == 400


def test_register_missing_name_rejected(client):
    resp = register(client, name="")
    assert resp.status_code == 400


# ---- Login ----

def test_login_success(client):
    register(client)
    client.post("/api/auth/logout")
    resp = login(client)
    assert resp.status_code == 200
    assert resp.get_json()["data"]["email"] == "ada@example.com"


def test_login_wrong_password_rejected(client):
    register(client)
    client.post("/api/auth/logout")
    resp = login(client, password="wrongpassword")
    assert resp.status_code == 401


def test_login_nonexistent_user_rejected(client):
    resp = login(client, email="nobody@example.com", password="whatever123")
    assert resp.status_code == 401


def test_login_error_message_identical_for_bad_user_and_bad_password(client):
    """Prevents account enumeration via distinguishable error messages."""
    register(client)
    client.post("/api/auth/logout")

    resp_wrong_pw = login(client, password="wrongpassword")
    resp_no_user = login(client, email="doesnotexist@example.com", password="whatever123")

    assert resp_wrong_pw.get_json()["message"] == resp_no_user.get_json()["message"]


# ---- Logout ----

def test_logout_requires_login(client):
    resp = client.post("/api/auth/logout")
    assert resp.status_code == 401


def test_logout_clears_session(client):
    register(client)
    resp = client.post("/api/auth/logout")
    assert resp.status_code == 200

    me_resp = client.get("/api/auth/me")
    assert me_resp.status_code == 401


# ---- Access control ----

def test_me_requires_login(client):
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401
    assert resp.get_json()["status"] == "error"


def test_protected_chat_page_redirects_when_logged_out(client):
    resp = client.get("/chat", follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_protected_chat_page_accessible_when_logged_in(client):
    register(client)
    resp = client.get("/chat")
    assert resp.status_code == 200
    assert b"Ada Lovelace" in resp.data


def test_logged_in_user_redirected_away_from_login_page(client):
    register(client)
    resp = client.get("/login", follow_redirects=False)
    assert resp.status_code == 302
