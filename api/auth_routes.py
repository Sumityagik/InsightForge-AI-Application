"""
Authentication routes.

Two blueprints live in this module:
- auth_bp     -> HTML page routes: GET /login, GET /register
- auth_api_bp -> JSON API routes:  POST /api/auth/register, /login, /logout,
                                     GET /api/auth/me

Kept separate (rather than one blueprint mixing HTML and JSON) so the
url_prefix stays clean and each route's response type is unambiguous.
"""

import logging

from flask import Blueprint, render_template, request, redirect, url_for
from flask_login import login_user, logout_user, login_required, current_user
from sqlalchemy.exc import IntegrityError

from src.db import db
from models import User
from utils.responses import success, error
from utils.validators import validate_registration, validate_login

logger = logging.getLogger("insightforge")

auth_bp = Blueprint("auth", __name__)
auth_api_bp = Blueprint("auth_api", __name__, url_prefix="/api/auth")


# ============================================================
# PAGE ROUTES (render HTML)
# ============================================================

@auth_bp.route("/login", methods=["GET"])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for("chat_page"))
    return render_template("login.html")


@auth_bp.route("/register", methods=["GET"])
def register_page():
    if current_user.is_authenticated:
        return redirect(url_for("chat_page"))
    return render_template("register.html")


# ============================================================
# API ROUTES (JSON)
# ============================================================

@auth_api_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    confirm_password = data.get("confirm_password") or ""

    errors = validate_registration(name, email, password, confirm_password)
    if errors:
        return error(errors[0], 400)

    if User.query.filter_by(email=email).first():
        return error("An account with this email already exists.", 409)

    user = User(name=name, email=email)
    user.set_password(password)

    try:
        db.session.add(user)
        db.session.commit()
    except IntegrityError:
        # Handles a race condition: two simultaneous registrations with the
        # same email both pass the check above but only one insert succeeds.
        db.session.rollback()
        return error("An account with this email already exists.", 409)
    except Exception:
        db.session.rollback()
        logger.exception("Failed to create user during registration")
        return error("Could not create account. Please try again.", 500)

    login_user(user)
    return success(user.to_dict(), "Account created successfully.", 201)


@auth_api_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    remember = bool(data.get("remember", False))

    errors = validate_login(email, password)
    if errors:
        return error(errors[0], 400)

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        # Intentionally identical message for "no such user" and "wrong
        # password" so login attempts can't be used to enumerate accounts.
        return error("Invalid email or password.", 401)

    login_user(user, remember=remember)
    return success(user.to_dict(), "Logged in successfully.")


@auth_api_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return success(message="Logged out successfully.")


@auth_api_bp.route("/me", methods=["GET"])
@login_required
def me():
    return success(current_user.to_dict())
