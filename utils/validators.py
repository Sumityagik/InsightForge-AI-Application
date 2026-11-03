"""Input validators for authentication endpoints."""

from email_validator import validate_email as _validate_email, EmailNotValidError

MIN_PASSWORD_LENGTH = 8


def validate_registration(name: str, email: str, password: str, confirm_password: str) -> list[str]:
    """Returns a list of human-readable error messages (empty list = valid)."""
    errors = []

    if not name or not name.strip():
        errors.append("Name is required.")
    elif len(name.strip()) > 120:
        errors.append("Name is too long.")

    if not email or not email.strip():
        errors.append("Email is required.")
    else:
        try:
            _validate_email(email, check_deliverability=False)
        except EmailNotValidError:
            errors.append("Please enter a valid email address.")

    if not password:
        errors.append("Password is required.")
    elif len(password) < MIN_PASSWORD_LENGTH:
        errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")

    if password and confirm_password and password != confirm_password:
        errors.append("Passwords do not match.")
    elif password and not confirm_password:
        errors.append("Please confirm your password.")

    return errors


def validate_login(email: str, password: str) -> list[str]:
    errors = []
    if not email or not email.strip():
        errors.append("Email is required.")
    if not password:
        errors.append("Password is required.")
    return errors
