"""
Upload service: validates an uploaded file and stores it safely on disk.

Safety measures:
- only whitelisted extensions are accepted
- the real file content is checked against the extension (a text file
  renamed to .pdf is rejected)
- the original filename is never used on disk (we store a random name), so
  path tricks like "../../evil.txt" cannot escape the uploads folder
- files are stored per user: uploads/<user_id>/<random>.<ext>
"""

import os
import uuid

from flask import current_app
from werkzeug.utils import secure_filename

# Types we can actually process right now.
SUPPORTED_TYPES = {"pdf", "docx", "txt", "xlsx", "xls"}
# Accepted by the product spec but not built yet (needs OCR).
IMAGE_TYPES = {"png", "jpg", "jpeg"}


class UploadError(Exception):
    """Problem with the uploaded file. Message is safe to show the user."""

    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def get_extension(filename: str) -> str:
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def validate_upload(file_storage) -> str:
    """Checks the file's name/type. Returns the lowercase extension."""
    if file_storage is None or not file_storage.filename:
        raise UploadError("No file was selected.")

    ext = get_extension(file_storage.filename)

    if ext in IMAGE_TYPES:
        raise UploadError("Image upload isn't available yet. It will be added in a later update.")
    if ext not in SUPPORTED_TYPES:
        raise UploadError("Unsupported file type. Please upload a PDF, DOCX, TXT, XLSX or XLS file.")
    return ext


def _check_signature(path: str, ext: str) -> None:
    """Makes sure the file's real content matches its extension."""
    with open(path, "rb") as f:
        head = f.read(1024)

    if len(head) == 0:
        raise UploadError("The file is empty.")

    if ext == "pdf" and b"%PDF" not in head:
        raise UploadError("This file isn't a valid PDF.")
    if ext in ("docx", "xlsx") and not head.startswith(b"PK"):
        raise UploadError(f"This file isn't a valid .{ext} document.")
    if ext == "xls" and not head.startswith(b"\xd0\xcf\x11\xe0"):
        raise UploadError("This file isn't a valid .xls spreadsheet.")


def save_upload(file_storage, user_id: int, ext: str):
    """
    Saves the file and returns (display_name, relative_path).
    `relative_path` is relative to the uploads folder, e.g. "3/9f2c....pdf".
    """
    stem = file_storage.filename.rsplit(".", 1)[0] if "." in file_storage.filename else file_storage.filename
    display_name = f"{secure_filename(stem) or 'document'}.{ext}"

    stored_name = f"{uuid.uuid4().hex}.{ext}"
    user_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], str(user_id))
    os.makedirs(user_dir, exist_ok=True)

    full_path = os.path.join(user_dir, stored_name)
    file_storage.save(full_path)

    try:
        _check_signature(full_path, ext)
    except UploadError:
        delete_file(f"{user_id}/{stored_name}")
        raise

    return display_name, f"{user_id}/{stored_name}"


def resolve_path(relative_path: str):
    """Returns the absolute path, or None if it would fall outside the uploads folder."""
    base = os.path.abspath(current_app.config["UPLOAD_FOLDER"])
    full = os.path.abspath(os.path.join(base, relative_path))
    if not full.startswith(base + os.sep):
        return None
    return full


def delete_file(relative_path: str) -> None:
    full = resolve_path(relative_path)
    if full and os.path.isfile(full):
        try:
            os.remove(full)
        except OSError:
            pass


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"