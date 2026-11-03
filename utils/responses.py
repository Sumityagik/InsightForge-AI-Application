"""
Standard JSON response builders.

Every API route should return one of these so the frontend can rely on a
single, predictable envelope shape:

    {"status": "success", "data": {...}, "message": "..."}
    {"status": "error", "message": "..."}
"""

from flask import jsonify


def success(data=None, message: str = "", status_code: int = 200):
    payload = {"status": "success", "data": data if data is not None else {}, "message": message}
    return jsonify(payload), status_code


def error(message: str = "Something went wrong.", status_code: int = 400):
    payload = {"status": "error", "message": message}
    return jsonify(payload), status_code
