"""Text analysis API — standalone NLP + LLM analysis of arbitrary pasted text."""

import logging

from flask import Blueprint, request
from flask_login import login_required

from src.orchestrator import generate_text_analysis
from src.llm import AIServiceError
from utils.responses import success, error

logger = logging.getLogger("insightforge")

analysis_bp = Blueprint("analysis_api", __name__, url_prefix="/api/analyze")

MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 20000


@analysis_bp.route("", methods=["POST"])
@login_required
def analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if len(text) < MIN_TEXT_LENGTH:
        return error(f"Please provide at least {MIN_TEXT_LENGTH} characters of text to analyze.", 400)
    if len(text) > MAX_TEXT_LENGTH:
        return error(f"Text is too long (max {MAX_TEXT_LENGTH} characters).", 400)

    try:
        result = generate_text_analysis(text)
    except AIServiceError as e:
        return error(str(e), 503)
    except Exception:
        logger.exception("Text analysis failed")
        return error("Something went wrong while analyzing this text. Please try again.", 500)

    return success(result)