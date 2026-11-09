"""Document API: upload a file for analysis, list your documents, delete one."""

import logging

from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user

from services import document_service
from services.upload_service import UploadError
from services.document_service import PartialUploadError
from src.document_loader import DocumentError
from utils.responses import success, error

logger = logging.getLogger("insightforge")

document_bp = Blueprint("document_api", __name__, url_prefix="/api")

MAX_INSTRUCTION_LENGTH = 2000


@document_bp.route("/upload", methods=["POST"])
@login_required
def upload():
    file = request.files.get("file")
    instruction = (request.form.get("message") or "").strip() or None

    if instruction and len(instruction) > MAX_INSTRUCTION_LENGTH:
        return error(f"Your message is too long (max {MAX_INSTRUCTION_LENGTH} characters).", 400)

    conversation_id = None
    raw_id = request.form.get("conversation_id")
    if raw_id:
        if not raw_id.isdigit():
            return error("Invalid conversation.", 400)
        conversation_id = int(raw_id)

    try:
        result = document_service.process_upload(
            current_user.id, file, conversation_id=conversation_id, instruction=instruction
        )
    except UploadError as e:
        return error(str(e), e.status_code)
    except DocumentError as e:
        return error(str(e), 422)
    except PartialUploadError as e:
        # The file was stored fine; only the AI step failed. Return the
        # conversation id so the page can keep using that conversation.
        return jsonify({
            "status": "error",
            "message": str(e),
            "conversation_id": e.conversation_id,
        }), 503
    except Exception:
        logger.exception("Upload failed")
        return error("Something went wrong while processing your file. Please try again.", 500)

    return success(result, "Document analysed.", 201)


@document_bp.route("/documents", methods=["GET"])
@login_required
def list_documents():
    docs = document_service.list_user_documents(current_user.id)
    return success([d.to_dict() for d in docs])


@document_bp.route("/documents/<int:document_id>", methods=["DELETE"])
@login_required
def delete_document(document_id):
    if not document_service.delete_document(current_user.id, document_id):
        return error("Document not found.", 404)
    return success(message="Document deleted.")