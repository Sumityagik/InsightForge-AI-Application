"""Chat API routes — all protected by @login_required, all scoped to current_user."""

import logging

from flask import Blueprint, request
from flask_login import login_required, current_user

from services import chat_service
from src.llm import AIServiceError
from utils.responses import success, error

logger = logging.getLogger("insightforge")

chat_bp = Blueprint("chat_api", __name__, url_prefix="/api/chat")


@chat_bp.route("/new", methods=["POST"])
@login_required
def new_conversation():
    convo = chat_service.create_conversation(current_user.id)
    return success(convo.to_dict(), "New conversation created.", 201)


@chat_bp.route("/history", methods=["GET"])
@login_required
def history():
    conversations = chat_service.get_user_conversations(current_user.id)
    return success([c.to_dict() for c in conversations])


@chat_bp.route("/<int:conversation_id>/messages", methods=["GET"])
@login_required
def get_messages(conversation_id):
    messages = chat_service.get_conversation_messages(current_user.id, conversation_id)
    if messages is None:
        return error("Conversation not found.", 404)
    return success([m.to_dict() for m in messages])


@chat_bp.route("", methods=["POST"])
@login_required
def send_message():
    data = request.get_json(silent=True) or {}
    content = (data.get("message") or "").strip()
    conversation_id = data.get("conversation_id")

    if not content:
        return error("Message cannot be empty.", 400)
    if len(content) > 8000:
        return error("Message is too long (max 8000 characters).", 400)

    # Resolve or create the conversation, always scoped to current_user.
    if conversation_id:
        convo = chat_service.get_owned_conversation(current_user.id, conversation_id)
        if convo is None:
            return error("Conversation not found.", 404)
    else:
        convo = chat_service.create_conversation(current_user.id, title="New Chat")

    is_first_message = len(convo.messages) == 0
   
    history = [{"role": m.role, "content": m.content} for m in convo.messages]

    try:
        user_msg = chat_service.add_message(convo.id, "user", content)
        reply_text = chat_service.generate_reply(content, history)
        ai_msg = chat_service.add_message(convo.id, "assistant", reply_text)

        new_title = chat_service.auto_title_from_message(content) if is_first_message else None
        chat_service.touch_conversation(convo, new_title=new_title)
    except AIServiceError as e:
        # The user's message is already saved; only the AI reply failed.
        # Return a clean, specific error instead of a generic 500.
        return error(str(e), 503)
    except Exception:
        logger.exception("Failed to process chat message")
        return error("Something went wrong while processing your message. Please try again.", 500)
    return success({
        "conversation": convo.to_dict(),
        "user_message": user_msg.to_dict(),
        "ai_message": ai_msg.to_dict(),
    })


@chat_bp.route("/<int:conversation_id>", methods=["DELETE"])
@login_required
def delete_conversation(conversation_id):
    deleted = chat_service.delete_conversation(current_user.id, conversation_id)
    if not deleted:
        return error("Conversation not found.", 404)
    return success(message="Conversation deleted.")