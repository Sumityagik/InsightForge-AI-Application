"""
Chat service — all conversation/message business logic lives here, not in
the routes. Every function is scoped by user_id so one user can never read
or modify another user's conversations (checked explicitly, not just via
the URL).
"""

import logging

from src.db import db
from models import Conversation, Message
from src.orchestrator import generate_chat_reply
from src.llm import AIServiceError
from services import document_service 

logger = logging.getLogger("insightforge")


def create_conversation(user_id: int, title: str = "New Chat") -> Conversation:
    convo = Conversation(user_id=user_id, title=title)
    db.session.add(convo)
    db.session.commit()
    return convo


def get_user_conversations(user_id: int) -> list[Conversation]:
    return (
        Conversation.query.filter_by(user_id=user_id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )


def get_owned_conversation(user_id: int, conversation_id: int) -> Conversation | None:
    """Returns the conversation only if it belongs to this user, else None."""
    return Conversation.query.filter_by(id=conversation_id, user_id=user_id).first()


def get_conversation_messages(user_id: int, conversation_id: int) -> list[Message] | None:
    convo = get_owned_conversation(user_id, conversation_id)
    if convo is None:
        return None
    return convo.messages


def add_message(conversation_id: int, role: str, content: str) -> Message:
    msg = Message(conversation_id=conversation_id, role=role, content=content)
    db.session.add(msg)
    db.session.commit()
    return msg


def touch_conversation(conversation: Conversation, new_title: str | None = None) -> None:
    """Updates updated_at (for sidebar ordering) and optionally sets a title."""
    if new_title:
        conversation.title = new_title
    db.session.add(conversation)
    db.session.commit()


def auto_title_from_message(message: str, max_len: int = 48) -> str:
    cleaned = " ".join(message.strip().split())
    if len(cleaned) <= max_len:
        return cleaned or "New Chat"
    return cleaned[:max_len].rstrip() + "..."


def delete_conversation(user_id: int, conversation_id: int) -> bool:
    convo = get_owned_conversation(user_id, conversation_id)
    if convo is None:
        return False
    document_service.delete_conversation_documents(user_id, conversation_id)
    db.session.delete(convo)
    db.session.commit()
    return True


def generate_reply(user_message: str, history: list[dict], document_context:str | None = None) -> str:
    """
    Generates a real AI reply via Groq, grounded in the conversation's
    prior messages so multi-turn context is preserved. `history` must be
    the messages that existed BEFORE this turn (oldest first) — callers
    must fetch it before inserting the current user message, otherwise
    the current message would be duplicated in the prompt.

    Raises AIServiceError on any failure (missing key, rate limit,
    network, etc.) — the route layer converts that into a clean JSON
    error response rather than a 500 traceback.
    """
    try:
        return generate_chat_reply(user_message, history=history, document_context = document_context)
    except AIServiceError:
        raise
    except Exception:
        logger.exception("Unexpected error while generating AI reply")
        raise AIServiceError("Something went wrong generating a response. Please try again.")