"""
Document service: the full "upload a file and analyse it" flow.

    validate -> save -> extract text -> create conversation/document rows
             -> AI analysis -> save messages -> return result

Everything is scoped to the logged-in user. Raw library errors never reach
the user; they are turned into UploadError / DocumentError / AIServiceError
with friendly messages by the layers below.
"""

import logging
import os

from src.db import db
from models import Document
from src.document_loader import load_document, DocumentError
from src.llm import AIServiceError
from src.orchestrator import generate_chat_reply, generate_text_analysis
from services import chat_service, upload_service, rag_service
from services.upload_service import UploadError

logger = logging.getLogger("insightforge")

# How much of a document we hand to the AI in one go. Retrieval (RAG) in the
# next sprint replaces this with "only the relevant chunks".
MAX_ANALYSIS_CHARS = 12_000
MAX_CONTEXT_CHARS = 12_000

SPREADSHEET_PROMPT = (
    "Analyse the spreadsheet in the document context. Describe what each sheet contains, "
    "what the columns mean, any notable numbers or patterns, and anything that looks "
    "incomplete or inconsistent. Use short Markdown sections."
)


class PartialUploadError(Exception):
    """The file was saved, but the AI step failed. Carries the conversation id."""

    def __init__(self, message, conversation_id):
        super().__init__(message)
        self.conversation_id = conversation_id


# ------------------------------------------------------------------ context
def build_document_context(conversation):
    """
    Text of the documents attached to this conversation, trimmed to a safe
    size. Returned to the chat flow so follow-up questions can use it.
    """
    docs = [d for d in conversation.documents if d.extracted_text]
    if not docs:
        return None
    per_doc = MAX_CONTEXT_CHARS // len(docs)
    return "\n\n".join(f"[Document: {d.filename}]\n{d.extracted_text[:per_doc]}" for d in docs)


# ------------------------------------------------------------------ upload
def process_upload(user_id, file_storage, conversation_id=None, instruction=None):
    ext = upload_service.validate_upload(file_storage)

    # Check the conversation first, so we don't store a file for nothing.
    conversation = None
    if conversation_id is not None:
        conversation = chat_service.get_owned_conversation(user_id, conversation_id)
        if conversation is None:
            raise UploadError("Conversation not found.", 404)

    display_name, relative_path = upload_service.save_upload(file_storage, user_id, ext)

    # Extract text. If this fails, remove the stored file again.
    try:
        full_path = upload_service.resolve_path(relative_path)
        size_label = upload_service.format_size(os.path.getsize(full_path))
        loaded = load_document(full_path, ext)
    except DocumentError:
        upload_service.delete_file(relative_path)
        raise

    text = loaded["text"]

    # Create the conversation only now that the file is known to be readable.
    is_new_conversation = conversation is None
    if is_new_conversation:
        conversation = chat_service.create_conversation(user_id, title=f"Analysis: {display_name}"[:60])
    history = [{"role": m.role, "content": m.content} for m in conversation.messages]

    document = Document(
        user_id=user_id,
        conversation_id=conversation.id,
        filename=display_name,
        file_type=ext,
        file_path=relative_path,
        extracted_text=text,
    )
    db.session.add(document)
    db.session.commit()

    user_text = f"**Uploaded:** {display_name} ({ext.upper()}, {size_label})"
    if instruction:
        user_text += f"\n\n{instruction}"
    user_msg = chat_service.add_message(conversation.id, "user", user_text)

    # ---- AI step
    context = f"[Document: {display_name}]\n{text[:MAX_CONTEXT_CHARS]}"
    try:
        if instruction:
            reply = generate_chat_reply(instruction, history=history, document_context=context)
        elif ext in ("xlsx", "xls"):
            reply = generate_chat_reply(SPREADSHEET_PROMPT, history=history, document_context=context)
        else:
            reply = generate_text_analysis(text[:MAX_ANALYSIS_CHARS])["narrative"]
    except AIServiceError as e:
        chat_service.touch_conversation(conversation)
        raise PartialUploadError(str(e), conversation.id)
    except Exception:
        logger.exception("Unexpected error during document analysis")
        chat_service.touch_conversation(conversation)
        raise PartialUploadError("Something went wrong while analysing the document.", conversation.id)

    if len(text) > MAX_ANALYSIS_CHARS and not instruction:
        reply += (
            f"\n\n*Note: this document is long, so only the first {MAX_ANALYSIS_CHARS:,} "
            "characters were analysed.*"
        )

    ai_msg = chat_service.add_message(conversation.id, "assistant", reply)
    chat_service.touch_conversation(conversation)

    return {
        "conversation": conversation.to_dict(),
        "user_message": user_msg.to_dict(),
        "ai_message": ai_msg.to_dict(),
        "document": document.to_dict(),
    }


# ------------------------------------------------------------------ list / delete
def list_user_documents(user_id):
    return (
        Document.query.filter_by(user_id=user_id)
        .order_by(Document.created_at.desc())
        .all()
    )


def delete_document(user_id, document_id) -> bool:
    document = Document.query.filter_by(id=document_id, user_id=user_id).first()
    if document is None:
        return False
    upload_service.delete_file(document.file_path)
    rag_service.delete_document_index(document)
    db.session.delete(document)
    db.session.commit()
    return True

def delete_conversation_documents(user_id, conversation_id) -> int:
    """Removes every document (file, search index, rows) of one conversation."""
    documents = Document.query.filter_by(user_id=user_id, conversation_id=conversation_id).all()
    for document in documents:
        upload_service.delete_file(document.file_path)
        rag_service.delete_document_index(document)
        db.session.delete(document)
    db.session.commit()
    return len(documents)