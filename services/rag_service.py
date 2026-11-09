"""
RAG service (Retrieval-Augmented Generation).

    Document text -> chunks -> embeddings -> FAISS index        (index_document)
    Question      -> embedding -> nearest chunks -> context     (retrieve_context)
    context + question -> language model -> grounded answer     (done by the chat flow)

How a question gets its context:

  * No documents in the conversation        -> no context
  * Documents are short (fit in the prompt) -> send the full text; retrieval is
                                               not needed and would only lose detail
  * Documents are long                      -> send only the most relevant chunks
  * Retrieval fails (model missing, etc.)   -> fall back to the start of the text,
                                               so chat keeps working

Every search is limited to the documents of ONE conversation, and every index
file belongs to ONE document, so text never leaks between users or chats.
"""

import logging
import os
from typing import NamedTuple, Optional

from flask import current_app

from src.db import db
from models import DocumentChunk
from src.chunker import split_text, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from src.embeddings import embed_texts, embed_query
from src import vector_store

logger = logging.getLogger("insightforge")

TOP_K = 5                      # how many chunks to send to the model
FULL_TEXT_LIMIT = 6_000        # at or below this many characters, send everything
MAX_CONTEXT_CHARS = 12_000     # hard cap on context size in "full" / "truncated" mode


class RetrievalResult(NamedTuple):
    context: Optional[str]     # text to give the model (None = no documents)
    mode: str                  # "none" | "full" | "retrieval" | "truncated"
    documents: list            # file names that contributed
    sections: int              # number of chunks used (retrieval mode)


# ------------------------------------------------------------------ paths
def index_path(document) -> str:
    """indexes/<user_id>/<document_id>.faiss, next to (not inside) the user's uploads."""
    return os.path.join(
        current_app.config["UPLOAD_FOLDER"], "indexes", str(document.user_id), f"{document.id}.faiss"
    )


def is_indexed(document) -> bool:
    if not vector_store.index_exists(index_path(document)):
        return False
    return DocumentChunk.query.filter_by(document_id=document.id).first() is not None


# ------------------------------------------------------------------ indexing
def index_document(document) -> int:
    """
    Splits the document's text into chunks, embeds them and stores the index.
    Replaces any earlier index for this document. Returns the number of chunks.
    Raises EmbeddingError / VectorStoreError if it cannot be done.
    """
    chunks = split_text(document.extracted_text or "", DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
    if not chunks:
        return 0

    vectors = embed_texts(chunks)              # may raise EmbeddingError
    path = index_path(document)
    vector_store.save_index(vectors, path)     # may raise VectorStoreError

    try:
        DocumentChunk.query.filter_by(document_id=document.id).delete()
        db.session.add_all(
            DocumentChunk(document_id=document.id, chunk_index=i, content=text)
            for i, text in enumerate(chunks)
        )
        db.session.commit()
    except Exception:
        db.session.rollback()
        vector_store.delete_index(path)        # don't leave an index without matching rows
        raise

    logger.info("Indexed document %s into %d chunks", document.id, len(chunks))
    return len(chunks)


def delete_document_index(document) -> None:
    vector_store.delete_index(index_path(document))
    # The chunk rows are removed together with the document (cascade).


# ------------------------------------------------------------------ retrieval
def retrieve_context(conversation, query: Optional[str]) -> RetrievalResult:
    docs = [
        d for d in conversation.documents
        if d.extracted_text and d.user_id == conversation.user_id
    ]
    if not docs:
        return RetrievalResult(None, "none", [], 0)

    total_chars = sum(len(d.extracted_text) for d in docs)

    if total_chars <= FULL_TEXT_LIMIT:
        return _full_text(docs, "full")
    if not (query or "").strip():
        return _full_text(docs, "truncated")

    try:
        return _retrieve(docs, query)
    except Exception:
        # Retrieval is an improvement, not a requirement: never break chat because of it.
        logger.exception("Retrieval failed; falling back to the start of the document text")
        db.session.rollback()
        return _full_text(docs, "truncated")


def _full_text(docs, mode) -> RetrievalResult:
    per_doc = MAX_CONTEXT_CHARS // len(docs)
    context = "\n\n".join(f"[Document: {d.filename}]\n{d.extracted_text[:per_doc]}" for d in docs)
    return RetrievalResult(context, mode, [d.filename for d in docs], 0)


def _retrieve(docs, query: str) -> RetrievalResult:
    query_vector = embed_query(query)

    hits = []  # (score, document, chunk_index)
    for doc in docs:
        if not is_indexed(doc):
            index_document(doc)  # e.g. documents uploaded before indexing existed
        for position, score in vector_store.search(index_path(doc), query_vector, TOP_K):
            hits.append((score, doc, position))

    if not hits:
        return _full_text(docs, "truncated")

    hits.sort(key=lambda h: h[0], reverse=True)
    best = hits[:TOP_K]

    wanted = {}  # document id -> set of chunk positions
    for _score, doc, position in best:
        wanted.setdefault(doc.id, set()).add(position)

    blocks = []
    used_docs = []
    for doc in docs:  # keep the order the documents were added in
        positions = wanted.get(doc.id)
        if not positions:
            continue
        rows = (
            DocumentChunk.query.filter(
                DocumentChunk.document_id == doc.id,
                DocumentChunk.chunk_index.in_(positions),
            )
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
        for row in rows:  # reading order inside the document
            blocks.append(f"[Document: {doc.filename} | excerpt {row.chunk_index + 1}]\n{row.content}")
        if rows:
            used_docs.append(doc.filename)

    if not blocks:
        return _full_text(docs, "truncated")

    return RetrievalResult("\n\n".join(blocks), "retrieval", used_docs, len(blocks))


def source_note(result: RetrievalResult) -> Optional[str]:
    """A short line shown under the answer when it was built from retrieved sections."""
    if result.mode != "retrieval":
        return None
    names = ", ".join(result.documents)
    plural = "section" if result.sections == 1 else "sections"
    return f"*Answered using {result.sections} relevant {plural} of: {names}*"
