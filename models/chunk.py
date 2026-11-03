"""
DocumentChunk model — a text chunk of a document, used for RAG retrieval.

The actual vector embedding lives in the per-document FAISS index on disk
(see src/vector_store.py); this row stores the chunk text and its position
so a FAISS search result (an index position) can be mapped back to content.
"""

from src.db import db


class DocumentChunk(db.Model):
    __tablename__ = "document_chunks"

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(
        db.Integer, db.ForeignKey("documents.id"), nullable=False, index=True
    )
    chunk_index = db.Column(db.Integer, nullable=False)  # position within the FAISS index
    content = db.Column(db.Text, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
        }

    def __repr__(self):
        return f"<DocumentChunk id={self.id} document_id={self.document_id} index={self.chunk_index}>"
