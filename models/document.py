"""Document model — metadata for an uploaded file, scoped to a user."""

from datetime import datetime, timezone
from src.db import db


class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    conversation_id = db.Column(
        db.Integer, db.ForeignKey("conversations.id"), nullable=True, index=True
    )
    filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(20), nullable=False)  # pdf, docx, txt, xlsx, xls, png, jpg
    file_path = db.Column(db.String(500), nullable=False)
    extracted_text = db.Column(db.Text, nullable=True)  # cached raw extraction
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    chunks = db.relationship(
        "DocumentChunk", backref="document", lazy=True, cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "file_type": self.file_type,
            "conversation_id": self.conversation_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<Document id={self.id} filename={self.filename!r} user_id={self.user_id}>"
