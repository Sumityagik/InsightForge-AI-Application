"""
Models package.

Importing this package registers all SQLAlchemy model classes with the
shared `db` metadata, which is required before calling db.create_all().
"""

from .user import User
from .conversation import Conversation
from .message import Message
from .document import Document
from .chunk import DocumentChunk

__all__ = ["User", "Conversation", "Message", "Document", "DocumentChunk"]
