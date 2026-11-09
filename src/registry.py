"""
Model registry — singleton loader for heavyweight NLP models.

Loading spaCy or KeyBERT fresh on every request is slow and wastes memory.
This module loads each model exactly once, lazily (only when first needed),
and reuses that instance for every subsequent request. Import from here
rather than instantiating these models anywhere else in the codebase.
"""

import logging
import threading

logger = logging.getLogger("insightforge")

_lock = threading.Lock()
_spacy_model = None
_keybert_model = None
_embedding_model = None


def get_spacy_model():
    """Returns the shared spaCy English pipeline, loading it on first use."""
    global _spacy_model
    if _spacy_model is None:
        with _lock:
            if _spacy_model is None:  # re-check inside the lock
                try:
                    import spacy
                    logger.info("Loading spaCy model 'en_core_web_sm' (first use)...")
                    _spacy_model = spacy.load("en_core_web_sm")
                    logger.info("spaCy model loaded.")
                except OSError as e:
                    logger.error(
                        "spaCy model 'en_core_web_sm' is not installed. "
                        "Run: python -m spacy download en_core_web_sm"
                    )
                    raise RuntimeError(
                        "The spaCy language model isn't installed on this server."
                    ) from e
    return _spacy_model


def get_keybert_model():
    """Returns the shared KeyBERT model, loading it on first use."""
    global _keybert_model
    if _keybert_model is None:
        with _lock:
            if _keybert_model is None:
                from keybert import KeyBERT
                logger.info("Loading KeyBERT model (first use)...")
                _keybert_model = KeyBERT()
                logger.info("KeyBERT model loaded.")
    return _keybert_model


def warm_up():
    """
    Optionally called at app startup to pre-load models instead of paying
    the load cost on the first user request. Not required — models still
    lazy-load on demand if this is never called.
    """
    try:
        get_spacy_model()
    except Exception:
        logger.warning("Could not warm up spaCy model at startup.")
    try:
        get_keybert_model()
    except Exception:
        logger.warning("Could not warm up KeyBERT model at startup.")


def get_embedding_model():
    """Returns the shared sentence-embedding model, loading it on first use."""
    global _embedding_model
    if _embedding_model is None:
        with _lock:
            if _embedding_model is None:
                from sentence_transformers import SentenceTransformer
                from src.embeddings import get_model_name

                name = get_model_name()
                logger.info("Loading embedding model '%s' (first use, may download ~90 MB)...", name)
                _embedding_model = SentenceTransformer(name)
                logger.info("Embedding model loaded.")
    return _embedding_model