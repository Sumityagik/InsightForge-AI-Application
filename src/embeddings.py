"""
Embeddings: turns text into vectors (lists of numbers) so that pieces of text
with similar meaning end up close together. This is what makes semantic
search possible: "How did they train the model?" can match a chunk that says
"The network was fitted on 10,000 images" even though the words differ.

Two backends, chosen with the EMBEDDING_BACKEND environment variable:

  sentence-transformers  (default)  real semantic model, all-MiniLM-L6-v2.
                                    Downloaded once (about 90 MB), then cached.
  hash                              tiny word-overlap stand-in. No download, no
                                    GPU. Used by the automated tests, and as an
                                    emergency offline option. It matches on
                                    shared words only, NOT on meaning.

All vectors are float32 and length-normalised, so a dot product between two
of them equals their cosine similarity (1.0 = identical direction).
"""

import hashlib
import logging
import os
import re

import numpy as np

logger = logging.getLogger("insightforge")

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
HASH_DIMENSIONS = 384  # same size as MiniLM, so the two backends are interchangeable


class EmbeddingError(Exception):
    """The embedding step failed. Message is safe to log; callers fall back gracefully."""


def get_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)


def _backend() -> str:
    return os.getenv("EMBEDDING_BACKEND", "sentence-transformers").strip().lower()


def embed_texts(texts) -> np.ndarray:
    """Returns an (n, dim) float32 array of normalised vectors, one row per text."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, HASH_DIMENSIONS), dtype="float32")

    backend = _backend()
    if backend == "hash":
        return _hash_embed(texts)
    if backend in ("sentence-transformers", "st"):
        return _sentence_transformer_embed(texts)
    raise EmbeddingError(f"Unknown EMBEDDING_BACKEND '{backend}'. Use 'sentence-transformers' or 'hash'.")


def embed_query(text: str) -> np.ndarray:
    """Returns a (1, dim) array for a single question."""
    return embed_texts([text])


# ------------------------------------------------------------ real model
def _sentence_transformer_embed(texts: list) -> np.ndarray:
    try:
        from src.registry import get_embedding_model

        model = get_embedding_model()
        vectors = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype="float32")
    except Exception as e:
        logger.exception("Embedding model failed")
        raise EmbeddingError(
            "The embedding model could not be loaded or run. Check that sentence-transformers is "
            "installed and that the model can be downloaded (first use needs internet)."
        ) from e


# ------------------------------------------------------------ hash stand-in
_WORD = re.compile(r"[a-z0-9]+")


def _hash_embed(texts: list) -> np.ndarray:
    vectors = np.zeros((len(texts), HASH_DIMENSIONS), dtype="float32")
    for row, text in enumerate(texts):
        words = [w for w in _WORD.findall(text.lower()) if len(w) >= 3]
        features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for feature in features:
            digest = int(hashlib.md5(feature.encode("utf-8")).hexdigest(), 16)
            vectors[row, digest % HASH_DIMENSIONS] += 1.0

        norm = np.linalg.norm(vectors[row])
        if norm > 0:
            vectors[row] /= norm
    return vectors
