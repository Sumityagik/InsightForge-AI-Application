"""
Vector store: saves vectors in a FAISS index file and finds the nearest ones.

One small index file per document. Keeping each document in its own index
means a search can never return text from somebody else's document.

Index type: IndexFlatIP, an exact search using inner product. Because our
vectors are length-normalised, inner product = cosine similarity. Exact search
is perfectly fast for the few hundred chunks a document produces.

The file is read and written through plain Python file I/O (not FAISS's own
file functions). That avoids two Windows problems: FAISS failing on paths with
non-ASCII characters, and files staying locked after being read.
"""

import logging
import os

import numpy as np

logger = logging.getLogger("insightforge")


class VectorStoreError(Exception):
    """Index could not be written or read."""


def _faiss():
    try:
        import faiss
        return faiss
    except ImportError as e:
        raise VectorStoreError("The 'faiss-cpu' package is not installed on this server.") from e


def save_index(vectors: np.ndarray, path: str) -> None:
    """Builds an index from `vectors` (n x dim) and writes it to `path` atomically."""
    faiss = _faiss()
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise VectorStoreError("There are no vectors to index.")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    data = faiss.serialize_index(index).tobytes()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)  # a half-written file never replaces a good one


def load_index(path: str):
    faiss = _faiss()
    try:
        with open(path, "rb") as f:
            raw = f.read()
        return faiss.deserialize_index(np.frombuffer(raw, dtype="uint8"))
    except Exception as e:
        raise VectorStoreError("The search index could not be read.") from e


def search(path: str, query_vector: np.ndarray, k: int) -> list:
    """
    Returns up to k (position, score) pairs, best first. `position` is the
    chunk number inside the document; `score` is cosine similarity.
    """
    index = load_index(path)
    if index.ntotal == 0:
        return []

    query = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
    if query.shape[1] != index.d:
        raise VectorStoreError("The search index was built with a different embedding size. Re-index the document.")

    scores, positions = index.search(query, min(k, index.ntotal))
    return [(int(p), float(s)) for p, s in zip(positions[0], scores[0]) if p != -1]


def index_exists(path: str) -> bool:
    return os.path.isfile(path)


def delete_index(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        logger.warning("Could not delete search index file %s", path)
