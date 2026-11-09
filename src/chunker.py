"""
Text chunker: splits a long document into overlapping pieces for retrieval.

Why chunks? A language model can only read a limited amount of text per
request. Instead of sending a whole document, we split it into small pieces,
find the few pieces most relevant to the user's question, and send only those.

How it splits (in order of preference):
  1. at line / paragraph breaks
  2. at sentence ends, if a single paragraph is too long
  3. at word boundaries, if a single sentence is too long
  4. hard cut, only for a "word" longer than the chunk size (rare)

Chunks overlap a little, so a sentence that sits on a boundary still appears
whole in at least one chunk.

Guarantee: no chunk is longer than chunk_size + overlap + 1 characters, and
every word of the original text appears in at least one chunk.
"""

import re

DEFAULT_CHUNK_SIZE = 1000   # characters
DEFAULT_OVERLAP = 200       # characters

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list:
    """Returns a list of chunk strings (empty list for empty text)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    units = _split_into_units(text, chunk_size)

    chunks = []
    current = ""
    for unit_text, separator in units:
        if current and len(current) + len(separator) + len(unit_text) > chunk_size:
            chunks.append(current)
            tail = _overlap_tail(current, overlap)
            current = f"{tail}{separator}{unit_text}" if tail else unit_text
        else:
            current = f"{current}{separator}{unit_text}" if current else unit_text

    if current:
        chunks.append(current)
    return chunks


def _split_into_units(text: str, max_len: int) -> list:
    """
    Breaks text into small pieces, each at most max_len characters.
    Returns [(piece, separator_to_put_before_it)], where the separator is a
    newline between lines and a space between sentences of the same line.
    """
    units = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        pieces = [line] if len(line) <= max_len else _split_long_line(line, max_len)
        for i, piece in enumerate(pieces):
            units.append((piece, "\n" if i == 0 else " "))
    return units


def _split_long_line(line: str, max_len: int) -> list:
    pieces = []
    for sentence in _SENTENCE_END.split(line):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_len:
            pieces.append(sentence)
        else:
            pieces.extend(_split_by_words(sentence, max_len))
    return pieces


def _split_by_words(sentence: str, max_len: int) -> list:
    pieces = []
    current = ""
    for word in sentence.split():
        # a single "word" longer than max_len (e.g. a long URL): cut it up
        while len(word) > max_len:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_len])
            word = word[max_len:]
        if current and len(current) + 1 + len(word) > max_len:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(chunk: str, overlap: int) -> str:
    """The last ~`overlap` characters of a chunk, starting at a word boundary."""
    if overlap <= 0 or len(chunk) <= overlap:
        return ""
    tail = chunk[-overlap:]
    # Don't start in the middle of a word: skip to just after the first space.
    cut = min((i for i in (tail.find(" "), tail.find("\n")) if i != -1), default=-1)
    if cut != -1:
        tail = tail[cut + 1:]
    return tail.strip()
