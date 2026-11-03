"""
Local NLP tools — fast, deterministic, no LLM call required.

These run before the LLM call in the pipeline (see orchestrator.py) and
their output is both: (a) returned directly to the frontend as structured
data, and (b) fed into the LLM prompt as grounding context so the model's
narrative explanation stays consistent with what was actually extracted.
"""

import logging
import re
from collections import Counter

from src.registry import get_spacy_model, get_keybert_model

logger = logging.getLogger("insightforge")

# Entity labels we surface to the user, mapped to friendly group names.
ENTITY_GROUPS = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "DATE": "Date",
    "TIME": "Date",
    "MONEY": "Money",
    "PERCENT": "Percent",
    "PRODUCT": "Product",
    "EVENT": "Event",
}


def extractive_summary(text: str, max_sentences: int = 4) -> str:
    """
    Lightweight frequency-based extractive summarizer: scores each sentence
    by the frequency of its non-stopword tokens and keeps the top N,
    restoring original order. Deliberately simple (no heavy transformer
    model) — the LLM produces the polished narrative summary; this exists
    to ground that summary in the source text and to give a fast fallback
    if the LLM call fails.
    """
    text = (text or "").strip()
    if not text:
        return ""

    nlp = get_spacy_model()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    word_freq = Counter(
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    )
    if not word_freq:
        return " ".join(sentences[:max_sentences])

    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] /= max_freq

    sentence_scores = []
    for sent in doc.sents:
        clean = sent.text.strip()
        if not clean:
            continue
        score = sum(word_freq.get(t.lemma_.lower(), 0) for t in sent)
        sentence_scores.append((clean, score))

    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    top_texts = {s for s, _ in top_sentences}
    ordered = [s for s in sentences if s in top_texts]
    return " ".join(ordered)


def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    """Returns the top N keywords/keyphrases using KeyBERT."""
    text = (text or "").strip()
    if not text:
        return []

    try:
        model = get_keybert_model()
        results = model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
        )
        return [phrase for phrase, score in results]
    except Exception:
        logger.exception("KeyBERT keyword extraction failed; falling back to simple extraction")
        return _fallback_keywords(text, top_n)


def _fallback_keywords(text: str, top_n: int) -> list[str]:
    """Simple frequency-based fallback if KeyBERT is unavailable or errors."""
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    stopwords = {"this", "that", "with", "from", "have", "which", "their", "about", "there"}
    filtered = [w for w in words if w not in stopwords]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Returns named entities grouped by friendly category, e.g.:
    {"Person": ["Marie Curie"], "Organization": ["Anthropic"], ...}
    Only entity types in ENTITY_GROUPS are surfaced; duplicates removed
    while preserving first-seen order.
    """
    text = (text or "").strip()
    if not text:
        return {}

    nlp = get_spacy_model()
    doc = nlp(text)

    grouped: dict[str, list[str]] = {}
    for ent in doc.ents:
        label = ENTITY_GROUPS.get(ent.label_)
        if not label:
            continue
        grouped.setdefault(label, [])
        if ent.text not in grouped[label]:
            grouped[label].append(ent.text)

    return grouped


def analyze_text(text: str) -> dict:
    """
    Runs the full local NLP pipeline (summary + keywords + entities) and
    returns a single structured dict. This is the grounding data passed
    to the LLM for the final polished response.
    """
    return {
        "summary": extractive_summary(text),
        "keywords": extract_keywords(text),
        "entities": extract_entities(text),
    }