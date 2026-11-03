"""
Orchestrator — the single entry point services call to get an AI response.

Keeps the decision of "what kind of AI request is this and what pipeline
does it need" out of the routes/services layer. Right now it handles:
  - conversational chat replies (with history)
  - standalone text analysis (structured, NLP-grounded)

Document-grounded RAG answers will be added here in a later sprint without
changing this module's public interface.
"""

import logging

from src.llm import chat_completion, AIServiceError
from src.prompts import build_chat_messages, build_analysis_messages
from src.nlp_tools import analyze_text

logger = logging.getLogger("insightforge")

MAX_HISTORY_MESSAGES = 12  # keep the last N turns so prompts don't grow unbounded


def generate_chat_reply(user_message: str, history: list[dict] | None = None,
                         document_context: str | None = None) -> str:
    """
    Generates a conversational reply. `history` should be prior messages in
    the conversation as [{"role": "user"|"assistant", "content": str}, ...],
    oldest first. Raises AIServiceError on any failure — callers should
    catch this and turn it into a clean HTTP error, never a raw 500.
    """
    trimmed_history = (history or [])[-MAX_HISTORY_MESSAGES:]
    messages = build_chat_messages(user_message, trimmed_history, document_context)
    return chat_completion(messages)


def generate_text_analysis(text: str) -> dict:
    """
    Runs the full pipeline for the standalone "Text Analysis" feature:
    local NLP extraction (summary/keywords/entities) followed by an LLM
    call that turns that data into the structured, narrative response
    format described in the product spec.

    Returns a dict with both the raw NLP data and the final formatted
    narrative, so the frontend can render either.
    """
    nlp_data = analyze_text(text)
    messages = build_analysis_messages(text, nlp_data)
    narrative = chat_completion(messages)

    return {
        "narrative": narrative,
        "summary": nlp_data.get("summary", ""),
        "keywords": nlp_data.get("keywords", []),
        "entities": nlp_data.get("entities", {}),
    }