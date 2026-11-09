"""
LLM wrapper — the only module that talks to Groq. Every failure mode
(missing key, auth failure, rate limit, network error, model error) is
caught here and converted into an AIServiceError with a clean, user-safe
message, so calling code never has to deal with raw Groq/LangChain
exceptions or leak a traceback to the frontend.
"""

import logging

from src.config import get_config

logger = logging.getLogger("insightforge")


class AIServiceError(Exception):
    """Raised for any AI-service failure. Message is safe to show the user."""
    pass


_llm_instance = None


def _get_llm():
    """Lazily builds the ChatGroq client once and reuses it."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    config = get_config()
    if not config.GROQ_API_KEY:
        raise AIServiceError(
            "The AI service isn't configured yet. Please set GROQ_API_KEY in the server's .env file."
        )

    try:
        from langchain_groq import ChatGroq
        _llm_instance = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_MODEL,
            temperature=0.4,
            max_tokens=3000,
            timeout=30,
        )
        return _llm_instance
    except Exception as e:
        logger.exception("Failed to initialize Groq client")
        raise AIServiceError("Could not initialize the AI service. Please try again later.") from e


def chat_completion(messages: list[dict]) -> str:
    """
    Sends a list of {"role": "system"|"user"|"assistant", "content": str}
    messages to Groq and returns the assistant's reply text.

    Any failure (bad key, rate limit, timeout, network error, empty
    response) is converted to AIServiceError — never a raw exception.
    """
    llm = _get_llm()

    try:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        lc_messages = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        response = llm.invoke(lc_messages)
        text = (response.content or "").strip()

        if not text:
            raise AIServiceError("The AI service returned an empty response. Please try again.")

        return text

    except AIServiceError:
        raise
    except Exception as e:
        logger.exception("Groq chat completion failed")
        error_str = str(e).lower()

        if "authentication" in error_str or "api key" in error_str or "401" in error_str:
            message = "The AI service rejected the configured API key. Please check GROQ_API_KEY."
        elif "rate limit" in error_str or "429" in error_str:
            message = "The AI service is currently rate-limited. Please wait a moment and try again."
        elif "timeout" in error_str or "timed out" in error_str:
            message = "The AI service took too long to respond. Please try again."
        elif "connection" in error_str or "network" in error_str:
            message = "Could not reach the AI service. Please check your internet connection."
        else:
            message = "The AI service encountered an error. Please try again."

        raise AIServiceError(message) from e