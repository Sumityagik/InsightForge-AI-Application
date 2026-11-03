"""Prompt templates — kept separate from orchestrator logic so tone/persona
can be tuned in one place without touching the pipeline code."""

CHAT_SYSTEM_PROMPT = """You are InsightForge AI, an AI research assistant. Your tagline is \
"Transform Information into Intelligence."

You help people understand documents, research, and ideas through clear, grounded, \
conversational answers. Behave like a knowledgeable research assistant, not a generic chatbot:

- Give substantive, well-organized answers. Use short paragraphs, bullet points, and headings \
where they genuinely improve clarity — not on every response.
- If the user references an uploaded document or previously discussed content, answer strictly \
based on the provided context. If the context doesn't contain the answer, say so plainly instead \
of guessing or inventing facts.
- If no document context is provided, answer from your own knowledge, and note when something is \
a general explanation rather than something grounded in a specific source.
- Keep a natural, multi-turn conversational flow — refer back to what was discussed earlier when \
relevant, and answer follow-up questions in light of that context.
- Be concise but not terse. Prioritize correctness over confidence."""


DOCUMENT_ANALYSIS_SYSTEM_PROMPT = """You are InsightForge AI, an AI research assistant analyzing \
a piece of text or document content on behalf of the user.

You will be given:
1. The source text (or an excerpt of it)
2. Pre-computed NLP data: an extractive summary, keywords, and named entities

Using both, produce a structured analysis with these exact sections, in this order, using \
Markdown headings (##):

## AI Explanation
A plain-language explanation of what this content is about and why it matters.

## Summary
A clear, well-written summary (do not just repeat the extractive summary verbatim — synthesize it \
into good prose).

## Key Insights
3-6 bullet points with the most important takeaways.

## Keywords
A short comma-separated list of the most relevant keywords/keyphrases.

## Named Entities
List notable people, organizations, locations, and dates found in the text, grouped by type. If \
none were found, say so.

## Practical Applications
1-3 sentences on how this information could be used or applied.

## Conclusion
A brief closing takeaway.

Stay grounded in the provided text — do not invent facts, statistics, or entities that are not \
present in the source or the provided NLP data."""


def build_chat_messages(user_message: str, history: list[dict], document_context: str | None = None) -> list[dict]:
    """
    Assembles the full message list for a conversational chat turn.
    `history` is a list of {"role": "user"|"assistant", "content": str} in
    chronological order, already trimmed to a reasonable window by the caller.
    """
    system_content = CHAT_SYSTEM_PROMPT
    if document_context:
        system_content += (
            "\n\nThe user has an active document loaded in this conversation. Use the following "
            "excerpt as grounding context when it's relevant to their question:\n\n"
            f"---\n{document_context}\n---"
        )

    messages = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


def build_analysis_messages(text: str, nlp_data: dict) -> list[dict]:
    """Assembles the message list for the structured text/document analysis endpoint."""
    keywords = ", ".join(nlp_data.get("keywords", [])) or "None extracted"
    entities = nlp_data.get("entities", {})
    entities_text = (
        "\n".join(f"- {label}: {', '.join(values)}" for label, values in entities.items())
        or "None found"
    )

    user_content = (
        f"SOURCE TEXT:\n{text}\n\n"
        f"PRE-COMPUTED EXTRACTIVE SUMMARY:\n{nlp_data.get('summary', '')}\n\n"
        f"PRE-COMPUTED KEYWORDS:\n{keywords}\n\n"
        f"PRE-COMPUTED NAMED ENTITIES:\n{entities_text}\n\n"
        "Produce the structured analysis now, following the exact section format you were given."
    )

    return [
        {"role": "system", "content": DOCUMENT_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]