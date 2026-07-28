"""Prompt template for the CrediTrust complaint RAG pipeline.

The template injects retrieved complaint chunks into a structured prompt
that tells the model to act as a financial analyst, answer only from the
provided context, and say so explicitly when the context is insufficient.
"""

from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_CONTEXT: str = (
    "You are a financial analyst assistant at CrediTrust Financial. "
    "Your job is to answer questions about customer complaints based solely "
    "on the complaint excerpts provided. Be concise, structured, and factual. "
    "If the provided excerpts do not contain enough information to answer the "
    "question, say so explicitly rather than speculating."
)


def build_prompt(chunks: List[Dict[str, Any]], question: str) -> str:
    """Build a RAG prompt from retrieved chunks and a user question.

    Parameters
    ----------
    chunks : list of dict
        Retrieved complaint chunks, each containing at minimum a
        ``chunk_text`` key and optionally metadata (product_category,
        company, etc.).
    question : str
        The user's plain-English question.

    Returns
    -------
    str
        A fully-formatted prompt ready to be sent to the generator.
    """
    context_parts: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        text = (chunk.get("chunk_text") or chunk.get("document") or "").strip()
        product = chunk.get("product_category", "")
        company = chunk.get("company", "")
        meta = " | ".join(filter(None, [product, company]))
        header = f"[Source {i}" + (f" — {meta}" if meta else "") + "]"
        context_parts.append(f"{header}\n{text}")

    context_block: str = "\n\n".join(context_parts)

    return (
        f"{SYSTEM_CONTEXT}\n\n"
        f"Customer complaint excerpts:\n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
