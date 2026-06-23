"""Prompt engineering for Task 3.

A single, robust template that instructs the LLM to act as a financial
analyst, rely only on the retrieved context, and explicitly say so when the
context doesn't contain the answer (this is what keeps the chatbot from
hallucinating complaint details that were never actually reported).
"""

from __future__ import annotations

from typing import List

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""


def build_context_block(chunks: List[dict], text_col: str = "chunk_text") -> str:
    """Format retrieved chunks into a numbered context block, so the model
    (and a human reviewer) can trace each part of the answer back to a
    specific source chunk."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get(text_col) or chunk.get("text") or ""
        product = chunk.get("product_category", "Unknown product")
        lines.append(f"[Source {i} | {product}]: {text}")
    return "\n\n".join(lines)


def build_prompt(chunks: List[dict], question: str, text_col: str = "chunk_text") -> str:
    context = build_context_block(chunks, text_col=text_col)
    return PROMPT_TEMPLATE.format(context=context, question=question)
