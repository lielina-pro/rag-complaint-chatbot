"""The full RAG pipeline for Task 3: retrieve -> build prompt -> generate.

This is the module Task 4's app.py will import directly.
"""

from __future__ import annotations

try:
    from .prompt_template import build_prompt
    from .retriever import Retriever
    from .generator import Generator
except ImportError:  # running as a standalone script, not as part of the src package
    from prompt_template import build_prompt
    from retriever import Retriever
    from generator import Generator


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator, k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.k = k

    def answer(self, question: str, k: int | None = None) -> dict:
        """Run the full pipeline for a single question.

        Returns a dict with:
          - "answer": the LLM's generated response (str)
          - "sources": the retrieved chunks used as context (list of dicts,
            each including the chunk text, its metadata, and similarity score)
          - "prompt": the exact prompt sent to the LLM (useful for debugging
            and for the evaluation table in the report)
        """
        k = k if k is not None else self.k
        chunks = self.retriever.retrieve(question, k=k)
        prompt = build_prompt(chunks, question)
        answer = self.generator.generate(prompt)
        return {"answer": answer, "sources": chunks, "prompt": prompt}
