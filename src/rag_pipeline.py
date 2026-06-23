"""The full RAG pipeline for Task 3: retrieve -> build prompt -> generate.

This is the module Task 4's app.py imports directly.
"""

from __future__ import annotations

from typing import Iterator, Tuple, List

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

    def answer_stream(self, question: str, k: int | None = None) -> Iterator[Tuple[str, List[dict]]]:
        """Streaming variant for the Task 4 UI.

        Retrieval happens once up front (it's fast and doesn't change), then
        the answer is yielded as a growing string each time a new token
        arrives. Each yield is (answer_so_far, sources) -- sources stay
        constant across the loop but are included every time so the caller
        (e.g. a Gradio generator callback) can update both outputs from a
        single iterator without extra bookkeeping.
        """
        k = k if k is not None else self.k
        chunks = self.retriever.retrieve(question, k=k)
        prompt = build_prompt(chunks, question)

        partial = ""
        for token in self.generator.generate_stream(prompt):
            partial += token
            yield partial, chunks

        if partial == "":
            # Some providers occasionally return zero delta chunks for a very
            # short answer -- fall back to a non-streaming call so the user
            # still gets a response instead of a blank box.
            partial = self.generator.generate(prompt)
            yield partial, chunks
