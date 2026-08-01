"""RAG pipeline orchestrator — ties retriever, prompt, and generator together."""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

from generator import Generator as LLMGenerator
from prompt_template import build_prompt
from retriever import Retriever


class RAGPipeline:
    """End-to-end RAG pipeline: retrieve → prompt → generate.

    Parameters
    ----------
    retriever : Retriever
        Embedding + FAISS search component.
    generator : LLMGenerator
        HF Inference API generation component.
    k : int
        Default number of chunks to retrieve. Used when ``answer()`` or
        ``answer_stream()`` are called without an explicit ``k`` argument.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: LLMGenerator,
        k: int = 5,
    ) -> None:
        self.retriever: Retriever = retriever
        self.generator: LLMGenerator = generator
        self.k: int = k

    @classmethod
    def from_config(cls, config: "RAGConfig") -> "RAGPipeline":  # type: ignore[name-defined]
        """Build a complete pipeline from a RAGConfig."""
        from embedding import Embedder
        from generator import Generator as LLMGenerator
        from vector_index import load_index_and_metadata

        index, metadata_df = load_index_and_metadata(
            config.index_path, config.metadata_path
        )
        embedder = Embedder(config.embedding_model)
        retriever = Retriever(index, metadata_df, embedder)
        generator = LLMGenerator.from_config(config)
        return cls(retriever, generator, k=config.top_k)

    def answer(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Retrieve relevant chunks and generate a sourced answer (blocking).

        Parameters
        ----------
        question : str
            Plain-English question from the user.
        k : int, optional
            Number of chunks to retrieve. Defaults to ``self.k``.

        Returns
        -------
        dict
            Keys: ``answer`` (str), ``sources`` (list of dict), ``prompt`` (str).
        """
        k = k if k is not None else self.k
        chunks: List[Dict[str, Any]] = self.retriever.retrieve(question, k=k)
        prompt: str = build_prompt(chunks, question)
        answer_text: str = self.generator.generate(prompt)
        return {"answer": answer_text, "sources": chunks, "prompt": prompt}

    def answer_stream(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> Generator[Any, None, None]:
        """Retrieve chunks then stream the answer as growing (text, sources) tuples.

        Every yield is a tuple ``(partial_answer: str, sources: list)`` so
        callers can always unpack the same way regardless of position in the
        stream.  If the generator yields no tokens (some providers return
        nothing for very short answers), falls back to a blocking
        ``generate()`` call and yields one final tuple.

        Parameters
        ----------
        question : str
            Plain-English question from the user.
        k : int, optional
            Number of chunks to retrieve. Defaults to ``self.k``.
        """
        k = k if k is not None else self.k
        chunks: List[Dict[str, Any]] = self.retriever.retrieve(question, k=k)
        prompt: str = build_prompt(chunks, question)

        accumulated: str = ""
        yielded_any: bool = False

        for token in self.generator.generate_stream(prompt):
            accumulated += token
            yielded_any = True
            yield (accumulated, chunks)

        if not yielded_any:
            # Fallback: provider returned an empty stream — use blocking generate
            answer_text: str = self.generator.generate(prompt)
            yield (answer_text, chunks)
