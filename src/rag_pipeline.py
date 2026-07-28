"""RAG pipeline orchestrator — ties retriever, prompt, and generator together.

Usage
-----
>>> from src.rag_pipeline import RAGPipeline
>>> result = pipeline.answer("Why are people unhappy with Credit Cards?", k=5)
>>> print(result["answer"])
>>> for source in result["sources"]:
...     print(source["product_category"], source["score"])
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List

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
        Default number of chunks to retrieve per question.
        Can be overridden per-call in ``answer()`` and ``answer_stream()``.
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
        """Build a complete pipeline from a RAGConfig.

        This is the recommended way to instantiate the pipeline in production
        scripts and the Gradio app — one import, one call, everything wired.

        Example
        -------
        >>> from src.config import RAGConfig
        >>> from src.rag_pipeline import RAGPipeline
        >>> pipeline = RAGPipeline.from_config(RAGConfig.from_env())
        >>> result = pipeline.answer("What billing issues are most common?")
        """
        from embedding import Embedder
        from generator import Generator as LLMGenerator
        from vector_index import load_index_and_metadata

        index, metadata_df = load_index_and_metadata(
            config.index_path, config.metadata_path
        )
        embedder = Embedder(config.embedding_model)
        retriever = Retriever(index, metadata_df, embedder)
        generator = LLMGenerator.from_config(config)
        return cls(retriever, generator)

    def answer(
        self,
        question: str,
        k: int = 5,
    ) -> Dict[str, Any]:
        """Retrieve relevant chunks and generate a sourced answer (blocking).

        Parameters
        ----------
        question : str
            Plain-English question from the user.
        k : int
            Number of complaint chunks to retrieve.

        Returns
        -------
        dict with keys:
            ``answer``  : str — the generated answer text
            ``sources`` : list of dict — retrieved chunks with metadata + score
            ``prompt``  : str — the full prompt sent to the generator
        """
        chunks: List[Dict[str, Any]] = self.retriever.retrieve(question, k=k)
        prompt: str = build_prompt(chunks, question)
        answer_text: str = self.generator.generate(prompt)
        return {"answer": answer_text, "sources": chunks, "prompt": prompt}

    def answer_stream(
        self,
        question: str,
        k: int = 5,
    ) -> Generator[Any, None, None]:
        """Retrieve chunks then stream the answer token-by-token.

        Yields
        ------
        dict
            First yield: ``{"sources": [...]}`` — the retrieved chunks,
            emitted before generation starts so the UI can show sources
            immediately.
        str
            Subsequent yields: individual tokens from the generator stream.
        """
        chunks: List[Dict[str, Any]] = self.retriever.retrieve(question, k=k)
        prompt: str = build_prompt(chunks, question)
        yield {"sources": chunks}
        yield from self.generator.generate_stream(prompt)
