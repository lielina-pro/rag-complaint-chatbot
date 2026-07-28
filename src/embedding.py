"""Embedding model wrapper for the RAG pipeline.

Wraps sentence-transformers so the rest of the pipeline never imports
SentenceTransformer directly — making it easy to swap models or inject
a fake embedder in tests without mocking the library.
"""

from __future__ import annotations

from typing import List

import numpy as np

DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Thin wrapper around a SentenceTransformer model.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.  Defaults to all-MiniLM-L6-v2, which produces
        384-dim embeddings, runs on CPU, and matches the pre-built vector store.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model_name: str = model_name
        self.model: SentenceTransformer = SentenceTransformer(model_name)

    def embed(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a list of strings and return an (n, dim) float32 array.

        Embeddings are L2-normalised so that inner product == cosine similarity,
        matching the normalisation applied when the FAISS index was built.
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
