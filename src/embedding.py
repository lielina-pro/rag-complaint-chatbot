"""Embedding model wrapper for Task 2.

We use sentence-transformers/all-MiniLM-L6-v2:
- Small and fast (~80MB, 384-dim embeddings), so encoding 10K-15K chunks runs
  in a few minutes on CPU — no GPU required.
- Strong general-purpose semantic similarity performance for its size,
  widely used as a baseline for retrieval tasks.
- Matches the embedding model used for the pre-built full-dataset vector
  store (Tasks 3-4), so this sample-based pipeline and the full pipeline are
  directly comparable and a retriever built here would generalize.
"""

from __future__ import annotations

from typing import List

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Thin wrapper around a SentenceTransformer model.

    Kept as a class (rather than calling SentenceTransformer directly
    everywhere) so tests can substitute a dummy embedder without needing to
    download model weights or have network access.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so cosine similarity == dot product
        )
