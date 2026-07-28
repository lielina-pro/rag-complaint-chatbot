"""Central configuration for the CrediTrust RAG pipeline.

All tuneable parameters used across src/ modules live here as a single
dataclass, replacing the scattered default-argument values that were
previously duplicated across build_faiss_index.py, retriever.py,
generator.py, and app.py.

Usage
-----
>>> from src.config import RAGConfig
>>> cfg = RAGConfig()                          # all defaults
>>> cfg = RAGConfig(top_k=8, temperature=0.5)  # overrides
>>> cfg = RAGConfig.from_env()                 # reads HF_TOKEN from env/.env
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository root — used to build default absolute paths so the app can be
# launched from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RAGConfig:
    """All tuneable knobs for the RAG pipeline in one place.

    Attributes
    ----------
    index_path : str
        Path to the persisted FAISS index file.
    metadata_path : str
        Path to the companion metadata parquet (same row order as the index).
    embedding_model : str
        HuggingFace model ID for the sentence-transformer used to embed both
        documents (at index-build time) and queries (at retrieval time).
        Must match the model used when the index was originally built.
    generator_model : str
        HuggingFace model ID served via the Inference API for answer generation.
        Use a standard instruct model (not a reasoning model) to avoid the
        hidden chain-of-thought token budget problem.
    top_k : int
        Number of complaint chunks to retrieve per question.
    chunk_size : int
        Character limit per chunk (used in Task 2 chunking; stored here for
        documentation / reproducibility).
    chunk_overlap : int
        Overlap between adjacent chunks in characters.
    max_new_tokens : int
        Maximum tokens the generator may produce per answer.
    temperature : float
        Sampling temperature for the generator (0.0 = deterministic).
    index_batch_size : int
        Rows processed per batch when streaming the embeddings parquet into
        the FAISS index.  Lower this if you hit memory errors.
    hf_token : str
        Hugging Face access token.  Read from the HF_TOKEN environment
        variable (or .env file) by default; never hard-code a real token here.
    """

    # ── Paths ────────────────────────────────────────────────────────────────
    index_path: str = field(
        default_factory=lambda: str(_REPO_ROOT / "vector_store" / "full_dataset.faiss")
    )
    metadata_path: str = field(
        default_factory=lambda: str(_REPO_ROOT / "vector_store" / "full_dataset_metadata.parquet")
    )

    # ── Models ───────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generator_model: str = "deepseek-ai/DeepSeek-V3-0324"

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k: int = 5

    # ── Chunking (Task 2 parameters — stored for reproducibility) ────────────
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── Generation ───────────────────────────────────────────────────────────
    max_new_tokens: int = 512
    temperature: float = 0.7

    # ── Index building ───────────────────────────────────────────────────────
    index_batch_size: int = 50_000

    # ── Auth ─────────────────────────────────────────────────────────────────
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))

    # ── Derived helpers ──────────────────────────────────────────────────────
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create a config whose hf_token is explicitly pulled from the
        environment (useful in scripts that don't call load_dotenv first)."""
        from dotenv import load_dotenv
        load_dotenv(override=True)
        return cls()

    def validate(self) -> None:
        """Raise ValueError for any obviously wrong config values."""
        if self.top_k < 1 or self.top_k > 20:
            raise ValueError(f"top_k must be between 1 and 20, got {self.top_k}")
        if self.chunk_size < 50:
            raise ValueError(f"chunk_size too small: {self.chunk_size}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.max_new_tokens < 64:
            raise ValueError(f"max_new_tokens too small: {self.max_new_tokens}")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature must be in [0.0, 2.0], got {self.temperature}")
