import numpy as np
import pandas as pd

from src.retriever import Retriever
from src.vector_index import build_faiss_index


class FakeEmbedder:
    """Stand-in for src.embedding.Embedder that doesn't need to download or
    run a real model -- just returns a fixed, pre-normalized vector so tests
    can control exactly which chunk should be "most similar"."""

    def __init__(self, vector: np.ndarray):
        self.vector = vector.astype("float32")

    def embed(self, texts, batch_size=64, show_progress=True):
        return np.vstack([self.vector for _ in texts])


def _make_index_and_metadata(n=10, dim=384, seed=0):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, dim)).astype("float32")
    df = pd.DataFrame({
        "complaint_id": range(n),
        "chunk_text": [f"complaint text {i}" for i in range(n)],
        "product_category": ["Credit Card"] * n,
        "embedding": list(embeddings),
    })
    index = build_faiss_index(df)
    metadata_df = df.drop(columns=["embedding"])
    return index, metadata_df, embeddings


def test_retrieve_returns_k_results_with_expected_fields():
    index, metadata_df, embeddings = _make_index_and_metadata(n=10)
    embedder = FakeEmbedder(embeddings[3])  # query vector == chunk 3's embedding
    retriever = Retriever(index, metadata_df, embedder)

    results = retriever.retrieve("any question", k=3)
    assert len(results) == 3
    assert all("score" in r and "chunk_text" in r and "complaint_id" in r for r in results)


def test_retrieve_finds_exact_match_as_top_result():
    index, metadata_df, embeddings = _make_index_and_metadata(n=10)
    embedder = FakeEmbedder(embeddings[7])
    retriever = Retriever(index, metadata_df, embedder)

    results = retriever.retrieve("any question", k=1)
    assert results[0]["complaint_id"] == 7
    assert results[0]["score"] > 0.999


def test_retrieve_results_sorted_by_descending_score():
    index, metadata_df, embeddings = _make_index_and_metadata(n=10)
    embedder = FakeEmbedder(embeddings[0])
    retriever = Retriever(index, metadata_df, embedder)

    results = retriever.retrieve("any question", k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_k_larger_than_index_does_not_crash():
    index, metadata_df, embeddings = _make_index_and_metadata(n=5)
    embedder = FakeEmbedder(embeddings[0])
    retriever = Retriever(index, metadata_df, embedder)

    results = retriever.retrieve("any question", k=100)
    assert len(results) == 5  # capped at total number of vectors
