"""Retriever — embeds a question and searches the FAISS index.

The retriever is the first active step in a RAG query: it converts the
user's plain-English question into a vector, runs an approximate (flat,
exact) nearest-neighbour search over the 1.37M indexed complaint chunks,
and returns the top-k most relevant records with their metadata and scores.
"""

from __future__ import annotations

from typing import Any, Dict, List

import faiss
import numpy as np
import pandas as pd

from embedding import Embedder


class Retriever:
    """Embed a question and return the top-k most similar complaint chunks.

    Parameters
    ----------
    index : faiss.IndexFlatIP
        A loaded FAISS index (inner-product, L2-normalised vectors).
    metadata_df : pd.DataFrame
        Parallel metadata dataframe — row i corresponds to vector i in the
        index.  Must contain at least a ``chunk_text`` column; any additional
        metadata columns (product_category, company, state, …) are passed
        through to the returned results.
    embedder : Embedder
        Embedding model.  Must be the *same* model used to build the index.
    """

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        metadata_df: pd.DataFrame,
        embedder: Embedder,
    ) -> None:
        self.index: faiss.IndexFlatIP = index
        self.metadata_df: pd.DataFrame = metadata_df
        self.embedder: Embedder = embedder

    def embed_query(self, question: str) -> np.ndarray:
        """Embed a single question and L2-normalise it.

        Returns a (1, dim) float32 array ready for ``index.search``.
        """
        vec = self.embedder.embed([question], show_progress=False)
        vec = np.ascontiguousarray(vec, dtype="float32")
        faiss.normalize_L2(vec)
        return vec

    def retrieve(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return the top-k most relevant chunks for *question*.

        Parameters
        ----------
        question : str
            The user's plain-English question.
        k : int
            Number of chunks to retrieve.

        Returns
        -------
        list of dict
            Each dict contains all metadata columns from ``metadata_df`` plus
            a ``score`` key (cosine similarity, higher = more relevant).
        """
        query_vec = self.embed_query(question)
        scores, indices = self.index.search(query_vec, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.metadata_df.iloc[idx].to_dict()
            row["score"] = float(score)
            results.append(row)
        return results
