"""Retriever for Task 3.

Embeds a user question with the same model used to build the vector store
(all-MiniLM-L6-v2) and runs a similarity search against the FAISS index to
get the top-k most relevant complaint chunks.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import faiss

try:
    from .embedding import Embedder
except ImportError:  # running as a standalone script, not as part of the src package
    from embedding import Embedder


class Retriever:
    def __init__(self, index: faiss.IndexFlatIP, metadata_df: pd.DataFrame, embedder: Embedder):
        self.index = index
        self.metadata_df = metadata_df
        self.embedder = embedder

    def retrieve(self, question: str, k: int = 5) -> List[dict]:
        """Return the top-k most relevant chunks for `question`, each as a
        dict of metadata plus a `score` field (cosine similarity, higher is
        more relevant)."""
        query_vec = self.embedder.embed([question], show_progress=False).astype("float32")
        # embedder already L2-normalizes, but normalize again defensively in
        # case this is ever called with a non-normalizing embedder.
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, k)
        scores, indices = scores[0], indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:  # FAISS pads with -1 if k > ntotal
                continue
            row = self.metadata_df.iloc[idx].to_dict()
            row["score"] = float(score)
            results.append(row)
        return results
