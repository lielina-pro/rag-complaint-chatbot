"""Vector store persistence for Task 2.

Builds a persisted ChromaDB collection from pre-computed chunk embeddings,
storing per-chunk metadata (complaint_id, product_category, etc.) so every
retrieved chunk can be traced back to its source complaint.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def build_chroma_store(
    persist_dir: str,
    collection_name: str,
    chunk_ids: List[str],
    embeddings: np.ndarray,
    documents: List[str],
    metadatas: List[dict],
    batch_size: int = 500,
    reset_if_exists: bool = True,
):
    """Create (or overwrite) a persisted Chroma collection and add all chunks.

    Inserted in batches because Chroma/sqlite has a limit on parameters per
    single ``add`` call, which large collections (10K+ chunks) can exceed.
    """
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)

    if reset_if_exists:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass  # collection didn't exist yet — fine

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    n = len(chunk_ids)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        collection.add(
            ids=chunk_ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return collection


def metadatas_from_dataframe(df: pd.DataFrame, columns: Iterable[str]) -> List[dict]:
    """Build Chroma-compatible metadata dicts (no NaN/None — Chroma rejects
    them) from the given columns of a chunk-level dataframe."""
    cols = list(columns)
    records = df[cols].copy()
    for col in cols:
        if records[col].dtype == object:
            records[col] = records[col].fillna("")
    return records.to_dict(orient="records")
