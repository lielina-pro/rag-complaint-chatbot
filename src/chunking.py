"""Text chunking utilities for Task 2.

Splits long complaint narratives into overlapping chunks before embedding.
The default parameters (500 chars / 50 overlap) match the pre-built
full-dataset vector store so the two are directly comparable.
"""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50


def build_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Create a RecursiveCharacterTextSplitter.

    Tries paragraph/sentence boundaries first, falls back to hard
    character cuts — keeping chunks more semantically coherent than a
    naive fixed-width split.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_dataframe(
    df: pd.DataFrame,
    text_col: str,
    id_col: str,
    metadata_cols: Iterable[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> pd.DataFrame:
    """Chunk every row's narrative into overlapping sub-texts.

    Parameters
    ----------
    df : pd.DataFrame
        Source complaint dataframe (one row per complaint).
    text_col : str
        Column containing the narrative text to chunk.
    id_col : str
        Column containing a unique complaint identifier.
    metadata_cols : iterable of str
        Columns to copy through from the source row into each chunk record.
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Characters of overlap between adjacent chunks.

    Returns
    -------
    pd.DataFrame
        One row per chunk, containing: chunk_id, complaint_id, chunk_index,
        total_chunks, chunk_text, and all requested metadata columns.
    """
    splitter = build_splitter(chunk_size, chunk_overlap)
    meta_cols: List[str] = list(metadata_cols)
    records: List[dict] = []

    for _, row in df.iterrows():
        text: str = row[text_col]
        if not isinstance(text, str) or not text.strip():
            continue
        chunks: List[str] = splitter.split_text(text)
        total: int = len(chunks)
        for idx, chunk_text in enumerate(chunks):
            record: dict = {
                "chunk_id": f"{row[id_col]}_{idx}",
                "complaint_id": row[id_col],
                "chunk_index": idx,
                "total_chunks": total,
                "chunk_text": chunk_text,
            }
            for col in meta_cols:
                record[col] = row[col]
            records.append(record)

    return pd.DataFrame.from_records(records)
