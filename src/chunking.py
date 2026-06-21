"""Text chunking utilities for Task 2.

Long complaint narratives are often ineffective when embedded as a single
vector — important details get diluted in one big embedding, and retrieval
granularity suffers. We split each narrative into overlapping chunks before
embedding.

Defaults (chunk_size=500 characters, chunk_overlap=50 characters) match the
specification of the pre-built vector store provided for Tasks 3-4, so the
two are directly comparable.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def build_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Create a RecursiveCharacterTextSplitter.

    RecursiveCharacterTextSplitter tries to split on paragraph/sentence
    boundaries first and only falls back to a hard character cut, which
    keeps chunks more semantically coherent than a naive fixed-width split.
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
    """Chunk every row's text into one or more overlapping chunks.

    Returns a new dataframe with one row per chunk, containing:
      - ``chunk_id``: f"{complaint_id}_{chunk_index}", unique per chunk
      - ``complaint_id``: copied from ``id_col``
      - ``chunk_index`` / ``total_chunks``: position within the source row
      - ``chunk_text``: the chunk's text
      - all columns listed in ``metadata_cols``, copied through unchanged
    """
    splitter = build_splitter(chunk_size, chunk_overlap)

    records = []
    for _, row in df.iterrows():
        text = row[text_col]
        if not isinstance(text, str) or not text.strip():
            continue

        chunks = splitter.split_text(text)
        total_chunks = len(chunks)
        for idx, chunk_text in enumerate(chunks):
            record = {
                "chunk_id": f"{row[id_col]}_{idx}",
                "complaint_id": row[id_col],
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "chunk_text": chunk_text,
            }
            for col in metadata_cols:
                record[col] = row[col]
            records.append(record)

    return pd.DataFrame.from_records(records)
