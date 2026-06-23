"""FAISS index utilities for Task 3.

The pre-built `complaint_embeddings.parquet` covers the full ~464K-complaint
dataset (~1.37M chunks). Rather than re-embedding anything, we load its
embeddings directly into a FAISS index for fast similarity search, and keep
the per-chunk metadata (text + complaint_id + product_category + ...) in a
parallel parquet file indexed by the same integer position used in FAISS.

We use a flat (exact, brute-force) index: at ~1.37M vectors x 384 dims,
a flat index still answers a query in well under a second and guarantees
exact nearest neighbors -- no approximate-search recall loss, and no extra
training step like IVF/HNSW would need.

IMPORTANT -- memory: `load_embeddings_parquet` below reads the whole file
into memory at once via pandas, which works fine for small/test files but
can exceed available RAM on the real ~1.37M-row file (pyarrow's table
representation, the pandas conversion, and the list-of-arrays expansion of
the embedding column each temporarily hold their own copy, multiplying peak
memory well past the ~2GB the final data actually needs). For the real file,
use `build_faiss_index_from_parquet` instead, which streams the file in
batches and adds each batch straight into the FAISS index, so peak memory
stays close to a single batch's size plus the index itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import faiss

EMBEDDING_DIM = 384

# "document"/"id" cover a ChromaDB-style export (id, document, embedding,
# metadata), which is what the real pre-built complaint_embeddings.parquet
# turned out to use -- a single `metadata` column holding a dict per row,
# rather than separate flat columns for product_category/company/etc.
TEXT_CANDIDATES = ("chunk_text", "text", "narrative_chunk", "narrative", "document")
EMBEDDING_CANDIDATES = ("embedding", "vector")
ID_CANDIDATES = ("chunk_id", "id")
METADATA_DICT_CANDIDATES = ("metadata", "meta")
# Only used as a fallback when there's no single metadata-dict column to
# expand (e.g. a flat export like Task 2's own sample store) -- when a dict
# column IS found, every key inside it is kept, whatever it's named.
METADATA_CANDIDATES = (
    "complaint_id", "product_category", "product", "issue", "sub_issue",
    "company", "state", "date_received", "chunk_index", "total_chunks",
)


def _find_column(names, *candidates: str) -> str:
    """Case/space-insensitive substring match against a list of column names.
    Fails loudly (rather than guessing) if nothing matches."""
    names = list(names)
    norm = {c: c.lower().replace(" ", "_") for c in names}
    for cand in candidates:
        cand_norm = cand.lower().replace(" ", "_")
        for c, c_norm in norm.items():
            if cand_norm in c_norm:
                return c
    raise ValueError(f"None of {candidates} found in columns: {names}")


def _find_exact_column(names, *candidates: str) -> str:
    """Like _find_column, but requires an exact (case/space-insensitive)
    match rather than substring containment. Needed for short, generic
    candidates like "id" -- a substring match would wrongly match
    "complaint_id" too (it ends in "id"), silently colliding two distinct
    columns into one."""
    names = list(names)
    norm = {c: c.lower().replace(" ", "_") for c in names}
    for cand in candidates:
        cand_norm = cand.lower().replace(" ", "_")
        for c, c_norm in norm.items():
            if cand_norm == c_norm:
                return c
    raise ValueError(f"None of {candidates} found (exact match) in columns: {names}")


def load_embeddings_parquet(path: str) -> pd.DataFrame:
    """Load a (small/test-sized) embeddings parquet fully into memory.

    Expects an `embedding` column containing a list/array per row (384-dim),
    plus metadata columns such as complaint_id, product_category, product,
    issue, sub_issue, company, state, date_received, chunk_index, and the
    chunk text itself (commonly named `chunk_text` or `text`).

    Do NOT use this on the full ~1.37M-row pre-built file -- see the module
    docstring. Use `build_faiss_index_from_parquet` for that instead.
    """
    df = pd.read_parquet(path)
    if "embedding" not in df.columns:
        raise ValueError(
            f"Expected an 'embedding' column in {path}. Found: {list(df.columns)}"
        )
    return df


def _find_text_column(df: pd.DataFrame) -> str:
    for candidate in TEXT_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not find a chunk text column (looked for chunk_text/text/"
        f"narrative_chunk/narrative). Found columns: {list(df.columns)}"
    )


def build_faiss_index(df: pd.DataFrame, normalize: bool = True) -> faiss.IndexFlatIP:
    """Build a flat FAISS index (inner-product) from the dataframe's
    `embedding` column. Embeddings are L2-normalized so inner product is
    equivalent to cosine similarity -- this MUST match how the query
    embedding is produced at retrieval time (see src/embedding.py, which
    also normalizes).

    Holds the full embeddings matrix in memory at once -- fine for small/
    test dataframes, but see `build_faiss_index_from_parquet` for the real
    ~1.37M-row file.
    """
    embeddings = np.vstack(df["embedding"].to_numpy()).astype("float32")

    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EMBEDDING_DIM}-dim embeddings, got shape {embeddings.shape}. "
            "Check that this parquet was built with all-MiniLM-L6-v2."
        )

    if normalize:
        faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    return index


def build_faiss_index_from_parquet(
    path: str,
    batch_rows: int = 50_000,
    normalize: bool = True,
    progress: bool = True,
) -> tuple[faiss.IndexFlatIP, pd.DataFrame]:
    """Memory-safer version of build_faiss_index for the full pre-built file.

    Streams the parquet file row-group-by-row-group (via pyarrow, never
    materializing the whole file as one pandas DataFrame) and adds each
    batch's embeddings to the FAISS index immediately. Metadata batches are
    still accumulated in memory and concatenated at the end, so this is a
    good fit for moderate file sizes or when you want the metadata back as
    a DataFrame to inspect (e.g. in a notebook) -- for the real ~1.37M-row
    file on a memory-constrained machine, prefer
    `stream_build_index_to_disk`, which never holds more than one batch of
    metadata at a time either (see its docstring for why this matters).

    Returns (index, metadata_df) -- metadata_df has no `embedding` column
    (it's redundant once the vectors are inside the FAISS index) and a
    `chunk_text` column regardless of what the source text column was named.
    """
    index = None
    metadata_batches = []
    processed = 0
    total_rows = pq.ParquetFile(path).metadata.num_rows

    for batch_df, batch_embeddings, text_col, embedding_col in _iter_parquet_batches(path, batch_rows):
        if index is None:
            _validate_embedding_dim(batch_embeddings)
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
        if normalize:
            faiss.normalize_L2(batch_embeddings)
        index.add(batch_embeddings)

        meta_batch = batch_df.drop(columns=[embedding_col]).rename(columns={text_col: "chunk_text"})
        metadata_batches.append(meta_batch)

        processed += len(batch_df)
        if progress:
            print(f"  Indexed {processed:,}/{total_rows:,} chunks...", end="\r")

    if progress:
        print()  # move past the \r-overwritten progress line

    metadata_df = pd.concat(metadata_batches, ignore_index=True)
    return index, metadata_df


def stream_build_index_to_disk(
    path: str,
    index_path: str,
    metadata_path: str,
    batch_rows: int = 50_000,
    normalize: bool = True,
    progress: bool = True,
) -> int:
    """Fully disk-streaming build for the real ~1.37M-row pre-built file.

    Unlike `build_faiss_index_from_parquet`, this never holds more than one
    batch's worth of metadata in memory -- each batch's metadata is written
    straight to `metadata_path` via a parquet writer instead of being
    accumulated in a Python list and concatenated at the end (that
    accumulate-then-concat pattern temporarily needs ~2x the full metadata
    size in memory during the final concat, which adds up across 1.37M rows
    of chunk text and can be a meaningful chunk of RAM on its own, on top of
    the ~2GB the FAISS index itself needs).

    Peak memory is roughly: one batch's embeddings + one batch's metadata +
    the FAISS index as it's built up (~2GB once complete for the full file --
    that part is unavoidable, since the index IS the 1.37M x 384 float
    vectors that need to live somewhere for exact search).

    Returns the number of vectors indexed.
    """
    import pyarrow as pa

    index = None
    writer = None
    processed = 0
    total_rows = pq.ParquetFile(path).metadata.num_rows

    try:
        for batch_df, batch_embeddings, text_col, embedding_col in _iter_parquet_batches(path, batch_rows):
            if index is None:
                _validate_embedding_dim(batch_embeddings)
                index = faiss.IndexFlatIP(EMBEDDING_DIM)
            if normalize:
                faiss.normalize_L2(batch_embeddings)
            index.add(batch_embeddings)

            meta_batch = batch_df.drop(columns=[embedding_col]).rename(columns={text_col: "chunk_text"})
            table = pa.Table.from_pandas(meta_batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(metadata_path, table.schema)
            else:
                # Later batches occasionally infer a slightly different type
                # for a column (e.g. int64 vs float64 if a field is null in
                # one batch but not another) -- cast to the schema the writer
                # was opened with so write_table doesn't reject the batch.
                table = table.cast(writer.schema)
            writer.write_table(table)

            processed += len(batch_df)
            if progress:
                print(f"  Indexed {processed:,}/{total_rows:,} chunks...", end="\r")
    finally:
        if writer is not None:
            writer.close()

    if progress:
        print()

    faiss.write_index(index, index_path)
    return index.ntotal


def _validate_embedding_dim(batch_embeddings: np.ndarray) -> None:
    if batch_embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EMBEDDING_DIM}-dim embeddings, got shape {batch_embeddings.shape}. "
            "Check that this parquet was built with all-MiniLM-L6-v2."
        )


def _expand_metadata_dict_column(series: pd.Series) -> pd.DataFrame:
    """Turn a column of per-row dicts (or JSON strings -- some exports
    serialize the dict as text rather than a native struct) into one column
    per key. Every key found is kept, regardless of name, so this adapts
    automatically to whatever fields the export actually included."""
    import json

    def _to_dict(value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    parsed = series.apply(_to_dict)
    return pd.DataFrame(parsed.tolist(), index=series.index)


def _detect_columns(schema_names):
    """Detect text/embedding/id/metadata columns from the parquet schema
    alone (no data read yet). Returns (text_col, embedding_col, id_col,
    metadata_dict_col, flat_metadata_cols) -- metadata_dict_col is set when
    the export bundles all metadata into one dict-valued column (e.g. a
    ChromaDB-style id/document/embedding/metadata export); flat_metadata_cols
    is only populated as a fallback when there's no such column."""
    text_col = _find_column(schema_names, *TEXT_CANDIDATES)
    embedding_col = _find_column(schema_names, *EMBEDDING_CANDIDATES)

    try:
        id_col = _find_exact_column(schema_names, *ID_CANDIDATES)
    except ValueError:
        id_col = None

    try:
        metadata_dict_col = _find_column(schema_names, *METADATA_DICT_CANDIDATES)
    except ValueError:
        metadata_dict_col = None

    flat_metadata_cols = []
    if metadata_dict_col is None:
        for candidate in METADATA_CANDIDATES:
            try:
                flat_metadata_cols.append(_find_column(schema_names, candidate))
            except ValueError:
                pass  # field not present in this export -- skip gracefully

    return text_col, embedding_col, id_col, metadata_dict_col, flat_metadata_cols


def _iter_parquet_batches(path: str, batch_rows: int):
    """Yield (batch_df, batch_embeddings, text_col, embedding_col) per
    row-group batch, with column names auto-detected once from the schema.
    `batch_df` always has a `chunk_id` column (if an id/chunk_id column
    exists in the source) and, if the source bundled metadata into a single
    dict column, that column is expanded into one column per key."""
    parquet_file = pq.ParquetFile(path)
    schema_names = parquet_file.schema_arrow.names
    text_col, embedding_col, id_col, metadata_dict_col, flat_metadata_cols = _detect_columns(schema_names)

    read_columns = [text_col, embedding_col] + flat_metadata_cols
    if metadata_dict_col:
        read_columns.append(metadata_dict_col)
    if id_col:
        read_columns.append(id_col)
    # de-duplicate while preserving order, in case any name collides
    seen = set()
    read_columns = [c for c in read_columns if not (c in seen or seen.add(c))]

    for record_batch in parquet_file.iter_batches(batch_size=batch_rows, columns=read_columns):
        batch_df = record_batch.to_pandas()

        if metadata_dict_col:
            expanded = _expand_metadata_dict_column(batch_df[metadata_dict_col])
            batch_df = pd.concat([batch_df.drop(columns=[metadata_dict_col]), expanded], axis=1)

        if id_col and id_col != "chunk_id":
            batch_df = batch_df.rename(columns={id_col: "chunk_id"})

        batch_embeddings = np.asarray(batch_df[embedding_col].tolist(), dtype="float32")
        yield batch_df, batch_embeddings, text_col, embedding_col


def save_index_and_metadata(index: faiss.IndexFlatIP, df: pd.DataFrame, index_path: str, metadata_path: str):
    """Persist the FAISS index plus a metadata parquet. If `df` still has an
    `embedding` column (the in-memory build_faiss_index path), it's dropped
    first -- it's redundant once it's inside the FAISS index, and dropping
    it keeps the metadata file far smaller. Output of
    `build_faiss_index_from_parquet` already has no embedding column."""
    faiss.write_index(index, index_path)
    metadata_df = df.drop(columns=["embedding"], errors="ignore").reset_index(drop=True)
    metadata_df.to_parquet(metadata_path, index=False)


def load_index_and_metadata(index_path: str, metadata_path: str) -> tuple[faiss.IndexFlatIP, pd.DataFrame]:
    index = faiss.read_index(index_path)
    metadata_df = pd.read_parquet(metadata_path)
    if index.ntotal != len(metadata_df):
        raise ValueError(
            f"Index/metadata mismatch: FAISS index has {index.ntotal} vectors but "
            f"metadata has {len(metadata_df)} rows. Rebuild both together."
        )
    return index, metadata_df
