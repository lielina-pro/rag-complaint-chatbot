import numpy as np
import pandas as pd
import pytest

from src.vector_index import (
    build_faiss_index,
    save_index_and_metadata,
    load_index_and_metadata,
    build_faiss_index_from_parquet,
    stream_build_index_to_disk,
)


def _make_synthetic_df(n=50, dim=384, seed=0):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, dim)).astype("float32")
    return pd.DataFrame({
        "complaint_id": range(n),
        "chunk_text": [f"complaint text {i}" for i in range(n)],
        "product_category": ["Credit Card"] * n,
        "embedding": list(embeddings),
    })


def test_build_faiss_index_returns_correct_count():
    df = _make_synthetic_df(n=50)
    index = build_faiss_index(df)
    assert index.ntotal == 50
    assert index.d == 384


def test_build_faiss_index_rejects_wrong_dimension():
    df = _make_synthetic_df(n=10, dim=128)
    with pytest.raises(ValueError):
        build_faiss_index(df)


def test_nearest_neighbor_of_a_vector_is_itself():
    df = _make_synthetic_df(n=50)
    index = build_faiss_index(df)

    query = np.vstack(df["embedding"].to_numpy()).astype("float32")[5:6]
    import faiss
    faiss.normalize_L2(query)
    scores, indices = index.search(query, 1)

    assert indices[0][0] == 5
    assert scores[0][0] > 0.999  # cosine similarity with itself ~= 1.0


def test_save_and_load_index_and_metadata_roundtrip(tmp_path):
    df = _make_synthetic_df(n=20)
    index = build_faiss_index(df)

    index_path = str(tmp_path / "test.faiss")
    metadata_path = str(tmp_path / "test_meta.parquet")
    save_index_and_metadata(index, df, index_path, metadata_path)

    loaded_index, loaded_metadata = load_index_and_metadata(index_path, metadata_path)
    assert loaded_index.ntotal == 20
    assert "embedding" not in loaded_metadata.columns
    assert list(loaded_metadata["complaint_id"]) == list(range(20))


def test_load_index_metadata_mismatch_raises(tmp_path):
    df = _make_synthetic_df(n=20)
    index = build_faiss_index(df)
    index_path = str(tmp_path / "test.faiss")
    metadata_path = str(tmp_path / "test_meta.parquet")
    save_index_and_metadata(index, df, index_path, metadata_path)

    # Corrupt the metadata to have fewer rows than the index
    bad_metadata = pd.read_parquet(metadata_path).iloc[:5]
    bad_metadata.to_parquet(metadata_path, index=False)

    with pytest.raises(ValueError):
        load_index_and_metadata(index_path, metadata_path)


def _write_synthetic_parquet(path, n=130, dim=384, text_col="chunk_text", embedding_col="embedding", seed=0):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, dim)).astype("float32")
    df = pd.DataFrame({
        "complaint_id": range(n),
        text_col: [f"complaint text {i}" for i in range(n)],
        "product_category": ["Credit Card"] * n,
        embedding_col: list(embeddings),
    })
    df.to_parquet(path, index=False)
    return df


def test_streaming_build_matches_in_memory_build(tmp_path):
    """Streaming in small batches (forcing multiple batches across a file
    smaller than one batch) should produce the same index contents as the
    plain in-memory build_faiss_index."""
    parquet_path = str(tmp_path / "embeddings.parquet")
    df = _write_synthetic_parquet(parquet_path, n=130)

    in_memory_index = build_faiss_index(df)
    streamed_index, metadata_df = build_faiss_index_from_parquet(parquet_path, batch_rows=50, progress=False)

    assert streamed_index.ntotal == in_memory_index.ntotal == 130
    assert len(metadata_df) == 130
    assert "embedding" not in metadata_df.columns
    assert "chunk_text" in metadata_df.columns

    # Same query should retrieve the same nearest neighbor from both indexes.
    query = np.vstack(df["embedding"].to_numpy()).astype("float32")[42:43]
    import faiss
    faiss.normalize_L2(query)

    _, idx_in_memory = in_memory_index.search(query, 1)
    _, idx_streamed = streamed_index.search(query, 1)
    assert idx_in_memory[0][0] == idx_streamed[0][0] == 42


def test_streaming_build_handles_uneven_batch_boundary(tmp_path):
    """130 rows with batch_rows=50 means batches of 50/50/30 -- make sure the
    last partial batch is handled correctly and no rows are dropped."""
    parquet_path = str(tmp_path / "embeddings.parquet")
    _write_synthetic_parquet(parquet_path, n=130)

    index, metadata_df = build_faiss_index_from_parquet(parquet_path, batch_rows=50, progress=False)
    assert index.ntotal == 130
    assert len(metadata_df) == 130
    assert list(metadata_df["complaint_id"]) == list(range(130))


def test_streaming_build_detects_alternate_column_names(tmp_path):
    """The real pre-built file might name columns slightly differently
    (e.g. 'text' instead of 'chunk_text', 'vector' instead of 'embedding') --
    confirm those are still picked up correctly."""
    parquet_path = str(tmp_path / "embeddings.parquet")
    _write_synthetic_parquet(parquet_path, n=20, text_col="text", embedding_col="vector")

    index, metadata_df = build_faiss_index_from_parquet(parquet_path, batch_rows=50, progress=False)
    assert index.ntotal == 20
    assert "chunk_text" in metadata_df.columns  # always normalized to this name
    assert "vector" not in metadata_df.columns


def test_streaming_build_rejects_wrong_dimension(tmp_path):
    parquet_path = str(tmp_path / "embeddings.parquet")
    _write_synthetic_parquet(parquet_path, n=10, dim=128)

    with pytest.raises(ValueError):
        build_faiss_index_from_parquet(parquet_path, batch_rows=50, progress=False)


def test_streaming_build_save_and_reload_roundtrip(tmp_path):
    parquet_path = str(tmp_path / "embeddings.parquet")
    _write_synthetic_parquet(parquet_path, n=60)

    index, metadata_df = build_faiss_index_from_parquet(parquet_path, batch_rows=25, progress=False)

    index_path = str(tmp_path / "out.faiss")
    metadata_path = str(tmp_path / "out_meta.parquet")
    save_index_and_metadata(index, metadata_df, index_path, metadata_path)

    loaded_index, loaded_metadata = load_index_and_metadata(index_path, metadata_path)
    assert loaded_index.ntotal == 60
    assert len(loaded_metadata) == 60


def test_stream_build_index_to_disk_roundtrip(tmp_path):
    parquet_path = str(tmp_path / "embeddings.parquet")
    _write_synthetic_parquet(parquet_path, n=130)

    index_path = str(tmp_path / "stream.faiss")
    metadata_path = str(tmp_path / "stream_meta.parquet")
    n_indexed = stream_build_index_to_disk(
        parquet_path, index_path, metadata_path, batch_rows=50, progress=False
    )
    assert n_indexed == 130

    loaded_index, loaded_metadata = load_index_and_metadata(index_path, metadata_path)
    assert loaded_index.ntotal == 130
    assert len(loaded_metadata) == 130
    assert "embedding" not in loaded_metadata.columns
    assert "chunk_text" in loaded_metadata.columns
    assert list(loaded_metadata["complaint_id"]) == list(range(130))


def test_stream_build_index_to_disk_matches_in_memory_results(tmp_path):
    """Same nearest-neighbor results whether built in-memory or streamed
    straight to disk -- confirms the disk-streaming path isn't silently
    producing a different (e.g. mis-ordered) index."""
    parquet_path = str(tmp_path / "embeddings.parquet")
    df = _write_synthetic_parquet(parquet_path, n=80)

    in_memory_index = build_faiss_index(df)

    index_path = str(tmp_path / "stream.faiss")
    metadata_path = str(tmp_path / "stream_meta.parquet")
    stream_build_index_to_disk(parquet_path, index_path, metadata_path, batch_rows=30, progress=False)
    streamed_index, _ = load_index_and_metadata(index_path, metadata_path)

    query = np.vstack(df["embedding"].to_numpy()).astype("float32")[17:18]
    import faiss
    faiss.normalize_L2(query)

    _, idx_in_memory = in_memory_index.search(query, 1)
    _, idx_streamed = streamed_index.search(query, 1)
    assert idx_in_memory[0][0] == idx_streamed[0][0] == 17


def _write_chromadb_style_parquet(path, n=130, dim=384, seed=0):
    """Mimics the real pre-built file's actual schema: id, document,
    embedding, metadata (a dict per row), discovered the hard way when the
    flat-column assumption above failed against the real file."""
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, dim)).astype("float32")
    categories = ["Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]
    df = pd.DataFrame({
        "id": [f"{i}_0" for i in range(n)],
        "document": [f"complaint text {i}" for i in range(n)],
        "embedding": list(embeddings),
        "metadata": [
            {
                "complaint_id": i,
                "product_category": categories[i % 4],
                "product": categories[i % 4],
                "company": "Acme Bank",
                "state": "CA",
            }
            for i in range(n)
        ],
    })
    df.to_parquet(path, index=False)
    return df


def test_chromadb_style_export_is_handled(tmp_path):
    """Regression test for the real pre-built file's actual schema (id /
    document / embedding / metadata-dict), which differs from the flat-
    column layout assumed everywhere else in this file."""
    parquet_path = str(tmp_path / "chromadb_style.parquet")
    _write_chromadb_style_parquet(parquet_path, n=130)

    index_path = str(tmp_path / "out.faiss")
    metadata_path = str(tmp_path / "out_meta.parquet")
    n_indexed = stream_build_index_to_disk(
        parquet_path, index_path, metadata_path, batch_rows=50, progress=False
    )
    assert n_indexed == 130

    loaded_index, loaded_metadata = load_index_and_metadata(index_path, metadata_path)
    assert loaded_index.ntotal == 130
    assert len(loaded_metadata) == 130

    # "document" must have been picked up as the text column and renamed.
    assert "chunk_text" in loaded_metadata.columns
    assert loaded_metadata["chunk_text"].iloc[0] == "complaint text 0"

    # The metadata dict must have been expanded into individual columns.
    assert "product_category" in loaded_metadata.columns
    assert "company" in loaded_metadata.columns
    assert set(loaded_metadata["product_category"].unique()) == {
        "Credit Card", "Personal Loan", "Savings Account", "Money Transfer"
    }

    # The top-level "id" column should survive, renamed to chunk_id.
    assert "chunk_id" in loaded_metadata.columns
    assert loaded_metadata["chunk_id"].iloc[0] == "0_0"


def test_chromadb_style_export_retrieval_correctness(tmp_path):
    """End-to-end sanity check: querying with a known row's own embedding
    should retrieve that exact row, same as the flat-column case."""
    parquet_path = str(tmp_path / "chromadb_style.parquet")
    df = _write_chromadb_style_parquet(parquet_path, n=80)

    index_path = str(tmp_path / "out.faiss")
    metadata_path = str(tmp_path / "out_meta.parquet")
    stream_build_index_to_disk(parquet_path, index_path, metadata_path, batch_rows=30, progress=False)
    index, metadata_df = load_index_and_metadata(index_path, metadata_path)

    query = np.vstack(df["embedding"].to_numpy()).astype("float32")[33:34]
    import faiss
    faiss.normalize_L2(query)
    scores, indices = index.search(query, 1)

    assert indices[0][0] == 33
    retrieved = metadata_df.iloc[indices[0][0]].to_dict()
    assert retrieved["product_category"] == df.iloc[33]["metadata"]["product_category"]
    assert retrieved["chunk_text"] == df.iloc[33]["document"]
