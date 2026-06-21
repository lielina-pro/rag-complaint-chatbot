import pandas as pd

from src.chunking import build_splitter, chunk_dataframe


def test_build_splitter_splits_long_text():
    splitter = build_splitter(chunk_size=50, chunk_overlap=10)
    text = "word " * 100  # 500 chars
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)  # allow a little slack for boundary search


def test_build_splitter_keeps_short_text_as_one_chunk():
    splitter = build_splitter(chunk_size=500, chunk_overlap=50)
    text = "This is a short complaint narrative."
    chunks = splitter.split_text(text)
    assert chunks == [text]


def test_chunk_dataframe_basic_structure():
    df = pd.DataFrame(
        {
            "complaint_id": [1, 2],
            "narrative": ["short text", "word " * 200],  # row 2 will be split
            "product_category": ["Credit Card", "Personal Loan"],
        }
    )
    chunks_df = chunk_dataframe(
        df,
        text_col="narrative",
        id_col="complaint_id",
        metadata_cols=["product_category"],
        chunk_size=100,
        chunk_overlap=20,
    )

    # complaint 1 -> 1 chunk, complaint 2 -> multiple chunks
    assert (chunks_df["complaint_id"] == 1).sum() == 1
    assert (chunks_df["complaint_id"] == 2).sum() > 1

    # chunk_id uniqueness
    assert chunks_df["chunk_id"].is_unique

    # metadata carried through correctly
    row1 = chunks_df[chunks_df["complaint_id"] == 1].iloc[0]
    assert row1["product_category"] == "Credit Card"

    # total_chunks matches actual count per complaint
    for cid, group in chunks_df.groupby("complaint_id"):
        assert (group["total_chunks"] == len(group)).all()
        assert sorted(group["chunk_index"].tolist()) == list(range(len(group)))


def test_chunk_dataframe_skips_empty_text():
    df = pd.DataFrame(
        {
            "complaint_id": [1, 2, 3],
            "narrative": ["valid text here", "", None],
            "product_category": ["Credit Card", "Credit Card", "Credit Card"],
        }
    )
    chunks_df = chunk_dataframe(
        df, text_col="narrative", id_col="complaint_id", metadata_cols=["product_category"]
    )
    assert set(chunks_df["complaint_id"]) == {1}
