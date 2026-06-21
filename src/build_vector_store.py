"""Task 2 — Text Chunking, Embedding, and Vector Store Indexing.

Pipeline:
    data/processed/filtered_complaints.csv  (Task 1 output)
        -> stratified sample (~10K-15K complaints)
        -> chunk each narrative (RecursiveCharacterTextSplitter)
        -> embed each chunk (sentence-transformers/all-MiniLM-L6-v2)
        -> persist to a ChromaDB collection in vector_store/
        -> also save chunks + embeddings + metadata to
           data/processed/sampled_chunks_with_embeddings.parquet
           (handy if you'd rather build your own FAISS index)

Usage:
    python src/build_vector_store.py
    python src/build_vector_store.py --sample-size 12000 --chunk-size 500 --chunk-overlap 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_dataframe
from embedding import DEFAULT_MODEL_NAME, Embedder
from sampling import stratified_sample, summarize_sampling
from vector_store import build_chroma_store, metadatas_from_dataframe

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "filtered_complaints.csv"
DEFAULT_SAMPLE_OUTPUT = REPO_ROOT / "data" / "processed" / "sampled_chunks_with_embeddings.parquet"
DEFAULT_VECTOR_STORE_DIR = REPO_ROOT / "vector_store"
COLLECTION_NAME = "complaint_chunks"

# Metadata carried through from the complaint level into every chunk record.
# Only columns that actually exist in the input CSV are kept (see main()).
CANDIDATE_METADATA_COLS = [
    "product_category",
    "product",
    "issue",
    "company",
    "state",
    "date_received",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--vector-store-dir", type=str, default=str(DEFAULT_VECTOR_STORE_DIR))
    parser.add_argument("--collection-name", type=str, default=COLLECTION_NAME)
    parser.add_argument("--parquet-output", type=str, default=str(DEFAULT_SAMPLE_OUTPUT))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[1/5] Loading cleaned dataset from {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"      {len(df):,} rows loaded.")

    metadata_cols = [c for c in CANDIDATE_METADATA_COLS if c in df.columns]
    if "product_category" not in metadata_cols:
        raise ValueError(
            "Expected a 'product_category' column (produced by the Task 1 "
            "notebook) in the input CSV."
        )

    print(f"\n[2/5] Building stratified sample (target n={args.sample_size:,}) ...")
    sample_df = stratified_sample(
        df, strat_col="product_category", target_n=args.sample_size, random_state=args.random_state
    )
    print(f"      Sampled {len(sample_df):,} rows.")
    print(summarize_sampling(df, sample_df, "product_category"))

    text_col = "clean_narrative" if "clean_narrative" in sample_df.columns else "consumer_complaint_narrative"
    id_col = next((c for c in sample_df.columns if "complaint_id" in c), sample_df.columns[0])

    print(
        f"\n[3/5] Chunking narratives "
        f"(chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}) ..."
    )
    t0 = time.time()
    chunks_df = chunk_dataframe(
        sample_df,
        text_col=text_col,
        id_col=id_col,
        metadata_cols=metadata_cols,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"      {len(chunks_df):,} chunks created from {len(sample_df):,} complaints "
          f"({len(chunks_df) / len(sample_df):.2f} chunks/complaint avg) in {time.time() - t0:.1f}s.")

    print(f"\n[4/5] Embedding chunks with {args.embedding_model} ...")
    t0 = time.time()
    embedder = Embedder(args.embedding_model)
    embeddings = embedder.embed(chunks_df["chunk_text"].tolist())
    print(f"      Embedded {len(chunks_df):,} chunks -> shape {embeddings.shape} in {time.time() - t0:.1f}s.")

    print(f"\n[5/5] Persisting vector store to {args.vector_store_dir} ...")
    metadata_records = metadatas_from_dataframe(
        chunks_df, columns=["complaint_id", "chunk_index", "total_chunks"] + metadata_cols
    )
    build_chroma_store(
        persist_dir=args.vector_store_dir,
        collection_name=args.collection_name,
        chunk_ids=chunks_df["chunk_id"].tolist(),
        embeddings=embeddings,
        documents=chunks_df["chunk_text"].tolist(),
        metadatas=metadata_records,
    )
    print(f"      Collection '{args.collection_name}' persisted "
          f"with {len(chunks_df):,} chunks.")

    # Also save a flat parquet (chunk text + metadata + embedding vector) for
    # anyone who'd rather build a FAISS index instead of using Chroma.
    out_df = chunks_df.copy()
    out_df["embedding"] = list(np.asarray(embeddings))
    out_df.to_parquet(args.parquet_output, index=False)
    print(f"      Also saved flat chunk/embedding table to {args.parquet_output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
