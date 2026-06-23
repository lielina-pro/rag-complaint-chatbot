"""Build the FAISS index for Task 3 from the pre-built `complaint_embeddings.parquet`
(the full ~464K-complaint / ~1.37M-chunk dataset provided for Tasks 3-4).

This does NOT re-embed anything -- it just loads the embeddings that are
already in the parquet and indexes them. The file is streamed in batches
(rather than loaded all at once) to keep memory usage manageable on
ordinary laptops -- see the module docstring in src/vector_index.py for why.

Usage:
    python src/build_faiss_index.py --input data/raw/complaint_embeddings.parquet
    python src/build_faiss_index.py --batch-size 20000   # lower this further if you still hit a memory error
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from vector_index import stream_build_index_to_disk

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "complaint_embeddings.parquet"
DEFAULT_INDEX_PATH = REPO_ROOT / "vector_store" / "full_dataset.faiss"
DEFAULT_METADATA_PATH = REPO_ROOT / "vector_store" / "full_dataset_metadata.parquet"
DEFAULT_BATCH_SIZE = 50_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--index-path", type=str, default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--metadata-path", type=str, default=str(DEFAULT_METADATA_PATH))
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Rows read/indexed per batch. Lower this (e.g. 10000 or 5000) if you "
             "still hit a memory error on your machine.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Streaming pre-built embeddings from {args.input} ...")
    print(f"(~1.37M rows total, processed and written in batches of {args.batch_size:,} "
          "rows at a time -- index and metadata are both written incrementally, "
          "so memory usage stays low throughout. This may still take a few minutes.)")
    t0 = time.time()
    n_vectors = stream_build_index_to_disk(
        args.input, args.index_path, args.metadata_path, batch_rows=args.batch_size
    )
    print(f"Indexed {n_vectors:,} vectors in {time.time() - t0:.1f}s.")
    print(f"Saved index to {args.index_path} and metadata to {args.metadata_path}.")
    print("Done.")


if __name__ == "__main__":
    main()
