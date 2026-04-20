"""Export minimal architecture benchmark data for plaintext kernels.

Produces a compact deterministic package:
- query.npy      shape: [d]
- db.npy         shape: [N, d]
- centroids.npy  shape: [K, d]
- metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export small deterministic architecture benchmark arrays."
    )
    parser.add_argument(
        "--doc-embeddings",
        default="data/processed/doc_embeddings.npy",
        help="Path to full document embedding matrix.",
    )
    parser.add_argument(
        "--query-embeddings",
        default="data/processed/query_embeddings.npy",
        help="Path to full query embedding matrix.",
    )
    parser.add_argument(
        "--centroids",
        default="data/processed/cluster_centroids.npy",
        help="Path to full centroid matrix.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/exported/arch_minimal",
        help="Directory to write exported arrays and metadata.",
    )
    parser.add_argument(
        "--n-docs",
        type=int,
        default=512,
        help="Number of DB vectors to export.",
    )
    parser.add_argument(
        "--k-centroids",
        type=int,
        default=64,
        help="Number of centroids to export.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Embedding dimension to export (prefix slice).",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=0,
        help="Query index in query embedding matrix.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for row downsampling.",
    )
    return parser


def _load_matrix(path: str | Path, *, name: str) -> np.ndarray:
    matrix = np.load(Path(path))
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    return np.asarray(matrix, dtype=np.float32)


def _validate_inputs(
    docs: np.ndarray,
    queries: np.ndarray,
    centroids: np.ndarray,
    *,
    n_docs: int,
    k_centroids: int,
    dim: int,
    query_index: int,
) -> None:
    if dim <= 0:
        raise ValueError("dim must be > 0.")
    if n_docs <= 0:
        raise ValueError("n_docs must be > 0.")
    if k_centroids <= 0:
        raise ValueError("k_centroids must be > 0.")
    if query_index < 0 or query_index >= queries.shape[0]:
        raise ValueError(f"query_index must be in [0, {queries.shape[0] - 1}].")

    full_dim = docs.shape[1]
    if queries.shape[1] != full_dim or centroids.shape[1] != full_dim:
        raise ValueError("docs, queries, and centroids must have matching dimensions.")
    if dim > full_dim:
        raise ValueError(f"dim ({dim}) cannot exceed available embedding dimension ({full_dim}).")
    if n_docs > docs.shape[0]:
        raise ValueError(f"n_docs ({n_docs}) cannot exceed docs count ({docs.shape[0]}).")
    if k_centroids > centroids.shape[0]:
        raise ValueError(
            f"k_centroids ({k_centroids}) cannot exceed centroid count ({centroids.shape[0]})."
        )


def _sample_rows(matrix: np.ndarray, *, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(matrix.shape[0], size=count, replace=False).astype(np.int32))
    return matrix[indices], indices


def main() -> int:
    args = _build_parser().parse_args()

    docs = _load_matrix(args.doc_embeddings, name="doc_embeddings")
    queries = _load_matrix(args.query_embeddings, name="query_embeddings")
    centroids = _load_matrix(args.centroids, name="centroids")

    _validate_inputs(
        docs,
        queries,
        centroids,
        n_docs=args.n_docs,
        k_centroids=args.k_centroids,
        dim=args.dim,
        query_index=args.query_index,
    )

    docs_small, doc_indices = _sample_rows(docs, count=args.n_docs, seed=args.seed)
    centroids_small, centroid_indices = _sample_rows(
        centroids, count=args.k_centroids, seed=args.seed + 1
    )
    query = queries[args.query_index]

    # Slice dimension prefix for compact export.
    docs_small = np.asarray(docs_small[:, : args.dim], dtype=np.float32)
    centroids_small = np.asarray(centroids_small[:, : args.dim], dtype=np.float32)
    query = np.asarray(query[: args.dim], dtype=np.float32)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    query_path = out_dir / "query.npy"
    db_path = out_dir / "db.npy"
    centroids_path = out_dir / "centroids.npy"
    metadata_path = out_dir / "metadata.json"

    np.save(query_path, query)
    np.save(db_path, docs_small)
    np.save(centroids_path, centroids_small)

    metadata = {
        "dataset": "arch_minimal_benchmark",
        "seed": int(args.seed),
        "source": {
            "doc_embeddings": str(Path(args.doc_embeddings).resolve()),
            "query_embeddings": str(Path(args.query_embeddings).resolve()),
            "centroids": str(Path(args.centroids).resolve()),
        },
        "shape": {
            "N": int(docs_small.shape[0]),
            "K": int(centroids_small.shape[0]),
            "d": int(args.dim),
            "query_shape": [int(x) for x in query.shape],
            "db_shape": [int(x) for x in docs_small.shape],
            "centroids_shape": [int(x) for x in centroids_small.shape],
        },
        "selection": {
            "query_index": int(args.query_index),
            "doc_indices_file": str((out_dir / "doc_indices.npy").resolve()),
            "centroid_indices_file": str((out_dir / "centroid_indices.npy").resolve()),
        },
        "files": {
            "query": str(query_path.resolve()),
            "db": str(db_path.resolve()),
            "centroids": str(centroids_path.resolve()),
        },
        "dtype": "float32",
        "layout": "row-major",
    }

    np.save(out_dir / "doc_indices.npy", doc_indices)
    np.save(out_dir / "centroid_indices.npy", centroid_indices)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")

    print("Exported minimal architecture data:")
    print(f"- {query_path.resolve()}")
    print(f"- {db_path.resolve()}")
    print(f"- {centroids_path.resolve()}")
    print(f"- {metadata_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
