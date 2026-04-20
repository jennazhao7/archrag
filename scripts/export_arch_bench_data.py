"""Export compact embedding artifacts for architecture benchmarks.

This script creates a deterministic package containing:
1) one query embedding
2) a centroid matrix (optionally downsampled)
3) an optional small document embedding matrix

Outputs are written as .npy or raw .bin (float32), plus metadata JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export compact benchmark data package for architecture simulation."
    )
    parser.add_argument(
        "--query-embeddings",
        default="data/processed/query_embeddings.npy",
        help="Path to query embeddings matrix (.npy).",
    )
    parser.add_argument(
        "--centroids",
        default="data/processed/cluster_centroids.npy",
        help="Path to centroid matrix (.npy).",
    )
    parser.add_argument(
        "--doc-embeddings",
        default="data/processed/doc_embeddings.npy",
        help="Path to document embeddings matrix (.npy).",
    )
    parser.add_argument(
        "--include-docs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to export a document embedding matrix sample.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/exported/arch_bench",
        help="Output directory for exported files.",
    )
    parser.add_argument(
        "--format",
        choices=("npy", "bin"),
        default="npy",
        help="Data file format for exported arrays.",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=0,
        help="Index of query embedding to export.",
    )
    parser.add_argument(
        "--centroid-count",
        type=int,
        default=0,
        help="Number of centroids to export (0 means all).",
    )
    parser.add_argument(
        "--doc-count",
        type=int,
        default=0,
        help="Number of document embeddings to export when --include-docs is enabled (0 means all).",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=("first", "random"),
        default="first",
        help="Downsampling strategy for centroid/doc rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic random downsampling.",
    )
    return parser


def _load_matrix(path: str | Path, *, name: str) -> np.ndarray:
    matrix = np.load(Path(path))
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D numpy array.")
    return np.asarray(matrix, dtype=np.float32)


def _validate_query_index(index: int, n_queries: int) -> None:
    if index < 0 or index >= n_queries:
        raise ValueError(f"query_index must be in [0, {n_queries - 1}], got {index}.")


def _sample_rows(
    matrix: np.ndarray,
    *,
    count: int,
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically sample rows and return (sampled_matrix, sampled_indices)."""
    n_rows = matrix.shape[0]
    if count <= 0 or count >= n_rows:
        indices = np.arange(n_rows, dtype=np.int32)
        return matrix.copy(), indices

    if strategy == "first":
        indices = np.arange(count, dtype=np.int32)
        return matrix[indices], indices

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_rows, size=count, replace=False).astype(np.int32))
    return matrix[indices], indices


def _save_array(array: np.ndarray, path_stem: Path, *, file_format: str) -> Path:
    if file_format == "npy":
        out_path = path_stem.with_suffix(".npy")
        np.save(out_path, array)
        return out_path

    out_path = path_stem.with_suffix(".bin")
    np.asarray(array, dtype=np.float32).tofile(out_path)
    return out_path


def _shape_list(array: np.ndarray) -> list[int]:
    return [int(x) for x in array.shape]


def main() -> int:
    args = _build_parser().parse_args()

    query_embeddings = _load_matrix(args.query_embeddings, name="query_embeddings")
    centroids = _load_matrix(args.centroids, name="centroids")
    docs = _load_matrix(args.doc_embeddings, name="doc_embeddings") if args.include_docs else None

    _validate_query_index(args.query_index, query_embeddings.shape[0])

    query_vector = np.asarray(query_embeddings[args.query_index], dtype=np.float32)[None, :]
    sampled_centroids, centroid_indices = _sample_rows(
        centroids,
        count=args.centroid_count,
        strategy=args.sample_strategy,
        seed=args.seed,
    )

    sampled_docs: np.ndarray | None = None
    doc_indices: np.ndarray | None = None
    if docs is not None:
        sampled_docs, doc_indices = _sample_rows(
            docs,
            count=args.doc_count,
            strategy=args.sample_strategy,
            seed=args.seed,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_path = _save_array(query_vector, output_dir / "query_embedding", file_format=args.format)
    centroid_path = _save_array(
        sampled_centroids,
        output_dir / "centroid_matrix",
        file_format=args.format,
    )
    doc_path: Path | None = None
    if sampled_docs is not None:
        doc_path = _save_array(
            sampled_docs,
            output_dir / "doc_embeddings_small",
            file_format=args.format,
        )

    metadata: dict[str, object] = {
        "export_type": "arch_bench_package",
        "format": args.format,
        "dtype": "float32",
        "sample_strategy": args.sample_strategy,
        "seed": int(args.seed),
        "query": {
            "source_path": str(Path(args.query_embeddings).resolve()),
            "source_count": int(query_embeddings.shape[0]),
            "selected_index": int(args.query_index),
            "shape": _shape_list(query_vector),
            "file": str(query_path.resolve()),
        },
        "centroids": {
            "source_path": str(Path(args.centroids).resolve()),
            "source_count": int(centroids.shape[0]),
            "export_count": int(sampled_centroids.shape[0]),
            "shape": _shape_list(sampled_centroids),
            "sampled_indices_file": None,
            "file": str(centroid_path.resolve()),
        },
        "docs": {
            "included": bool(sampled_docs is not None),
            "source_path": str(Path(args.doc_embeddings).resolve()) if args.include_docs else None,
            "source_count": int(docs.shape[0]) if docs is not None else 0,
            "export_count": int(sampled_docs.shape[0]) if sampled_docs is not None else 0,
            "shape": _shape_list(sampled_docs) if sampled_docs is not None else None,
            "sampled_indices_file": None,
            "file": str(doc_path.resolve()) if doc_path is not None else None,
        },
    }

    centroid_indices_path = output_dir / "centroid_indices.npy"
    np.save(centroid_indices_path, centroid_indices.astype(np.int32))
    metadata["centroids"]["sampled_indices_file"] = str(centroid_indices_path.resolve())  # type: ignore[index]

    if doc_indices is not None:
        doc_indices_path = output_dir / "doc_indices.npy"
        np.save(doc_indices_path, doc_indices.astype(np.int32))
        metadata["docs"]["sampled_indices_file"] = str(doc_indices_path.resolve())  # type: ignore[index]

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")

    print("Exported architecture benchmark package:")
    print(f"- query: {query_path.resolve()}")
    print(f"- centroids: {centroid_path.resolve()}")
    if doc_path is not None:
        print(f"- docs: {doc_path.resolve()}")
    print(f"- metadata: {metadata_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
